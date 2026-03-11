"""
VrunGPU Inference Server - Persistent LLM Service
Qwen3-8B, Qwen2.5, DeepSeek-R1 등 다양한 LLM 지원

Usage:
    python inference_server.py [--port 9826] [--model Qwen/Qwen3-8B]
    python inference_server.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --load-in-4bit
"""

import argparse
import os
import uuid
import threading
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datetime import datetime, timedelta
import uvicorn

# Fix: Newer transformers calls model_info() in _patch_mistral_regex, which fails without internet.
# Wrap the classmethod to catch network errors and skip (we only use Qwen, not Mistral).
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PTTB
    _orig_pmr = _PTTB._patch_mistral_regex.__func__
    @classmethod
    def _safe_patch_mistral_regex(cls, tokenizer, *args, **kwargs):
        try:
            return _orig_pmr(cls, tokenizer, *args, **kwargs)
        except Exception:
            return tokenizer
    _PTTB._patch_mistral_regex = _safe_patch_mistral_regex
except Exception:
    pass

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_PORT = 9826

# ============================================================================
# Request/Response Models
# ============================================================================

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="입력 프롬프트")
    max_new_tokens: int = Field(default=512, description="생성할 최대 토큰 수")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="샘플링 온도")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p 샘플링")
    top_k: int = Field(default=50, ge=0, description="Top-k 샘플링")
    do_sample: bool = Field(default=False, description="샘플링 사용 여부 (GPU 호환성 문제로 기본값 False)")
    system_prompt: Optional[str] = Field(default=None, description="시스템 프롬프트")


class ChatMessage(BaseModel):
    role: str = Field(..., description="메시지 역할 (system/user/assistant)")
    content: str = Field(..., description="메시지 내용")


class ChatRequest(BaseModel):
    messages: list[ChatMessage] | None = Field(default=None, description="대화 메시지 목록 (session_id 미사용 시 필수)")
    message: str | None = Field(default=None, description="단일 메시지 (session_id 사용 시)")
    session_id: str | None = Field(default=None, description="세션 ID (멀티턴 대화용)")
    max_new_tokens: int = Field(default=512, description="생성할 최대 토큰 수")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    do_sample: bool = Field(default=False)
    system_prompt: str | None = Field(default=None, description="시스템 프롬프트 (세션 생성 시 설정)")
    ei_gate: bool = Field(default=False, description="E-I gate computation")


class GenerateResponse(BaseModel):
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    model: str
    elapsed_time: float
    session_id: str | None = None
    ei_info: dict | None = None


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    updated_at: str
    message_count: int
    total_tokens: int
    system_prompt: str | None = None


# ============================================================================
# Session Store
# ============================================================================

SESSION_TTL_HOURS = 4  # 세션 만료 시간

class SessionStore:
    """서버 사이드 대화 세션 관리"""

    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str | None = None, system_prompt: str | None = None) -> str:
        sid = session_id or str(uuid.uuid4())[:8]
        now = datetime.now()
        with self._lock:
            self._sessions[sid] = {
                "messages": [],
                "system_prompt": system_prompt,
                "created_at": now,
                "updated_at": now,
                "total_tokens": 0,
            }
        return sid

    def get(self, session_id: str) -> dict | None:
        with self._lock:
            return self._sessions.get(session_id)

    def add_message(self, session_id: str, role: str, content: str, tokens: int = 0):
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session["messages"].append({"role": role, "content": content})
            session["updated_at"] = datetime.now()
            session["total_tokens"] += tokens

    def get_messages(self, session_id: str) -> list[dict]:
        """세션의 전체 메시지를 반환 (system_prompt 포함)"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            msgs = []
            if session["system_prompt"]:
                msgs.append({"role": "system", "content": session["system_prompt"]})
            msgs.extend(session["messages"])
            return msgs

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> list[dict]:
        with self._lock:
            result = []
            for sid, s in self._sessions.items():
                result.append({
                    "session_id": sid,
                    "created_at": s["created_at"].isoformat(),
                    "updated_at": s["updated_at"].isoformat(),
                    "message_count": len(s["messages"]),
                    "total_tokens": s["total_tokens"],
                    "system_prompt": s["system_prompt"],
                })
            return result

    def cleanup_expired(self):
        """TTL 초과 세션 정리"""
        cutoff = datetime.now() - timedelta(hours=SESSION_TTL_HOURS)
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if s["updated_at"] < cutoff]
            for sid in expired:
                del self._sessions[sid]
            if expired:
                print(f"[Session] {len(expired)}개 만료 세션 정리: {expired}")

session_store = SessionStore()


# ============================================================================
# Global Model State
# ============================================================================

model = None
tokenizer = None
model_name = None
device = None
quantization_mode = None  # None, "4bit", "8bit"

# E-I Gate state (loaded to CPU)
ei_gate_weights = None


def load_ei_gates(checkpoint_dir: str):
    global ei_gate_weights
    import torch, json as _json
    from safetensors import safe_open

    cfg_path = os.path.join(checkpoint_dir, "ei_config.json")
    if not os.path.exists(cfg_path):
        print(f"[E-I] Config not found: {cfg_path}")
        return False

    with open(cfg_path) as f:
        cfg = _json.load(f)

    ei_layers = cfg.get("ei_enabled_layers", [])
    risk_gate_hidden = cfg.get("risk_gate_hidden", 256)

    st_path = os.path.join(checkpoint_dir, "ei_gates_only.safetensors")
    if not os.path.exists(st_path):
        st_path = os.path.join(checkpoint_dir, "ei_weights.safetensors")
    if not os.path.exists(st_path):
        print(f"[E-I] Weights not found in {checkpoint_dir}")
        return False

    gate_weights = {}
    alpha_key_map = {}
    with safe_open(st_path, framework="pt", device="cpu") as sf:
        all_keys = list(sf.keys())
        for li in ei_layers:
            w = {}
            prefix = f"{li}."
            for k in all_keys:
                if k.startswith(prefix):
                    w[k[len(prefix):]] = sf.get_tensor(k)
            if w:
                gate_weights[li] = w
        for k in all_keys:
            if "alpha" in k:
                parts = k.split(".")
                try:
                    li = int(parts[0])
                    alpha_key_map[li] = sf.get_tensor(k)
                except (ValueError, IndexError):
                    pass

    ei_gate_weights = {
        "weights": gate_weights,
        "layers": ei_layers,
        "risk_gate_hidden": risk_gate_hidden,
        "alpha_values": alpha_key_map,
    }
    total_params = sum(sum(t.numel() for t in w.values()) for w in gate_weights.values())
    print(f"[E-I] Gates loaded to CPU: {len(gate_weights)} layers, {total_params:,} params, hidden={risk_gate_hidden}")
    return True


def compute_ei_gates(input_text: str):
    import torch
    import torch.nn.functional as F

    if ei_gate_weights is None or model is None:
        return None

    ei_layers = ei_gate_weights["layers"]
    weights = ei_gate_weights["weights"]
    rgh = ei_gate_weights["risk_gate_hidden"]

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    display_tokens = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]

    captured = {}
    hooks = []
    for li in ei_layers:
        def make_hook(layer_idx):
            def fn(module, inp, out):
                h = inp[0] if isinstance(inp, tuple) else inp
                captured[layer_idx] = h.mean(dim=1).detach().cpu()
                return out
            return fn
        try:
            layer = model.model.layers[li].mlp
            hooks.append(layer.register_forward_hook(make_hook(li)))
        except (AttributeError, IndexError) as e:
            print(f"[E-I] Hook error layer {li}: {e}")

    dev = next(model.parameters()).device
    with torch.no_grad():
        model(input_ids=inputs["input_ids"].to(dev), attention_mask=inputs["attention_mask"].to(dev))

    for h in hooks:
        h.remove()

    gate_list, alpha_list = [], []
    risk_ctx = None
    for li in ei_layers:
        if li not in weights or li not in captured:
            continue
        h = captured[li]
        w = weights[li]
        has_ln = "risk_gate.gate_mlp.1.weight" in w
        has_ctx = "risk_gate.context_proj.weight" in w
        out_idx = 4 if has_ln else 3
        with torch.no_grad():
            if has_ctx:
                new_ctx = F.linear(h.to(w["risk_gate.context_proj.weight"].dtype), w["risk_gate.context_proj.weight"], w.get("risk_gate.context_proj.bias"))
                gi = torch.cat([h, risk_ctx if risk_ctx is not None else torch.zeros(1, rgh, dtype=h.dtype)], dim=-1)
            else:
                gi = h
            # Cast to match gate weight dtype
            target_dtype = w["risk_gate.gate_mlp.0.weight"].dtype
            gi = gi.to(target_dtype)
            x = F.linear(gi, w["risk_gate.gate_mlp.0.weight"], w["risk_gate.gate_mlp.0.bias"])
            if has_ln:
                x = F.layer_norm(x, [rgh], w["risk_gate.gate_mlp.1.weight"], w["risk_gate.gate_mlp.1.bias"])
            x = F.gelu(x)
            x = F.linear(x, w[f"risk_gate.gate_mlp.{out_idx}.weight"], w[f"risk_gate.gate_mlp.{out_idx}.bias"])
            g = torch.sigmoid(x)
            gate_list.append(round(g[0, 0].item(), 4))
            av = ei_gate_weights.get("alpha_values", {})
            alpha_list.append(round(av[li].item(), 4) if li in av else 0.1)
            if has_ctx:
                risk_ctx = new_ctx

    captured.clear()
    mean_gate = sum(gate_list) / max(len(gate_list), 1)
    return {
        "mean_gate": round(mean_gate, 4),
        "max_gate": max(gate_list) if gate_list else 0,
        "min_gate": min(gate_list) if gate_list else 0,
        "gate_per_layer": gate_list,
        "alpha_values": alpha_list,
        "ei_layers": ei_layers,
        "tokens": display_tokens,
        "token_gates": [[g for g in gate_list] for _ in display_tokens],
    }



def check_mxfp4_support() -> bool:
    """현재 GPU가 MXFP4를 지원하는지 확인 (compute capability >= 7.5)"""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major > 7) or (major == 7 and minor >= 5)


def is_mxfp4_model(model_id: str) -> bool:
    """모델 config에서 MXFP4 양자화 여부 자동 감지"""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        # quantization_config 확인
        if hasattr(config, 'quantization_config'):
            quant_config = config.quantization_config
            if isinstance(quant_config, dict):
                quant_method = quant_config.get('quant_method', '').lower()
                if 'mxfp' in quant_method or 'mx' in quant_method:
                    print(f"[Auto-Detect] MXFP4 quantization detected in {model_id}")
                    return True
            elif hasattr(quant_config, 'quant_method'):
                if 'mxfp' in str(quant_config.quant_method).lower():
                    print(f"[Auto-Detect] MXFP4 quantization detected in {model_id}")
                    return True
        return False
    except Exception as e:
        print(f"[Auto-Detect] Could not check model config: {e}")
        return False


def load_model(model_id: str, load_in_4bit: bool = False, load_in_8bit: bool = False, lora_adapter: str | None = None):
    """모델 로드 (4-bit/8-bit 양자화 지원)"""
    global model, tokenizer, model_name, device, quantization_mode

    print(f"Loading model: {model_id}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # MXFP4 모델 감지
    is_mxfp = is_mxfp4_model(model_id)
    mxfp_fallback = False
    if is_mxfp and device == "cuda" and not check_mxfp4_support():
        major, minor = torch.cuda.get_device_capability()
        print(f"[MXFP4] GPU compute capability {major}.{minor} < 7.5")
        print(f"[MXFP4] Will dequantize to BF16 using CPU (V100 compatible mode)")
        mxfp_fallback = True

    if mxfp_fallback:
        print("Quantization: MXFP4 → BF16 (CPU dequantize)")
        quantization_mode = "mxfp4_fallback"
    elif load_in_4bit:
        print("Quantization: 4-bit (bitsandbytes NF4)")
        quantization_mode = "4bit"
    elif load_in_8bit:
        print("Quantization: 8-bit (bitsandbytes)")
        quantization_mode = "8bit"
    else:
        print("Quantization: None (FP16)")
        quantization_mode = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left"
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 양자화 설정
    if mxfp_fallback and device == "cuda":
        # MXFP4 모델: V100 등 compute < 7.5 GPU에서 CPU dequantization 사용
        import gc
        import tempfile

        print("[MXFP4] Loading model (CPU dequantize → GPU)...")
        print("[MXFP4] Patching dequantizer for CPU execution...")

        # Monkey-patch: MXFP4 dequantization을 CPU에서 실행
        import transformers.integrations.mxfp4 as mxfp4_module
        import math

        FP4_VALUES = mxfp4_module.FP4_VALUES

        def cpu_only_convert_moe_packed_tensors(
            blocks,
            scales,
            *,
            dtype: torch.dtype = torch.bfloat16,
            rows_per_chunk: int = 32768 * 1024,
        ) -> torch.Tensor:
            """CPU에서만 MXFP4 dequantization 수행"""
            original_device = blocks.device
            blocks = blocks.cpu()
            scales = scales.cpu()
            scales = scales.to(torch.int32) - 127

            assert blocks.shape[:-1] == scales.shape, f"{blocks.shape[:-1]=} != {scales.shape=}"

            lut = torch.tensor(FP4_VALUES, dtype=dtype, device="cpu")
            *prefix_shape, G, B = blocks.shape
            rows_total = math.prod(prefix_shape) * G

            blocks = blocks.reshape(rows_total, B)
            scales = scales.reshape(rows_total, 1)
            out = torch.empty(rows_total, B * 2, dtype=dtype, device="cpu")

            for r0 in range(0, rows_total, rows_per_chunk):
                r1 = min(r0 + rows_per_chunk, rows_total)
                blk, exp = blocks[r0:r1], scales[r0:r1]
                idx_lo = (blk & 0x0F).to(torch.long)
                idx_hi = (blk >> 4).to(torch.long)
                sub = out[r0:r1]
                sub[:, 0::2] = lut[idx_lo]
                sub[:, 1::2] = lut[idx_hi]
                torch.ldexp(sub, exp, out=sub)
                del idx_lo, idx_hi, blk, exp, sub

            out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
            del blocks, scales, lut
            result = out.transpose(1, 2).contiguous()
            return result.to(original_device) if original_device.type == "cuda" else result

        mxfp4_module.convert_moe_packed_tensors = cpu_only_convert_moe_packed_tensors
        print("[MXFP4] Dequantizer patched")

        # 디스크 오프로딩 폴더
        offload_folder = tempfile.mkdtemp(prefix="vrungpu_offload_")
        print(f"[MXFP4] Offload folder: {offload_folder}")

        # 메모리를 최소화하여 로드
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder=offload_folder,
            offload_state_dict=True,
            max_memory={0: "28GiB", "cpu": "15GiB"},
        )
        print(f"[MXFP4] Model loaded successfully")

        # 메모리 정리
        gc.collect()
        torch.cuda.empty_cache()
    elif load_in_4bit and device == "cuda":
        # 일반 모델: bitsandbytes 4-bit 적용
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    elif load_in_8bit and device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    if device == "cpu":
        model = model.to(device)

    # LoRA 어댑터 로딩
    if lora_adapter:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {lora_adapter}")
        model = PeftModel.from_pretrained(model, lora_adapter)
        model = model.merge_and_unload()  # 추론 속도를 위해 merge
        print(f"LoRA adapter merged successfully")

    model.eval()
    model_name = model_id

    print(f"Model loaded successfully!")
    if hasattr(model, 'dtype'):
        print(f"Model dtype: {model.dtype}")
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1024**3
        mem_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="VrunGPU Inference Server",
    description="Persistent LLM inference service with Qwen3-8B",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """서버 상태 확인"""
    return {
        "service": "VrunGPU Inference Server",
        "status": "running",
        "model": model_name,
        "device": device,
        "quantization": quantization_mode,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@app.get("/health")
async def health():
    """헬스 체크"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": model_name}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """텍스트 생성"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = datetime.now()

    # 프롬프트 구성 - chat template 사용
    if request.system_prompt:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.prompt}
        ]
    else:
        messages = [{"role": "user", "content": request.prompt}]

    # apply_chat_template 사용
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 토크나이즈 (attention_mask 포함)
    inputs = tokenizer(full_prompt, return_tensors="pt", return_attention_mask=True).to(device)
    prompt_tokens = inputs.input_ids.shape[1]

    # 생성
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature if request.do_sample else 1.0,
            top_p=request.top_p if request.do_sample else 1.0,
            top_k=request.top_k if request.do_sample else 0,
            do_sample=request.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 디코드
    generated_ids = outputs[0][prompt_tokens:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_tokens = len(generated_ids)

    elapsed_time = (datetime.now() - start_time).total_seconds()

    return GenerateResponse(
        generated_text=generated_text,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        total_tokens=prompt_tokens + generated_tokens,
        model=model_name,
        elapsed_time=elapsed_time
    )


@app.post("/chat", response_model=GenerateResponse)
async def chat(request: ChatRequest):
    """채팅 형식 추론 (세션 지원)

    사용법:
    1. Stateless (기존): messages 배열 직접 전달
    2. 세션 시작: message + session_id (없으면 자동 생성)
    3. 세션 이어가기: message + session_id (기존 세션)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 만료 세션 정리
    session_store.cleanup_expired()

    start_time = datetime.now()
    sid = request.session_id

    # 세션 모드: message (단일 메시지) + session_id
    if request.message is not None:
        # 세션이 없으면 새로 생성
        if sid is None or session_store.get(sid) is None:
            sid = session_store.create(sid, request.system_prompt)
            print(f"[Session] 새 세션 생성: {sid}")
        # 사용자 메시지 추가
        session_store.add_message(sid, "user", request.message)
        messages = session_store.get_messages(sid)

    # Stateless 모드: messages 배열 직접 전달
    elif request.messages is not None:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    else:
        raise HTTPException(status_code=400, detail="message 또는 messages 중 하나는 필수입니다.")

    # apply_chat_template 사용
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 토크나이즈 (attention_mask 포함)
    inputs = tokenizer(full_prompt, return_tensors="pt", return_attention_mask=True).to(device)
    prompt_tokens = inputs.input_ids.shape[1]

    # 생성
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature if request.do_sample else 1.0,
            top_p=request.top_p if request.do_sample else 1.0,
            top_k=request.top_k if request.do_sample else 0,
            do_sample=request.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 디코드
    generated_ids = outputs[0][prompt_tokens:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_tokens = len(generated_ids)

    # 세션에 assistant 응답 저장
    if sid and session_store.get(sid) is not None:
        session_store.add_message(sid, "assistant", generated_text, prompt_tokens + generated_tokens)


    # E-I gate computation
    ei_info = None
    if request.ei_gate and ei_gate_weights is not None:
        try:
            last_user_msg = None
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    last_user_msg = msg["content"]
                    break
            if last_user_msg:
                ei_info = compute_ei_gates(last_user_msg)
        except Exception as e:
            print(f"[E-I] Gate computation error: {e}")
            ei_info = {"error": str(e)}
    elapsed_time = (datetime.now() - start_time).total_seconds()

    return GenerateResponse(
        generated_text=generated_text,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        total_tokens=prompt_tokens + generated_tokens,
        model=model_name,
        elapsed_time=elapsed_time,
        session_id=sid,
        ei_info=ei_info,
    )


# ============================================================================
# Session Management Endpoints
# ============================================================================

@app.get("/sessions")
async def list_sessions():
    """활성 세션 목록 조회"""
    session_store.cleanup_expired()
    return {"sessions": session_store.list_sessions()}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """세션 상세 조회 (대화 이력 포함)"""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"세션 '{session_id}'을(를) 찾을 수 없습니다.")
    return {
        "session_id": session_id,
        "system_prompt": session["system_prompt"],
        "messages": session["messages"],
        "created_at": session["created_at"].isoformat(),
        "updated_at": session["updated_at"].isoformat(),
        "message_count": len(session["messages"]),
        "total_tokens": session["total_tokens"],
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    if session_store.delete(session_id):
        return {"message": f"세션 '{session_id}' 삭제 완료"}
    raise HTTPException(status_code=404, detail=f"세션 '{session_id}'을(를) 찾을 수 없습니다.")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VrunGPU Inference Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (default: auto)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit quantization (saves VRAM)")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit quantization")
    parser.add_argument("--lora-adapter", type=str, default=None, help="LoRA adapter path (fine-tuned model)")
    parser.add_argument("--ei-checkpoint", type=str, default=None, help="E-I gate checkpoint dir")
    args = parser.parse_args()

    # GPU 설정
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")

    print("=" * 60)
    print("VrunGPU Inference Server")
    print("=" * 60)

    # 모델 로드
    load_model(args.model, load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit, lora_adapter=args.lora_adapter)

    # E-I gate loading (CPU)
    if args.ei_checkpoint:
        ok = load_ei_gates(args.ei_checkpoint)
        if ok:
            print(f"[E-I] Gates loaded from: {args.ei_checkpoint}")
        else:
            print(f"[E-I] WARNING: Failed to load from: {args.ei_checkpoint}")

    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print("=" * 60)

    # 서버 시작
    uvicorn.run(app, host=args.host, port=args.port)
