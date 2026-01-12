"""
VrunGPU Inference Server - Persistent LLM Service
Qwen3-8B 모델을 상시 로드하여 빠른 추론 제공

Usage:
    python inference_server.py [--port 9826] [--model Qwen/Qwen3-8B]
"""

import argparse
import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import uvicorn

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
    messages: list[ChatMessage] = Field(..., description="대화 메시지 목록")
    max_new_tokens: int = Field(default=512, description="생성할 최대 토큰 수")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    do_sample: bool = Field(default=False)


class GenerateResponse(BaseModel):
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int
    model: str
    elapsed_time: float


# ============================================================================
# Global Model State
# ============================================================================

model = None
tokenizer = None
model_name = None
device = None


def load_model(model_id: str):
    """모델 로드"""
    global model, tokenizer, model_name, device

    print(f"Loading model: {model_id}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left"
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()
    model_name = model_id

    print(f"Model loaded successfully!")
    print(f"Model dtype: {model.dtype}")


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
    """채팅 형식 추론"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = datetime.now()

    # 메시지를 dict 형식으로 변환
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

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


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VrunGPU Inference Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use (default: auto)")
    args = parser.parse_args()

    # GPU 설정
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")

    print("=" * 60)
    print("VrunGPU Inference Server")
    print("=" * 60)

    # 모델 로드
    load_model(args.model)

    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print("=" * 60)

    # 서버 시작
    uvicorn.run(app, host=args.host, port=args.port)
