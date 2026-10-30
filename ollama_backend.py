"""
Ollama 백엔드 — vrungpu LLM API(/llm/*)를 Ollama 서버로 중계

model 이름이 "ollama:" 로 시작하면 inference_server.py 프로세스 대신 이 백엔드를 사용한다.
  예) POST /llm/start?model=ollama:qwen3.8:27b&gpu=1

- Ollama 서버는 별도 서비스(systemd)로 떠 있고, 특정 GPU에 고정되어 있다 (CUDA_VISIBLE_DEVICES).
  vrungpu는 그 GPU를 gpu_pool에 예약해 학습 작업과의 충돌을 막는다.
- Ollama에는 세션 개념이 없으므로 멀티턴 세션은 여기서 관리한다 (inference_server와 동일한 형식).
- 응답은 inference_server의 GenerateResponse 형식에 thinking / done_reason / backend 필드를 더해 반환한다.
"""

import os
import threading
import time
import uuid
from datetime import datetime, timedelta

import httpx

OLLAMA_PREFIX = "ollama:"
OLLAMA_URL = os.environ.get("VRUNGPU_OLLAMA_URL", "http://127.0.0.1:11434")
# Ollama 서비스가 고정된 GPU (ollama.service 의 CUDA_VISIBLE_DEVICES 와 일치해야 함)
OLLAMA_GPU = int(os.environ.get("VRUNGPU_OLLAMA_GPU", "1"))
# 요청당 컨텍스트 길이 (KV 캐시 메모리 상한)
OLLAMA_NUM_CTX = int(os.environ.get("VRUNGPU_OLLAMA_NUM_CTX", "32768"))
# thinking 모델은 한 응답에 수 분이 걸릴 수 있음
OLLAMA_REQUEST_TIMEOUT = float(os.environ.get("VRUNGPU_OLLAMA_TIMEOUT", "1800"))
OLLAMA_LOAD_TIMEOUT = float(os.environ.get("VRUNGPU_OLLAMA_LOAD_TIMEOUT", "600"))

SESSION_TTL_HOURS = 4  # inference_server 와 동일


def is_ollama_model(name: str | None) -> bool:
    return bool(name) and name.startswith(OLLAMA_PREFIX)


def strip_prefix(name: str) -> str:
    return name[len(OLLAMA_PREFIX):] if is_ollama_model(name) else name


class OllamaError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# ============================================================================
# Session Store (inference_server.SessionStore 와 같은 형식)
# ============================================================================

class _SessionStore:
    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()

    def create(self, session_id: str | None = None, system_prompt: str | None = None) -> str:
        sid = session_id or str(uuid.uuid4())[:8]
        now = datetime.now()
        with self._lock:
            self._sessions[sid] = {"messages": [], "system_prompt": system_prompt,
                                   "created_at": now, "updated_at": now, "total_tokens": 0}
        return sid

    def get(self, session_id: str) -> dict | None:
        with self._lock:
            return self._sessions.get(session_id)

    def add_message(self, session_id: str, role: str, content: str, tokens: int = 0):
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return
            s["messages"].append({"role": role, "content": content})
            s["updated_at"] = datetime.now()
            s["total_tokens"] += tokens

    def get_messages(self, session_id: str) -> list[dict]:
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return []
            msgs = [{"role": "system", "content": s["system_prompt"]}] if s["system_prompt"] else []
            return msgs + list(s["messages"])

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def info(self, sid: str, s: dict) -> dict:
        return {"session_id": sid, "created_at": s["created_at"].isoformat(),
                "updated_at": s["updated_at"].isoformat(), "message_count": len(s["messages"]),
                "total_tokens": s["total_tokens"], "system_prompt": s["system_prompt"]}

    def list_sessions(self) -> list[dict]:
        with self._lock:
            return [self.info(sid, s) for sid, s in self._sessions.items()]

    def detail(self, session_id: str) -> dict | None:
        with self._lock:
            s = self._sessions.get(session_id)
            if s is None:
                return None
            return {**self.info(session_id, s), "messages": list(s["messages"])}

    def cleanup_expired(self):
        cutoff = datetime.now() - timedelta(hours=SESSION_TTL_HOURS)
        with self._lock:
            for sid in [sid for sid, s in self._sessions.items() if s["updated_at"] < cutoff]:
                del self._sessions[sid]


# ============================================================================
# Ollama Instance — server.LLMInstance 와 같은 속성을 제공
# ============================================================================

class OllamaInstance:
    """GPU 한 대에 매핑된 Ollama 모델 (프로세스는 Ollama 서비스가 관리)"""
    is_ollama = True

    def __init__(self, model_name: str, gpu_id: int):
        self.model_name = model_name          # "ollama:qwen3.8:27b"
        self.ollama_model = strip_prefix(model_name)
        self.gpu_id = gpu_id
        self.port = int(OLLAMA_URL.rsplit(":", 1)[-1])
        self.process = None
        self.active = True
        self.sessions = _SessionStore()

    @property
    def alive(self) -> bool:
        return self.active

    @property
    def url(self) -> str:
        return OLLAMA_URL

    # ---- lifecycle ---------------------------------------------------------

    async def load(self):
        """모델 존재 확인 후 메모리에 상주시킴 (keep_alive=-1)"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                r = await client.post(f"{OLLAMA_URL}/api/show", json={"model": self.ollama_model})
            except httpx.ConnectError:
                raise OllamaError(503, f"Ollama 서버({OLLAMA_URL})에 연결할 수 없습니다.")
            if r.status_code == 404:
                raise OllamaError(404, f"Ollama에 '{self.ollama_model}' 모델이 없습니다. 먼저 `ollama pull {self.ollama_model}` 을 실행하세요.")
            if r.status_code != 200:
                raise OllamaError(r.status_code, r.text)
        async with httpx.AsyncClient(timeout=OLLAMA_LOAD_TIMEOUT) as client:
            r = await client.post(f"{OLLAMA_URL}/api/generate",
                                  json={"model": self.ollama_model, "keep_alive": -1,
                                        "options": {"num_ctx": OLLAMA_NUM_CTX}})
            if r.status_code != 200:
                raise OllamaError(r.status_code, f"모델 로드 실패: {r.text}")

    def unload(self):
        """모델을 GPU에서 내림 (동기 — _stop_llm_instance 에서 호출)"""
        self.active = False
        try:
            httpx.post(f"{OLLAMA_URL}/api/generate",
                       json={"model": self.ollama_model, "keep_alive": 0}, timeout=30.0)
        except Exception as e:
            print(f"[LLM] Ollama 모델 언로드 실패 ({self.ollama_model}): {e}")

    async def health(self) -> bool:
        """모델이 Ollama에 실제로 올라가 있는지 (/api/ps)"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{OLLAMA_URL}/api/ps")
                if r.status_code != 200:
                    return False
                return any(m.get("name") == self.ollama_model or m.get("model") == self.ollama_model
                           for m in r.json().get("models", []))
        except Exception:
            return False

    # ---- inference ---------------------------------------------------------

    @staticmethod
    def _options(payload: dict) -> dict:
        opts = {"num_ctx": OLLAMA_NUM_CTX, "num_predict": payload.get("max_new_tokens", 512)}
        if payload.get("do_sample", True):
            opts.update(temperature=payload.get("temperature", 0.7),
                        top_p=payload.get("top_p", 0.9), top_k=payload.get("top_k", 50))
        else:
            opts["temperature"] = 0.0  # greedy
        return opts

    async def _post(self, path: str, body: dict) -> dict:
        body = {**body, "model": self.ollama_model, "stream": False, "keep_alive": -1}
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_REQUEST_TIMEOUT) as client:
                r = await client.post(f"{OLLAMA_URL}{path}", json=body)
        except httpx.TimeoutException:
            raise OllamaError(504, "LLM 생성 타임아웃")
        except httpx.ConnectError:
            raise OllamaError(503, "Ollama 서버에 연결할 수 없습니다.")
        if r.status_code != 200:
            raise OllamaError(r.status_code, r.text)
        return r.json()

    def _response(self, data: dict, text: str, thinking: str | None, started: float, session_id: str | None) -> dict:
        prompt_tokens = data.get("prompt_eval_count", 0)
        generated_tokens = data.get("eval_count", 0)
        resp = {
            "generated_text": text,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": prompt_tokens + generated_tokens,
            "model": self.model_name,
            "elapsed_time": round(time.time() - started, 3),
            "session_id": session_id,
            "ei_info": None,
            "thinking": thinking or None,
            "done_reason": data.get("done_reason"),
            "backend": "ollama",
        }
        if not text and data.get("done_reason") == "length":
            resp["warning"] = "max_new_tokens 에 도달해 답변이 비었습니다 (thinking 이 토큰을 소진). max_new_tokens 를 늘리거나 think=false 로 호출하세요."
        return resp

    async def chat(self, payload: dict) -> dict:
        """/llm/chat — stateless(messages) 또는 세션(message + session_id)"""
        self.sessions.cleanup_expired()
        started = time.time()
        sid = payload.get("session_id")
        if payload.get("message") is not None:
            if sid is None or self.sessions.get(sid) is None:
                sid = self.sessions.create(sid, payload.get("system_prompt"))
            self.sessions.add_message(sid, "user", payload["message"])
            messages = self.sessions.get_messages(sid)
        elif payload.get("messages") is not None:
            messages = [{"role": m["role"], "content": m["content"]} for m in payload["messages"]]
            sid = None
        else:
            raise OllamaError(400, "message 또는 messages 중 하나는 필수입니다.")

        body = {"messages": messages, "options": self._options(payload)}
        if payload.get("think") is not None:
            body["think"] = payload["think"]
        data = await self._post("/api/chat", body)
        msg = data.get("message", {})
        text = msg.get("content", "")
        if sid:
            self.sessions.add_message(sid, "assistant", text,
                                      data.get("prompt_eval_count", 0) + data.get("eval_count", 0))
        return self._response(data, text, msg.get("thinking"), started, sid)

    # ---- OpenAI 호환 (/v1) 중계 ------------------------------------------

    async def openai_chat(self, body: dict) -> tuple[int, dict]:
        """/v1/chat/completions (stream=false) — 요청/응답을 그대로 중계 (tools 포함)"""
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_REQUEST_TIMEOUT) as client:
                r = await client.post(f"{OLLAMA_URL}/v1/chat/completions", json=body)
        except httpx.TimeoutException:
            raise OllamaError(504, "LLM 생성 타임아웃")
        except httpx.ConnectError:
            raise OllamaError(503, "Ollama 서버에 연결할 수 없습니다.")
        return r.status_code, r.json()

    async def openai_stream(self, body: dict):
        """/v1/chat/completions (stream=true) — 응답 상태를 먼저 확인한 뒤 SSE 바이트를 그대로 중계하는 iterator 반환"""
        client = httpx.AsyncClient(timeout=OLLAMA_REQUEST_TIMEOUT)
        try:
            resp = await client.send(client.build_request("POST", f"{OLLAMA_URL}/v1/chat/completions", json=body), stream=True)
        except httpx.ConnectError:
            await client.aclose()
            raise OllamaError(503, "Ollama 서버에 연결할 수 없습니다.")
        if resp.status_code != 200:
            detail = (await resp.aread()).decode(errors="replace")
            await resp.aclose()
            await client.aclose()
            raise OllamaError(resp.status_code, detail)

        async def relay():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()
        return relay()

    async def generate(self, payload: dict) -> dict:
        """/llm/generate — raw_prompt 면 템플릿 없이, 아니면 system+user 채팅으로"""
        started = time.time()
        if payload.get("raw_prompt"):
            body = {"prompt": payload["prompt"], "raw": True, "options": self._options(payload)}
            data = await self._post("/api/generate", body)
            return self._response(data, data.get("response", ""), None, started, None)

        messages = []
        if payload.get("system_prompt"):
            messages.append({"role": "system", "content": payload["system_prompt"]})
        messages.append({"role": "user", "content": payload["prompt"]})
        body = {"messages": messages, "options": self._options(payload)}
        if payload.get("think") is not None:
            body["think"] = payload["think"]
        data = await self._post("/api/chat", body)
        msg = data.get("message", {})
        return self._response(data, msg.get("content", ""), msg.get("thinking"), started, None)
