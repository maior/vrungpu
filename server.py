"""
VrunGPU - Remote GPU Execution Server v0.4.0
Python 코드를 원격으로 실행하고 GPU 리소스를 활용하는 REST API 서버

Features:
- 멀티 GPU 병렬 작업 지원
- WebSocket 실시간 스트리밍
- ZIP 프로젝트 업로드
- SQLite 영구 저장소
- 모델 관리 및 추론 API
"""

import asyncio
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File, Query, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent / "data"
WORKSPACES_DIR = BASE_DIR / "workspaces"
MODELS_DIR = BASE_DIR / "models"
UPLOADS_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "vrungpu.db"

# LLM Inference Server Config
LLM_SERVER_SCRIPT = Path(__file__).parent / "inference_server.py"
LLM_SERVER_PORT = 9826
LLM_SERVER_URL = f"http://localhost:{LLM_SERVER_PORT}"
DEFAULT_LLM_MODEL = "Qwen/Qwen3-8B"
DEFAULT_LLM_GPU = 0  # 단일 GPU 환경: LLM과 VrunGPU가 GPU 0을 번갈아 사용
LLM_STARTUP_TIMEOUT = 120  # LLM 서버 시작 대기 시간 (초)

# 디렉토리 생성
for d in [BASE_DIR, WORKSPACES_DIR, MODELS_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Database Schema & Operations
# ============================================================================

def init_database():
    """데이터베이스 초기화"""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                name TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                task_type TEXT DEFAULT 'training',
                gpu_id INTEGER,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                stdout TEXT,
                stderr TEXT,
                return_code INTEGER,
                error TEXT,
                work_dir TEXT,
                entry_point TEXT,
                config TEXT,
                progress REAL DEFAULT 0,
                progress_message TEXT,
                model_id TEXT,
                parent_task_id TEXT
            );

            CREATE TABLE IF NOT EXISTS models (
                model_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                task_id TEXT,
                model_type TEXT,
                framework TEXT DEFAULT 'pytorch',
                created_at TEXT NOT NULL,
                updated_at TEXT,
                model_path TEXT,
                config TEXT,
                metrics TEXT,
                description TEXT,
                status TEXT DEFAULT 'ready',
                file_size INTEGER,
                FOREIGN KEY (task_id) REFERENCES tasks(task_id)
            );

            CREATE TABLE IF NOT EXISTS inference_logs (
                log_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                input_data TEXT,
                output_data TEXT,
                latency_ms REAL,
                gpu_id INTEGER,
                status TEXT,
                error TEXT,
                FOREIGN KEY (model_id) REFERENCES models(model_id)
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);
            CREATE INDEX IF NOT EXISTS idx_models_task ON models(task_id);
        """)


@contextmanager
def get_db():
    """데이터베이스 연결 컨텍스트 매니저"""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ============================================================================
# Models (Pydantic)
# ============================================================================

class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"


class TaskResult(BaseModel):
    task_id: str
    name: str | None = None
    status: TaskStatus
    task_type: TaskType = TaskType.TRAINING
    gpu_id: int | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    stdout: str | None = None
    stderr: str | None = None
    return_code: int | None = None
    error: str | None = None
    work_dir: str | None = None
    entry_point: str | None = None
    config: dict | None = None
    progress: float = 0
    progress_message: str | None = None
    model_id: str | None = None


class ModelInfo(BaseModel):
    model_id: str
    name: str
    task_id: str | None = None
    model_type: str | None = None
    framework: str = "pytorch"
    created_at: datetime
    updated_at: datetime | None = None
    model_path: str | None = None
    config: dict | None = None
    metrics: dict | None = None
    description: str | None = None
    status: str = "ready"
    file_size: int | None = None


class CodeRequest(BaseModel):
    code: str = Field(..., description="실행할 Python 코드")
    name: str | None = Field(default=None, description="작업 이름")
    timeout: int | None = Field(default=None, description="타임아웃 (초), None=무제한")
    gpu_id: int | None = Field(default=None, description="사용할 GPU ID")
    save_model: bool = Field(default=False, description="학습 완료 후 모델 저장")
    model_name: str | None = Field(default=None, description="저장할 모델 이름")


class InferenceRequest(BaseModel):
    input_data: Any = Field(..., description="추론 입력 데이터")
    gpu_id: int | None = Field(default=None, description="사용할 GPU ID")
    timeout: int | None = Field(default=None, description="타임아웃 (초), None=무제한")


class AsyncTaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    gpu_id: int | None = None
    message: str


class GPUInfo(BaseModel):
    available: bool
    devices: list[dict[str, Any]] = []
    error: str | None = None


class GPUPoolStatus(BaseModel):
    total_gpus: int
    available_gpus: list[int]
    busy_gpus: dict[str, int]


class ProgressUpdate(BaseModel):
    progress: float = Field(..., ge=0, le=100, description="진행률 (0-100)")
    message: str | None = Field(default=None, description="진행 상태 메시지")


# LLM Service Models
class LLMGenerateRequest(BaseModel):
    prompt: str = Field(..., description="입력 프롬프트")
    max_new_tokens: int = Field(default=512, description="생성할 최대 토큰 수")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="샘플링 온도")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p 샘플링")
    top_k: int = Field(default=50, ge=0, description="Top-k 샘플링")
    do_sample: bool = Field(default=True, description="샘플링 사용 여부")
    system_prompt: str | None = Field(default=None, description="시스템 프롬프트")


class LLMChatMessage(BaseModel):
    role: str = Field(..., description="메시지 역할 (system/user/assistant)")
    content: str = Field(..., description="메시지 내용")


class LLMChatRequest(BaseModel):
    messages: list[LLMChatMessage] = Field(..., description="대화 메시지 목록")
    max_new_tokens: int = Field(default=512, description="생성할 최대 토큰 수")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    do_sample: bool = Field(default=True)


class LLMServiceStatus(BaseModel):
    running: bool
    model: str | None = None
    port: int = LLM_SERVER_PORT
    pid: int | None = None
    message: str = "LLM API는 /llm/generate, /llm/chat 엔드포인트를 통해 호출하세요"


# ============================================================================
# GPU Pool Manager
# ============================================================================

class GPUPool:
    """GPU 풀 관리자"""

    def __init__(self):
        self._lock = threading.Lock()
        self._gpu_count = self._detect_gpus()
        self._busy: dict[int, str] = {}

    def _detect_gpus(self) -> int:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return len(result.stdout.strip().split("\n"))
        except Exception:
            pass
        return 0

    @property
    def gpu_count(self) -> int:
        return self._gpu_count

    def acquire(self, task_id: str, gpu_id: int | None = None) -> int | None:
        with self._lock:
            if gpu_id is not None:
                if gpu_id < 0 or gpu_id >= self._gpu_count:
                    return None
                if gpu_id in self._busy:
                    return None
                self._busy[gpu_id] = task_id
                return gpu_id
            else:
                for i in range(self._gpu_count):
                    if i not in self._busy:
                        self._busy[i] = task_id
                        return i
                return None

    def release(self, gpu_id: int):
        with self._lock:
            if gpu_id in self._busy:
                del self._busy[gpu_id]

    def get_status(self) -> dict:
        with self._lock:
            available = [i for i in range(self._gpu_count) if i not in self._busy]
            busy = {task_id: gpu_id for gpu_id, task_id in self._busy.items()}
            return {
                "total_gpus": self._gpu_count,
                "available_gpus": available,
                "busy_gpus": busy,
            }


# ============================================================================
# Database Task Operations
# ============================================================================

def save_task(task: TaskResult):
    """작업을 DB에 저장"""
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO tasks
            (task_id, name, status, task_type, gpu_id, created_at, started_at,
             completed_at, stdout, stderr, return_code, error, work_dir,
             entry_point, config, progress, progress_message, model_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.task_id, task.name, task.status.value, task.task_type.value,
            task.gpu_id, task.created_at.isoformat(),
            task.started_at.isoformat() if task.started_at else None,
            task.completed_at.isoformat() if task.completed_at else None,
            task.stdout, task.stderr, task.return_code, task.error,
            task.work_dir, task.entry_point,
            json.dumps(task.config) if task.config else None,
            task.progress, task.progress_message, task.model_id
        ))


def get_task(task_id: str) -> TaskResult | None:
    """DB에서 작업 조회"""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row:
            return _row_to_task(row)
    return None


def get_tasks(limit: int = 50, status: str | None = None, task_type: str | None = None) -> list[TaskResult]:
    """작업 목록 조회"""
    with get_db() as conn:
        query = "SELECT * FROM tasks WHERE 1=1"
        params = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [_row_to_task(row) for row in rows]


def _row_to_task(row) -> TaskResult:
    """DB row를 TaskResult로 변환"""
    return TaskResult(
        task_id=row["task_id"],
        name=row["name"],
        status=TaskStatus(row["status"]),
        task_type=TaskType(row["task_type"]) if row["task_type"] else TaskType.TRAINING,
        gpu_id=row["gpu_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        stdout=row["stdout"],
        stderr=row["stderr"],
        return_code=row["return_code"],
        error=row["error"],
        work_dir=row["work_dir"],
        entry_point=row["entry_point"],
        config=json.loads(row["config"]) if row["config"] else None,
        progress=row["progress"] or 0,
        progress_message=row["progress_message"],
        model_id=row["model_id"],
    )


# ============================================================================
# Model Operations
# ============================================================================

def save_model(model: ModelInfo):
    """모델을 DB에 저장"""
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO models
            (model_id, name, task_id, model_type, framework, created_at, updated_at,
             model_path, config, metrics, description, status, file_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model.model_id, model.name, model.task_id, model.model_type,
            model.framework, model.created_at.isoformat(),
            model.updated_at.isoformat() if model.updated_at else None,
            model.model_path,
            json.dumps(model.config) if model.config else None,
            json.dumps(model.metrics) if model.metrics else None,
            model.description, model.status, model.file_size
        ))


def get_model(model_id: str) -> ModelInfo | None:
    """DB에서 모델 조회"""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM models WHERE model_id = ?", (model_id,)).fetchone()
        if row:
            return _row_to_model(row)
    return None


def get_models(limit: int = 50, status: str | None = None) -> list[ModelInfo]:
    """모델 목록 조회"""
    with get_db() as conn:
        query = "SELECT * FROM models WHERE 1=1"
        params = []
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [_row_to_model(row) for row in rows]


def _row_to_model(row) -> ModelInfo:
    """DB row를 ModelInfo로 변환"""
    return ModelInfo(
        model_id=row["model_id"],
        name=row["name"],
        task_id=row["task_id"],
        model_type=row["model_type"],
        framework=row["framework"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        model_path=row["model_path"],
        config=json.loads(row["config"]) if row["config"] else None,
        metrics=json.loads(row["metrics"]) if row["metrics"] else None,
        description=row["description"],
        status=row["status"],
        file_size=row["file_size"],
    )


# ============================================================================
# Global Objects
# ============================================================================

gpu_pool = GPUPool()
websocket_clients: set[WebSocket] = set()
running_tasks: dict[str, TaskResult] = {}  # 메모리 캐시 (실행 중인 작업만)

# LLM Inference Server Process
llm_process: subprocess.Popen | None = None
llm_model_name: str | None = None
llm_switching_lock = asyncio.Lock()  # LLM 서비스 전환 동기화


async def ensure_llm_stopped():
    """VrunGPU 작업 전 LLM 서비스 중지 (GPU 확보)"""
    global llm_process, llm_model_name

    async with llm_switching_lock:
        # Check if LLM is running
        if llm_process is not None and llm_process.poll() is None:
            print("[Auto-Switch] LLM 서비스 중지 중...")
            llm_process.terminate()
            try:
                llm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                llm_process.kill()
                llm_process.wait()
            llm_process = None
            # llm_model_name은 유지 (재시작 시 사용)
            print("[Auto-Switch] LLM 서비스 중지 완료")
            await asyncio.sleep(1)  # GPU 메모리 해제 대기
            return True
    return False


async def ensure_llm_running(model: str | None = None):
    """LLM API 호출 전 LLM 서비스 자동 시작"""
    global llm_process, llm_model_name

    async with llm_switching_lock:
        # Check if already running
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{LLM_SERVER_URL}/health")
                if resp.status_code == 200:
                    return True  # Already running
        except Exception:
            pass

        # Start LLM service
        target_model = model or llm_model_name or DEFAULT_LLM_MODEL
        print(f"[Auto-Switch] LLM 서비스 시작 중... ({target_model})")

        if not LLM_SERVER_SCRIPT.exists():
            raise HTTPException(status_code=500, detail="inference_server.py 파일을 찾을 수 없습니다.")

        llm_process = subprocess.Popen(
            [sys.executable, str(LLM_SERVER_SCRIPT), "--model", target_model,
             "--port", str(LLM_SERVER_PORT), "--gpu", str(DEFAULT_LLM_GPU)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(LLM_SERVER_SCRIPT.parent),
        )
        llm_model_name = target_model

        # Wait for server to be ready
        start_time = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start_time < LLM_STARTUP_TIMEOUT:
            if llm_process.poll() is not None:
                output = llm_process.stdout.read() if llm_process.stdout else ""
                raise HTTPException(status_code=500, detail=f"LLM 서버 시작 실패: {output[:500]}")

            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(f"{LLM_SERVER_URL}/health")
                    if resp.status_code == 200:
                        print(f"[Auto-Switch] LLM 서비스 시작 완료 (PID: {llm_process.pid})")
                        return True
            except Exception:
                pass

            await asyncio.sleep(2)

        raise HTTPException(status_code=504, detail="LLM 서버 시작 타임아웃 (모델 로딩 시간 초과)")


# 데이터베이스 초기화
init_database()

app = FastAPI(
    title="VrunGPU",
    description="원격 GPU 학습/추론 실행 서버 (SQLite 영구 저장 + 모델 관리)",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# WebSocket & Broadcasting
# ============================================================================

async def broadcast(message: dict):
    """모든 WebSocket 클라이언트에 메시지 전송"""
    dead_clients = set()
    for ws in websocket_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead_clients.add(ws)
    websocket_clients.difference_update(dead_clients)


async def broadcast_task_update(task: TaskResult):
    """작업 상태 업데이트 브로드캐스트"""
    await broadcast({
        "type": "task_update",
        "task": {
            "task_id": task.task_id,
            "name": task.name,
            "status": task.status.value,
            "task_type": task.task_type.value,
            "gpu_id": task.gpu_id,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "stdout": task.stdout,
            "stderr": task.stderr,
            "return_code": task.return_code,
            "error": task.error,
            "progress": task.progress,
            "progress_message": task.progress_message,
            "model_id": task.model_id,
        }
    })


async def broadcast_gpu_update():
    """GPU 상태 업데이트 브로드캐스트"""
    gpu_info = get_gpu_info()
    pool_status = gpu_pool.get_status()
    await broadcast({
        "type": "gpu_update",
        "gpus": gpu_info.devices,
        "pool": pool_status,
    })


def get_gpu_info() -> GPUInfo:
    """GPU 정보 조회"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return GPUInfo(available=False, error=result.stderr)

        devices = []
        pool_status = gpu_pool.get_status()
        busy_gpus = {v: k for k, v in pool_status["busy_gpus"].items()}

        for line in result.stdout.strip().split("\n"):
            if line:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    gpu_idx = int(parts[0])
                    devices.append({
                        "index": gpu_idx,
                        "name": parts[1],
                        "memory_total_mb": int(parts[2]),
                        "memory_used_mb": int(parts[3]),
                        "memory_free_mb": int(parts[4]),
                        "utilization_percent": int(parts[5]),
                        "task_id": busy_gpus.get(gpu_idx),
                    })
        return GPUInfo(available=True, devices=devices)
    except FileNotFoundError:
        return GPUInfo(available=False, error="nvidia-smi not found")
    except Exception as e:
        return GPUInfo(available=False, error=str(e))


# ============================================================================
# Task Execution
# ============================================================================

async def run_task_with_streaming(
    task_id: str,
    work_dir: Path,
    entry_point: str,
    timeout: int | None,
    gpu_id: int | None,
    save_model: bool = False,
    model_name: str | None = None,
):
    """실시간 스트리밍과 함께 작업 실행"""
    task = running_tasks.get(task_id) or get_task(task_id)
    if not task:
        return

    running_tasks[task_id] = task

    # GPU 할당 대기
    assigned_gpu = None
    while assigned_gpu is None:
        assigned_gpu = gpu_pool.acquire(task_id, gpu_id)
        if assigned_gpu is None:
            task.status = TaskStatus.QUEUED
            save_task(task)
            await broadcast_task_update(task)
            await asyncio.sleep(1)

    task.gpu_id = assigned_gpu
    task.status = TaskStatus.RUNNING
    task.started_at = datetime.now()
    save_task(task)
    await broadcast_task_update(task)
    await broadcast_gpu_update()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["VRUNGPU_TASK_ID"] = task_id
    env["VRUNGPU_MODEL_DIR"] = str(MODELS_DIR)

    script_path = work_dir / entry_point
    stdout_lines = []
    stderr_lines = []

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(work_dir),
            env=env,
        )

        async def read_stream(stream, lines, stream_name):
            while True:
                line = await stream.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace")
                lines.append(decoded)

                # 진행률 파싱 (특별한 형식: [PROGRESS:50.5:Training epoch 5/10])
                if decoded.startswith("[PROGRESS:"):
                    try:
                        parts = decoded[10:-2].split(":", 1)
                        task.progress = float(parts[0])
                        task.progress_message = parts[1] if len(parts) > 1 else None
                        save_task(task)
                    except:
                        pass

                # 실시간 스트리밍
                await broadcast({
                    "type": "task_output",
                    "task_id": task_id,
                    "stream": stream_name,
                    "line": decoded,
                })

        try:
            gather_task = asyncio.gather(
                read_stream(process.stdout, stdout_lines, "stdout"),
                read_stream(process.stderr, stderr_lines, "stderr"),
            )
            if timeout:
                await asyncio.wait_for(gather_task, timeout=timeout)
            else:
                await gather_task  # 무제한 실행
            await process.wait()
            return_code = process.returncode
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            stderr_lines.append(f"Timeout after {timeout} seconds\n")
            return_code = -1

        task.stdout = "".join(stdout_lines)
        task.stderr = "".join(stderr_lines)
        task.return_code = return_code
        task.status = TaskStatus.COMPLETED if return_code == 0 else TaskStatus.FAILED
        task.progress = 100 if return_code == 0 else task.progress

        # 모델 저장 처리
        if save_model and return_code == 0:
            await _save_trained_model(task, work_dir, model_name)

    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error = str(e)
    finally:
        task.completed_at = datetime.now()
        save_task(task)
        gpu_pool.release(assigned_gpu)
        running_tasks.pop(task_id, None)
        await broadcast_task_update(task)
        await broadcast_gpu_update()


async def _save_trained_model(task: TaskResult, work_dir: Path, model_name: str | None):
    """학습 완료된 모델 저장"""
    model_id = str(uuid.uuid4())[:8]
    model_dir = MODELS_DIR / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # 모델 파일 찾기 (.pt, .pth, .onnx, .h5, .pkl)
    model_extensions = [".pt", ".pth", ".onnx", ".h5", ".pkl", ".bin", ".safetensors"]
    model_files = []
    for ext in model_extensions:
        model_files.extend(work_dir.rglob(f"*{ext}"))

    total_size = 0
    for mf in model_files:
        dest = model_dir / mf.name
        shutil.copy2(mf, dest)
        total_size += dest.stat().st_size

    # config 파일 복사
    for cf in work_dir.rglob("*.json"):
        if "config" in cf.name.lower():
            shutil.copy2(cf, model_dir / cf.name)

    model = ModelInfo(
        model_id=model_id,
        name=model_name or f"model_{task.task_id[:8]}",
        task_id=task.task_id,
        model_type="pytorch",
        framework="pytorch",
        created_at=datetime.now(),
        model_path=str(model_dir),
        status="ready",
        file_size=total_size,
        description=f"Trained from task {task.task_id}",
    )
    save_model(model)
    task.model_id = model_id


# ============================================================================
# API Endpoints - Core
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 연결 - 실시간 모니터링"""
    await websocket.accept()
    websocket_clients.add(websocket)

    # 초기 데이터 전송
    gpu_info = get_gpu_info()
    pool_status = gpu_pool.get_status()
    recent_tasks = get_tasks(limit=20)

    await websocket.send_json({
        "type": "init",
        "gpus": gpu_info.devices,
        "pool": pool_status,
        "tasks": [
            {
                "task_id": t.task_id,
                "name": t.name,
                "status": t.status.value,
                "task_type": t.task_type.value,
                "gpu_id": t.gpu_id,
                "created_at": t.created_at.isoformat(),
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                "stdout": t.stdout,
                "stderr": t.stderr,
                "return_code": t.return_code,
                "error": t.error,
                "progress": t.progress,
                "progress_message": t.progress_message,
            }
            for t in recent_tasks
        ],
    })

    try:
        while True:
            await asyncio.sleep(2)
            await broadcast_gpu_update()
    except WebSocketDisconnect:
        websocket_clients.discard(websocket)
    except Exception:
        websocket_clients.discard(websocket)


@app.get("/")
async def root():
    pool_status = gpu_pool.get_status()
    with get_db() as conn:
        total_tasks = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        total_models = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    return {
        "service": "VrunGPU",
        "status": "running",
        "version": "0.4.0",
        "gpu_count": pool_status["total_gpus"],
        "available_gpus": len(pool_status["available_gpus"]),
        "total_tasks": total_tasks,
        "total_models": total_models,
        "storage": {
            "workspaces": str(WORKSPACES_DIR),
            "models": str(MODELS_DIR),
            "database": str(DB_PATH),
        }
    }


@app.get("/gpu", response_model=GPUInfo)
async def get_gpu_status():
    return get_gpu_info()


@app.get("/gpu/pool", response_model=GPUPoolStatus)
async def get_gpu_pool_status():
    return gpu_pool.get_status()


# ============================================================================
# API Endpoints - Tasks
# ============================================================================

@app.post("/run/sync", response_model=TaskResult)
async def run_sync(request: CodeRequest):
    """동기 실행 (결과를 기다림, LLM 자동 중지)"""
    # GPU 확보를 위해 LLM 서비스 자동 중지
    await ensure_llm_stopped()

    task_id = str(uuid.uuid4())
    work_dir = WORKSPACES_DIR / task_id
    work_dir.mkdir(parents=True)

    script_path = work_dir / "main.py"
    script_path.write_text(request.code)

    assigned_gpu = gpu_pool.acquire(task_id, request.gpu_id)
    if assigned_gpu is None and request.gpu_id is not None:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=409, detail=f"GPU {request.gpu_id}가 사용 중입니다.")

    task = TaskResult(
        task_id=task_id,
        name=request.name or f"sync_{task_id[:8]}",
        status=TaskStatus.RUNNING,
        task_type=TaskType.TRAINING,
        gpu_id=assigned_gpu,
        created_at=datetime.now(),
        started_at=datetime.now(),
        work_dir=str(work_dir),
        entry_point="main.py",
    )
    save_task(task)

    env = os.environ.copy()
    if assigned_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=request.timeout if request.timeout else None,
            cwd=str(work_dir),
            env=env,
        )
        task.stdout = result.stdout
        task.stderr = result.stderr
        task.return_code = result.returncode
        task.status = TaskStatus.COMPLETED if result.returncode == 0 else TaskStatus.FAILED
        task.progress = 100 if result.returncode == 0 else 0
    except subprocess.TimeoutExpired:
        task.status = TaskStatus.FAILED
        task.error = f"Timeout after {request.timeout} seconds" if request.timeout else "Timeout"
    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error = str(e)
    finally:
        task.completed_at = datetime.now()
        if assigned_gpu is not None:
            gpu_pool.release(assigned_gpu)
        save_task(task)

    return task


@app.post("/run/async", response_model=AsyncTaskResponse)
async def run_async(request: CodeRequest, background_tasks: BackgroundTasks):
    """비동기 실행 (즉시 반환, 백그라운드 실행, LLM 자동 중지)"""
    # GPU 확보를 위해 LLM 서비스 자동 중지
    await ensure_llm_stopped()

    task_id = str(uuid.uuid4())
    work_dir = WORKSPACES_DIR / task_id
    work_dir.mkdir(parents=True)

    script_path = work_dir / "main.py"
    script_path.write_text(request.code)

    task = TaskResult(
        task_id=task_id,
        name=request.name or f"async_{task_id[:8]}",
        status=TaskStatus.PENDING,
        task_type=TaskType.TRAINING,
        created_at=datetime.now(),
        work_dir=str(work_dir),
        entry_point="main.py",
    )
    save_task(task)
    running_tasks[task_id] = task

    background_tasks.add_task(
        run_task_with_streaming, task_id, work_dir, "main.py",
        request.timeout, request.gpu_id, request.save_model, request.model_name
    )

    return AsyncTaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="작업이 큐에 등록되었습니다.",
    )


@app.post("/run/project", response_model=AsyncTaskResponse)
async def run_project(
    file: UploadFile = File(...),
    name: str = Form(default=None),
    entry_point: str = Form(default="main.py"),
    timeout: int | None = Form(default=None),
    gpu_id: int | None = Form(default=None),
    save_model: bool = Form(default=False),
    model_name: str | None = Form(default=None),
    background_tasks: BackgroundTasks = None,
):
    """프로젝트 업로드 후 실행 (스트리밍 업로드, LLM 자동 중지)"""
    # GPU 확보를 위해 LLM 서비스 자동 중지
    await ensure_llm_stopped()

    task_id = str(uuid.uuid4())
    work_dir = WORKSPACES_DIR / task_id
    work_dir.mkdir(parents=True)

    filename = file.filename or "upload"

    try:
        if filename.endswith(".zip"):
            zip_path = work_dir / "upload.zip"
            # 스트리밍 업로드: 1MB 청크 단위로 파일 저장
            async with aiofiles.open(zip_path, 'wb') as f:
                while chunk := await file.read(1024 * 1024):  # 1MB chunks
                    await f.write(chunk)

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(work_dir)
            zip_path.unlink()

            items = list(work_dir.iterdir())
            if len(items) == 1 and items[0].is_dir():
                nested_dir = items[0]
                for item in nested_dir.iterdir():
                    shutil.move(str(item), str(work_dir))
                nested_dir.rmdir()

        elif filename.endswith(".py"):
            script_path = work_dir / filename
            # 스트리밍 업로드
            async with aiofiles.open(script_path, 'wb') as f:
                while chunk := await file.read(1024 * 1024):
                    await f.write(chunk)
            entry_point = filename
        else:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="ZIP 또는 .py 파일만 지원합니다.")

        if not (work_dir / entry_point).exists():
            files = [f.name for f in work_dir.rglob("*.py")]
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail=f"Entry point '{entry_point}' not found. Available: {files}"
            )

    except zipfile.BadZipFile:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="잘못된 ZIP 파일입니다.")

    task = TaskResult(
        task_id=task_id,
        name=name or f"project_{task_id[:8]}",
        status=TaskStatus.PENDING,
        task_type=TaskType.TRAINING,
        created_at=datetime.now(),
        work_dir=str(work_dir),
        entry_point=entry_point,
    )
    save_task(task)
    running_tasks[task_id] = task

    background_tasks.add_task(
        run_task_with_streaming, task_id, work_dir, entry_point,
        timeout, gpu_id, save_model, model_name
    )

    return AsyncTaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=f"프로젝트가 업로드되었습니다. Entry: {entry_point}",
    )


@app.get("/task/{task_id}", response_model=TaskResult)
async def get_task_status(task_id: str):
    """작업 상태 조회"""
    # 먼저 실행 중인 작업 캐시에서 확인
    if task_id in running_tasks:
        return running_tasks[task_id]
    # DB에서 조회
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return task


@app.get("/tasks", response_model=list[TaskResult])
async def list_tasks(
    limit: int = Query(default=50, le=200),
    status: str | None = Query(default=None),
    task_type: str | None = Query(default=None),
):
    """작업 목록 조회"""
    return get_tasks(limit=limit, status=status, task_type=task_type)


@app.put("/task/{task_id}/progress")
async def update_task_progress(task_id: str, update: ProgressUpdate):
    """작업 진행률 업데이트 (외부에서 호출 가능)"""
    task = running_tasks.get(task_id) or get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")

    task.progress = update.progress
    task.progress_message = update.message
    save_task(task)
    await broadcast_task_update(task)

    return {"message": "Progress updated", "progress": update.progress}


@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """작업 삭제"""
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")

    if task.work_dir:
        shutil.rmtree(task.work_dir, ignore_errors=True)

    with get_db() as conn:
        conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))

    running_tasks.pop(task_id, None)
    return {"message": "작업이 삭제되었습니다."}


# ============================================================================
# API Endpoints - Models
# ============================================================================

@app.get("/models", response_model=list[ModelInfo])
async def list_models(
    limit: int = Query(default=50, le=200),
    status: str | None = Query(default=None),
):
    """모델 목록 조회"""
    return get_models(limit=limit, status=status)


@app.get("/model/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id: str):
    """모델 상세 정보 조회"""
    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다.")
    return model


@app.post("/model/{model_id}/inference", response_model=AsyncTaskResponse)
async def run_inference(
    model_id: str,
    request: InferenceRequest,
    background_tasks: BackgroundTasks,
):
    """모델로 추론 실행 (LLM 자동 중지)"""
    # GPU 확보를 위해 LLM 서비스 자동 중지
    await ensure_llm_stopped()

    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다.")

    if model.status != "ready":
        raise HTTPException(status_code=400, detail="모델이 준비되지 않았습니다.")

    task_id = str(uuid.uuid4())
    work_dir = WORKSPACES_DIR / f"inference_{task_id}"
    work_dir.mkdir(parents=True)

    # 추론 스크립트 생성
    inference_script = f'''
import json
import sys
import torch
from pathlib import Path

MODEL_PATH = Path("{model.model_path}")
INPUT_DATA = {json.dumps(request.input_data)}

def load_model():
    """모델 로드 (사용자 정의 필요)"""
    model_files = list(MODEL_PATH.glob("*.pt")) + list(MODEL_PATH.glob("*.pth"))
    if model_files:
        return torch.load(model_files[0], map_location="cuda" if torch.cuda.is_available() else "cpu")
    raise FileNotFoundError("No model file found")

def run_inference(model, input_data):
    """추론 실행 (사용자 정의 필요)"""
    # 기본 구현: 입력 데이터를 텐서로 변환하여 모델에 전달
    if hasattr(model, 'eval'):
        model.eval()
    with torch.no_grad():
        if isinstance(input_data, list):
            input_tensor = torch.tensor(input_data)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()
            output = model(input_tensor)
            return output.cpu().tolist() if hasattr(output, 'cpu') else output
    return {{"message": "Inference completed", "input": input_data}}

if __name__ == "__main__":
    try:
        model = load_model()
        result = run_inference(model, INPUT_DATA)
        print(json.dumps({{"success": True, "result": result}}))
    except Exception as e:
        print(json.dumps({{"success": False, "error": str(e)}}), file=sys.stderr)
        sys.exit(1)
'''

    script_path = work_dir / "inference.py"
    script_path.write_text(inference_script)

    task = TaskResult(
        task_id=task_id,
        name=f"inference_{model.name}",
        status=TaskStatus.PENDING,
        task_type=TaskType.INFERENCE,
        created_at=datetime.now(),
        work_dir=str(work_dir),
        entry_point="inference.py",
        model_id=model_id,
        config={"input_data": request.input_data},
    )
    save_task(task)
    running_tasks[task_id] = task

    background_tasks.add_task(
        run_task_with_streaming, task_id, work_dir, "inference.py",
        request.timeout, request.gpu_id, False, None
    )

    return AsyncTaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=f"추론 작업이 시작되었습니다. Model: {model.name}",
    )


@app.post("/model/register", response_model=ModelInfo)
async def register_model(
    name: str = Form(...),
    model_file: UploadFile = File(...),
    model_type: str = Form(default="pytorch"),
    framework: str = Form(default="pytorch"),
    description: str | None = Form(default=None),
):
    """외부 모델 등록 (스트리밍 업로드 지원)"""
    model_id = str(uuid.uuid4())[:8]
    model_dir = MODELS_DIR / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / (model_file.filename or "model.pt")

    # 스트리밍 업로드: 1MB 청크 단위로 파일 저장
    file_size = 0
    async with aiofiles.open(model_path, 'wb') as f:
        while chunk := await model_file.read(1024 * 1024):  # 1MB chunks
            await f.write(chunk)
            file_size += len(chunk)

    model = ModelInfo(
        model_id=model_id,
        name=name,
        model_type=model_type,
        framework=framework,
        created_at=datetime.now(),
        model_path=str(model_dir),
        description=description,
        status="ready",
        file_size=file_size,
    )
    save_model(model)

    return model


@app.delete("/model/{model_id}")
async def delete_model(model_id: str):
    """모델 삭제"""
    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다.")

    if model.model_path:
        shutil.rmtree(model.model_path, ignore_errors=True)

    with get_db() as conn:
        conn.execute("DELETE FROM models WHERE model_id = ?", (model_id,))

    return {"message": "모델이 삭제되었습니다."}


@app.get("/model/{model_id}/download")
async def download_model(model_id: str):
    """모델 파일 다운로드"""
    model = get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다.")

    if not model.model_path:
        raise HTTPException(status_code=404, detail="모델 경로가 없습니다.")

    model_dir = Path(model.model_path)
    model_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth"))

    if not model_files:
        raise HTTPException(status_code=404, detail="모델 파일이 없습니다.")

    return FileResponse(
        model_files[0],
        filename=model_files[0].name,
        media_type="application/octet-stream"
    )


# ============================================================================
# API Endpoints - Statistics
# ============================================================================

@app.get("/stats")
async def get_statistics():
    """통계 정보 조회"""
    with get_db() as conn:
        # 작업 통계
        task_stats = conn.execute("""
            SELECT
                status,
                COUNT(*) as count,
                AVG(CASE WHEN completed_at IS NOT NULL AND started_at IS NOT NULL
                    THEN (julianday(completed_at) - julianday(started_at)) * 86400
                    ELSE NULL END) as avg_duration_sec
            FROM tasks
            GROUP BY status
        """).fetchall()

        # 최근 24시간 작업 수
        recent_tasks = conn.execute("""
            SELECT COUNT(*) FROM tasks
            WHERE created_at > datetime('now', '-24 hours')
        """).fetchone()[0]

        # 모델 통계
        model_stats = conn.execute("""
            SELECT status, COUNT(*) as count
            FROM models
            GROUP BY status
        """).fetchall()

        # GPU 사용 통계
        gpu_usage = conn.execute("""
            SELECT gpu_id, COUNT(*) as count
            FROM tasks
            WHERE gpu_id IS NOT NULL
            GROUP BY gpu_id
        """).fetchall()

    return {
        "tasks": {
            "by_status": {row["status"]: row["count"] for row in task_stats},
            "avg_duration": {
                row["status"]: round(row["avg_duration_sec"], 2) if row["avg_duration_sec"] else None
                for row in task_stats
            },
            "last_24h": recent_tasks,
        },
        "models": {
            "by_status": {row["status"]: row["count"] for row in model_stats},
            "total": sum(row["count"] for row in model_stats),
        },
        "gpu": {
            "usage_count": {f"gpu_{row['gpu_id']}": row["count"] for row in gpu_usage},
            "current": gpu_pool.get_status(),
        }
    }


# ============================================================================
# API Endpoints - LLM Service
# ============================================================================

@app.post("/llm/start", response_model=LLMServiceStatus)
async def start_llm_service(
    model: str = Query(default=DEFAULT_LLM_MODEL, description="HuggingFace 모델 이름"),
    gpu: int = Query(default=DEFAULT_LLM_GPU, description="사용할 GPU ID"),
):
    """LLM 추론 서버 시작 (Qwen3-8B 등)"""
    global llm_process, llm_model_name

    if llm_process is not None and llm_process.poll() is None:
        raise HTTPException(status_code=400, detail="LLM 서비스가 이미 실행 중입니다.")

    if not LLM_SERVER_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="inference_server.py 파일을 찾을 수 없습니다.")

    # Start the inference server as a subprocess (GPU 1 for LLM by default)
    llm_process = subprocess.Popen(
        [sys.executable, str(LLM_SERVER_SCRIPT), "--model", model, "--port", str(LLM_SERVER_PORT), "--gpu", str(gpu)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(LLM_SERVER_SCRIPT.parent),
    )
    llm_model_name = model

    # Wait a moment and check if it started
    await asyncio.sleep(2)
    if llm_process.poll() is not None:
        output = llm_process.stdout.read() if llm_process.stdout else ""
        raise HTTPException(status_code=500, detail=f"LLM 서버 시작 실패: {output}")

    return LLMServiceStatus(
        running=True,
        model=model,
        port=LLM_SERVER_PORT,
        pid=llm_process.pid,
    )


@app.post("/llm/stop")
async def stop_llm_service():
    """LLM 추론 서버 중지"""
    global llm_process, llm_model_name

    if llm_process is None or llm_process.poll() is not None:
        return {"message": "LLM 서비스가 실행 중이 아닙니다.", "running": False}

    # Gracefully terminate
    llm_process.terminate()
    try:
        llm_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        llm_process.kill()
        llm_process.wait()

    llm_process = None
    llm_model_name = None

    return {"message": "LLM 서비스가 중지되었습니다.", "running": False}


@app.get("/llm/status", response_model=LLMServiceStatus)
async def get_llm_status():
    """LLM 서비스 상태 확인"""
    global llm_process, llm_model_name

    running = False
    detected_model = llm_model_name

    # Try to ping the server (works even if started externally)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{LLM_SERVER_URL}/")
            if resp.status_code == 200:
                running = True
                data = resp.json()
                detected_model = data.get("model", llm_model_name)
    except Exception:
        running = False

    return LLMServiceStatus(
        running=running,
        model=detected_model if running else None,
        port=LLM_SERVER_PORT,
        pid=llm_process.pid if llm_process and llm_process.poll() is None else None,
    )


@app.post("/llm/generate")
async def llm_generate(request: LLMGenerateRequest):
    """LLM 텍스트 생성 (자동 서비스 시작)"""
    # 자동으로 LLM 서비스 시작
    await ensure_llm_running()

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{LLM_SERVER_URL}/generate",
                json=request.model_dump(),
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM 생성 타임아웃")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="LLM 서버에 연결할 수 없습니다.")


@app.post("/llm/chat")
async def llm_chat(request: LLMChatRequest):
    """LLM 채팅 (자동 서비스 시작)"""
    # 자동으로 LLM 서비스 시작
    await ensure_llm_running()

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{LLM_SERVER_URL}/chat",
                json=request.model_dump(),
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM 생성 타임아웃")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="LLM 서버에 연결할 수 없습니다.")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print(f"VrunGPU Server v0.4.0 starting with {gpu_pool.gpu_count} GPUs")
    print(f"Database: {DB_PATH}")
    print(f"Workspaces: {WORKSPACES_DIR}")
    print(f"Models: {MODELS_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=9825)
