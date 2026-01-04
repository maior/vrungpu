"""
VrunGPU - Remote GPU Execution Server
Python 코드를 원격으로 실행하고 GPU 리소스를 활용하는 REST API 서버
멀티 GPU 병렬 작업 지원 + WebSocket 실시간 스트리밍 + ZIP 프로젝트 업로드
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import zipfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, File, Query, WebSocket, WebSocketDisconnect, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskResult(BaseModel):
    task_id: str
    status: TaskStatus
    gpu_id: int | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    stdout: str | None = None
    stderr: str | None = None
    return_code: int | None = None
    error: str | None = None
    work_dir: str | None = None


class CodeRequest(BaseModel):
    code: str = Field(..., description="실행할 Python 코드")
    timeout: int = Field(default=300, description="타임아웃 (초)", ge=1, le=3600)
    gpu_id: int | None = Field(default=None, description="사용할 GPU ID (None이면 자동 할당)")


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


# 전역 객체
tasks: dict[str, TaskResult] = {}
gpu_pool = GPUPool()
websocket_clients: set[WebSocket] = set()
WORK_DIR_BASE = Path(tempfile.gettempdir()) / "vrungpu_workspaces"
WORK_DIR_BASE.mkdir(exist_ok=True)

app = FastAPI(
    title="VrunGPU",
    description="원격 GPU 학습/추론 실행 서버 (멀티 GPU + WebSocket 스트리밍)",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
            "status": task.status.value,
            "gpu_id": task.gpu_id,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "stdout": task.stdout,
            "stderr": task.stderr,
            "return_code": task.return_code,
            "error": task.error,
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


def execute_in_workdir(
    work_dir: Path,
    entry_point: str,
    timeout: int = 300,
    gpu_id: int | None = None
) -> tuple[str, str, int]:
    """작업 디렉토리에서 Python 스크립트 실행"""
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    script_path = work_dir / entry_point
    if not script_path.exists():
        return "", f"Entry point not found: {entry_point}", -1

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(work_dir),
            env=env,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Timeout after {timeout} seconds", -1
    except Exception as e:
        return "", str(e), -1


async def run_task_with_streaming(
    task_id: str,
    work_dir: Path,
    entry_point: str,
    timeout: int,
    gpu_id: int | None
):
    """실시간 스트리밍과 함께 작업 실행"""
    task = tasks[task_id]

    # GPU 할당 대기
    assigned_gpu = None
    while assigned_gpu is None:
        assigned_gpu = gpu_pool.acquire(task_id, gpu_id)
        if assigned_gpu is None:
            task.status = TaskStatus.QUEUED
            await broadcast_task_update(task)
            await asyncio.sleep(1)

    task.gpu_id = assigned_gpu
    task.status = TaskStatus.RUNNING
    task.started_at = datetime.now()
    await broadcast_task_update(task)
    await broadcast_gpu_update()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
    env["PYTHONUNBUFFERED"] = "1"

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
                # 실시간 스트리밍
                await broadcast({
                    "type": "task_output",
                    "task_id": task_id,
                    "stream": stream_name,
                    "line": decoded,
                })

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    read_stream(process.stdout, stdout_lines, "stdout"),
                    read_stream(process.stderr, stderr_lines, "stderr"),
                ),
                timeout=timeout
            )
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

    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error = str(e)
    finally:
        task.completed_at = datetime.now()
        gpu_pool.release(assigned_gpu)
        await broadcast_task_update(task)
        await broadcast_gpu_update()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 연결 - 실시간 모니터링"""
    await websocket.accept()
    websocket_clients.add(websocket)

    # 초기 데이터 전송
    gpu_info = get_gpu_info()
    pool_status = gpu_pool.get_status()
    sorted_tasks = sorted(tasks.values(), key=lambda t: t.created_at, reverse=True)[:20]

    await websocket.send_json({
        "type": "init",
        "gpus": gpu_info.devices,
        "pool": pool_status,
        "tasks": [
            {
                "task_id": t.task_id,
                "status": t.status.value,
                "gpu_id": t.gpu_id,
                "created_at": t.created_at.isoformat(),
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                "stdout": t.stdout,
                "stderr": t.stderr,
                "return_code": t.return_code,
                "error": t.error,
            }
            for t in sorted_tasks
        ],
    })

    try:
        while True:
            # 주기적으로 GPU 상태 업데이트
            await asyncio.sleep(2)
            await broadcast_gpu_update()
    except WebSocketDisconnect:
        websocket_clients.discard(websocket)
    except Exception:
        websocket_clients.discard(websocket)


@app.get("/")
async def root():
    pool_status = gpu_pool.get_status()
    return {
        "service": "VrunGPU",
        "status": "running",
        "version": "0.3.0",
        "gpu_count": pool_status["total_gpus"],
        "available_gpus": len(pool_status["available_gpus"]),
    }


@app.get("/gpu", response_model=GPUInfo)
async def get_gpu_status():
    return get_gpu_info()


@app.get("/gpu/pool", response_model=GPUPoolStatus)
async def get_gpu_pool_status():
    return gpu_pool.get_status()


@app.post("/run/sync", response_model=TaskResult)
async def run_sync(request: CodeRequest):
    """동기 실행"""
    task_id = str(uuid.uuid4())
    work_dir = WORK_DIR_BASE / task_id
    work_dir.mkdir(parents=True)

    # 코드를 파일로 저장
    script_path = work_dir / "main.py"
    script_path.write_text(request.code)

    assigned_gpu = gpu_pool.acquire(task_id, request.gpu_id)
    if assigned_gpu is None and request.gpu_id is not None:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=409, detail=f"GPU {request.gpu_id}가 사용 중입니다.")

    task = TaskResult(
        task_id=task_id,
        status=TaskStatus.RUNNING,
        gpu_id=assigned_gpu,
        created_at=datetime.now(),
        started_at=datetime.now(),
        work_dir=str(work_dir),
    )

    try:
        loop = asyncio.get_event_loop()
        stdout, stderr, return_code = await loop.run_in_executor(
            None, execute_in_workdir, work_dir, "main.py", request.timeout, assigned_gpu
        )
        task.stdout = stdout
        task.stderr = stderr
        task.return_code = return_code
        task.status = TaskStatus.COMPLETED if return_code == 0 else TaskStatus.FAILED
    except Exception as e:
        task.status = TaskStatus.FAILED
        task.error = str(e)
    finally:
        task.completed_at = datetime.now()
        if assigned_gpu is not None:
            gpu_pool.release(assigned_gpu)
        shutil.rmtree(work_dir, ignore_errors=True)

    return task


@app.post("/run/async", response_model=AsyncTaskResponse)
async def run_async(request: CodeRequest, background_tasks: BackgroundTasks):
    """비동기 실행"""
    task_id = str(uuid.uuid4())
    work_dir = WORK_DIR_BASE / task_id
    work_dir.mkdir(parents=True)

    script_path = work_dir / "main.py"
    script_path.write_text(request.code)

    task = TaskResult(
        task_id=task_id,
        status=TaskStatus.PENDING,
        created_at=datetime.now(),
        work_dir=str(work_dir),
    )
    tasks[task_id] = task

    background_tasks.add_task(
        run_task_with_streaming, task_id, work_dir, "main.py", request.timeout, request.gpu_id
    )

    return AsyncTaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="작업이 큐에 등록되었습니다.",
    )


@app.post("/run/project", response_model=AsyncTaskResponse)
async def run_project(
    file: UploadFile = File(..., description="ZIP 파일 (프로젝트 전체) 또는 단일 .py 파일"),
    entry_point: str = Form(default="main.py", description="실행할 메인 파일 (예: train.py)"),
    timeout: int = Form(default=300),
    gpu_id: int | None = Form(default=None),
    background_tasks: BackgroundTasks = None,
):
    """
    프로젝트 업로드 후 실행
    - ZIP 파일: 여러 Python 파일 + 데이터셋 포함 가능
    - 단일 .py 파일: 간단한 스크립트 실행
    """
    task_id = str(uuid.uuid4())
    work_dir = WORK_DIR_BASE / task_id
    work_dir.mkdir(parents=True)

    filename = file.filename or "upload"
    content = await file.read()

    try:
        if filename.endswith(".zip"):
            # ZIP 파일 압축 해제
            zip_path = work_dir / "upload.zip"
            zip_path.write_bytes(content)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(work_dir)
            zip_path.unlink()

            # 최상위에 단일 폴더만 있으면 그 안의 내용을 올림
            items = list(work_dir.iterdir())
            if len(items) == 1 and items[0].is_dir():
                nested_dir = items[0]
                for item in nested_dir.iterdir():
                    shutil.move(str(item), str(work_dir))
                nested_dir.rmdir()

        elif filename.endswith(".py"):
            # 단일 Python 파일
            script_path = work_dir / filename
            script_path.write_bytes(content)
            entry_point = filename
        else:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(status_code=400, detail="ZIP 또는 .py 파일만 지원합니다.")

        # entry_point 파일 존재 확인
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
        status=TaskStatus.PENDING,
        created_at=datetime.now(),
        work_dir=str(work_dir),
    )
    tasks[task_id] = task

    background_tasks.add_task(
        run_task_with_streaming, task_id, work_dir, entry_point, timeout, gpu_id
    )

    return AsyncTaskResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message=f"프로젝트가 업로드되었습니다. Entry: {entry_point}",
    )


@app.get("/task/{task_id}", response_model=TaskResult)
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return tasks[task_id]


@app.get("/tasks", response_model=list[TaskResult])
async def list_tasks(limit: int = 20):
    sorted_tasks = sorted(
        tasks.values(), key=lambda t: t.created_at, reverse=True
    )
    return sorted_tasks[:limit]


@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    task = tasks[task_id]
    if task.work_dir:
        shutil.rmtree(task.work_dir, ignore_errors=True)
    del tasks[task_id]
    return {"message": "작업이 삭제되었습니다."}


if __name__ == "__main__":
    import uvicorn
    print(f"VrunGPU Server v0.3.0 starting with {gpu_pool.gpu_count} GPUs")
    uvicorn.run(app, host="0.0.0.0", port=9825)
