"""
VrunGPU Client - 원격 GPU 서버 클라이언트
멀티 GPU 병렬 실행 지원
"""

import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


class VrunGPUClient:
    def __init__(self, server_url: str = "http://localhost:9825"):
        self.server_url = server_url.rstrip("/")

    def check_health(self) -> dict:
        """서버 상태 확인"""
        resp = requests.get(f"{self.server_url}/")
        resp.raise_for_status()
        return resp.json()

    def get_gpu_info(self) -> dict:
        """GPU 정보 조회"""
        resp = requests.get(f"{self.server_url}/gpu")
        resp.raise_for_status()
        return resp.json()

    def get_gpu_pool(self) -> dict:
        """GPU 풀 상태 조회"""
        resp = requests.get(f"{self.server_url}/gpu/pool")
        resp.raise_for_status()
        return resp.json()

    def run_sync(self, code: str, timeout: int = 300, gpu_id: int | None = None) -> dict:
        """동기 실행 - 완료까지 대기"""
        payload = {"code": code, "timeout": timeout}
        if gpu_id is not None:
            payload["gpu_id"] = gpu_id
        resp = requests.post(f"{self.server_url}/run/sync", json=payload)
        resp.raise_for_status()
        return resp.json()

    def run_async(self, code: str, timeout: int = 300, gpu_id: int | None = None) -> str:
        """비동기 실행 - task_id 반환"""
        payload = {"code": code, "timeout": timeout}
        if gpu_id is not None:
            payload["gpu_id"] = gpu_id
        resp = requests.post(f"{self.server_url}/run/async", json=payload)
        resp.raise_for_status()
        return resp.json()["task_id"]

    def get_task_status(self, task_id: str) -> dict:
        """작업 상태 조회"""
        resp = requests.get(f"{self.server_url}/task/{task_id}")
        resp.raise_for_status()
        return resp.json()

    def wait_for_task(self, task_id: str, poll_interval: float = 1.0) -> dict:
        """작업 완료까지 대기"""
        while True:
            status = self.get_task_status(task_id)
            if status["status"] in ("completed", "failed"):
                return status
            time.sleep(poll_interval)

    def run_file(self, filepath: str, timeout: int = 300, gpu_id: int | None = None) -> str:
        """파일 업로드 후 비동기 실행"""
        params = {"timeout": timeout}
        if gpu_id is not None:
            params["gpu_id"] = gpu_id
        with open(filepath, "rb") as f:
            resp = requests.post(
                f"{self.server_url}/run/file",
                files={"file": f},
                params=params,
            )
        resp.raise_for_status()
        return resp.json()["task_id"]

    def run_parallel(self, codes: list[str], timeout: int = 300) -> list[dict]:
        """여러 코드를 병렬로 실행 (각각 다른 GPU에 할당)"""
        task_ids = []
        for code in codes:
            task_id = self.run_async(code, timeout)
            task_ids.append(task_id)

        results = []
        for task_id in task_ids:
            result = self.wait_for_task(task_id)
            results.append(result)
        return results

    def list_tasks(self, limit: int = 20) -> list[dict]:
        """작업 목록 조회"""
        resp = requests.get(f"{self.server_url}/tasks", params={"limit": limit})
        resp.raise_for_status()
        return resp.json()


# 사용 예제
if __name__ == "__main__":
    client = VrunGPUClient("http://localhost:9825")

    # 1. 서버 상태 확인
    print("=== 서버 상태 ===")
    health = client.check_health()
    print(f"Service: {health['service']} v{health['version']}")
    print(f"GPUs: {health['gpu_count']} total, {health['available_gpus']} available")

    # 2. GPU 정보 확인
    print("\n=== GPU 정보 ===")
    gpu_info = client.get_gpu_info()
    for gpu in gpu_info["devices"]:
        print(f"  GPU {gpu['index']}: {gpu['name']}")
        print(f"    Memory: {gpu['memory_free_mb']}/{gpu['memory_total_mb']} MB free")

    # 3. 동기 실행 예제
    print("\n=== 동기 실행 ===")
    code = """
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""
    result = client.run_sync(code)
    print(f"Status: {result['status']}, GPU: {result['gpu_id']}")
    print(f"Output:\n{result['stdout']}")

    # 4. 특정 GPU 지정
    print("\n=== 특정 GPU 지정 (GPU 1) ===")
    result = client.run_sync("import os; print(f\"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES')}\")", gpu_id=1)
    print(f"Output: {result['stdout'].strip()}")

    # 5. 병렬 실행 예제
    print("\n=== 병렬 실행 (2개 작업) ===")
    codes = [
        "import time, os; time.sleep(2); print(f'Task A on GPU {os.environ.get(\"CUDA_VISIBLE_DEVICES\")}')",
        "import time, os; time.sleep(2); print(f'Task B on GPU {os.environ.get(\"CUDA_VISIBLE_DEVICES\")}')",
    ]

    start = time.time()
    task_ids = [client.run_async(code) for code in codes]
    print(f"Submitted {len(task_ids)} tasks")

    # GPU 풀 상태 확인
    pool = client.get_gpu_pool()
    print(f"GPU Pool: {pool['available_gpus']} available, {len(pool['busy_gpus'])} busy")

    # 완료 대기
    results = [client.wait_for_task(tid) for tid in task_ids]
    elapsed = time.time() - start

    for i, r in enumerate(results):
        print(f"  Task {i+1}: {r['stdout'].strip()} (GPU {r['gpu_id']})")
    print(f"Total time: {elapsed:.2f}s (parallel execution)")
