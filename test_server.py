"""
VrunGPU 서버 테스트 스크립트
서버가 실행 중인 상태에서 실행하세요: .venv/bin/python test_server.py
"""

import time
import requests
import json

SERVER_URL = "http://localhost:9825"


def test_health():
    """서버 상태 확인"""
    print("=" * 50)
    print("1. 서버 상태 확인")
    print("=" * 50)
    resp = requests.get(f"{SERVER_URL}/")
    data = resp.json()
    print(f"Status: {resp.status_code}")
    print(f"Response: {json.dumps(data, indent=2)}")
    assert data["status"] == "running"
    assert data["gpu_count"] > 0
    print("PASSED\n")
    return data["gpu_count"]


def test_gpu_info():
    """GPU 정보 조회"""
    print("=" * 50)
    print("2. GPU 정보 조회")
    print("=" * 50)
    resp = requests.get(f"{SERVER_URL}/gpu")
    data = resp.json()
    print(f"Available: {data['available']}")
    for gpu in data["devices"]:
        print(f"  GPU {gpu['index']}: {gpu['name']}")
        print(f"    Memory: {gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB")
        print(f"    Utilization: {gpu['utilization_percent']}%")
    assert data["available"] == True
    print("PASSED\n")


def test_sync_execution():
    """동기 실행 테스트"""
    print("=" * 50)
    print("3. 동기 실행 테스트")
    print("=" * 50)
    code = """
import sys
print(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
print("Hello from VrunGPU!")
result = sum(range(1000))
print(f"Sum: {result}")
"""
    resp = requests.post(
        f"{SERVER_URL}/run/sync",
        json={"code": code, "timeout": 30}
    )
    data = resp.json()
    print(f"Status: {data['status']}")
    print(f"GPU ID: {data['gpu_id']}")
    print(f"Output:\n{data['stdout']}")
    assert data["status"] == "completed"
    assert "Hello from VrunGPU!" in data["stdout"]
    print("PASSED\n")


def test_sync_gpu_execution():
    """GPU 사용 동기 실행 테스트"""
    print("=" * 50)
    print("4. GPU 사용 동기 실행 테스트")
    print("=" * 50)
    code = """
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.randn(100, 100).cuda()
    y = torch.matmul(x, x)
    print(f"Matrix multiplication done on GPU")
"""
    resp = requests.post(
        f"{SERVER_URL}/run/sync",
        json={"code": code, "timeout": 60}
    )
    data = resp.json()
    print(f"Status: {data['status']}")
    print(f"GPU ID: {data['gpu_id']}")
    print(f"Output:\n{data['stdout']}")
    if data["stderr"]:
        print(f"Stderr:\n{data['stderr']}")
    assert data["status"] == "completed"
    print("PASSED\n")


def test_async_execution():
    """비동기 실행 테스트"""
    print("=" * 50)
    print("5. 비동기 실행 테스트")
    print("=" * 50)
    code = """
import time
for i in range(5):
    print(f"Step {i+1}/5")
    time.sleep(0.5)
print("Done!")
"""
    # 작업 제출
    resp = requests.post(
        f"{SERVER_URL}/run/async",
        json={"code": code, "timeout": 60}
    )
    data = resp.json()
    task_id = data["task_id"]
    print(f"Task ID: {task_id}")
    print(f"Initial Status: {data['status']}")

    # 완료 대기
    while True:
        resp = requests.get(f"{SERVER_URL}/task/{task_id}")
        status = resp.json()
        print(f"  Status: {status['status']}, GPU: {status['gpu_id']}")
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(1)

    print(f"\nFinal Output:\n{status['stdout']}")
    assert status["status"] == "completed"
    assert "Done!" in status["stdout"]
    print("PASSED\n")


def test_parallel_execution(gpu_count: int):
    """병렬 실행 테스트"""
    print("=" * 50)
    print(f"6. 병렬 실행 테스트 ({gpu_count} GPUs)")
    print("=" * 50)

    if gpu_count < 2:
        print("SKIPPED (need 2+ GPUs)\n")
        return

    code = """
import torch
import time
import os

gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")
print(f"Running on GPU: {gpu_id}")

# 간단한 GPU 작업
x = torch.randn(500, 500).cuda()
for i in range(10):
    x = torch.matmul(x, x.T)
    x = x / x.max()
    time.sleep(0.2)

print(f"Completed on GPU: {gpu_id}")
"""

    # 2개의 작업 동시 제출
    task_ids = []
    for i in range(2):
        resp = requests.post(
            f"{SERVER_URL}/run/async",
            json={"code": code, "timeout": 60}
        )
        task_id = resp.json()["task_id"]
        task_ids.append(task_id)
        print(f"Task {i+1} submitted: {task_id[:8]}...")

    # GPU 풀 상태 확인
    time.sleep(0.5)
    resp = requests.get(f"{SERVER_URL}/gpu/pool")
    pool = resp.json()
    print(f"\nGPU Pool: {len(pool['available_gpus'])} available, {len(pool['busy_gpus'])} busy")

    # 두 작업이 다른 GPU에 할당되었는지 확인
    gpu_assignments = set()
    for task_id in task_ids:
        resp = requests.get(f"{SERVER_URL}/task/{task_id}")
        gpu_id = resp.json().get("gpu_id")
        if gpu_id is not None:
            gpu_assignments.add(gpu_id)

    print(f"GPU assignments: {gpu_assignments}")
    if len(gpu_assignments) == 2:
        print("Both tasks running on different GPUs!")

    # 완료 대기
    print("\nWaiting for completion...")
    for task_id in task_ids:
        while True:
            resp = requests.get(f"{SERVER_URL}/task/{task_id}")
            status = resp.json()
            if status["status"] in ("completed", "failed"):
                print(f"Task {task_id[:8]}... -> {status['status']} (GPU {status['gpu_id']})")
                break
            time.sleep(0.5)

    assert len(gpu_assignments) == 2, "Tasks should run on different GPUs"
    print("PASSED\n")


def test_gpu_pool_status():
    """GPU 풀 상태 조회 테스트"""
    print("=" * 50)
    print("7. GPU 풀 상태 조회")
    print("=" * 50)
    resp = requests.get(f"{SERVER_URL}/gpu/pool")
    data = resp.json()
    print(f"Total GPUs: {data['total_gpus']}")
    print(f"Available: {data['available_gpus']}")
    print(f"Busy: {data['busy_gpus']}")
    assert data["total_gpus"] > 0
    print("PASSED\n")


def test_specific_gpu():
    """특정 GPU 지정 테스트"""
    print("=" * 50)
    print("8. 특정 GPU 지정 테스트")
    print("=" * 50)
    code = """
import os
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
"""
    # GPU 1 지정
    resp = requests.post(
        f"{SERVER_URL}/run/sync",
        json={"code": code, "timeout": 30, "gpu_id": 1}
    )
    data = resp.json()
    print(f"Requested GPU: 1")
    print(f"Assigned GPU: {data['gpu_id']}")
    print(f"Output: {data['stdout'].strip()}")
    assert data["gpu_id"] == 1
    assert "CUDA_VISIBLE_DEVICES: 1" in data["stdout"]
    print("PASSED\n")


def test_task_list():
    """작업 목록 조회"""
    print("=" * 50)
    print("9. 작업 목록 조회")
    print("=" * 50)
    resp = requests.get(f"{SERVER_URL}/tasks?limit=5")
    data = resp.json()
    print(f"Recent tasks: {len(data)}")
    for task in data[:3]:
        print(f"  {task['task_id'][:8]}... - {task['status']} (GPU {task['gpu_id']})")
    print("PASSED\n")


def main():
    print("\n" + "=" * 50)
    print("VrunGPU Server Test Suite")
    print("=" * 50 + "\n")

    try:
        requests.get(f"{SERVER_URL}/", timeout=2)
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Server not running at {SERVER_URL}")
        print("Start the server first: .venv/bin/python server.py")
        return

    try:
        gpu_count = test_health()
        test_gpu_info()
        test_sync_execution()
        test_sync_gpu_execution()
        test_async_execution()
        test_parallel_execution(gpu_count)
        test_gpu_pool_status()
        test_specific_gpu()
        test_task_list()

        print("=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
