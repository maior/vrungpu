# VrunGPU

원격 GPU 학습/추론 실행 서버. 노트북에서 REST API로 GPU 서버에 Python 코드를 보내서 실행하고 결과를 받을 수 있습니다.

## 주요 기능

- **동기/비동기 실행**: 짧은 작업은 동기, 긴 학습은 비동기로 실행
- **멀티 GPU 지원**: 여러 GPU에 작업을 자동 분배하여 병렬 실행
- **ZIP 프로젝트 업로드**: 여러 Python 파일 + 데이터셋을 ZIP으로 묶어서 업로드
- **WebSocket 실시간 스트리밍**: 학습 로그를 실시간으로 확인
- **웹 대시보드**: GPU 상태, 작업 현황을 시각적으로 모니터링

## 설치

```bash
# 가상환경 생성
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# PyTorch (CUDA) 설치
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 대시보드 설치 (선택)
cd dashboard && npm install && npm run build
```

## 서버 실행

```bash
# 백엔드 (포트 9825)
.venv/bin/python server.py

# 대시보드 (포트 9824)
cd dashboard && npm run start -- -p 9824
```

## 접속

| 서비스 | URL |
|--------|-----|
| Backend API | http://서버IP:9825 |
| Dashboard | http://서버IP:9824 |
| API 문서 | http://서버IP:9825/docs |

---

## ZIP 프로젝트 업로드

여러 Python 파일과 데이터셋을 함께 업로드해서 학습할 수 있습니다.

### 프로젝트 구조 예시

```
my_project/
├── train.py          # Entry point (메인 실행 파일)
├── model.py          # 모델 정의
├── utils.py          # 유틸리티 함수
├── config.json       # 설정 파일
└── data/             # 데이터셋 폴더
    ├── train.csv
    └── test.csv
```

### ZIP 압축 방법

```bash
# 방법 1: 폴더 전체를 압축
cd /path/to/projects
zip -r my_project.zip my_project/

# 방법 2: 폴더 안에서 파일들만 압축
cd my_project
zip -r ../my_project.zip .

# 방법 3: 특정 파일만 압축
zip my_project.zip train.py model.py utils.py config.json
zip -r my_project.zip data/  # 데이터 폴더 추가
```

### 업로드 및 실행

**curl로 업로드:**
```bash
curl -X POST http://서버IP:9825/run/project \
  -F "file=@my_project.zip" \
  -F "entry_point=train.py" \
  -F "timeout=3600"
```

**Python으로 업로드:**
```python
import requests

with open("my_project.zip", "rb") as f:
    response = requests.post(
        "http://서버IP:9825/run/project",
        files={"file": f},
        data={
            "entry_point": "train.py",
            "timeout": 3600,
            "gpu_id": 0  # 선택: 특정 GPU 지정
        }
    )

task_id = response.json()["task_id"]
print(f"Task started: {task_id}")
```

**대시보드에서 업로드:**
1. http://서버IP:9824 접속
2. "파일 업로드" 탭 선택
3. ZIP 파일 선택
4. Entry point 입력 (예: train.py)
5. "업로드 및 실행" 클릭

### 주의사항

- Entry point 파일은 ZIP 최상위 또는 단일 폴더 안에 있어야 함
- 상대 import 사용 가능 (예: `from model import MyModel`)
- 현재 작업 디렉토리에서 실행되므로 파일 경로는 상대경로 사용
- 대용량 데이터셋은 서버에 미리 두고 경로만 참조하는 것을 권장

---

## API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/` | GET | 서버 상태 및 GPU 수 확인 |
| `/gpu` | GET | GPU 상세 정보 (메모리, 사용률) |
| `/gpu/pool` | GET | GPU 풀 상태 (할당 현황) |
| `/ws` | WebSocket | 실시간 모니터링 연결 |
| `/run/sync` | POST | 동기 실행 (완료까지 대기) |
| `/run/async` | POST | 비동기 실행 (task_id 반환) |
| `/run/project` | POST | **ZIP/파일 업로드 후 실행** |
| `/task/{task_id}` | GET | 작업 상태/결과 조회 |
| `/tasks` | GET | 작업 목록 |
| `/task/{task_id}` | DELETE | 작업 삭제 |

---

## 사용 예제

### 1. 서버 상태 확인

```bash
curl http://서버IP:9825/
```

```json
{
  "service": "VrunGPU",
  "status": "running",
  "version": "0.3.0",
  "gpu_count": 2,
  "available_gpus": 2
}
```

### 2. 동기 실행 (짧은 작업)

```bash
curl -X POST http://서버IP:9825/run/sync \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import torch\nprint(torch.cuda.get_device_name(0))",
    "timeout": 30
  }'
```

### 3. 비동기 실행 (긴 학습)

```bash
# 작업 제출
curl -X POST http://서버IP:9825/run/async \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import time\nfor i in range(10):\n    print(f\"Step {i}\")\n    time.sleep(1)",
    "timeout": 300
  }'

# 결과: {"task_id": "abc-123", "status": "pending", ...}

# 작업 상태 확인
curl http://서버IP:9825/task/abc-123
```

### 4. 특정 GPU 지정

```bash
curl -X POST http://서버IP:9825/run/sync \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import torch\nprint(f\"Running on GPU: {torch.cuda.current_device()}\")",
    "gpu_id": 1
  }'
```

---

## WebSocket 실시간 모니터링

대시보드는 WebSocket으로 실시간 업데이트를 받습니다.

### 메시지 타입

```javascript
// 연결 시 초기 데이터
{ "type": "init", "gpus": [...], "tasks": [...] }

// GPU 상태 업데이트
{ "type": "gpu_update", "gpus": [...] }

// 작업 상태 변경
{ "type": "task_update", "task": {...} }

// 실행 중 출력 (실시간 로그)
{ "type": "task_output", "task_id": "...", "stream": "stdout", "line": "..." }
```

### JavaScript 연결 예시

```javascript
const ws = new WebSocket("ws://서버IP:9825/ws");

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === "task_output") {
    console.log(data.line);  // 실시간 로그 출력
  }
};
```

---

## Python 클라이언트

```python
from client import VrunGPUClient

client = VrunGPUClient("http://서버IP:9825")

# GPU 정보
print(client.get_gpu_info())

# 동기 실행
result = client.run_sync("print('Hello GPU!')")
print(result['stdout'])

# 비동기 실행 + 완료 대기
task_id = client.run_async(train_code)
result = client.wait_for_task(task_id)
print(result['stdout'])

# 파일 업로드
task_id = client.run_file("train.py", timeout=600, gpu_id=0)
result = client.wait_for_task(task_id)
```

---

## 병렬 실행

GPU가 여러 개일 때, 동시에 여러 작업을 제출하면 각각 다른 GPU에서 병렬 실행됩니다.

```bash
# 작업 1 제출 -> GPU 0에 할당
curl -X POST http://서버IP:9825/run/async -d '{"code": "..."}'

# 작업 2 제출 -> GPU 1에 할당
curl -X POST http://서버IP:9825/run/async -d '{"code": "..."}'

# GPU 풀 상태 확인
curl http://서버IP:9825/gpu/pool
# {"total_gpus": 2, "available_gpus": [], "busy_gpus": {"task1": 0, "task2": 1}}
```

모든 GPU가 사용 중이면 새 작업은 `queued` 상태로 대기하다가 GPU가 해제되면 자동으로 실행됩니다.

---

## 프로젝트 구조

```
vrungpu/
├── server.py           # FastAPI 서버 (메인)
├── client.py           # Python 클라이언트
├── requirements.txt    # 의존성
├── test_server.py      # 테스트 스크립트
├── README.md           # 문서
├── dashboard/          # Next.js 대시보드
│   ├── app/
│   │   └── page.tsx    # 대시보드 UI
│   └── package.json
└── examples/           # 예제 프로젝트
    └── mnist_project/
        ├── train.py
        ├── model.py
        └── utils.py
```

---

## 예제: MNIST 학습

```bash
# 예제 프로젝트 압축
cd examples
zip -r mnist_project.zip mnist_project/

# 업로드 및 실행
curl -X POST http://서버IP:9825/run/project \
  -F "file=@mnist_project.zip" \
  -F "entry_point=train.py" \
  -F "timeout=300"
```

출력 예시:
```
Device: cuda
GPU: Tesla V100S-PCIE-32GB

=== Training Started ===
Epochs: 5, Batch Size: 64, LR: 0.001
Train samples: 1000, Test samples: 200

Epoch 1/5 - Loss: 2.4748, Accuracy: 11.50%
Epoch 2/5 - Loss: 2.4555, Accuracy: 10.00%
...
=== Training Completed ===
Final Accuracy: 8.00%
Model saved to model.pt
```
