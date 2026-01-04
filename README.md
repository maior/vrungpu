# VrunGPU

원격 GPU 학습/추론 실행 서버. 노트북에서 REST API로 GPU 서버에 Python 코드를 보내서 실행하고 결과를 받을 수 있습니다.

## 주요 기능

- **동기/비동기 실행**: 짧은 작업은 동기, 긴 학습은 비동기로 실행
- **멀티 GPU 지원**: 여러 GPU에 작업을 자동 분배하여 병렬 실행
- **ZIP 프로젝트 업로드**: 여러 Python 파일 + 데이터셋을 ZIP으로 묶어서 업로드
- **WebSocket 실시간 스트리밍**: 학습 로그를 실시간으로 확인
- **웹 대시보드**: GPU 상태, 작업 현황을 시각적으로 모니터링 (D3.js 차트)
- **SQLite 영구 저장소**: 서버 재시작 후에도 작업/모델 데이터 유지
- **모델 관리**: 학습된 모델 등록, 조회, 다운로드
- **추론 API**: 등록된 모델로 추론 실행
- **진행상황 추적**: 학습 진행률을 실시간으로 확인

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

## 데이터 저장 구조

v0.4.0부터 모든 데이터는 영구적으로 저장됩니다.

```
vrungpu/
└── data/
    ├── vrungpu.db        # SQLite 데이터베이스 (작업, 모델 메타데이터)
    ├── workspaces/       # 작업별 실행 환경 및 결과물
    │   ├── {task_id}/
    │   │   ├── main.py
    │   │   └── model.pt
    │   └── inference_{task_id}/
    │       └── inference.py
    ├── models/           # 등록된 모델 파일
    │   └── {model_id}/
    │       └── model.pt
    └── uploads/          # 업로드된 ZIP 파일
```

---

## 진행상황 추적 (Progress Tracking)

학습 코드에서 진행상황을 출력하면 서버가 자동으로 파싱하여 추적합니다.

### 진행상황 출력 형식

```python
# 학습 코드에서 다음 형식으로 출력
print(f"[PROGRESS:{progress_percent}:{message}]")
```

### 예시 코드

```python
import torch
import torch.nn as nn

# 학습 시작
print("[PROGRESS:0:학습 시작]")

for epoch in range(1, 11):
    # 학습 로직...
    loss = train_one_epoch()

    # 진행상황 출력 (10%, 20%, ... 100%)
    progress = epoch * 10
    print(f"[PROGRESS:{progress}:Epoch {epoch}/10 완료, Loss: {loss:.4f}]")

print("[PROGRESS:100:학습 완료!]")

# 모델 저장
torch.save(model.state_dict(), "model.pt")
print("모델이 model.pt에 저장되었습니다.")
```

### 진행상황 조회

```bash
# 작업 상태에서 progress와 progress_message 확인
curl http://서버IP:9825/task/{task_id}
```

응답 예시:
```json
{
  "task_id": "abc-123",
  "status": "running",
  "progress": 60.0,
  "progress_message": "Epoch 6/10 완료, Loss: 0.0234",
  "task_type": "training",
  ...
}
```

---

## 모델 관리 API

### 모델 등록

학습 완료 후 모델 파일을 서버에 등록합니다.

```bash
curl -X POST http://서버IP:9825/model/register \
  -F "name=mnist-classifier-v1" \
  -F "model_file=@model.pt" \
  -F "model_type=classifier" \
  -F "framework=pytorch"
```

응답:
```json
{
  "model_id": "a1b2c3d4",
  "name": "mnist-classifier-v1",
  "model_type": "classifier",
  "framework": "pytorch",
  "status": "ready",
  "file_size": 12345,
  "created_at": "2026-01-05T12:00:00"
}
```

### 모델 목록 조회

```bash
curl http://서버IP:9825/models
```

### 특정 모델 조회

```bash
curl http://서버IP:9825/model/{model_id}
```

### 모델 다운로드

```bash
curl -O http://서버IP:9825/model/{model_id}/download
```

### 모델 삭제

```bash
curl -X DELETE http://서버IP:9825/model/{model_id}
```

---

## 추론 API

등록된 모델로 추론을 실행합니다.

### 추론 실행

```bash
curl -X POST http://서버IP:9825/model/{model_id}/inference \
  -H "Content-Type: application/json" \
  -d '{
    "input_data": {"image": "base64...", "batch_size": 1},
    "timeout": 60
  }'
```

응답:
```json
{
  "task_id": "inference-xyz",
  "status": "pending",
  "message": "추론 작업이 시작되었습니다. Model: mnist-classifier-v1"
}
```

### 추론 결과 조회

```bash
curl http://서버IP:9825/task/{task_id}
```

응답:
```json
{
  "task_id": "inference-xyz",
  "status": "completed",
  "task_type": "inference",
  "model_id": "a1b2c3d4",
  "stdout": "{\"success\": true, \"result\": {...}}",
  ...
}
```

### 커스텀 추론 로직

기본 추론 스크립트는 단순히 모델을 로드하고 입력 데이터를 반환합니다.
복잡한 추론이 필요하면 `/run/async` 엔드포인트를 사용하세요.

```python
code = """
import torch
import json

# 모델 로드
model = torch.load('/path/to/model.pt')
model.eval()

# 입력 처리 및 추론
input_tensor = preprocess(input_data)
with torch.no_grad():
    output = model(input_tensor)

# 결과 출력
result = {"prediction": output.argmax().item()}
print(json.dumps(result))
"""

response = requests.post(
    "http://서버IP:9825/run/async",
    json={"code": code, "timeout": 60}
)
```

---

## 통계 API

서버 전체 통계를 조회합니다.

```bash
curl http://서버IP:9825/stats
```

응답:
```json
{
  "tasks": {
    "by_status": {
      "completed": 150,
      "running": 2,
      "failed": 5
    },
    "avg_duration": {
      "training": 3600,
      "inference": 5
    },
    "last_24h": 25
  },
  "models": {
    "by_status": {
      "ready": 10,
      "archived": 3
    },
    "total": 13
  },
  "gpu": {
    "usage_count": {"0": 100, "1": 80},
    "current": {
      "total_gpus": 2,
      "available_gpus": [0, 1],
      "busy_gpus": {}
    }
  }
}
```

---

## API 엔드포인트 (전체)

### 기본 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/` | GET | 서버 상태 및 스토리지 정보 |
| `/gpu` | GET | GPU 상세 정보 (메모리, 사용률) |
| `/gpu/pool` | GET | GPU 풀 상태 (할당 현황) |
| `/ws` | WebSocket | 실시간 모니터링 연결 |
| `/stats` | GET | 서버 전체 통계 |

### 작업 실행 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/run/sync` | POST | 동기 실행 (완료까지 대기) |
| `/run/async` | POST | 비동기 실행 (task_id 반환) |
| `/run/project` | POST | ZIP/파일 업로드 후 실행 |
| `/task/{task_id}` | GET | 작업 상태/결과/진행률 조회 |
| `/tasks` | GET | 작업 목록 |
| `/task/{task_id}` | DELETE | 작업 삭제 |
| `/task/{task_id}/progress` | PUT | 진행상황 수동 업데이트 |

### 모델 관리 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/models` | GET | 모델 목록 |
| `/model/register` | POST | 모델 등록 (파일 업로드) |
| `/model/{model_id}` | GET | 모델 정보 조회 |
| `/model/{model_id}` | DELETE | 모델 삭제 |
| `/model/{model_id}/download` | GET | 모델 파일 다운로드 |
| `/model/{model_id}/inference` | POST | 모델로 추론 실행 |

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

## 사용 예제

### 1. 서버 상태 확인

```bash
curl http://서버IP:9825/
```

```json
{
  "service": "VrunGPU",
  "status": "running",
  "version": "0.4.0",
  "gpu_count": 2,
  "available_gpus": 2,
  "total_tasks": 150,
  "total_models": 10,
  "storage": {
    "workspaces": "/path/to/data/workspaces",
    "models": "/path/to/data/models",
    "database": "/path/to/data/vrungpu.db"
  }
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
    "code": "import time\nfor i in range(10):\n    print(f\"[PROGRESS:{(i+1)*10}:Step {i+1}/10]\")\n    time.sleep(1)",
    "timeout": 300
  }'

# 결과: {"task_id": "abc-123", "status": "pending", ...}

# 작업 상태 확인 (진행률 포함)
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
├── data/               # 영구 저장소
│   ├── vrungpu.db      # SQLite 데이터베이스
│   ├── workspaces/     # 작업 실행 환경
│   ├── models/         # 등록된 모델
│   └── uploads/        # 업로드 파일
├── dashboard/          # Next.js 대시보드
│   ├── app/
│   │   ├── page.tsx    # 대시보드 UI (D3.js 차트)
│   │   └── globals.css # 애니메이션 스타일
│   └── package.json
└── examples/           # 예제 프로젝트
    └── mnist_project/
        ├── train.py
        ├── model.py
        └── utils.py
```

---

## 예제: MNIST 학습 + 모델 등록 + 추론

### 1. 학습 실행

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

### 2. 학습 진행 확인

```bash
# 진행률 확인
curl http://서버IP:9825/task/{task_id}

# 응답에서 progress 확인
# "progress": 60.0, "progress_message": "Epoch 6/10..."
```

### 3. 모델 등록

학습 완료 후 workspace에서 모델 파일을 가져와 등록:

```bash
curl -X POST http://서버IP:9825/model/register \
  -F "name=mnist-v1" \
  -F "model_file=@/path/to/workspace/{task_id}/model.pt" \
  -F "model_type=classifier"
```

### 4. 추론 실행

```bash
curl -X POST http://서버IP:9825/model/{model_id}/inference \
  -H "Content-Type: application/json" \
  -d '{"input_data": {"image": "..."}, "timeout": 30}'
```

---

## 버전 히스토리

- **v0.4.0** - SQLite 영구 저장소, 모델 관리 API, 추론 API, 진행상황 추적
- **v0.3.0** - D3.js 대시보드, 부드러운 애니메이션
- **v0.2.0** - ZIP 프로젝트 업로드, 멀티 GPU 지원
- **v0.1.0** - 기본 동기/비동기 실행
