# Pro*C / C에서 vrungpu LLM 호출하기

증권사 대외계처럼 **Pro*C/C로 된 배치·온라인 프로그램**에서 vrungpu에 올린 모델(기본 `qwen3.8:27b`)을 호출하는 예제입니다.
vrungpu의 OpenAI 호환 API(`POST /v1/chat/completions`)를 **libcurl 없이 POSIX 소켓 + iconv**만으로 호출하므로 AIX/HP-UX/Linux 어디서나 C99 컴파일러만 있으면 됩니다.

| 파일 | 설명 |
|------|------|
| `llm_client.h/.c` | 호출 라이브러리. `llm_chat()`(UTF-8), `llm_chat_euckr()`(EUC-KR/CP949 입출력) |
| `llm_cli.c` | Oracle 없이 호출만 테스트하는 CLI |
| `llm_demo.pc` | Pro*C 예제: LLM 호출 → 결과를 `TB_LLM_LOG`에 INSERT/COMMIT |
| `tb_llm_log.sql` | 예제 테이블/시퀀스 DDL |
| `check_on_vrungpu.py` | `.pc`를 vrungpu 서버로 보내 **proc 전처리 + gcc + libclntsh 링크** 검증 (DB 실행 X) |
| `Makefile` | `make llm_cli`, `make llm_demo` |

## 1. 사전 확인

```bash
# 모델이 로드되어 있어야 함 (없으면 503) — 운영자가 /llm/start 로 올림
curl http://your-server:9825/v1/models
```

증권사 개발망은 망분리로 외부 IP가 막혀 있는 경우가 많습니다. **호출할 서버에서 위 curl이 되는지 먼저 확인**하세요.

## 2. 호출 테스트 (Oracle 불필요)

```bash
make llm_cli
./llm_cli your-server 9825 "Pro*C에서 VARCHAR 호스트 변수를 쓰는 이유를 한 줄로"

# EUC-KR 입출력 (대외계 소스/전문이 EUC-KR인 경우)
echo "고정길이 전문의 계좌번호 필드 검증 방법" | iconv -t CP949 \
  | ./llm_cli -e your-server 9825 | iconv -f CP949
```

옵션: `-e` EUC-KR, `-t` thinking 켜기(느림, 타임아웃 600초), `-m 모델명`. `VRUNGPU_API_TOKEN` 환경변수가 있으면 Bearer 토큰으로 보냅니다.

## 3. 내 프로그램에 붙이기

```c
#include "llm_client.h"

llm_conf conf = { 0 };
conf.host = "your-server";
conf.port = "9825";
conf.system_prompt = "항상 한국어로 간결하게 답하라.";   /* llm_chat_euckr이면 EUC-KR 문자열 */
conf.timeout_sec = 120;          /* 연결/송신/수신 타임아웃 */
/* conf.model = NULL → 서버에 로드된 기본 모델, conf.think = 0 → reasoning_effort="none" (2~5초) */

char answer[16384];
int rc = llm_chat_euckr(&conf, question_euckr, answer, sizeof answer);
if (rc != LLM_OK) { /* 아래 반환 코드 참고 — LLM 실패가 본 업무 처리를 막지 않게 */ }
```

| 반환 | 의미 |
|------|------|
| `LLM_OK` (0) | 성공 |
| `LLM_ERR_IO` (-1) | 연결/송수신 실패, 타임아웃 |
| `LLM_ERR_HTTP` (-2) | HTTP 200 아님 (예: 503 모델 미로드). `answer`에 응답 앞부분 |
| `LLM_ERR_PARSE` (-3) | 응답에 content 없음 |
| `LLM_ERR_SIZE` (-4) | 요청/응답이 버퍼 초과 (`answer`는 잘린 채로 채워짐) |
| `LLM_ERR_ICONV` (-5) | EUC-KR ↔ UTF-8 변환 실패 |

**소켓 코드는 `.pc`에 넣지 말고 `llm_client.c`처럼 별도 `.c`로 컴파일해서 링크하세요.** `.pc` 안에서 `<sys/socket.h>` 등을 include하면 proc가 `linux/stddef.h`의 GNU 가변 매크로를 해석하지 못해 `PCC-S-02014`가 납니다. `.pc`에는 `EXEC SQL`과 업무 로직만 두는 구조가 안전합니다.

스레드(`EXEC SQL ENABLE THREADS` + `CONTEXT USE`)에서 호출해도 됩니다. `llm_chat`은 전역 상태가 없고 호출마다 연결을 새로 엽니다.

## 4. Pro*C 데모 빌드/실행

```bash
sqlplus user/pass@DB @tb_llm_log.sql

# Instant Client(SDK + Precompiler) 기준. sys_include 등은 pcscfg.cfg로 지정
make ORACLE_HOME=/opt/oracle/instantclient_21_19 CONFIG=/path/pcscfg.cfg llm_demo
# 정식 ORACLE_HOME이면
make ORA_INC=$ORACLE_HOME/precomp/public ORA_LIB=$ORACLE_HOME/lib PROC=$ORACLE_HOME/bin/proc llm_demo

export NLS_LANG=KOREAN_KOREA.KO16MSWIN949     # 질문/답변을 EUC-KR(CP949)로 DB에 저장
ORA_USERID=user/pass@DB LLM_HOST=your-server ./llm_demo "$(echo '대외계 응답코드 0000의 의미' | iconv -t CP949)"
```

답변은 `VARCHAR2(4000)`에 맞춰 EUC-KR 2바이트 문자 중간에서 끊기지 않게 잘라 저장합니다.

## 5. Pro*C 빌드를 vrungpu에서 검증

로컬에 Precompiler가 없거나, **에이전트(LLM)가 생성한 Pro*C 코드를 자동 검증**할 때 사용합니다. vrungpu 서버의 Instant Client 21.19로 proc → gcc → `-lclntsh` 링크까지 수행하고 결과를 돌려줍니다 (`/run/sync`, `device=cpu`, GPU 사용 안 함).

```bash
python3 check_on_vrungpu.py http://your-server:9825 llm_demo.pc llm_client.c llm_client.h
# === proc llm_demo.pc: rc=0 / === gcc ...: rc=0 / === link: rc=0 / RESULT: OK   (exit 0)

python3 check_on_vrungpu.py http://your-server:9825 mt_job.pc --proc-opt threads=yes --ldlib -lpthread
```

실패하면 `PCC-S-02201, Encountered the symbol ...` 같은 proc 오류가 그대로 나오고 exit 1이므로, 에이전트의 "생성 → 빌드 검증 → 수정" 루프에 바로 넣을 수 있습니다. **DB에 실제로 실행하지는 않으므로** `sqlcode` 처리, 배열 FETCH 등 런타임 동작은 개발 DB에서 따로 검증해야 합니다.

## 주의사항

- **보안/규정**: 현재 vrungpu는 HTTP 평문이고 인증이 꺼져 있을 수 있습니다. 실제 고객정보·계좌번호·운영 전문을 보내지 말고 가상 데이터로 테스트하세요.
- **답변 검증 필수**: 테스트 중 모델이 "MODE=ORACLE에서 NO DATA FOUND의 sqlcode"를 `100`이라고 답했습니다 (정답은 `1403`, `100`은 MODE=ANSI). Pro*C 세부 동작에 대한 모델 답은 컴파일·DB 테스트로 확인하세요.
- **EUC-KR**: JSON은 UTF-8이어야 하므로 EUC-KR을 그대로 보내면 깨집니다. `llm_chat_euckr()`이 CP949 ↔ UTF-8 변환을 대신합니다. CP949에 없는 문자(이모지 등)는 `?`로 바뀝니다.
- **타임아웃**: thinking off면 보통 2~6초, on이면 수 분 걸릴 수 있습니다. 온라인 거래 경로에서는 동기 호출을 피하고 배치/비동기로 쓰는 것을 권장합니다.
- 응답은 1MB까지 받습니다(`RESP_CAP`). 스트리밍(`stream`)은 지원하지 않습니다.
