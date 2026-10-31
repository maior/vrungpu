/* vrungpu /v1/chat/completions C 클라이언트 (libcurl 없이 POSIX 소켓 + iconv)
 *
 * Pro*C(.pc) 파일에 직접 넣지 말고 이 .c/.h를 별도로 컴파일해서 링크한다.
 * proc가 소켓 헤더(linux/stddef.h)의 GNU 매크로를 파싱하지 못해 PCC-S-02014가 날 수 있다.
 */
#ifndef LLM_CLIENT_H
#define LLM_CLIENT_H

#include <stddef.h>

typedef struct {
    const char *host;           /* vrungpu 서버, 예: "your-server" */
    const char *port;           /* 예: "9825" */
    const char *model;          /* NULL이면 서버에 로드된 기본 모델 */
    const char *system_prompt;  /* UTF-8, NULL이면 생략 */
    const char *api_token;      /* VRUNGPU_API_TOKEN 사용 시, NULL이면 생략 */
    int max_tokens;             /* 0이면 1024 */
    int timeout_sec;            /* 연결/송신/수신 타임아웃, 0이면 120 */
    int think;                  /* 0: reasoning_effort="none"(빠름), 1: 모델 기본 */
} llm_conf;

/* 반환 코드 */
#define LLM_OK          0
#define LLM_ERR_IO     -1   /* 연결/송수신 실패, 타임아웃 */
#define LLM_ERR_HTTP   -2   /* HTTP 200이 아님 (answer에 응답 앞부분) */
#define LLM_ERR_PARSE  -3   /* 응답 JSON에 content 없음 */
#define LLM_ERR_SIZE   -4   /* 요청/응답이 버퍼보다 큼 (answer는 잘린 채로 채워짐) */
#define LLM_ERR_ICONV  -5   /* EUC-KR <-> UTF-8 변환 실패 */

/* 질문/답변 모두 UTF-8 */
int llm_chat(const llm_conf *conf, const char *question, char *answer, size_t cap);

/* 질문/답변(및 conf->system_prompt) 모두 EUC-KR(CP949). 대외계 전문/소스가 EUC-KR일 때 사용 */
int llm_chat_euckr(const llm_conf *conf, const char *question, char *answer, size_t cap);

/* 문자셋 변환 유틸 (from/to 예: "CP949", "UTF-8"). 성공 시 출력 길이, 변환 실패 -1, 버퍼 부족 -2 */
long llm_iconv(const char *from, const char *to, const char *src, char *dst, size_t cap);

#endif
