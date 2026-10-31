/* vrungpu /v1/chat/completions C 클라이언트 — 설명은 llm_client.h 참고 */
#include "llm_client.h"

#include <errno.h>
#include <iconv.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#define REQ_CAP   (256 * 1024)
#define RESP_CAP  (1024 * 1024)

/* UTF-8 문자열을 JSON 문자열 리터럴 내용으로 이스케이프. 넘치면 -1 */
static long json_escape(const char *src, char *dst, size_t cap) {
    size_t n = 0;
    for (; *src; src++) {
        unsigned char c = (unsigned char)*src;
        if (n + 7 >= cap) return -1;
        if (c == '"' || c == '\\') { dst[n++] = '\\'; dst[n++] = c; }
        else if (c == '\n') { dst[n++] = '\\'; dst[n++] = 'n'; }
        else if (c == '\r') { dst[n++] = '\\'; dst[n++] = 'r'; }
        else if (c == '\t') { dst[n++] = '\\'; dst[n++] = 't'; }
        else if (c < 0x20) n += snprintf(dst + n, cap - n, "\\u%04x", c);
        else dst[n++] = c;
    }
    dst[n] = 0;
    return (long)n;
}

static size_t put_utf8(unsigned cp, char *out) {
    if (cp < 0x80) { out[0] = cp; return 1; }
    if (cp < 0x800) { out[0] = 0xC0 | (cp >> 6); out[1] = 0x80 | (cp & 0x3F); return 2; }
    if (cp < 0x10000) { out[0] = 0xE0 | (cp >> 12); out[1] = 0x80 | ((cp >> 6) & 0x3F); out[2] = 0x80 | (cp & 0x3F); return 3; }
    out[0] = 0xF0 | (cp >> 18); out[1] = 0x80 | ((cp >> 12) & 0x3F);
    out[2] = 0x80 | ((cp >> 6) & 0x3F); out[3] = 0x80 | (cp & 0x3F); return 4;
}

static int hex4(const char *p, unsigned *v) {
    unsigned r = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        r <<= 4;
        if (c >= '0' && c <= '9') r |= c - '0';
        else if (c >= 'a' && c <= 'f') r |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') r |= c - 'A' + 10;
        else return -1;
    }
    *v = r;
    return 0;
}

/* choices[0].message.content 문자열을 꺼내 UTF-8로 언이스케이프 */
static int extract_content(const char *json, char *out, size_t cap) {
    const char *p = strstr(json, "\"message\"");
    size_t n = 0;
    if (!p || !(p = strstr(p, "\"content\""))) return LLM_ERR_PARSE;
    p += 9;
    while (*p == ' ' || *p == ':') p++;
    if (*p != '"') return LLM_ERR_PARSE;          /* null: 도구 호출 응답 등 */
    for (p++; *p && *p != '"'; p++) {
        char tmp[4];
        size_t k = 1;
        if (*p != '\\') tmp[0] = *p;
        else {
            unsigned cp;
            switch (*++p) {
            case 'n': tmp[0] = '\n'; break;
            case 'r': tmp[0] = '\r'; break;
            case 't': tmp[0] = '\t'; break;
            case 'b': tmp[0] = '\b'; break;
            case 'f': tmp[0] = '\f'; break;
            case 'u':
                if (hex4(p + 1, &cp)) return LLM_ERR_PARSE;
                p += 4;
                if (cp >= 0xD800 && cp < 0xDC00 && p[1] == '\\' && p[2] == 'u') {   /* 서로게이트 쌍 */
                    unsigned lo;
                    if (hex4(p + 3, &lo)) return LLM_ERR_PARSE;
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    p += 6;
                }
                k = put_utf8(cp, tmp);
                break;
            case 0: return LLM_ERR_PARSE;
            default: tmp[0] = *p;                    /* \" \\ \/ */
            }
        }
        if (n + k >= cap) { out[n] = 0; return LLM_ERR_SIZE; }
        memcpy(out + n, tmp, k);
        n += k;
    }
    out[n] = 0;
    return *p == '"' ? LLM_OK : LLM_ERR_PARSE;
}

/* 대소문자 무시 부분 문자열 검색 (strcasestr은 AIX/HP-UX에 없음) */
static int has_header(const char *hdr, const char *needle) {
    size_t n = strlen(needle);
    for (; *hdr; hdr++) {
        size_t i = 0;
        while (i < n && hdr[i] && (hdr[i] | 0x20) == (needle[i] | 0x20)) i++;
        if (i == n) return 1;
    }
    return 0;
}

/* Transfer-Encoding: chunked 본문을 제자리에서 이어붙임 (중간 프록시 대비) */
static void dechunk(char *body) {
    char *r = body, *w = body;
    for (;;) {
        char *end;
        long sz = strtol(r, &end, 16);
        if (end == r || sz <= 0) break;
        r = strstr(end, "\r\n");
        if (!r) break;
        r += 2;
        memmove(w, r, sz);
        w += sz;
        r += sz + 2;
    }
    *w = 0;
}

static int connect_to(const char *host, const char *port, int timeout_sec) {
    struct addrinfo hints, *ai, *a;
    struct timeval tv = { timeout_sec, 0 };
    int fd = -1;

    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &ai) != 0) return -1;
    for (a = ai; a; a = a->ai_next) {
        fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
        /* Linux에서 SO_SNDTIMEO는 connect()에도 적용된다 */
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
        if (connect(fd, a->ai_addr, a->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(ai);
    return fd;
}

static int write_all(int fd, const char *buf, size_t len) {
    while (len > 0) {
        ssize_t w = write(fd, buf, len);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        buf += w;
        len -= (size_t)w;
    }
    return 0;
}

int llm_chat(const llm_conf *conf, const char *question, char *answer, size_t cap) {
    int timeout = conf->timeout_sec > 0 ? conf->timeout_sec : 120;
    int max_tokens = conf->max_tokens > 0 ? conf->max_tokens : 1024;
    char *q = malloc(REQ_CAP), *sys = malloc(REQ_CAP), *body = malloc(REQ_CAP), *resp = malloc(RESP_CAP);
    char hdr[1024], model[256] = "", auth[512] = "", *p;
    int rc = LLM_ERR_IO, fd = -1, len, hlen;
    size_t got = 0;
    ssize_t r;

    if (cap) answer[0] = 0;
    if (!q || !sys || !body || !resp) goto out;
    rc = LLM_ERR_SIZE;
    if (json_escape(question, q, REQ_CAP) < 0) goto out;
    if (json_escape(conf->system_prompt ? conf->system_prompt : "", sys, REQ_CAP) < 0) goto out;
    if (conf->model) snprintf(model, sizeof model, "\"model\":\"%s\",", conf->model);
    if (conf->api_token) snprintf(auth, sizeof auth, "Authorization: Bearer %s\r\n", conf->api_token);

    len = snprintf(body, REQ_CAP, "{%s%s\"max_tokens\":%d,\"messages\":[%s%s%s{\"role\":\"user\",\"content\":\"%s\"}]}",
                   model, conf->think ? "" : "\"reasoning_effort\":\"none\",", max_tokens,
                   conf->system_prompt ? "{\"role\":\"system\",\"content\":\"" : "",
                   conf->system_prompt ? sys : "",
                   conf->system_prompt ? "\"}," : "", q);
    if (len < 0 || len >= REQ_CAP) goto out;
    hlen = snprintf(hdr, sizeof hdr,
                    "POST /v1/chat/completions HTTP/1.1\r\nHost: %s:%s\r\n%s"
                    "Content-Type: application/json; charset=utf-8\r\nContent-Length: %d\r\nConnection: close\r\n\r\n",
                    conf->host, conf->port, auth, len);
    if (hlen < 0 || hlen >= (int)sizeof hdr) goto out;

    rc = LLM_ERR_IO;
    if ((fd = connect_to(conf->host, conf->port, timeout)) < 0) goto out;
    if (write_all(fd, hdr, hlen) || write_all(fd, body, len)) goto out;
    while (got < RESP_CAP - 1 && (r = read(fd, resp + got, RESP_CAP - 1 - got)) != 0) {
        if (r < 0) { if (errno == EINTR) continue; goto out; }   /* EAGAIN = 수신 타임아웃 */
        got += (size_t)r;
    }
    resp[got] = 0;
    if (got == 0) goto out;

    if (strncmp(resp, "HTTP/1.", 7) != 0 || strncmp(resp + 9, "200", 3) != 0) {
        snprintf(answer, cap, "%.300s", resp);
        rc = LLM_ERR_HTTP;
        goto out;
    }
    rc = LLM_ERR_PARSE;
    if (!(p = strstr(resp, "\r\n\r\n"))) goto out;
    *p = 0;
    if (has_header(resp, "transfer-encoding: chunked")) dechunk(p + 4);
    rc = extract_content(p + 4, answer, cap);

out:
    if (fd >= 0) close(fd);
    free(q); free(sys); free(body); free(resp);
    return rc;
}

long llm_iconv(const char *from, const char *to, const char *src, char *dst, size_t cap) {
    iconv_t cd = iconv_open(to, from);
    char *in = (char *)src, *outp = dst;
    size_t inleft = strlen(src), outleft = cap ? cap - 1 : 0;
    size_t res;

    if (cd == (iconv_t)-1 || cap == 0) return -1;
    res = iconv(cd, &in, &inleft, &outp, &outleft);
    if (res == (size_t)-1 && errno == E2BIG) res = (size_t)-2;
    iconv_close(cd);
    *outp = 0;
    return res == (size_t)-1 ? -1 : res == (size_t)-2 ? -2 : (long)(outp - dst);
}

int llm_chat_euckr(const llm_conf *conf, const char *question, char *answer, size_t cap) {
    size_t qcap = strlen(question) * 3 / 2 + 1, acap = cap * 2;   /* 한글 2바이트 -> UTF-8 3바이트 */
    char *uq = malloc(qcap), *ua = malloc(acap);
    llm_conf c = *conf;
    char *usys = NULL;
    int rc = LLM_ERR_ICONV;

    if (cap) answer[0] = 0;
    if (!uq || !ua) goto out;
    if (llm_iconv("CP949", "UTF-8", question, uq, qcap) < 0) goto out;
    if (conf->system_prompt) {
        size_t scap = strlen(conf->system_prompt) * 3 / 2 + 1;
        if (!(usys = malloc(scap)) || llm_iconv("CP949", "UTF-8", conf->system_prompt, usys, scap) < 0) goto out;
        c.system_prompt = usys;
    }
    rc = llm_chat(&c, uq, ua, acap);
    if (rc == LLM_OK || rc == LLM_ERR_SIZE || rc == LLM_ERR_HTTP) {
        /* CP949에 없는 문자(이모지 등)는 //TRANSLIT로 '?' 등으로 대체 */
        long n = llm_iconv("UTF-8", "CP949//TRANSLIT", ua, answer, cap);
        if (n == -2 && rc == LLM_OK) rc = LLM_ERR_SIZE;
        else if (n < 0 && rc == LLM_OK) rc = LLM_ERR_ICONV;
    }
out:
    free(uq); free(ua); free(usys);
    return rc;
}
