/* llm_client 단독 테스트 (Oracle 불필요)
 *   ./llm_cli [-e] [-m model] [-t] host port ["질문"]    질문 생략 시 stdin
 *   -e: 입출력 EUC-KR(CP949)   -t: thinking 켬   -m: 모델 지정
 * 예) echo "MODE=ORACLE에서 NO DATA FOUND의 sqlcode는?" | iconv -t CP949 | ./llm_cli -e your-server 9825 | iconv -f CP949
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "llm_client.h"

int main(int argc, char **argv) {
    llm_conf conf = { 0 };
    static char question[65536], answer[65536];
    int euckr = 0, opt, rc;
    struct timespec t0, t1;

    while ((opt = getopt(argc, argv, "em:t")) != -1) {
        if (opt == 'e') euckr = 1;
        else if (opt == 'm') conf.model = optarg;
        else if (opt == 't') conf.think = 1;
        else { fprintf(stderr, "usage: %s [-e] [-m model] [-t] host port [question]\n", argv[0]); return 2; }
    }
    if (argc - optind < 2) { fprintf(stderr, "usage: %s [-e] [-m model] [-t] host port [question]\n", argv[0]); return 2; }
    conf.host = argv[optind];
    conf.port = argv[optind + 1];
    conf.api_token = getenv("VRUNGPU_API_TOKEN");
    conf.timeout_sec = conf.think ? 600 : 120;

    if (argc - optind >= 3) snprintf(question, sizeof question, "%s", argv[optind + 2]);
    else {
        size_t n = fread(question, 1, sizeof question - 1, stdin);
        question[n] = 0;
    }

    clock_gettime(CLOCK_MONOTONIC, &t0);
    rc = euckr ? llm_chat_euckr(&conf, question, answer, sizeof answer)
               : llm_chat(&conf, question, answer, sizeof answer);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "rc=%d (%.1fs)\n", rc, (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9);
    printf("%s\n", answer);
    return rc != LLM_OK;
}
