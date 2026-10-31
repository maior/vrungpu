# Pro*C + vrungpu LLM 호출 예제
#   make llm_cli                       Oracle 없이 클라이언트만 (호출 테스트용)
#   make ORACLE_HOME=/path llm_demo    Pro*C 데모 (proc 전처리 + 컴파일 + libclntsh 링크)
#
# Instant Client(SDK + Precompiler 패키지) 기준 경로. 정식 ORACLE_HOME이면 아래처럼 바꿔서 실행:
#   make ORA_INC=$ORACLE_HOME/precomp/public ORA_LIB=$ORACLE_HOME/lib PROC=$ORACLE_HOME/bin/proc llm_demo
ORACLE_HOME ?= $(HOME)/oracle/instantclient_21_19
PROC    ?= $(ORACLE_HOME)/sdk/proc
ORA_INC ?= $(ORACLE_HOME)/sdk/include
ORA_LIB ?= $(ORACLE_HOME)
# sys_include 등은 pcscfg.cfg에 두는 것을 권장 (CONFIG=경로 로 지정)
PROCFLAGS ?= code=ANSI_C mode=ORACLE parse=partial ltype=none include=$(ORA_INC) $(if $(CONFIG),config=$(CONFIG))

CC      ?= gcc
CFLAGS  ?= -O2 -Wall

all: llm_cli

llm_client.o: llm_client.c llm_client.h
	$(CC) $(CFLAGS) -c llm_client.c

llm_cli: llm_cli.c llm_client.o
	$(CC) $(CFLAGS) -o $@ llm_cli.c llm_client.o

llm_demo.c: llm_demo.pc
	LD_LIBRARY_PATH=$(ORA_LIB):$$LD_LIBRARY_PATH $(PROC) iname=llm_demo.pc oname=llm_demo.c $(PROCFLAGS)

llm_demo: llm_demo.c llm_client.o
	$(CC) $(CFLAGS) -Wno-unused-variable -I$(ORA_INC) -o $@ llm_demo.c llm_client.o -L$(ORA_LIB) -lclntsh -Wl,-rpath,$(ORA_LIB)

clean:
	rm -f *.o llm_cli llm_demo llm_demo.c llm_demo.lis

.PHONY: all clean
