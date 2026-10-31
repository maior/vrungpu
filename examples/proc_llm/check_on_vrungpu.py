"""Pro*C 소스를 vrungpu 서버로 보내 proc 전처리 + gcc 컴파일 + libclntsh 링크까지 검증 (DB 실행은 하지 않음)

로컬에 Oracle Precompiler가 없어도, 에이전트/개발자가 만든 .pc가 빌드되는지 확인할 수 있다.
서버의 Oracle Instant Client(기본 ~/oracle/instantclient_21_19, pcscfg.cfg)를 사용한다.

    python3 check_on_vrungpu.py http://your-server:9825 llm_demo.pc llm_client.c llm_client.h
    python3 check_on_vrungpu.py http://your-server:9825 mt_job.pc --proc-opt threads=yes --ldlib -lpthread

.pc 파일은 각각 proc → gcc -c 하고, .c 파일은 gcc -c 한 뒤 전부 모아 -lclntsh로 링크한다.
소스는 바이트 그대로 전송하므로 EUC-KR 소스도 그대로 넘기면 된다. 표준 라이브러리만 사용 (requests 불필요).
"""
import argparse
import base64
import json
import os
import sys
import urllib.request

REMOTE = r'''
import base64, json, os, subprocess, pathlib
IC = os.path.expanduser("~/oracle/instantclient_21_19")
CFG = os.path.expanduser("~/oracle/pcscfg.cfg")
env = dict(os.environ, LD_LIBRARY_PATH=IC, ORACLE_HOME=IC)
files = json.loads(base64.b64decode(FILES))
for name, data in files.items():
    pathlib.Path(name).write_bytes(base64.b64decode(data))

def run(step, cmd):
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, errors="replace")
    out = (r.stdout + r.stderr).strip()
    print(f"=== {step}: rc={r.returncode}")
    if out:
        print(out[-4000:])
    return r.returncode == 0

ok, objs = True, []
for pc in [n for n in files if n.endswith(".pc")]:
    c = pc[:-3] + ".c"
    ok &= run(f"proc {pc}", [f"{IC}/sdk/proc", f"iname={pc}", f"oname={c}", f"config={CFG}",
                             "code=ANSI_C", "mode=ORACLE", "parse=partial", "ltype=none"] + PROC_OPTS)
    ok = ok and run(f"gcc {c}", ["gcc", "-c", "-Wall", "-Wno-unused-variable", f"-I{IC}/sdk/include", c])
    objs.append(pc[:-3] + ".o")
for c in [n for n in files if n.endswith(".c")]:
    ok = ok and run(f"gcc {c}", ["gcc", "-c", "-Wall", c])
    objs.append(c[:-2] + ".o")
if ok:
    ok = run("link", ["gcc", "-o", "a.out"] + objs + [f"-L{IC}", "-lclntsh", f"-Wl,-rpath,{IC}"] + LDLIBS)
print("RESULT:", "OK" if ok else "FAIL")
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("server", help="vrungpu URL, e.g. http://your-server:9825")
    ap.add_argument("files", nargs="+", help=".pc / .c / .h files")
    ap.add_argument("--proc-opt", action="append", default=[], help="extra proc option, e.g. --proc-opt threads=yes")
    ap.add_argument("--ldlib", action="append", default=[], help="extra linker arg, e.g. --ldlib -lpthread")
    ap.add_argument("--token", default=os.environ.get("VRUNGPU_API_TOKEN"), help="API token if the server requires one")
    args = ap.parse_args()

    files = {}
    for f in args.files:
        with open(f, "rb") as fp:
            files[os.path.basename(f)] = base64.b64encode(fp.read()).decode()
    if not any(n.endswith(".pc") for n in files):
        sys.exit("at least one .pc file is required")

    code = (f"FILES = {base64.b64encode(json.dumps(files).encode()).decode()!r}\n"
            f"PROC_OPTS = {args.proc_opt!r}\nLDLIBS = {args.ldlib!r}\n" + REMOTE)
    headers = {"Content-Type": "application/json"}
    if args.token:
        headers["Authorization"] = f"Bearer {args.token}"
    req = urllib.request.Request(f"{args.server.rstrip('/')}/run/sync", method="POST", headers=headers,
                                 data=json.dumps({"code": code, "device": "cpu", "timeout": 300}).encode())
    with urllib.request.urlopen(req, timeout=330) as r:
        res = json.load(r)
    print(res.get("stdout", ""))
    if res.get("stderr"):
        print(res["stderr"], file=sys.stderr)
    sys.exit(0 if "RESULT: OK" in res.get("stdout", "") else 1)


if __name__ == "__main__":
    main()
