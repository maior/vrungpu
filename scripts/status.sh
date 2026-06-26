#!/usr/bin/env bash
# Show vrungpu service status: PID, uptime, HTTP health, GPU + CPU pool, GPU memory.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"

pid="$(read_pid 2>/dev/null || true)"

echo "=== vrungpu status ==="
if [ -z "${pid:-}" ]; then
    echo "PID file:   (none)  $PID_FILE"
elif pid_alive "$pid"; then
    etime="$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')"
    echo "PID:        $pid  (uptime: ${etime:-?})"
else
    echo "PID file:   $pid  (NOT alive — stale)"
fi

if server_responds; then
    info="$(curl -sf --max-time 3 "http://$HOST:$PORT/")"
    echo "HTTP:       OK   http://$HOST:$PORT/  version=$(printf '%s' "$info" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("version","?"))' 2>/dev/null)"
else
    echo "HTTP:       DOWN  (http://$HOST:$PORT/ not responding)"
fi

echo
echo "--- Pool ---"
if pool="$(curl -sf --max-time 3 "http://$HOST:$PORT/pool")"; then
    printf '%s\n' "$pool" | "$PYTHON_BIN" -c '
import json, sys
d = json.load(sys.stdin)
g = d.get("gpu", {})
c = d.get("cpu", {})
busy = g.get("busy_gpus", {}) or {}
avail = g.get("available_gpus", []) or []
print("GPU: total={}  available={}  busy={}".format(g.get("total_gpus","?"), avail, busy))
print("CPU: {}/{} slots  auto={}  tasks={}".format(c.get("in_use",0), c.get("max_slots","?"), c.get("auto_detected",False), c.get("tasks",[])))
' || echo "$pool"
else
    echo "(server not reachable)"
fi

echo
echo "--- nvidia-smi ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader 2>/dev/null \
        | awk -F',' '{ printf "GPU %s%s  mem=%s /%s  util=%s  temp=%s\n", $1, $2, $3, $4, $5, $6 }'
else
    echo "(nvidia-smi not available)"
fi

echo
echo "--- Logs ---"
if [ -L "$CURRENT_LOG" ]; then
    target="$(readlink "$CURRENT_LOG")"
    echo "current:   $CURRENT_LOG -> $target"
fi
ls -1t "$LOGS_DIR"/server-*.log 2>/dev/null | head -3 | sed 's/^/recent:    /'

# Running tasks summary
echo
echo "--- Running tasks ---"
if running="$(curl -sf --max-time 3 "http://$HOST:$PORT/tasks?status=running")"; then
    printf '%s\n' "$running" | "$PYTHON_BIN" -c '
import json, sys
d = json.load(sys.stdin)
tasks = d.get("tasks", d) if isinstance(d, dict) else d
if not tasks:
    print("(none)")
else:
    for t in tasks:
        tid = (t.get("task_id") or t.get("id") or "?")
        name = (t.get("name") or "")
        gpu = t.get("gpu_id")
        dev = t.get("device","gpu")
        started = t.get("started_at","")
        print("  {}  {:40}  device={:3}  gpu={}  started={}".format(tid[:8], name[:40], dev, gpu, started))
' || echo "$running"
else
    echo "(server not reachable)"
fi
