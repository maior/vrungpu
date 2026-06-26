#!/usr/bin/env bash
# Stop the vrungpu server: SIGTERM, wait, then SIGKILL if still alive.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"

pid="$(read_pid 2>/dev/null || true)"
if [ -z "${pid:-}" ]; then
    # Fall back to process search if PID file is missing.
    pid="$(pgrep -f "[p]ython.* $SERVER_PY" | head -1 || true)"
fi

if [ -z "${pid:-}" ]; then
    echo "vrungpu is not running (no PID file or matching process)."
    rm -f "$PID_FILE"
    exit 0
fi

if ! pid_alive "$pid"; then
    echo "PID $pid not alive — cleaning up stale PID file."
    rm -f "$PID_FILE"
    exit 0
fi

echo "stopping vrungpu PID=$pid (SIGTERM, grace=${STOP_GRACE_SECONDS}s)..."
kill -TERM "$pid" 2>/dev/null || true

waited=0
while [ "$waited" -lt "$STOP_GRACE_SECONDS" ]; do
    if ! pid_alive "$pid"; then
        echo "stopped after ${waited}s"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
    waited=$((waited + 1))
done

echo "still alive after ${STOP_GRACE_SECONDS}s → SIGKILL"
kill -KILL "$pid" 2>/dev/null || true
sleep 1

if pid_alive "$pid"; then
    echo "ERROR: PID $pid did not die." >&2
    exit 1
fi

rm -f "$PID_FILE"
echo "killed."
