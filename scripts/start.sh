#!/usr/bin/env bash
# Start the vrungpu server in the background with a PID file and rotated logs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_common.sh
source "$SCRIPT_DIR/_common.sh"

ensure_dirs

if pid="$(current_pid_if_alive)"; then
    echo "vrungpu already running (PID=$pid). Use scripts/restart.sh to restart." >&2
    exit 1
fi

# Stale PID file from a previous crash — clean it up.
if [ -f "$PID_FILE" ]; then
    rm -f "$PID_FILE"
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python not found at $PYTHON_BIN. Set VRUNGPU_PYTHON or create .venv." >&2
    exit 1
fi
if [ ! -f "$SERVER_PY" ]; then
    echo "server.py not found at $SERVER_PY" >&2
    exit 1
fi

# Prune old log files (keep current.log symlink intact).
if [ "$LOG_RETENTION_DAYS" -gt 0 ] 2>/dev/null; then
    find "$LOGS_DIR" -maxdepth 1 -type f -name 'server-*.log' \
        -mtime "+$LOG_RETENTION_DAYS" -delete 2>/dev/null || true
fi

ts="$(date +%Y%m%d-%H%M%S)"
log_file="$LOGS_DIR/server-$ts.log"
touch "$log_file"

# Update current.log → newest log
ln -sfn "$(basename "$log_file")" "$CURRENT_LOG"

IFS=':' read -ra extra_dirs <<< "$EXTRA_PATH"
for ((i = ${#extra_dirs[@]} - 1; i >= 0; i--)); do
    d="${extra_dirs[i]}"
    if [ -n "$d" ] && [ -d "$d" ]; then
        PATH="$d:$PATH"
    fi
done
export PATH

cd "$PROJECT_ROOT"
nohup "$PYTHON_BIN" "$SERVER_PY" >> "$log_file" 2>&1 </dev/null &
new_pid=$!
echo "$new_pid" > "$PID_FILE"

disown 2>/dev/null || true

echo "spawned PID=$new_pid → $log_file"
echo -n "waiting for http://$HOST:$PORT/ ..."

# Wait up to 60s for the server to become ready.
deadline=$((SECONDS + 60))
while [ "$SECONDS" -lt "$deadline" ]; do
    if ! pid_alive "$new_pid"; then
        echo " FAILED (process died)"
        echo "Tail of $log_file:" >&2
        tail -n 50 "$log_file" >&2 || true
        rm -f "$PID_FILE"
        exit 1
    fi
    if server_responds; then
        echo " ready"
        echo "vrungpu v$(server_version) running on http://$HOST:$PORT (PID=$new_pid)"
        echo "log:  $CURRENT_LOG → $log_file"
        echo "pid:  $PID_FILE"
        exit 0
    fi
    sleep 1
done

echo " TIMEOUT after 60s"
echo "Process is still alive at PID=$new_pid but did not respond. Check $log_file" >&2
exit 1
