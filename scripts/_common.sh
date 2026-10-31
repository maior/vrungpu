# shellcheck shell=bash
# Common variables and helpers for vrungpu service scripts.
# Source this from start.sh / stop.sh / restart.sh / status.sh.

set -u

# Resolve project root as the directory containing this scripts/ folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PORT="${VRUNGPU_PORT:-9825}"
HOST="${VRUNGPU_HOST:-127.0.0.1}"

PYTHON_BIN="${VRUNGPU_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
SERVER_PY="$PROJECT_ROOT/server.py"

RUN_DIR="$PROJECT_ROOT/run"
PID_FILE="$RUN_DIR/vrungpu.pid"

LOGS_DIR="$PROJECT_ROOT/logs"
CURRENT_LOG="$LOGS_DIR/current.log"

# Toolchain dirs put ahead of PATH for submitted jobs (e.g. rustup's latest Rust
# before the distro rustc). Colon-separated; missing dirs are skipped.
EXTRA_PATH="${VRUNGPU_EXTRA_PATH:-$HOME/.cargo/bin}"

# Stop tuning
STOP_GRACE_SECONDS="${VRUNGPU_STOP_GRACE:-10}"

# Log retention (days). Files in logs/ older than this are removed on start.
LOG_RETENTION_DAYS="${VRUNGPU_LOG_RETENTION_DAYS:-30}"

ensure_dirs() {
    mkdir -p "$RUN_DIR" "$LOGS_DIR"
}

pid_alive() {
    local pid="$1"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

read_pid() {
    [ -f "$PID_FILE" ] || return 1
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null | tr -d '[:space:]')"
    [ -n "$pid" ] || return 1
    printf '%s' "$pid"
}

current_pid_if_alive() {
    local pid
    pid="$(read_pid)" || return 1
    if pid_alive "$pid"; then
        printf '%s' "$pid"
        return 0
    fi
    return 1
}

server_responds() {
    curl -sf -o /dev/null --max-time 2 "http://$HOST:$PORT/" 2>/dev/null
}

server_version() {
    curl -sf --max-time 3 "http://$HOST:$PORT/" 2>/dev/null \
        | "$PYTHON_BIN" -c 'import json,sys; d=json.load(sys.stdin); print(d.get("version","?"))' 2>/dev/null \
        || echo "?"
}
