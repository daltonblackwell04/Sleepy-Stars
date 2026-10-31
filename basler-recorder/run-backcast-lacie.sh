#!/usr/bin/env bash
# Hands-off hybrid LaCie rollover. Runs detached in the background and:
#   1. waits for the primary back-cast (backcast.pid) to finish, so the two
#      never contend for the GPU;
#   2. copies the rewritten StarSeg sidecars primary -> LaCie for overlapping
#      videos (no GPU);
#   3. back-casts the LaCie-only videos (the synced overlap is auto-skipped as
#      already-done), gently, on the LaCie root.
#
# Usage:
#   ./run-backcast-lacie.sh            # start detached in background
#   Extra args are forwarded to the LaCie back-cast step (e.g. --frame-delay 0.2)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${BASLER_PYTHON:-python3}"  # a Python with pypylon (and torch/ultralytics for StarSeg)
BACKUP_ROOT="${BASLER_BACKUP_ROOT:-/media/$USER/LaCie/Basler Loop Recordings}"
STATE_DIR="/tmp/basler_loop_recorder_state"
LOG_FILE="$STATE_DIR/backcast_lacie.log"
PID_FILE="$STATE_DIR/backcast_lacie.pid"
PRIMARY_PID_FILE="$STATE_DIR/backcast.pid"

mkdir -p "$STATE_DIR"

NICE=(nice -n 19)
IONICE=()
if command -v ionice >/dev/null 2>&1; then
  IONICE=(ionice -c3)
fi

# Worker mode: the actual long-running pipeline (re-exec'd detached below).
if [ "${1:-}" = "__worker__" ]; then
  shift
  echo "=== [$(date -Iseconds)] LaCie hybrid rollover starting (pid $$) ==="

  if [ -f "$PRIMARY_PID_FILE" ]; then
    ppid="$(cat "$PRIMARY_PID_FILE")"
    while kill -0 "$ppid" 2>/dev/null; do
      echo "[$(date -Iseconds)] waiting for primary back-cast (pid $ppid) to finish..."
      sleep 300
    done
  fi
  echo "[$(date -Iseconds)] primary back-cast finished; syncing overlap to backup"

  "${NICE[@]}" "${IONICE[@]}" "$PYTHON" "$PROJECT_DIR/sync_starseg_to_backup.py" --backup "$BACKUP_ROOT"

  echo "[$(date -Iseconds)] back-casting LaCie-only videos"
  "${NICE[@]}" "${IONICE[@]}" "$PYTHON" "$PROJECT_DIR/backcast_starseg.py" \
    --root "$BACKUP_ROOT" \
    --state-file "$HOME/.config/basler-loop-recorder/backcast_lacie_state.json" \
    "$@"

  echo "=== [$(date -Iseconds)] LaCie hybrid rollover done ==="
  exit 0
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "LaCie rollover already running (pid $(cat "$PID_FILE")); tail -f $LOG_FILE" >&2
  exit 1
fi

# Launch the worker fully detached so it survives this shell/session closing.
setsid nohup "$0" __worker__ "$@" >>"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "LaCie rollover started (pid $(cat "$PID_FILE")); logs: $LOG_FILE"
