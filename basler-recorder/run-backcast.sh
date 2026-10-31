#!/usr/bin/env bash
# Launch the StarSeg back-cast as a low-priority background process that runs
# concurrently with live recording without interrupting it.
#
# Resource gentleness:
#   - nice -n 19           : lowest CPU scheduling priority
#   - ionice -c3 (idle)    : only use disk IO when otherwise idle (if available)
#   - the Python tool runs batch-of-1 streaming inference, caps torch threads,
#     and sleeps briefly between frames/videos so the live worker keeps GPU time.
#
# Usage:
#   ./run-backcast.sh                 # start in background (logs to state dir)
#   ./run-backcast.sh --dry-run       # see what would be processed (foreground)
#   ./run-backcast.sh --force         # reprocess everything
#   Any extra args are passed through to backcast_starseg.py.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${BASLER_PYTHON:-python3}"  # a Python with pypylon (and torch/ultralytics for StarSeg)
APP="$PROJECT_DIR/backcast_starseg.py"
STATE_DIR="/tmp/basler_loop_recorder_state"
LOG_FILE="$STATE_DIR/backcast.log"
PID_FILE="$STATE_DIR/backcast.pid"

mkdir -p "$STATE_DIR"

NICE=(nice -n 19)
IONICE=()
if command -v ionice >/dev/null 2>&1; then
  IONICE=(ionice -c3)
fi

# Dry-run (or other --foreground-y inspection) stays in the foreground.
for arg in "$@"; do
  if [ "$arg" = "--dry-run" ] || [ "$arg" = "--foreground" ]; then
    exec "${NICE[@]}" "${IONICE[@]}" "$PYTHON" "$APP" "$@"
  fi
done

# Refuse to start a second back-cast if one is already running.
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "back-cast already running (pid $(cat "$PID_FILE")); tail -f $LOG_FILE" >&2
  exit 1
fi

"${NICE[@]}" "${IONICE[@]}" nohup "$PYTHON" "$APP" "$@" >>"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "back-cast started (pid $(cat "$PID_FILE")); logs: $LOG_FILE"
