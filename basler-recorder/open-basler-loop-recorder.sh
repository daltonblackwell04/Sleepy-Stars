#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${BASLER_PYTHON:-python3}"  # a Python with pypylon (and torch/ultralytics for StarSeg)
APP="$PROJECT_DIR/basler_loop_video_recorder.py"
PORT="${BASLER_STREAM_PORT:-8000}"
STATE_DIR="/tmp/basler_loop_recorder_state"
LOG_FILE="$STATE_DIR/server.log"
PID_FILE="$STATE_DIR/server.pid"
URL="http://127.0.0.1:${PORT}/"

mkdir -p "$STATE_DIR"

server_responds() {
  curl -fsS --max-time 1 "http://127.0.0.1:${PORT}/status.json" >/dev/null 2>&1
}

if ! server_responds; then
  # Clean up stale pid file if present.
  if [ -f "$PID_FILE" ] && ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    rm -f "$PID_FILE"
  fi

  cd "$PROJECT_DIR"
  BASLER_STREAM_PORT="$PORT" \
  BASLER_STREAM_FPS="${BASLER_STREAM_FPS:-1}" \
  BASLER_SEGMENT_SECONDS="${BASLER_SEGMENT_SECONDS:-600}" \
  BASLER_JPEG_QUALITY="${BASLER_JPEG_QUALITY:-80}" \
  BASLER_X264_CRF="${BASLER_X264_CRF:-34}" \
  BASLER_X264_PRESET="${BASLER_X264_PRESET:-medium}" \
  BASLER_DELETE_FAILED_SEGMENTS="${BASLER_DELETE_FAILED_SEGMENTS:-1}" \
  ${BASLER_STARSEG_MODEL:+BASLER_STARSEG_MODEL="$BASLER_STARSEG_MODEL"} \
  nohup "$PYTHON" "$APP" >>"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"

  # Give the HTTP server a short chance to come up before opening the browser.
  for _ in $(seq 1 20); do
    if server_responds; then
      break
    fi
    sleep 0.25
  done
fi

xdg-open "$URL"
