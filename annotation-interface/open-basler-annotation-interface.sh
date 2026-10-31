#!/usr/bin/env bash
set -euo pipefail
PORT="${BASLER_ANNOTATION_PORT:-8010}"
URL="http://127.0.0.1:${PORT}/"
HEALTH="http://127.0.0.1:${PORT}/api/health"
responds() { curl -fsS --max-time 2 "$HEALTH" >/dev/null 2>&1; }
if ! responds; then
  systemctl --user restart basler-annotation-interface.service
  for _ in $(seq 1 60); do responds && break; sleep 0.25; done
fi
if ! responds; then
  systemctl --user --no-pager status basler-annotation-interface.service >&2 || true
  exit 1
fi
xdg-open "$URL" >/dev/null 2>&1 || true
printf '%s\n' "$URL"
