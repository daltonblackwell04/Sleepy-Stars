#!/usr/bin/env bash
set -euo pipefail

url="http://127.0.0.1:8081"

systemctl --user start loop-injector.service

for _ in $(seq 1 40); do
  if curl -fsS --max-time 1 "$url/health" >/dev/null 2>&1 \
    || curl -fsS --max-time 1 "$url/" >/dev/null 2>&1; then
    exec /usr/bin/xdg-open "$url"
  fi
  sleep 0.15
done

exec /usr/bin/xdg-open "$url"
