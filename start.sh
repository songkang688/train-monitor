#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PID_FILE="./board.pid"
LOG_FILE="./board.log"

case "${1:-start}" in
  daemon)
    shift || true
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "通用训练台已在运行 (PID $(cat "$PID_FILE"))"
      exit 0
    fi
    nohup python3 server.py "$@" >> "$LOG_FILE" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    sleep 0.6
    echo "已在后台启动 (PID $(cat "$PID_FILE"))，日志见 $LOG_FILE"
    tail -n 3 "$LOG_FILE" || true
    ;;
  stop)
    if [ -f "$PID_FILE" ]; then
      PID="$(cat "$PID_FILE")"
      if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        echo "已停止通用训练台 (PID $PID)"
      fi
      rm -f "$PID_FILE"
    else
      echo "未找到运行中的 board.pid"
    fi
    ;;
  status)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "运行中 (PID $(cat "$PID_FILE"))"
      tail -n 5 "$LOG_FILE" || true
    else
      echo "未运行"
    fi
    ;;
  start)
    shift || true
    exec python3 server.py "$@"
    ;;
  *)
    exec python3 server.py "$@"
    ;;
esac
