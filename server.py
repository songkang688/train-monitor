"""通用训练监控台 HTTP 服务（纯 Python 标准库，支持 CLI 参数覆盖）。"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import board

_LOCK = threading.Lock()
_CACHED = {"at": 0.0, "body": b""}
STATIC = board._HERE / "static" / "index.html"


def board_body() -> bytes:
    now = time.time()
    with _LOCK:
        if _CACHED["body"] and now - _CACHED["at"] < 2.0:
            return _CACHED["body"]
        body = json.dumps(board.snapshot(), ensure_ascii=False).encode("utf-8")
        _CACHED["at"] = time.time()
        _CACHED["body"] = body
        return body


def invalidate_cache() -> None:
    with _LOCK:
        _CACHED["at"] = 0.0
        _CACHED["body"] = b""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            try:
                body = STATIC.read_bytes()
            except OSError:
                self._send(500, b"missing static/index.html", "text/plain; charset=utf-8")
                return
            self._send(200, body, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/board":
            self._send(200, board_body(), "application/json; charset=utf-8")
            return
        if parsed.path == "/api/history":
            q = parse_qs(parsed.query)
            payload = board.history((q.get("path") or [""])[0], (q.get("log") or [""])[0])
            if payload is None:
                self._send(404, b'{"error":"run not found"}', "application/json; charset=utf-8")
                return
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            return
        if parsed.path == "/api/log":
            q = parse_qs(parsed.query)
            try:
                lines = max(20, min(600, int((q.get("lines") or ["120"])[0])))
            except ValueError:
                lines = 120
            payload = board.log_tail((q.get("path") or [""])[0], (q.get("log") or [""])[0], lines=lines)
            if payload is None:
                self._send(404, b'{"error":"log not found"}', "application/json; charset=utf-8")
                return
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            return
        if parsed.path == "/img":
            q = parse_qs(parsed.query)
            folder = Path((q.get("path") or [""])[0])
            name = Path((q.get("name") or [""])[0]).name
            image = folder / name
            if (
                not name
                or image.suffix.lower() not in board.IMAGE_EXTS
                or not image.is_file()
                or image.resolve().parent != folder.resolve()
                or not board._is_allowed_path(folder, board.load_config(), set())
            ):
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            kind = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
            self._send(200, image.read_bytes(), kind.get(image.suffix.lower(), "application/octet-stream"))
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/api/note", "/api/test_result"}:
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError:
                length = 0
            if length <= 0 or length > 32768:
                self._send(400, b'{"error":"invalid body size"}', "application/json; charset=utf-8")
                return
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                key = str(body.get("key") or body.get("path") or "").strip()
                if parsed.path == "/api/note":
                    saved = board.save_note(key, str(body.get("note") or ""))
                else:
                    saved = board.save_manual_test(key, body)
                invalidate_cache()
            except Exception as exc:
                self._send(400, json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
                return
            self._send(200, json.dumps(saved, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="通用 GPU 训练监控台（零依赖）")
    parser.add_argument("--host", default="", help="监听地址，默认读取 config.json（建议 127.0.0.1 配合 SSH 转发）")
    parser.add_argument("--port", type=int, default=0, help="监听端口，默认 8788")
    parser.add_argument("--root", action="append", default=[], help="追加监视的实验输出根目录（可多次传入）")
    parser.add_argument("--log-dir", action="append", default=[], help="追加监视的日志目录（可多次传入）")
    parser.add_argument("--config", default="", help="指定自定义 config.json 路径")
    args = parser.parse_args()

    board.set_cli_overrides({
        "host": args.host or None,
        "port": args.port or None,
        "roots": args.root,
        "log_dirs": args.log_dir,
        "config": args.config or None,
    })
    cfg = board.load_config()
    host = str(cfg.get("host") or "127.0.0.1")
    port = int(cfg.get("port") or 8788)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"通用训练监控台已启动: http://{host}:{port}/", flush=True)
    if cfg.get("roots"):
        print(f"监视实验目录: {', '.join(cfg['roots'])}", flush=True)
    else:
        print("当前未配置 roots：自动监视本机所有 GPU 与当前用户实时训练进程（可用 --root 或 config.json 追加历史目录）", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
