"""通用训练监控台数据引擎（零外部依赖，纯 Python 标准库）。

支持任意深度学习项目（PyTorch / Ultralytics / HuggingFace / MMDetection / Lightning / 自定义脚本）：
1. 显卡与进程实时归属：调用 nvidia-smi 精确区分本人训练任务、本人辅助脚本与他人占用，自动识别可排空闲卡。
2. 零配置自动发现：即使未配置 roots，也会自动发现当前用户占用 GPU 或匹配关键字的实时训练进程，穿透 tee 管道读取日志。
3. 多格式指标解析与控制台兜底：
   - 优先解析结构化文件：results.csv / train_log.csv / metrics.csv / history.csv / metrics.jsonl / trainer_state.json
   - 无结构化文件时自动从控制台 .log 日志按严格白名单提取合法 ML 指标（拦截 batch_size/lr/time/mem 等噪音）
4. 独立测试结果（Test）与实验备注：
   - 自动采集目录内 eval*.json / test_metrics.json / summary.json 以及控制台 [Test] 输出
   - 支持手动录入/编辑测试结果（test_results.json）与实验备注（notes.json），服务端持久化
"""
from __future__ import annotations

import csv
import email.utils
import json
import os
import pwd
import re
import subprocess
import threading
import time
from pathlib import Path

import console_metrics as cm

_HERE = Path(__file__).resolve().parent
CONFIG_PATH = _HERE / "config.json"
NOTES_PATH = _HERE / "notes.json"
TEST_RESULTS_PATH = _HERE / "test_results.json"

_NOTES_LOCK = threading.Lock()
_TESTS_LOCK = threading.Lock()
_CLI_OVERRIDES: dict = {}

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
ULTRALYTICS_PROG = re.compile(
    r"(\d+)/(\d+)\s+(\S+)\s+.*?:\s*(\d+)%\s+\S+\s+(\d+)/(\d+)\s+(\S+)\s+(\S+)<(\S+)"
)
TQDM_PROG = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^<\]]+)<([^,\]]+)(?:,\s*([^\]]+))?\]")
TRAINER_FIELD = re.compile(r"\b(name|project|device|epochs|batch)=\s*([^,\n]+)")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
IMAGE_FIRST = (
    "results.png", "confusion_matrix_normalized.png", "confusion_matrix.png",
    "PR_curve.png", "F1_curve.png", "BoxPR_curve.png", "BoxF1_curve.png",
    "val_batch0_pred.jpg", "val_batch0_labels.jpg", "labels.jpg",
)
STATE_ZH = {
    "running": "在跑",
    "starting": "启动",
    "stalled": "停更",
    "failed": "失败",
    "stopped": "停了",
    "done": "结束",
}
SKIP_WALK_DIRS = {
    ".git", "__pycache__", "node_modules", "weights", "checkpoints", "ckpt",
    "images", "labels", "preds", "data", "datasets", "venv", ".venv", "env",
    "site-packages", "wandb", "tensorboard", "tb_logs",
}
CATEGORY_ZH = {
    "detection": "检测",
    "seg_vis": "分割/视觉",
    "cls_nlp": "分类/NLP/LLM",
    "general": "通用/其他",
}

_summary_cache: dict[str, tuple] = {}
_scan_cache: dict = {"at": 0.0, "key": "", "dirs": []}
_recent_allowed_paths: set[str] = set()


def set_cli_overrides(overrides: dict) -> None:
    _CLI_OVERRIDES.clear()
    _CLI_OVERRIDES.update(overrides or {})


def load_config() -> dict:
    fallback = {
        "roots": [],
        "log_dirs": [],
        "script_needles": ["train", "finetune", "pretrain", "run_", "torchrun", "accelerate", "deepspeed", "yolo"],
        "max_scan_depth": 4,
        "max_age_days": 30,
        "host": "127.0.0.1",
        "port": 8788,
    }
    cfg_path = Path(_CLI_OVERRIDES.get("config") or CONFIG_PATH)
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raw = {}
    except (OSError, json.JSONDecodeError):
        raw = {}
    for k, v in fallback.items():
        raw.setdefault(k, v)
    if _CLI_OVERRIDES.get("roots"):
        merged = list(raw.get("roots") or [])
        for r in _CLI_OVERRIDES["roots"]:
            if r not in merged:
                merged.append(r)
        raw["roots"] = merged
    if _CLI_OVERRIDES.get("log_dirs"):
        merged_l = list(raw.get("log_dirs") or [])
        for ld in _CLI_OVERRIDES["log_dirs"]:
            if ld not in merged_l:
                merged_l.append(ld)
        raw["log_dirs"] = merged_l
    if _CLI_OVERRIDES.get("host"):
        raw["host"] = _CLI_OVERRIDES["host"]
    if _CLI_OVERRIDES.get("port"):
        raw["port"] = int(_CLI_OVERRIDES["port"])
    return raw


def _f(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if val != val or val in (float("inf"), float("-inf")):
        return None
    return val


def _file_sig(path: Path | None) -> tuple | None:
    if path is None:
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _my_username() -> str:
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return os.environ.get("USER") or "user"


def _pid_user(pid: int | str) -> str:
    try:
        return pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
    except (OSError, KeyError):
        return ""


def _cmdline(pid: int | str) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _pid_cvd(pid: int | str) -> str:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().decode("utf-8", "replace")
    except OSError:
        return ""
    for entry in raw.split("\0"):
        if entry.startswith("CUDA_VISIBLE_DEVICES="):
            return entry.split("=", 1)[1].strip()
    return ""


def _pid_ancestors(pid: int) -> list[int]:
    chain = [pid]
    cur = str(pid)
    for _ in range(4):
        ppid = ""
        try:
            for line in Path(f"/proc/{cur}/status").read_text().splitlines():
                if line.startswith("PPid:"):
                    ppid = line.split()[1]
                    break
        except OSError:
            break
        if not ppid.isdigit() or int(ppid) <= 1:
            break
        chain.append(int(ppid))
        cur = ppid
    return chain


def _clean_device(raw: str, path: str = "", cvd: str = "", is_live: bool = False) -> str:
    text = str(raw or "").strip().strip("'\"").strip("[]").replace("cuda:", "").replace(" ", "")
    if text.lower() in {"none", "null"}:
        return ""
    if text.lower() == "cpu":
        return "cpu"
    parts = [p for p in text.split(",") if p.isdigit()]
    vis = [p.strip() for p in str(cvd or "").split(",") if p.strip().isdigit()]
    if vis:
        if parts:
            mapped = [vis[int(p)] if int(p) < len(vis) else p for p in parts]
            return ",".join(dict.fromkeys(mapped))
        return vis[0]
    cleaned = ",".join(dict.fromkeys(parts))
    if not is_live and cleaned in {"", "0"} and path:
        for seg in reversed(path.strip("/").split("/")[-2:]):
            m = re.search(r"(?:^|[_-])gpu([0-7]+)(?:$|[_-])", seg, re.I)
            if m:
                return ",".join(dict.fromkeys(m.group(1)))
    return cleaned


def _tail(path: Path, size: int = 32768) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            end = fh.tell()
            fh.seek(max(0, end - size))
            raw = fh.read()
    except OSError:
        return ""
    return ANSI_RE.sub("", raw.decode("utf-8", errors="replace"))


def _clean_log_lines(raw_text: str, limit: int) -> list[str]:
    cleaned = []
    for raw_line in raw_text.split("\n"):
        segs = [s.rstrip() for s in raw_line.split("\r") if s.strip()]
        if not segs:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        cleaned.append(segs[-1])
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return cleaned[-limit:]


# ---------- 笔记 & 手动测试持久化 ----------

def load_notes() -> dict[str, str]:
    with _NOTES_LOCK:
        try:
            raw = json.loads(NOTES_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {str(k): str(v) for k, v in raw.items() if isinstance(v, str) and v.strip()} if isinstance(raw, dict) else {}


def save_note(key: str, text: str) -> dict:
    clean_key = str(key or "").strip()
    if not clean_key:
        raise ValueError("missing key")
    clean_text = str(text or "").strip()[:1000]
    with _NOTES_LOCK:
        try:
            data = json.loads(NOTES_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, json.JSONDecodeError):
            data = {}
        if clean_text:
            data[clean_key] = clean_text
        else:
            data.pop(clean_key, None)
        tmp = NOTES_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, NOTES_PATH)
    return {"ok": True, "key": clean_key, "note": clean_text}


def load_manual_tests() -> dict[str, dict]:
    with _TESTS_LOCK:
        try:
            raw = json.loads(TEST_RESULTS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return raw if isinstance(raw, dict) else {}


def save_manual_test(key: str, payload: dict) -> dict:
    clean_key = str(key or "").strip()
    if not clean_key:
        raise ValueError("missing key")
    with _TESTS_LOCK:
        try:
            data = json.loads(TEST_RESULTS_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, json.JSONDecodeError):
            data = {}
        if payload.get("clear"):
            data.pop(clean_key, None)
            saved_entry = None
        else:
            ckpt = str(payload.get("ckpt") or "").strip()[:80]
            raw_sets = payload.get("sets") or {}
            clean_sets: dict[str, dict] = {}
            if isinstance(raw_sets, dict):
                for sname, svals in raw_sets.items():
                    sn = str(sname or "").strip()[:48]
                    if not sn or not isinstance(svals, dict):
                        continue
                    item: dict[str, float] = {}
                    for mk, mv in svals.items():
                        fv = _f(mv)
                        if fv is not None:
                            item[str(mk).strip()[:24]] = round(fv, 6)
                    if item:
                        clean_sets[sn] = item
            if not clean_sets:
                data.pop(clean_key, None)
                saved_entry = None
            else:
                saved_entry = {"ckpt": ckpt or "手动录入", "updated": time.time(), "sets": clean_sets}
                data[clean_key] = saved_entry
        tmp = TEST_RESULTS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, TEST_RESULTS_PATH)
    return {"ok": True, "key": clean_key, "entry": saved_entry}


# ---------- GPU 与进程发现 ----------

def _gpus() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    cards = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7 or not parts[0].isdigit():
            continue
        try:
            p_draw = _f(parts[7]) if len(parts) > 7 else None
            p_lim = _f(parts[8]) if len(parts) > 8 else None
            cards.append({
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_used": int(float(parts[3])),
                "memory_total": int(float(parts[4])),
                "util": int(float(parts[5])),
                "temp": int(float(parts[6])),
                "power_draw": round(p_draw) if p_draw is not None else None,
                "power_limit": round(p_lim) if p_lim is not None else None,
            })
        except ValueError:
            continue
    return cards


def _gpu_compute_apps() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_gpu_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    apps = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4 or not parts[1].isdigit():
            continue
        apps.append({
            "uuid": parts[0],
            "pid": int(parts[1]),
            "mem_mb": int(float(parts[2])) if _f(parts[2]) is not None else 0,
            "proc_name": parts[3],
        })
    return apps


def _tee_pipe_logs() -> dict[str, str]:
    mapping: dict[str, str] = {}
    try:
        out = subprocess.check_output(["pgrep", "-af", "tee"], text=True, timeout=4)
    except (OSError, subprocess.SubprocessError):
        return mapping
    for line in out.splitlines():
        pid, _, cmd = line.partition(" ")
        if not pid.isdigit():
            continue
        try:
            fd0 = os.readlink(f"/proc/{pid}/fd/0")
        except OSError:
            continue
        if not fd0.startswith("pipe:"):
            continue
        for tok in reversed(cmd.split()):
            clean = tok.strip("'\"")
            if clean.startswith("/") and clean.endswith((".log", ".txt", ".out")):
                mapping[fd0] = clean
                break
    return mapping


def _aux_proc_label(pid: int, fallback: str) -> str:
    cmd = _cmdline(pid)
    if not cmd:
        return Path(fallback).name or f"PID {pid}"
    for tok in cmd.split():
        clean = tok.strip("'\"")
        if clean.endswith(".py"):
            return Path(clean).stem[:22]
    if " -c " in cmd:
        return "python -c"
    if " -m " in cmd:
        m = re.search(r"-m\s+([A-Za-z0-9_.]+)", cmd)
        if m:
            return m.group(1).rsplit(".", 1)[-1][:22]
    return Path(fallback).name[:20] or f"PID {pid}"


def _infer_cmd_dir_and_name(cmd: str, cwd: str) -> tuple[str, str]:
    for flag in ("--output_dir", "--save_dir", "--work_dir", "--out_dir", "--log_dir", "--project", "--checkpoint_dir"):
        m = re.search(rf"{flag}(?:\s+|=)(\S+)", cmd)
        if m:
            p = Path(m.group(1).strip("'\""))
            if not p.is_absolute() and cwd:
                p = Path(cwd) / p
            if p.is_dir():
                return str(p), p.name
    for tok in cmd.split():
        clean = tok.strip("'\"")
        if clean.endswith(".py"):
            return "", Path(clean).stem
    return "", ""


def _header_ultralytics(path: Path) -> dict:
    try:
        with path.open("rb") as fh:
            raw = fh.read(400000)
    except OSError:
        return {}
    text = ANSI_RE.sub("", raw.decode("utf-8", errors="replace"))
    found = {}
    for line in text.splitlines():
        if not found.get("name") and line.startswith("engine/trainer:"):
            for k, v in TRAINER_FIELD.findall(line):
                found[k] = v.strip().strip("'\"")
        if "Logging results to " in line:
            sd = line.split("Logging results to ", 1)[1].strip().strip("'\"")
            if sd:
                found["save_dir"] = sd
        if found.get("save_dir") and found.get("name"):
            break
    return found


def _progress_from_log(path: Path) -> dict:
    text = _tail(path, 28000)
    pieces = [p.strip() for p in re.split(r"[\r\n]", text) if p.strip()]
    found: dict = {}
    last_train = -1
    last_val = -1
    for i, piece in enumerate(pieces):
        m = ULTRALYTICS_PROG.search(piece)
        if m:
            ep, tot, mem, pct, cur, batches, spd, elapsed, rem = m.groups()
            found = {
                "epoch_now": int(ep), "total": int(tot), "gpu_mem": mem,
                "batch_pct": int(pct), "batch": int(cur), "batches": int(batches),
                "speed": spd, "elapsed": elapsed, "remain": rem, "phase": "训练",
            }
            last_train = i
        elif "Class" in piece and "mAP50" in piece:
            last_val = i
    if found:
        found["phase"] = "验证" if last_val > last_train else "训练"
    else:
        for piece in reversed(pieces):
            tm = TQDM_PROG.search(piece)
            if tm:
                pct, cur, tot, elapsed, rem, spd = tm.groups()
                found = {
                    "batch_pct": int(pct), "batch": int(cur), "batches": int(tot),
                    "elapsed": elapsed.strip(), "remain": rem.strip(), "speed": (spd or "").strip(),
                    "phase": "验证" if re.search(r"\b(?:val|eval|test)\b", piece, re.I) else "训练",
                }
                break
        for piece in reversed(pieces):
            em = cm.EPOCH_RE.search(piece)
            if em:
                ep = int(em.group(1) or em.group(3))
                tot = em.group(2) or em.group(4)
                found.setdefault("epoch_now", ep)
                if tot and tot.isdigit():
                    found.setdefault("total", int(tot))
                found.setdefault("phase", "验证" if re.search(r"\b(?:val|eval|valid)\b", piece, re.I) else "训练")
                break
    alert = ""
    for piece in reversed(pieces[-120:]):
        if re.search(r"Traceback|CUDA out of memory|OutOfMemory|ChildFailedError|RuntimeError:|AssertionError", piece):
            alert = piece[:180]
            break
    if alert:
        found["alert"] = alert
    return found


def live_processes(cfg: dict, gpu_pids: set[int]) -> list[dict]:
    my_user = _my_username()
    needles = [n.lower() for n in (cfg.get("script_needles") or []) if n]
    try:
        out = subprocess.check_output(["pgrep", "-af", "python"], text=True, timeout=6)
    except (OSError, subprocess.SubprocessError):
        return []
    tee_map = _tee_pipe_logs()
    cmds: dict[str, str] = {}
    for line in out.splitlines():
        pid_s, _, cmd = line.partition(" ")
        if not pid_s.isdigit() or "python" not in cmd:
            continue
        if _pid_user(pid_s) != my_user:
            continue
        low = cmd.lower()
        if any(skip in low for skip in ("pgrep", "server.py", "train_board", "通用版", "ipykernel", "jupyter", "language_server")):
            continue
        pid_i = int(pid_s)
        if pid_i in gpu_pids or any(n in low for n in needles):
            cmds[pid_s] = cmd

    jobs = []
    for pid_s, cmd in cmds.items():
        if " -c " in cmd:
            continue
        ppid = ""
        try:
            for ln in Path(f"/proc/{pid_s}/status").read_text().splitlines():
                if ln.startswith("PPid:"):
                    ppid = ln.split()[1]
                    break
        except OSError:
            continue
        if ppid in cmds:
            continue
        stdout = ""
        try:
            stdout = os.readlink(f"/proc/{pid_s}/fd/1")
        except OSError:
            stdout = ""
        if stdout in tee_map:
            stdout = tee_map[stdout]
        cvd = _pid_cvd(pid_s)
        m_dev = re.search(r"--(?:device|gpu|gpus)(?:\s+|=)(?:cuda:)?([0-9,]+)", cmd)
        device = _clean_device(m_dev.group(1) if m_dev else "", "", cvd, is_live=True)
        cwd = ""
        try:
            cwd = os.readlink(f"/proc/{pid_s}/cwd")
        except OSError:
            cwd = ""
        jobs.append({"pid": int(pid_s), "cmd": cmd, "stdout": stdout, "device": device, "cvd": cvd, "cwd": cwd})
    return jobs


# ---------- 指标加载（结构化 CSV/JSONL + 控制台自动兜底） ----------

def _read_csv_rows(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return []
    if text and not text.endswith(("\n", "\r")) and time.time() - path.stat().st_mtime < 10:
        cut = text.rfind("\n")
        text = text[: cut + 1] if cut >= 0 else ""
    out = []
    for row in csv.DictReader(text.splitlines()):
        clean = {(k or "").strip(): (v or "").strip() for k, v in row.items() if k is not None}
        if any(clean.values()):
            out.append(clean)
    return out


def _find_best_log(folder: Path | None, log_path: str = "") -> Path | None:
    if log_path and Path(log_path).is_file():
        return Path(log_path)
    if folder and folder.is_dir():
        for sub in (folder, folder / "logs"):
            if not sub.is_dir():
                continue
            try:
                logs = sorted(
                    [p for p in sub.iterdir() if p.is_file() and p.suffix in {".log", ".out"} and p.name not in {"queue.log", "board.log"}],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            except OSError:
                logs = []
            if logs:
                return logs[0]
    return None


def _parse_structured_csv(folder: Path) -> dict | None:
    """读取 results.csv / train_log.csv / triple_val.csv（纯标准库，零外部依赖）。"""
    csv_sources = []
    for fname, prefix in (("triple_val.csv", "val"), ("results.csv", "res"), ("train_log.csv", "log")):
        p = folder / fname
        if p.is_file():
            rows = _read_csv_rows(p)
            if rows:
                csv_sources.append((prefix, rows))
    if not csv_sources:
        return None

    raw_points: dict[str, list[tuple[int, float]]] = {}
    meta_by_key: dict[str, dict] = {}
    for prefix, rows in csv_sources:
        for idx, row in enumerate(rows, start=1):
            ep_f = _f(row.get("epoch") or row.get("Epoch") or row.get("step"))
            ep = int(round(ep_f)) if ep_f is not None else idx
            for col_name, raw_val in row.items():
                if not col_name or col_name.lower() in {"epoch", "time", "step"} or col_name.lower().startswith("lr"):
                    continue
                val = _f(raw_val)
                if val is None:
                    continue
                meta = cm.classify_metric_key(col_name)
                if meta is None:
                    continue
                ck = meta["canonical"]
                meta_by_key[ck] = meta
                raw_points.setdefault(ck, []).append((ep, round(val, 6)))
    if not raw_points:
        return None
    return cm._finalize_parsed(raw_points, meta_by_key, {}, None, "epoch", max_metrics=14)


def _parse_dur_sec(text: str) -> float | None:
    s = str(text or "").strip().lower()
    if not s or "?" in s:
        return None
    m_s = re.match(r"^(\d+(?:\.\d+)?)s$", s)
    if m_s:
        return float(m_s.group(1))
    parts = s.split(":")
    if len(parts) in (2, 3) and all(re.match(r"^\d+(?:\.\d+)?$", p) for p in parts):
        nums = [float(p) for p in parts]
        return nums[0] * 60.0 + nums[1] if len(nums) == 2 else nums[0] * 3600.0 + nums[1] * 60.0 + nums[2]
    return None


def _estimate_epoch_sec(folder: Path | None) -> float | None:
    if not folder or not folder.is_dir():
        return None
    et_file = folder / "epoch_times.txt"
    if et_file.is_file():
        try:
            vals = [v for ln in et_file.read_text(encoding="utf-8", errors="replace").splitlines()
                    if ln.strip() and (v := _f(ln.strip().split()[-1])) is not None and 0.5 <= v <= 86400]
            if vals:
                rec = sorted(vals[-6:])
                mid = rec[1:-1] if len(rec) >= 4 else rec
                return round(sum(mid) / len(mid), 1)
        except OSError:
            pass
    for fn in ("results.csv", "train_log.csv", "metrics.csv", "history.csv"):
        p = folder / fn
        if not p.is_file():
            continue
        rows = _read_csv_rows(p)
        if not rows:
            continue
        pts = []
        for idx, row in enumerate(rows, start=1):
            t = _f(row.get("time") or row.get("elapsed") or row.get("elapsed_sec"))
            if t is None or t <= 0:
                continue
            ep_f = _f(row.get("epoch") or row.get("Epoch"))
            ep = int(round(ep_f)) if ep_f is not None else idx
            pts.append((ep, t))
        if len(pts) >= 2:
            deltas = [(t1 - t0) / (e1 - e0) for (e0, t0), (e1, t1) in zip(pts[:-1], pts[1:])
                      if (e1 - e0) > 0 and 0.5 <= ((t1 - t0) / (e1 - e0)) <= 86400]
            if deltas:
                rec = sorted(deltas[-6:])
                mid = rec[1:-1] if len(rec) >= 4 else rec
                return round(sum(mid) / len(mid), 1)
        elif len(pts) == 1 and pts[0][0] >= 1 and 0.5 <= (pts[0][1] / pts[0][0]) <= 86400:
            return round(pts[0][1] / pts[0][0], 1)
        durs = [d for row in rows[-8:] if (d := _f(row.get("epoch_time") or row.get("epoch_sec") or row.get("duration"))) is not None and 0.5 <= d <= 86400]
        if durs:
            return round(sum(durs) / len(durs), 1)
    return None


def _compute_eta_fields(base_ep_sec: float | None, prog: dict, ep_done: int, ep_now: int, total: int | None, is_active: bool) -> tuple[float | None, int | None]:
    ep_sec = base_ep_sec
    if ep_sec is None and is_active:
        el = _parse_dur_sec(prog.get("elapsed") or "")
        rem = _parse_dur_sec(prog.get("remain") or "")
        if el is not None and rem is not None and (el + rem) >= 1.0:
            ep_sec = round(el + rem, 1)
    if not is_active or ep_sec is None or not total or int(total) <= ep_done:
        return ep_sec, None
    rem_epochs = max(0.0, float(total) - float(ep_done))
    bp = prog.get("batch_pct")
    if ep_now > ep_done and bp is not None and 0 < float(bp) <= 100:
        rem_epochs = max(0.0, rem_epochs - min(0.95, float(bp) / 100.0))
    return ep_sec, max(0, int(round(rem_epochs * ep_sec)))


def metrics_of(folder: Path | None, log_path: str = "") -> dict:
    fb_log = _find_best_log(folder, log_path)
    watched = []
    if folder and folder.is_dir():
        for nm in ("results.csv", "triple_val.csv", "train_log.csv", "metrics.csv", "history.csv", "log.csv", "metrics.jsonl", "trainer_state.json", "epoch_times.txt"):
            watched.append(_file_sig(folder / nm))
    sig = (tuple(watched), _file_sig(fb_log))
    ckey = f"{folder or ''}|{fb_log or ''}"
    hit = _summary_cache.get(ckey)
    if hit and hit[0] == sig:
        return hit[1]

    parsed = None
    source_kind = ""
    if folder and folder.is_dir():
        parsed = _parse_structured_csv(folder)
        if parsed and parsed.get("latest"):
            source_kind = "csv"
        else:
            parsed = cm.parse_generic_files(folder, max_metrics=10)
            if parsed and parsed.get("latest"):
                source_kind = "generic"
    if (not parsed or not parsed.get("latest")) and fb_log is not None:
        parsed = cm.parse_console_log(fb_log, max_metrics=8)
        if parsed and (parsed.get("latest") or parsed.get("test_metrics")):
            source_kind = "console"
    elif parsed and not parsed.get("test_metrics") and fb_log is not None:
        log_parsed = cm.parse_console_log(fb_log, max_metrics=8)
        if log_parsed and log_parsed.get("test_metrics"):
            parsed = dict(parsed)
            parsed["test_metrics"] = log_parsed["test_metrics"]

    ep_sec = _estimate_epoch_sec(folder)
    if not parsed:
        payload = {"rows": [], "columns": [], "latest": {}, "latest_epoch": {}, "best": {}, "test_metrics": {}, "epoch_done": 0, "total": None, "source_kind": "", "auto_summary": [], "epoch_sec": ep_sec}
    else:
        cols = parsed.get("columns") or []
        latest = parsed.get("latest") or {}
        best = parsed.get("best") or {}
        latest_epoch: dict[str, int] = {}
        for row_pt in parsed.get("rows") or []:
            ep_i = row_pt.get("epoch")
            if ep_i is None:
                continue
            for k in latest:
                if row_pt.get(k) is not None:
                    latest_epoch[k] = int(ep_i)
        auto_summary = []
        for col in cols[:4]:
            k = col["key"]
            if k not in latest:
                continue
            b = best.get(k) or {}
            auto_summary.append({
                "key": k,
                "label": col["label"] if col["group"] in {"自动指标", "验证指标"} else f"{col['group']} {col['label']}",
                "latest": latest[k],
                "latest_ep": latest_epoch.get(k),
                "best": b.get("value"),
                "best_ep": b.get("epoch"),
                "higher": col.get("higher", True),
                "is_pct": col.get("is_pct", False),
            })
        payload = {**parsed, "latest_epoch": latest_epoch, "source_kind": source_kind, "auto_summary": auto_summary, "epoch_sec": ep_sec}
    _summary_cache[ckey] = (sig, payload)
    return payload


def infer_category(path_str: str, columns: list[dict]) -> str:
    keys_blob = " ".join(c.get("key", "") for c in columns).lower() + " " + (path_str or "").lower()
    if any(k in keys_blob for k in ("map50", "map75", "box_loss", "yolo", "detr", "rcnn", "mmdet")):
        return "detection"
    if any(k in keys_blob for k in ("miou", "mdsc", "dice", "psnr", "ssim", "lpips", "sam3", "sam2", "unet", "seg", "matte")):
        return "seg_vis"
    if any(k in keys_blob for k in ("acc", "top1", "f1", "auc", "ppl", "bleu", "rouge", "pass@1", "em", "trainer_state", "llm", "bert", "lora")):
        return "cls_nlp"
    return "general"


# ---------- 额外测试结果自动采集 ----------

def _parse_eval_file(path: Path) -> dict | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        mtime = path.stat().st_mtime
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    if isinstance(raw.get("tests"), dict) and raw["tests"]:
        sets = {}
        for sn, sv in raw["tests"].items():
            if not isinstance(sv, dict):
                continue
            clean = {k: round(v, 6) for k, raw_v in sv.items() if (v := _f(raw_v)) is not None}
            if clean:
                sets[str(sn).strip()] = clean
        if sets:
            return {"ckpt": str(raw.get("ckpt") or path.stem), "source": path.name, "updated": mtime, "sets": sets}
    if isinstance(raw.get("rows"), list):
        sets = {}
        for row in raw["rows"]:
            if not isinstance(row, dict):
                continue
            sn = str(row.get("测试集") or row.get("split") or row.get("name") or "").strip()
            if not sn or "平均" in sn:
                continue
            clean = {k: round(v, 6) for k, raw_v in row.items() if k not in {"测试集", "split", "name"} and (v := _f(raw_v)) is not None}
            if clean:
                sets[sn] = clean
        if sets:
            return {"ckpt": path.parent.name if path.name == "summary.json" else path.stem, "source": f"{path.parent.name}/{path.name}", "updated": mtime, "sets": sets}
    # 扁平 test_metrics.json 或 best_summary.json["final_test"]
    block = raw.get("final_test") if isinstance(raw.get("final_test"), dict) else raw
    if isinstance(block, dict):
        flat = {}
        for k, v in block.items():
            fv = _f(v)
            if fv is None:
                continue
            if cm.classify_metric_key(k) is not None:
                flat[k] = round(fv, 6)
        if flat:
            ep = block.get("checkpoint_epoch") or block.get("epoch")
            ckpt = Path(str(block.get("checkpoint_used") or block.get("weights") or path.stem)).name
            if ep is not None:
                ckpt = f"{ckpt} (E{ep})"
            return {"ckpt": ckpt, "source": path.name, "updated": mtime, "sets": {"Test": flat}}
    return None


def tests_for_run(folder: Path | None, key: str, manual_map: dict, console_tests: dict) -> dict:
    sets: dict[str, dict] = {}
    ckpt = ""
    source = ""
    updated = 0.0
    if folder and folder.is_dir():
        cands = [folder / n for n in ("eval_best.json", "eval_tests.json", "test_eval.json", "test_metrics.json", "best_summary.json") if (folder / n).is_file()]
        cands.extend(sorted(folder.glob("eval_*/summary.json"), key=lambda p: p.stat().st_mtime))
        for cf in cands:
            parsed = _parse_eval_file(cf)
            if not parsed:
                continue
            if not sets or parsed["updated"] >= updated:
                sets.update(parsed["sets"])
                ckpt = parsed["ckpt"]
                source = parsed["source"]
                updated = max(updated, parsed["updated"])
    if console_tests:
        sets.setdefault("控制台 Test", {}).update(console_tests)
        source = source or "控制台日志"
    manual = manual_map.get(key)
    if isinstance(manual, dict) and isinstance(manual.get("sets"), dict):
        for sn, sv in manual["sets"].items():
            sets[sn] = sv
        ckpt = manual.get("ckpt") or ckpt or "手动录入"
        source = "手动录入" if not source else f"{source} + 手动"
        updated = max(updated, float(manual.get("updated") or 0))
    return {"has_test": bool(sets), "ckpt": ckpt, "source": source, "updated": updated, "sets": sets}


# ---------- 目录发现与快照构建 ----------

def _discover_run_dirs(cfg: dict, now: float) -> list[tuple[Path, float]]:
    roots = [str(Path(r).expanduser()) for r in (cfg.get("roots") or []) if r]
    max_depth = int(cfg.get("max_scan_depth") or 4)
    max_age = float(cfg.get("max_age_days") or 30) * 86400
    cache_key = "|".join(roots) + f"|{max_depth}"
    if now - float(_scan_cache.get("at") or 0) > 60 or _scan_cache.get("key") != cache_key:
        found: list[Path] = []
        markers = {"results.csv", "train_log.csv", "metrics.csv", "history.csv", "log.csv", "metrics.jsonl", "trainer_state.json", "args.yaml"}
        for rtxt in roots:
            root = Path(rtxt)
            if not root.is_dir():
                continue
            for dp, dns, fns in os.walk(root):
                rel = Path(dp).relative_to(root)
                if len(rel.parts) >= max_depth:
                    dns[:] = []
                dns[:] = [d for d in dns if d not in SKIP_WALK_DIRS and not d.startswith(".")]
                if any(fn in markers for fn in fns):
                    found.append(Path(dp))
        _scan_cache["at"] = now
        _scan_cache["key"] = cache_key
        _scan_cache["dirs"] = found

    fresh: list[tuple[Path, float]] = []
    for folder in _scan_cache["dirs"]:
        mtimes = []
        for fn in ("results.csv", "triple_val.csv", "train_log.csv", "metrics.csv", "history.csv", "metrics.jsonl", "trainer_state.json", "args.yaml"):
            p = folder / fn
            try:
                if p.is_file():
                    mtimes.append(p.stat().st_mtime)
            except OSError:
                pass
        if not mtimes:
            continue
        upd = max(mtimes)
        if now - upd <= max_age:
            fresh.append((folder, upd))
    return fresh


def _pretty_title(name: str, path: str) -> str:
    if re.match(r"^\d{8}[_-]\d{4,6}", name) and path:
        parts = [p for p in path.strip("/").split("/") if p]
        if len(parts) >= 2:
            return f"{parts[-2]} / {name}"
    return name or "未命名实验"


def _args_yaml_val(folder: Path, key: str) -> str:
    p = folder / "args.yaml"
    if not p.is_file():
        return ""
    try:
        for ln in p.read_text(encoding="utf-8", errors="replace")[:15000].splitlines():
            m = re.match(rf"^\s*{re.escape(key)}\s*:\s*(.+)$", ln)
            if m:
                return m.group(1).strip().strip("'\"")
    except OSError:
        pass
    return ""


def snapshot() -> dict:
    cfg = load_config()
    now = time.time()
    cards = _gpus()
    apps = _gpu_compute_apps()
    my_user = _my_username()
    my_gpu_pids = {a["pid"] for a in apps if _pid_user(a["pid"]) == my_user}

    procs = live_processes(cfg, my_gpu_pids)
    live_by_path: dict[str, dict] = {}
    orphan_lives: list[dict] = []

    for proc in procs:
        stdout = proc["stdout"]
        log_file = stdout if stdout.startswith("/") and Path(stdout).is_file() else ""
        hdr = _header_ultralytics(Path(log_file)) if log_file else {}
        prog = _progress_from_log(Path(log_file)) if log_file else {}
        save_dir = hdr.get("save_dir") or ""
        run_dir = save_dir if save_dir and Path(save_dir).is_dir() else (str(Path(hdr["project"]) / hdr["name"]) if hdr.get("project") and hdr.get("name") else "")
        name = hdr.get("name") or (Path(run_dir).name if run_dir else "")
        if not run_dir and log_file and Path(log_file).parent.name == "logs":
            sib_runs = Path(log_file).parent.parent / "runs"
            if sib_runs.is_dir():
                stem = Path(log_file).stem
                try:
                    for cand in sib_runs.iterdir():
                        if cand.is_dir() and (cand.name == stem or stem.endswith(cand.name)):
                            run_dir = str(cand)
                            name = name or cand.name
                            break
                except OSError:
                    pass
        if not run_dir:
            inf_d, inf_n = _infer_cmd_dir_and_name(proc["cmd"], proc["cwd"])
            if inf_d:
                run_dir = inf_d
                name = name or inf_n
            elif inf_n and not name:
                name = inf_n
        if not run_dir and log_file and Path(log_file).parent.name == "logs":
            cand = Path(log_file).parent.parent
            if any((cand / f).is_file() for f in ("results.csv", "train_log.csv", "metrics.csv", "args.yaml")):
                run_dir = str(cand)
                name = name or cand.name
        if not run_dir and proc["cwd"]:
            cdir = Path(proc["cwd"])
            if any((cdir / f).is_file() for f in ("results.csv", "train_log.csv", "metrics.csv", "args.yaml")):
                run_dir = proc["cwd"]
                name = name or cdir.name
        dev = proc["device"] or _clean_device(hdr.get("device") or "", run_dir, proc["cvd"], is_live=True)
        mtime = Path(log_file).stat().st_mtime if log_file else now
        item = {
            "path": run_dir,
            "name": name or (Path(log_file).stem if log_file else f"pid_{proc['pid']}"),
            "log": log_file,
            "pid": proc["pid"],
            "device": dev,
            "cmd": proc["cmd"],
            "progress": prog,
            "log_mtime": mtime,
        }
        if run_dir:
            live_by_path[run_dir] = item
        elif log_file or proc["pid"] in my_gpu_pids:
            orphan_lives.append(item)

    # 扫描配置的 log_dirs 近期日志
    for ldir_txt in cfg.get("log_dirs") or []:
        ldir = Path(ldir_txt).expanduser()
        if not ldir.is_dir():
            continue
        for lf in ldir.glob("*.log"):
            if lf.name in {"queue.log", "board.log"}:
                continue
            try:
                mt = lf.stat().st_mtime
                if now - mt > 1800:
                    continue
            except OSError:
                continue
            if any(x.get("log") == str(lf) for x in list(live_by_path.values()) + orphan_lives):
                continue
            hdr = _header_ultralytics(lf)
            run_dir = str(Path(hdr["project"]) / hdr["name"]) if hdr.get("project") and hdr.get("name") else ""
            item = {
                "path": run_dir,
                "name": hdr.get("name") or lf.stem,
                "log": str(lf),
                "pid": None,
                "device": _clean_device(hdr.get("device") or "", run_dir),
                "cmd": "",
                "progress": _progress_from_log(lf),
                "log_mtime": mt,
            }
            if run_dir:
                live_by_path.setdefault(run_dir, item)
            else:
                orphan_lives.append(item)

    notes = load_notes()
    manual_tests = load_manual_tests()
    runs_by_key: dict[str, dict] = {}
    all_cols_map: dict[str, dict] = {}

    def build_row(folder: Path | None, live: dict | None, disk_updated: float = 0.0) -> dict:
        log_path = (live or {}).get("log") or ""
        met = metrics_of(folder, log_path)
        prog = (live or {}).get("progress") or {}
        pid = (live or {}).get("pid")
        path_str = str(folder) if folder else ""
        raw_name = (folder.name if folder else "") or (live or {}).get("name") or "未命名"
        dev = (live or {}).get("device") or (_clean_device(_args_yaml_val(folder, "device"), path_str) if folder else "")
        total = prog.get("total") or met.get("total") or (_f(_args_yaml_val(folder, "epochs")) if folder else None)
        if total is not None:
            total = int(total)
        ep_done = int(met.get("epoch_done") or 0)
        ep_now = int(prog.get("epoch_now") or ep_done)
        updated = max(disk_updated, float((live or {}).get("log_mtime") or 0))
        if pid:
            state = "running"
        elif prog.get("alert"):
            state = "failed"
        elif total and ep_done >= total:
            state = "done"
        elif updated and now - updated <= 45 and log_path:
            state = "running"
        else:
            state = "stopped" if (total and ep_done < total) else "done"
        is_active = state in {"running", "starting"}
        ep_now_val = ep_now if is_active else ep_done
        ep_sec_val, eta_total_sec = _compute_eta_fields(met.get("epoch_sec"), prog, ep_done, ep_now_val, total, is_active)

        cat = infer_category(path_str or raw_name, met.get("columns") or [])
        nkey = path_str or log_path or raw_name
        tinfo = tests_for_run(folder, nkey, manual_tests, met.get("test_metrics") or {})
        for c in met.get("columns") or []:
            cm_copy = dict(c)
            cm_copy["cat"] = cat
            all_cols_map.setdefault(cm_copy["key"], cm_copy)

        return {
            "key": nkey,
            "path": path_str,
            "name": raw_name,
            "title": _pretty_title(raw_name, path_str),
            "category": cat,
            "category_zh": CATEGORY_ZH.get(cat, "通用"),
            "source_kind": met.get("source_kind") or "",
            "state": state,
            "state_zh": STATE_ZH.get(state, state),
            "device": dev,
            "pid": pid,
            "log": log_path or (str(_find_best_log(folder)) if _find_best_log(folder) else ""),
            "epoch_done": ep_done,
            "epoch_now": ep_now_val,
            "total": total,
            "phase": (prog.get("phase") or "") if is_active else "",
            "batch": prog.get("batch") if is_active else None,
            "batches": prog.get("batches") if is_active else None,
            "batch_pct": prog.get("batch_pct") if is_active else None,
            "speed": (prog.get("speed") or "") if is_active else "",
            "remain": (prog.get("remain") or "") if is_active else "",
            "epoch_sec": ep_sec_val,
            "eta_total_sec": eta_total_sec,
            "gpu_mem": (prog.get("gpu_mem") or "") if is_active else "",
            "alert": prog.get("alert") or "",
            "latest": met.get("latest") or {},
            "latest_epoch": met.get("latest_epoch") or {},
            "best": met.get("best") or {},
            "auto_summary": met.get("auto_summary") or [],
            "note": notes.get(nkey, ""),
            "tests": tinfo,
            "updated": updated or now,
        }

    for folder, upd in _discover_run_dirs(cfg, now):
        k = str(folder)
        runs_by_key[k] = build_row(folder, live_by_path.pop(k, None), upd)
    for rdir, live in live_by_path.items():
        f = Path(rdir)
        runs_by_key[rdir] = build_row(f if f.is_dir() else None, live, live.get("log_mtime") or now)
    for live in orphan_lives:
        k = live.get("log") or f"pid:{live['pid']}"
        if k not in runs_by_key:
            runs_by_key[k] = build_row(None, live, live.get("log_mtime") or now)

    runs = list(runs_by_key.values())

    # GPU 进程实时关联
    by_pid = {int(r["pid"]): r for r in runs if r.get("pid")}
    apps_by_uuid: dict[str, list[dict]] = {}
    for a in apps:
        apps_by_uuid.setdefault(a["uuid"], []).append(a)

    for card in cards:
        idx_s = str(card["index"])
        job_details = []
        seen_runs: set[str] = set()
        other_map: dict[str, dict] = {}
        for app in apps_by_uuid.get(card.get("uuid") or "", []):
            pid = app["pid"]
            mem_mb = app["mem_mb"]
            user = _pid_user(pid)
            if user and user != my_user:
                ent = other_map.setdefault(user, {"user": user, "procs": 0, "mem_mb": 0})
                ent["procs"] += 1
                ent["mem_mb"] += mem_mb
                continue
            matched = None
            for anc in _pid_ancestors(pid):
                if anc in by_pid:
                    matched = by_pid[anc]
                    break
            if matched is not None:
                devs = [d for d in str(matched.get("device") or "").split(",") if d]
                if idx_s not in devs:
                    devs.append(idx_s)
                    matched["device"] = ",".join(devs)
                if not matched.get("gpu_mem") and mem_mb > 0:
                    matched["gpu_mem"] = f"{mem_mb / 1024:.1f}G"
                rk = matched["key"]
                if rk not in seen_runs:
                    seen_runs.add(rk)
                    job_details.append({"label": matched["name"][:22], "pid": matched["pid"], "mem_mb": mem_mb, "kind": "run", "path": matched["path"], "log": matched["log"]})
                else:
                    for jd in job_details:
                        if jd.get("pid") == matched["pid"]:
                            jd["mem_mb"] += mem_mb
                            break
            else:
                job_details.append({"label": _aux_proc_label(pid, app["proc_name"]), "pid": pid, "mem_mb": mem_mb, "kind": "aux", "path": "", "log": ""})
        other_users = sorted(other_map.values(), key=lambda x: -x["mem_mb"])
        card["job_details"] = job_details
        card["other_users"] = other_users
        card["free"] = (len(job_details) == 0 and len(other_users) == 0 and card["util"] < 10 and card["memory_used"] < 1500)

    def _run_sort_key(r: dict) -> tuple:
        live_rank = 0 if r.get("pid") else 1 if r.get("state") in {"running", "starting"} else 2
        if live_rank < 2:
            dev_nums = [int(p) for p in str(r.get("device") or "").split(",") if p.strip().isdigit()]
            return (live_rank, min(dev_nums) if dev_nums else 99, str(r.get("key") or ""))
        return (live_rank, -float(r.get("updated") or 0), str(r.get("key") or ""))

    runs.sort(key=_run_sort_key)
    # 记录当前快照中可见的运行目录与日志，确保 /data、/autodl-tmp 等非 $HOME 目录也能正常查看曲线与日志
    allowed_now: set[str] = set()
    for r in runs:
        for ptxt in (r.get("path"), r.get("log")):
            if ptxt:
                try:
                    rp = Path(ptxt).resolve()
                    allowed_now.add(str(rp))
                    allowed_now.add(str(rp.parent))
                except OSError:
                    pass
    _recent_allowed_paths.clear()
    _recent_allowed_paths.update(allowed_now)
    test_sets = sorted({sn for r in runs for sn in ((r.get("tests") or {}).get("sets") or {}).keys()})
    categories = sorted({r["category"] for r in runs})
    columns = sorted(all_cols_map.values(), key=lambda c: (0 if "验证" in c["group"] else 1 if "指标" in c["group"] else 2, c["label"]))
    return {
        "now": now,
        "user": my_user,
        "roots": [str(Path(r).expanduser()) for r in (cfg.get("roots") or [])],
        "runs": runs,
        "columns": columns,
        "categories": [{"id": c, "label": CATEGORY_ZH.get(c, c)} for c in categories],
        "test_sets": test_sets,
        "gpus": cards,
        "users": _server_users(cards, now),
        "user_events": list(_USER_EVENTS[:8]),
    }


_LASTLOG_CACHE: tuple[float, dict[str, tuple[float | None, str]]] = (0.0, {})
_WTMP_CACHE: tuple[float, dict[str, tuple[float, str]]] = (0.0, {})
USER_SEEN_PATH = _HERE / "user_seen.json"
_USER_SEEN_LOCK = threading.Lock()
_USER_SEEN_MEM: dict[str, dict] = {}
_USER_SEEN_LAST_SAVE: float = 0.0
_PREV_USER_ONLINE: set[str] | None = None
_PREV_USER_GPUS: dict[str, set[int]] = {}
_USER_EVENTS: list[dict] = []


def _load_user_seen() -> dict[str, dict]:
    global _USER_SEEN_MEM
    if _USER_SEEN_MEM:
        return _USER_SEEN_MEM
    if USER_SEEN_PATH.is_file():
        try:
            raw = json.loads(USER_SEEN_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                out: dict[str, dict] = {}
                for k, v in raw.items():
                    if isinstance(v, (int, float)):
                        out[str(k)] = {"ts": float(v), "via": "历史在线记录"}
                    elif isinstance(v, dict) and isinstance(v.get("ts"), (int, float)):
                        out[str(k)] = {"ts": float(v["ts"]), "via": str(v.get("via") or "历史在线记录")}
                _USER_SEEN_MEM = out
        except (OSError, ValueError):
            pass
    return _USER_SEEN_MEM


def _record_user_seen(active_map: dict[str, str], now: float) -> dict[str, dict]:
    global _USER_SEEN_LAST_SAVE
    with _USER_SEEN_LOCK:
        seen = _load_user_seen()
        changed = False
        force_save = False
        for u, via in active_map.items():
            prev = seen.get(u)
            if prev is None:
                seen[u] = {"ts": now, "via": via}
                changed = True
                force_save = True
            elif now - prev["ts"] >= 20.0 or (via and prev.get("via") != via):
                seen[u] = {"ts": now, "via": via or prev.get("via") or ""}
                changed = True
        if changed and (force_save or now - _USER_SEEN_LAST_SAVE >= 20.0):
            try:
                tmp = USER_SEEN_PATH.with_suffix(".tmp")
                tmp.write_text(json.dumps(seen, ensure_ascii=False, indent=2), encoding="utf-8")
                tmp.replace(USER_SEEN_PATH)
                _USER_SEEN_LAST_SAVE = now
            except OSError:
                pass
        return {k: dict(v) for k, v in seen.items()}


def _parse_w_idle_sec(token: str) -> float | None:
    t = (token or "").strip()
    if not t or t == "?":
        return None
    if t.endswith("s"):
        try:
            return float(t[:-1])
        except ValueError:
            return None
    if t.endswith("days"):
        try:
            return float(t[:-4]) * 86400.0
        except ValueError:
            return None
    if t.endswith("m") and ":" in t:
        p = t[:-1].split(":")
        if len(p) == 2 and p[0].isdigit() and p[1].isdigit():
            return int(p[0]) * 3600.0 + int(p[1]) * 60.0
        return None
    if ":" in t:
        p = t.split(":")
        if len(p) == 2 and p[0].isdigit() and p[1].isdigit():
            return int(p[0]) * 60.0 + int(p[1])
    return None


def _parse_wtmp(now: float) -> dict[str, tuple[float, str]]:
    global _WTMP_CACHE
    cached_at, cached_map = _WTMP_CACHE
    if now - cached_at < 30.0 and cached_map:
        return cached_map
    result: dict[str, tuple[float, str]] = {}
    try:
        out = subprocess.check_output(["last", "-F", "-w", "-n", "300"], text=True, timeout=4, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return cached_map
    for line in out.splitlines():
        if not line.strip() or line.startswith(("wtmp ", "reboot ", "shutdown ")):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        user = parts[0]
        if user in result:
            continue
        for idx in range(2, min(6, len(parts) - 4)):
            cand = " ".join(parts[idx:idx + 5])
            try:
                tm = time.strptime(cand, "%a %b %d %H:%M:%S %Y")
                result[user] = (time.mktime(tm), parts[2] if idx >= 3 else "")
                break
            except ValueError:
                continue
    _WTMP_CACHE = (now, result)
    return result


def _parse_lastlog(now: float) -> dict[str, tuple[float | None, str]]:
    global _LASTLOG_CACHE
    cached_at, cached_map = _LASTLOG_CACHE
    if now - cached_at < 30.0 and cached_map:
        return cached_map
    result: dict[str, tuple[float | None, str]] = {}
    try:
        out = subprocess.check_output(["lastlog"], text=True, timeout=4, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return cached_map
    for line in out.splitlines()[1:]:
        line = line.rstrip()
        if not line or "Never logged in" in line or "**从未登录过**" in line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        user = parts[0]
        ts_val: float | None = None
        from_host = ""
        for idx in (3, 2):
            if len(parts) <= idx + 4:
                continue
            cand = " ".join(parts[idx:])
            try:
                dt = email.utils.parsedate_to_datetime(cand)
                ts_val = dt.timestamp()
                if idx == 3:
                    from_host = parts[2]
                break
            except (TypeError, ValueError):
                continue
        if ts_val is not None:
            result[user] = (ts_val, from_host)
    _LASTLOG_CACHE = (now, result)
    return result


def _server_users(gpus: list[dict], now: float) -> list[dict]:
    """全渠道统计本机真实用户的在线/调用状态、多源合并最近活跃时间、后台进程与占卡情况。"""
    global _PREV_USER_ONLINE
    my_user = _my_username()
    human_users: set[str] = set()
    try:
        for ent in pwd.getpwall():
            if 1000 <= ent.pw_uid < 65534:
                shell = ent.pw_shell or ""
                if shell.endswith("nologin") or shell.endswith("/false"):
                    continue
                human_users.add(ent.pw_name)
    except OSError:
        pass
    if my_user:
        human_users.add(my_user)

    live_signals: dict[str, dict] = {}
    proc_info: dict[str, dict] = {}
    ide_active_re = re.compile(
        r"extensionHost|ptyHost|server-main\.js|bootstrap-fork|code-server|jupyter-lab|jupyter-notebook|tensorboard|streamlit|sftp-server|rsync|scp\b",
        re.I,
    )
    ignore_bg_comm = {"systemd", "(sd-pam)", "dbus-daemon"}
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "user:32,pcpu,etimes,comm,args", "--no-headers"],
            text=True,
            timeout=4,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            parts = line.strip().split(None, 4)
            if len(parts) < 4:
                continue
            u, cpu_s, et_s, comm = parts[0], parts[1], parts[2], parts[3]
            args = parts[4] if len(parts) > 4 else comm
            if u == "root":
                continue
            try:
                pcpu = float(cpu_s)
            except ValueError:
                pcpu = 0.0
            try:
                et = max(0, int(et_s))
            except ValueError:
                et = 0
            if u not in human_users and comm != "sshd":
                continue
            human_users.add(u)
            p = proc_info.setdefault(u, {"count": 0, "min_etimes": et, "cpu_sum": 0.0, "newest_comm": comm})
            if comm not in ignore_bg_comm:
                p["count"] += 1
            p["cpu_sum"] += pcpu
            if et < p["min_etimes"]:
                p["min_etimes"] = et
                p["newest_comm"] = comm

            def _mark_live(reason: str, etimes_val: int, is_sess: bool = False) -> None:
                info = live_signals.setdefault(u, {"sessions": 0, "min_etimes": etimes_val, "reasons": []})
                if is_sess:
                    info["sessions"] += 1
                if etimes_val < info["min_etimes"]:
                    info["min_etimes"] = etimes_val
                if reason and reason not in info["reasons"]:
                    info["reasons"].append(reason)

            if comm == "sshd":
                _mark_live("SSH终端" if "@pts/" in args else "SSH/IDE", et, is_sess=True)
            elif ide_active_re.search(args) and "agent host" not in args:
                tag = "Jupyter/Web" if re.search(r"jupyter|tensorboard|streamlit", args, re.I) else (
                    "文件传输" if re.search(r"sftp-server|rsync|scp\b", args, re.I) else "IDE远程"
                )
                _mark_live(tag, et, is_sess=True)
            elif et <= 300 and comm not in ignore_bg_comm:
                _mark_live(f"调用:{comm}", et, is_sess=False)
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        out = subprocess.check_output(["who"], text=True, timeout=3, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            parts = line.split()
            if not parts:
                continue
            u = parts[0]
            if u == "root" or "tmux(" in line or ":pts/" in line or "SCREEN" in line:
                continue
            human_users.add(u)
            info = live_signals.setdefault(u, {"sessions": 0, "min_etimes": 0, "reasons": []})
            info["sessions"] = max(info["sessions"], 1)
            if "直连终端" not in info["reasons"]:
                info["reasons"].append("直连终端")
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        out = subprocess.check_output(["w", "-h", "-s"], text=True, timeout=3, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            raw_u, idle_tok = parts[0], parts[3]
            idle_sec = _parse_w_idle_sec(idle_tok)
            if idle_sec is None or idle_sec > 600.0:
                continue
            matched_u = raw_u if raw_u in human_users else next((hu for hu in human_users if hu.startswith(raw_u)), "")
            if not matched_u:
                continue
            info = live_signals.setdefault(matched_u, {"sessions": 0, "min_etimes": int(idle_sec), "reasons": []})
            if int(idle_sec) < info["min_etimes"]:
                info["min_etimes"] = int(idle_sec)
            if "终端活跃" not in info["reasons"]:
                info["reasons"].append("终端活跃")
    except (OSError, subprocess.SubprocessError):
        pass

    gpu_by_user: dict[str, dict] = {}
    for card in gpus:
        idx = card.get("index")
        my_mem = sum(int(jd.get("mem_mb") or 0) for jd in (card.get("job_details") or []))
        if my_user and my_mem > 0:
            entry = gpu_by_user.setdefault(my_user, {"indices": [], "mem_mb": 0})
            if idx not in entry["indices"]:
                entry["indices"].append(idx)
            entry["mem_mb"] += my_mem
        for ou in card.get("other_users") or []:
            u = ou.get("user")
            if not u:
                continue
            human_users.add(u)
            entry = gpu_by_user.setdefault(u, {"indices": [], "mem_mb": 0})
            if idx not in entry["indices"]:
                entry["indices"].append(idx)
            entry["mem_mb"] += int(ou.get("mem_mb") or 0)

    for u, ginfo in gpu_by_user.items():
        cur_set = set(ginfo["indices"])
        prev_set = _PREV_USER_GPUS.get(u)
        if prev_set is not None and (cur_set - prev_set):
            info = live_signals.setdefault(u, {"sessions": 0, "min_etimes": 0, "reasons": []})
            info["min_etimes"] = 0
            tag = f"新占卡#{','.join(str(i) for i in sorted(cur_set - prev_set))}"
            if tag not in info["reasons"]:
                info["reasons"].append(tag)
        _PREV_USER_GPUS[u] = cur_set
    for u in list(_PREV_USER_GPUS.keys()):
        if u not in gpu_by_user:
            _PREV_USER_GPUS[u] = set()

    active_via_map = {u: " / ".join(info["reasons"][:2]) for u, info in live_signals.items()}
    seen_map = _record_user_seen(active_via_map, now)

    cur_online_others = {u for u in live_signals if u != my_user}
    if _PREV_USER_ONLINE is not None:
        for u in sorted(cur_online_others - _PREV_USER_ONLINE):
            _USER_EVENTS.insert(0, {"user": u, "ts": now, "via": active_via_map.get(u) or "登录/调用"})
        del _USER_EVENTS[15:]
    _PREV_USER_ONLINE = cur_online_others

    lastlog_map = _parse_lastlog(now)
    wtmp_map = _parse_wtmp(now)
    users_out: list[dict] = []
    for u in sorted(human_users):
        live = live_signals.get(u)
        online = live is not None
        connected_since = (now - live["min_etimes"]) if live is not None else None
        ll_ts, ll_from = lastlog_map.get(u, (None, ""))
        wt_ts, wt_from = wtmp_map.get(u, (None, ""))
        pinfo = proc_info.get(u)
        proc_ts = (now - pinfo["min_etimes"]) if pinfo is not None else None
        seen_entry = seen_map.get(u)
        seen_ts = seen_entry["ts"] if seen_entry else None
        via_str = " / ".join(live["reasons"][:2]) if live else ""

        cands: list[tuple[float, str]] = []
        if connected_since is not None:
            cands.append((connected_since, via_str or "实时在线"))
        if seen_ts is not None:
            cands.append((seen_ts, (seen_entry.get("via") if seen_entry else "") or "监控台记录"))
        if wt_ts is not None:
            cands.append((wt_ts, "wtmp 登录/终端"))
        if ll_ts is not None:
            cands.append((ll_ts, "lastlog 登录"))
        if proc_ts is not None:
            cands.append((proc_ts, f"进程/IDE ({pinfo['newest_comm']})"))

        last_login: float | None = None
        last_source = ""
        if cands:
            last_login, last_source = max(cands, key=lambda x: x[0])

        ginfo = gpu_by_user.get(u) or {"indices": [], "mem_mb": 0}
        cpu_sum = round(pinfo["cpu_sum"], 0) if pinfo else 0.0
        users_out.append({
            "user": u,
            "is_me": u == my_user,
            "online": online,
            "sessions": live["sessions"] if live else 0,
            "via": via_str,
            "connected_since": connected_since,
            "last_login": last_login,
            "last_source": last_source,
            "bg_procs": pinfo["count"] if (pinfo and not online) else 0,
            "cpu_pct": int(cpu_sum) if cpu_sum >= 5.0 else 0,
            "from_host": ll_from or wt_from,
            "gpu_indices": sorted(ginfo["indices"]),
            "gpu_mem_gb": round(ginfo["mem_mb"] / 1024.0, 1) if ginfo["mem_mb"] > 0 else 0.0,
        })

    def _u_sort(item: dict) -> tuple:
        if item["online"] and not item["is_me"]:
            group = 0
        elif item["online"] and item["is_me"]:
            group = 1
        else:
            group = 2
        return (group, -(item["last_login"] or 0.0), item["user"])

    users_out.sort(key=_u_sort)
    return users_out


def _safe_roots(cfg: dict) -> list[Path]:
    roots = [Path(r).expanduser().resolve() for r in (cfg.get("roots") or []) + (cfg.get("log_dirs") or []) if r]
    return roots


def _is_allowed_path(target: Path, cfg: dict, active_paths: set[str]) -> bool:
    if str(target) in active_paths:
        return True
    try:
        res = target.resolve()
    except OSError:
        return False
    res_s = str(res)
    if res_s in _recent_allowed_paths or str(res.parent) in _recent_allowed_paths:
        return True
    for r in _safe_roots(cfg):
        if res == r or r in res.parents:
            return True
    # 允许当前用户主目录下的活跃运行目录/日志
    try:
        home = Path.home().resolve()
        if home in res.parents:
            return True
    except OSError:
        pass
    return False


def _images(folder: Path | None) -> list[str]:
    if not folder or not folder.is_dir():
        return []
    try:
        names = {p.name for p in folder.iterdir() if p.is_file()}
    except OSError:
        return []
    found = [n for n in IMAGE_FIRST if n in names]
    for n in sorted(names):
        if n not in found and Path(n).suffix.lower() in IMAGE_EXTS:
            found.append(n)
        if len(found) >= 12:
            break
    return found[:12]


def history(path_text: str = "", log_text: str = "") -> dict | None:
    cfg = load_config()
    folder = Path(path_text) if path_text else None
    if folder and not folder.is_dir():
        folder = None
    log_p = Path(log_text) if log_text else None
    if log_p and not log_p.is_file():
        log_p = None
    if folder is None and log_p is None:
        return None
    if not _is_allowed_path(folder or log_p, cfg, set()):
        return None

    met = metrics_of(folder, str(log_p) if log_p else "")
    name = (folder.name if folder else log_p.stem)
    key = str(folder) if folder else str(log_p)
    cols = met.get("columns") or []
    rows = met.get("rows") or []
    tinfo = tests_for_run(folder, key, load_manual_tests(), met.get("test_metrics") or {})
    return {
        "key": key,
        "path": str(folder) if folder else "",
        "log": str(log_p) if log_p else (str(_find_best_log(folder)) if _find_best_log(folder) else ""),
        "name": name,
        "title": _pretty_title(name, str(folder) if folder else ""),
        "category": infer_category(key, cols),
        "source_kind": met.get("source_kind") or "",
        "x_axis": met.get("x_axis") or "epoch",
        "note": load_notes().get(key, ""),
        "tests": tinfo,
        "columns": cols,
        "rows": rows,
        "images": _images(folder),
    }


def log_tail(path_text: str = "", log_text: str = "", lines: int = 120) -> dict | None:
    cfg = load_config()
    candidates: list[Path] = []
    if log_text and Path(log_text).is_file() and _is_allowed_path(Path(log_text), cfg, set()):
        candidates.append(Path(log_text))
    folder = Path(path_text) if path_text else None
    if folder and folder.is_dir() and _is_allowed_path(folder, cfg, set()):
        for sub in (folder, folder / "logs"):
            if not sub.is_dir():
                continue
            try:
                local = sorted(
                    [p for p in sub.iterdir() if p.is_file() and p.suffix in {".log", ".out", ".txt"} and p.name != "classes.txt"],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
            except OSError:
                local = []
            for p in local:
                if p not in candidates:
                    candidates.append(p)
        if not candidates and (folder / "args.yaml").is_file():
            candidates.append(folder / "args.yaml")
    if not candidates:
        return None
    target = candidates[0]
    try:
        st = target.stat()
    except OSError:
        return None
    raw = _tail(target, size=min(max(262144, lines * 2600), 1200000))
    tail_lines = _clean_log_lines(raw, lines)
    has_err = any(re.search(r"Traceback|Error:|Exception:|CUDA out of memory|OutOfMemory", ln) for ln in tail_lines)
    return {
        "log_path": str(target),
        "name": target.name,
        "mtime": st.st_mtime,
        "size": st.st_size,
        "lines": tail_lines,
        "has_error": has_err,
        "available": [{"name": c.name, "path": str(c)} for c in candidates[:6]],
    }
