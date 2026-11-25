"""通用控制台日志与零散指标文件自动提取器（带严格防杂乱兜底）。

设计目标：
1. 任何深度学习脚本（PyTorch / HF Trainer / Lightning / MMDetection / timm / 自定义脚本）
   只要在控制台日志或 metrics.csv / metrics.jsonl / trainer_state.json 里打印标准指标，
   就能自动识别指标名、高低方向、逐轮/逐脉冲曲线与最好值。
2. 严格拦截超参（batch_size/lr/seed/imgsz/workers）、耗时、显存、端口、报错栈、
   argparse 参数回显等非指标数字，绝不让零碎小实验把监控表撑乱。
"""
from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

# 明确屏蔽的非指标字段（超参、耗时、显存、尺寸、进程/编号）
BLACKLIST_KEYS = {
    "epoch", "epochs", "ep", "step", "steps", "global_step", "iter", "iters", "iteration",
    "batch", "batches", "bs", "batch_size", "micro_batch", "minibatch",
    "lr", "learning_rate", "base_lr", "min_lr", "max_lr", "wd", "weight_decay",
    "momentum", "beta", "beta1", "beta2", "eps", "epsilon", "warmup", "warmup_epochs",
    "patience", "seed", "device", "gpu", "gpus", "cuda", "rank", "local_rank", "world_size",
    "workers", "num_workers", "threads", "imgsz", "img_size", "image_size", "crop_size",
    "size", "width", "height", "h", "w", "c", "b", "n", "k", "x", "y", "ch", "channels",
    "dim", "embed_dim", "hidden", "hidden_size", "layers", "num_layers", "heads", "depth",
    "num_classes", "classes", "nc", "vocab_size", "seq_len", "max_len",
    "params", "trainable_params", "total_params", "flops", "gflops", "macs",
    "time", "sec", "secs", "ms", "min", "mins", "eta", "remain", "elapsed", "duration",
    "speed", "fps", "throughput", "samples_per_sec", "it_s", "step_time", "data_time",
    "mem", "memory", "vram", "ram", "peak_mb", "max_mem", "allocated", "reserved",
    "port", "pid", "ppid", "uid", "id", "idx", "index", "count", "total", "num", "len",
    "length", "samples", "images", "instances", "boxes", "version", "code", "status",
    "exit", "retry", "timeout", "conf", "iou_thres", "nms", "threshold", "thresh",
    "alpha_thresh", "temp", "temperature", "scale", "loss_scale", "grad_scale",
    "clip", "grad_norm", "gnorm", "max_norm", "dropout", "drop", "ratio", "weight",
    "lambda", "gamma", "kappa", "tau", "sigma", "mu", "std", "mean_ms", "median_ms",
    "ratio_fair", "eval_max_diff", "seg_logit_diff",
}

# 越低越好的核心指标词根
LOWER_METRICS = {
    "loss", "total_loss", "box_loss", "cls_loss", "dfl_loss", "seg_loss", "ce_loss",
    "bce_loss", "mse_loss", "mae_loss", "l1_loss", "l2_loss", "kl_loss", "dice_loss",
    "focal_loss", "iou_loss", "distill_loss", "reg_loss", "obj_loss", "recon_loss",
    "gan_loss", "ctc_loss", "nll_loss", "aux_loss", "contrastive_loss", "align_loss",
    "l1", "l2", "nll", "brier", "err", "error", "top1_err", "top5_err",
    "wer", "cer", "per", "ppl", "perplexity", "fid", "kid", "lpips",
    "sad", "mse", "mae", "rmse", "mape", "smape", "grad", "conn",
    "hd95", "assd", "epe", "abs_rel", "sq_rel", "niqe", "brisque",
}

# 越高越好的核心指标词根
HIGHER_METRICS = {
    "acc", "accuracy", "top1", "top5", "top1_acc", "top5_acc", "bal_acc", "balanced_acc",
    "f1", "f1_score", "macro_f1", "micro_f1", "weighted_f1",
    "precision", "recall", "prec", "rec",
    "auc", "auroc", "auprc", "roc_auc", "pr_auc",
    "ap", "ap50", "ap75", "aps", "apm", "apl", "map", "map50", "map75", "map50_95", "map50-95",
    "miou", "iou", "biou", "fw_iou", "mdsc", "dsc", "dice", "mdice", "pa", "mpa", "oa",
    "kappa", "mcc", "r2", "psnr", "ssim", "ms_ssim",
    "bleu", "bleu1", "bleu2", "bleu3", "bleu4",
    "rouge", "rouge1", "rouge2", "rougel", "rouge_l",
    "meteor", "cider", "spice", "bertscore", "ndcg", "mrr", "hit", "hits1", "hits5", "hits10",
    "em", "exact_match", "pass1", "pass5", "pass10", "pass@1", "pass@5", "pass@10",
    "reward", "return", "win_rate", "score", "fscore", "gmean", "sensitivity", "specificity",
}

# 适合按百分数（×100）显示的指标词根（PSNR/SSIM/BLEU/PPL/R2/Loss 不乘 100）
PERCENT_STEMS = {
    "acc", "accuracy", "top1", "top5", "top1_acc", "top5_acc", "bal_acc", "balanced_acc",
    "f1", "f1_score", "macro_f1", "micro_f1", "weighted_f1",
    "precision", "recall", "prec", "rec",
    "auc", "auroc", "auprc", "roc_auc", "pr_auc",
    "ap", "ap50", "ap75", "aps", "apm", "apl", "map", "map50", "map75", "map50_95", "map50-95",
    "miou", "iou", "biou", "fw_iou", "mdsc", "dsc", "dice", "mdice", "pa", "mpa", "oa",
    "em", "exact_match", "pass1", "pass5", "pass10", "pass@1", "pass@5", "pass@10",
    "win_rate", "sensitivity", "specificity", "top1_err", "top5_err", "wer", "cer",
}

SPLIT_PREFIXES = {
    "train": "train", "trn": "train", "tr": "train",
    "val": "val", "valid": "val", "validation": "val", "eval": "val", "dev": "val",
    "val1": "val1", "valid1": "val1", "val2": "val2", "valid2": "val2", "val3": "val3",
    "test": "test", "tst": "test", "test500": "test",
}

SKIP_LINE_RE = re.compile(
    r"^\s*(?:Namespace\(|engine/trainer:|Args:|Config:|Traceback|File \"|"
    r"UserWarning|FutureWarning|DeprecationWarning|RuntimeWarning|"
    r"\[PASS\]|\[FAIL\]|\[SUMMARY\]|--[a-zA-Z])"
    r"|https?://|argparse|pip install|conda ",
    re.I,
)
EPOCH_RE = re.compile(
    r"(?:\bepoch\s*\[?\s*(\d+)\s*(?:/\s*(\d+))?\s*\]?|\bep\s*[:=\[]\s*(\d+)(?:\s*/\s*(\d+))?)",
    re.I,
)
STEP_RE = re.compile(r"\b(?:global_step|step|iter|iteration)\s*[:=\[]\s*(\d+)(?:\s*/\s*(\d+))?", re.I)
LINE_SPLIT_RE = re.compile(r"(?:^|[\[\(|\s])(train|val\d?|valid\d?|eval|dev|test)(?:[\]\):|\s]|$)", re.I)

# key=value 或 key: value，拒绝后面紧跟时间/显存/频率等非指标单位
KV_RE = re.compile(
    r"(?<![A-Za-z0-9_./@-])"
    r"([A-Za-z][A-Za-z0-9_./@-]{1,36})"
    r"\s*(?:=|:)\s*"
    r"([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(%?)"
    r"(?![A-Za-z0-9_/])"
)
# 行内 dict / JSON 对象提取
DICT_RE = re.compile(r"\{[^{}]{8,1200}\}")


def _to_float(raw) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v


def classify_metric_key(raw_key: str) -> dict | None:
    """校验并规范化一个指标键名；非合法 ML 指标返回 None。"""
    key = str(raw_key or "").strip().strip("'\"")
    if not key or len(key) > 48:
        return None
    low = key.lower().replace(" ", "_")
    # 剥离 metrics/ 前缀
    if low.startswith("metrics/"):
        low = low[len("metrics/"):]
    low = re.sub(r"\([bmp]\)$", "", low)
    if low in BLACKLIST_KEYS:
        return None

    # 切分 split 前缀（支持 val/acc、eval_loss、test-mIoU 等）
    split = ""
    rest = low
    m_split = re.match(r"^(train|trn|tr|val\d?|valid\d?|eval|dev|test\d*)[/_.-]+(.+)$", low)
    if m_split:
        sp_raw, rest = m_split.group(1), m_split.group(2)
        split = SPLIT_PREFIXES.get(sp_raw, "test" if sp_raw.startswith("test") else sp_raw)

    # 忽略嵌套学习率等
    if rest in BLACKLIST_KEYS or rest.startswith("lr") or rest.endswith(("_time", "_sec", "_ms", "_mem", "_mb", "_gb", "_size", "_count", "_num")):
        return None

    # 常见同义词归一（例如 eval_accuracy 与 val_acc 合并为同一曲线）
    alias_map = {
        "accuracy": "acc",
        "f1_score": "f1",
        "perplexity": "ppl",
        "roc_auc": "auroc",
        "pr_auc": "auprc",
        "map50_95": "map50-95",
        "rouge_l": "rougel",
        "exact_match": "em",
    }
    norm_rest = alias_map.get(rest.replace("/", "_").replace("-", "_"), rest.replace("/", "_").replace("-", "_"))
    rest = alias_map.get(rest, rest)
    stem = norm_rest
    matched_kind = None  # "lower" or "higher"

    if stem in LOWER_METRICS or stem.endswith("_loss") or stem.endswith("_err") or stem.endswith("_error") or stem.endswith("_mse") or stem.endswith("_mae"):
        matched_kind = "lower"
    elif stem in HIGHER_METRICS:
        matched_kind = "higher"
    else:
        # 尾部词根匹配，例如 flame_miou, smoke_ap50, class0_f1
        parts = [p for p in re.split(r"[_/]", norm_rest) if p]
        if len(parts) >= 2:
            tail1 = parts[-1]
            tail2 = "_".join(parts[-2:])
            if tail2 in LOWER_METRICS or tail1 in LOWER_METRICS:
                matched_kind = "lower"
                stem = tail2 if tail2 in LOWER_METRICS else tail1
            elif tail2 in HIGHER_METRICS or tail1 in HIGHER_METRICS:
                matched_kind = "higher"
                stem = tail2 if tail2 in HIGHER_METRICS else tail1

    if matched_kind is None:
        return None

    higher = matched_kind == "higher"
    is_pct = stem in PERCENT_STEMS
    canonical = f"{split}/{rest}" if split else rest
    label = _pretty_label(rest)
    group = _metric_group(split, matched_kind == "lower")
    return {
        "canonical": canonical,
        "split": split,
        "rest": rest,
        "stem": stem,
        "label": label,
        "group": group,
        "higher": higher,
        "is_pct": is_pct,
        "priority": _metric_priority(split, stem, higher),
    }


def _pretty_label(rest: str) -> str:
    clean = rest.replace("mAP50-95", "50-95").replace("map50_95", "50-95").replace("map50-95", "50-95")
    clean = clean.replace("map50", "mAP50").replace("miou", "mIoU").replace("mdsc", "mDSC")
    clean = clean.replace("accuracy", "Acc").replace("precision", "P").replace("recall", "R")
    clean = clean.replace("_", " ").strip()
    return clean[:18] if clean else rest


def _metric_group(split: str, is_loss: bool) -> str:
    if split == "test":
        return "控制台 Test"
    if is_loss:
        return "损失" if not split or split == "train" else f"验证损失 ({split})"
    if split.startswith("val"):
        return f"验证 {split}" if split != "val" else "验证指标"
    if split == "train":
        return "训练指标"
    return "自动指标"


def _metric_priority(split: str, stem: str, higher: bool) -> int:
    # 数值越小优先级越高（用于挑选最多 4 个重点列，避免杂乱）
    score = 50
    if split.startswith("val"):
        score -= 25
    elif not split:
        score -= 10
    elif split == "train":
        score += 5
    if higher:
        score -= 10
    if stem in {"map50_95", "map50-95", "map50", "miou", "f1", "acc", "accuracy", "top1", "auc", "auroc", "dice", "mdsc", "psnr", "ssim", "bleu", "bleu4", "rougel", "em", "pass@1"}:
        score -= 12
    elif stem in {"loss", "total_loss"}:
        score -= 6
    return score


def _sane_value(meta: dict, value: float, has_pct_sign: bool) -> float | None:
    if abs(value) > 1e6:
        return None
    if meta["is_pct"]:
        if value < -1.0 or value > 100.5:
            return None
        if has_pct_sign and value > 1.5:
            return round(value / 100.0, 6)
    elif meta["stem"] in {"ssim", "ms_ssim", "r2"}:
        if value < -5.0 or value > 100.5:
            return None
    elif meta["stem"] == "loss" or meta["stem"].endswith("_loss"):
        if value < -1e4 or value > 1e5:
            return None
    return round(value, 6)


def _collapse_cr(text: str) -> list[str]:
    out = []
    for raw_line in text.split("\n"):
        segs = [s.strip() for s in raw_line.split("\r") if s.strip()]
        if segs:
            out.append(segs[-1])
    return out


def _read_tail_text(path: Path, max_bytes: int = 1500000) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            end = fh.tell()
            fh.seek(max(0, end - max_bytes))
            raw = fh.read()
    except OSError:
        return ""
    return ANSI_RE.sub("", raw.decode("utf-8", errors="replace"))


def _extract_dict_kvs(line: str) -> list[tuple[str, float, bool]]:
    found: list[tuple[str, float, bool]] = []
    for m in DICT_RE.finditer(line):
        chunk = m.group(0)
        obj = None
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            try:
                obj = json.loads(chunk.replace("'", '"').replace("True", "true").replace("False", "false").replace("None", "null"))
            except json.JSONDecodeError:
                obj = None
        if not isinstance(obj, dict):
            continue
        for k, v in obj.items():
            fv = _to_float(v)
            if fv is not None:
                found.append((str(k), fv, False))
    return found


def parse_console_log(path: Path, max_metrics: int = 8) -> dict:
    """从任意训练控制台 .log 文件自动抽取逐轮/逐脉冲指标与 Test 摘要。"""
    text = _read_tail_text(path)
    if not text:
        return {"rows": [], "columns": [], "latest": {}, "best": {}, "test_metrics": {}, "epoch_done": 0, "total": None, "x_axis": "epoch"}

    lines = _collapse_cr(text)
    cur_epoch = 0
    cur_step = 0
    total_epochs: int | None = None
    saw_explicit_epoch = False
    saw_explicit_step = False
    seq_idx = 0

    # raw_points[canonical] = list[(x, val)]
    raw_points: dict[str, list[tuple[int, float]]] = {}
    meta_by_key: dict[str, dict] = {}
    test_metrics: dict[str, float] = {}

    for line in lines:
        if len(line) < 4 or SKIP_LINE_RE.search(line):
            continue

        em = EPOCH_RE.search(line)
        if em:
            ep = int(em.group(1) or em.group(3))
            tot = em.group(2) or em.group(4)
            if 0 <= ep <= 50000:
                cur_epoch = ep
                saw_explicit_epoch = True
                if tot and tot.isdigit() and int(tot) >= ep:
                    total_epochs = int(tot)
        sm = STEP_RE.search(line)
        if sm:
            st = int(sm.group(1))
            if 0 <= st <= 10000000:
                cur_step = st
                saw_explicit_step = True

        # 行级 split 提示
        line_split = ""
        lsp = LINE_SPLIT_RE.search(line)
        if lsp:
            sp = lsp.group(1).lower()
            line_split = SPLIT_PREFIXES.get(sp, sp)

        pairs: list[tuple[str, float, bool]] = []
        if "{" in line and "}" in line:
            pairs.extend(_extract_dict_kvs(line))
        if not pairs:
            for km in KV_RE.finditer(line):
                k, v_str, pct = km.groups()
                fv = _to_float(v_str)
                if fv is not None:
                    pairs.append((k, fv, pct == "%"))
        if not pairs:
            continue

        # 判断本行是否有合法指标
        valid_pairs = []
        for k, fv, has_pct in pairs:
            kl = k.lower()
            if kl in {"epoch", "ep"} and 0 <= fv <= 50000:
                cur_epoch = int(round(fv))
                saw_explicit_epoch = True
                continue
            if kl in {"step", "global_step", "iter"} and 0 <= fv <= 10000000:
                cur_step = int(round(fv))
                saw_explicit_step = True
                continue
            eff_key = k if ("/" in k or "_" in k and k.split("_", 1)[0].lower() in SPLIT_PREFIXES) else (f"{line_split}/{k}" if line_split else k)
            meta = classify_metric_key(eff_key)
            if meta is None:
                continue
            val = _sane_value(meta, fv, has_pct)
            if val is None:
                continue
            valid_pairs.append((meta, val))

        if not valid_pairs:
            continue

        if saw_explicit_epoch:
            x = cur_epoch
        elif saw_explicit_step:
            x = cur_step
        else:
            seq_idx += 1
            x = seq_idx

        for meta, val in valid_pairs:
            if meta["split"] == "test":
                test_metrics[meta["label"]] = val
                continue
            ck = meta["canonical"]
            meta_by_key[ck] = meta
            raw_points.setdefault(ck, []).append((x, val))

    return _finalize_parsed(raw_points, meta_by_key, test_metrics, total_epochs, "epoch" if saw_explicit_epoch else ("step" if saw_explicit_step else "idx"), max_metrics)


def parse_generic_files(folder: Path, max_metrics: int = 8) -> dict | None:
    """支持常见非 Ultralytics 结构化日志：metrics.csv / history.csv / metrics.jsonl / trainer_state.json。"""
    # 1. HuggingFace trainer_state.json
    ts = folder / "trainer_state.json"
    if ts.is_file():
        try:
            obj = json.loads(ts.read_text(encoding="utf-8", errors="replace"))
            history = obj.get("log_history") if isinstance(obj, dict) else None
            if isinstance(history, list) and history:
                return _parse_record_list(history, total=int(obj.get("num_train_epochs") or 0) or None, max_metrics=max_metrics)
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    # 2. JSONL 文件
    for name in ("metrics.jsonl", "log.jsonl", "scalars.jsonl", "history.jsonl"):
        jp = folder / name
        if not jp.is_file():
            continue
        recs = []
        try:
            for line in jp.read_text(encoding="utf-8", errors="replace").splitlines()[-3000:]:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    recs.append(item)
        except (OSError, json.JSONDecodeError):
            continue
        if recs:
            return _parse_record_list(recs, max_metrics=max_metrics)

    # 3. 通用 CSV 文件（跳过 results.csv / triple_val.csv / train_log.csv，这些由专用解析器处理）
    for name in ("metrics.csv", "history.csv", "log.csv", "scalars.csv", "training_log.csv"):
        cp = folder / name
        if not cp.is_file():
            continue
        try:
            text = cp.read_text(encoding="utf-8-sig", errors="replace")
            recs = list(csv.DictReader(text.splitlines()))
        except OSError:
            continue
        if recs:
            return _parse_record_list(recs, max_metrics=max_metrics)
    return None


def _parse_record_list(records: list[dict], total: int | None = None, max_metrics: int = 8) -> dict:
    raw_points: dict[str, list[tuple[int, float]]] = {}
    meta_by_key: dict[str, dict] = {}
    test_metrics: dict[str, float] = {}
    saw_epoch = False
    saw_step = False
    for idx, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            continue
        ep = _to_float(rec.get("epoch"))
        st = _to_float(rec.get("step") or rec.get("global_step") or rec.get("iter"))
        if ep is not None:
            saw_epoch = True
            x = int(round(ep))
        elif st is not None:
            saw_step = True
            x = int(round(st))
        else:
            x = idx
        for k, v in rec.items():
            fv = _to_float(v)
            if fv is None:
                continue
            meta = classify_metric_key(k)
            if meta is None:
                continue
            val = _sane_value(meta, fv, False)
            if val is None:
                continue
            if meta["split"] == "test":
                test_metrics[meta["label"]] = val
                continue
            ck = meta["canonical"]
            meta_by_key[ck] = meta
            raw_points.setdefault(ck, []).append((x, val))
    return _finalize_parsed(raw_points, meta_by_key, test_metrics, total, "epoch" if saw_epoch else ("step" if saw_step else "idx"), max_metrics)


def _finalize_parsed(
    raw_points: dict[str, list[tuple[int, float]]],
    meta_by_key: dict[str, dict],
    test_metrics: dict[str, float],
    total_epochs: int | None,
    x_axis: str,
    max_metrics: int,
) -> dict:
    if not raw_points:
        return {"rows": [], "columns": [], "latest": {}, "best": {}, "test_metrics": test_metrics, "epoch_done": 0, "total": total_epochs, "x_axis": x_axis}

    # 按语义优先级截取前 max_metrics 列，防止杂乱脚本撑爆面板
    ranked_keys = sorted(
        raw_points.keys(),
        key=lambda ck: (meta_by_key[ck]["priority"], -len(raw_points[ck]), ck),
    )[:max_metrics]

    by_x: dict[int, dict] = {}
    latest: dict[str, float] = {}
    best: dict[str, dict] = {}
    columns: list[dict] = []
    max_x = 0

    for rank, ck in enumerate(ranked_keys):
        pts = raw_points[ck]
        meta = meta_by_key[ck]
        col_key = f"console|{ck}"
        # 同一 x 取最后一次打印的值
        folded: dict[int, float] = {}
        for x, val in pts:
            folded[x] = val
            if x > max_x:
                max_x = x
        xs_sorted = sorted(folded.keys())
        last_x = xs_sorted[-1]
        latest[col_key] = folded[last_x]
        best_x = max(xs_sorted, key=lambda x: folded[x]) if meta["higher"] else min(xs_sorted, key=lambda x: folded[x])
        best[col_key] = {"value": folded[best_x], "epoch": best_x}
        for x in xs_sorted:
            by_x.setdefault(x, {"epoch": x})[col_key] = folded[x]
        columns.append({
            "key": col_key,
            "group": meta["group"],
            "label": meta["label"],
            "cls": "all",
            "higher": meta["higher"],
            "is_pct": meta["is_pct"],
            "source": "console",
            "key_for": ["other"] if rank < 4 else [],
        })

    rows = [by_x[x] for x in sorted(by_x.keys())]
    return {
        "rows": rows,
        "columns": columns,
        "latest": latest,
        "best": best,
        "test_metrics": test_metrics,
        "epoch_done": max_x,
        "total": total_epochs,
        "x_axis": x_axis,
    }
