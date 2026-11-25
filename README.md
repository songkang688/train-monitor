# train-monitor · 通用 GPU 训练监控台

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-0d6b65?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero%20(Stdlib%20Only)-b85027?style=flat-square)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-2c3d47?style=flat-square)](./LICENSE)

一个 **零外部依赖（纯 Python 标准库）、零项目硬编码、换台电脑 `git clone` 即用** 的 Linux GPU 深度学习训练监控与多实验对比看板。

无需在训练代码里嵌入 TensorBoard / WandB SDK，也无需为 PyTorch、Ultralytics YOLO、HuggingFace Trainer、MMDetection、Lightning 或自写脚本单独适配指标名——它直接从显卡进程、结构化日志以及控制台输出中**自动识别实验进度、核心指标、异常报错与独立测试结果（Test）**。

---

## 核心特性

| 模块 | 能力亮点 |
|---|---|
| **零配置实时发现** | 即使不填 `config.json` 的 `roots`，启动即自动调用 `nvidia-smi` 与 `/proc` 识别全部显卡负载、温度/功耗、**本人实时训练进程**、本人辅助脚本以及**他人占卡详情**，清晰标出 **「✓ 空闲 · 可排卡」** 卡号。 |
| **穿透管道读日志** | 自动穿透 `python train.py ... 2>&1 \| tee train.log` 的管道定位真实 `.log` 文件，实时解析进度条（Ultralytics / tqdm / Epoch）、剩余时间与 OOM/Traceback 报错。 |
| **多格式解析 + 控制台智能兜底** | 优先读取 `results.csv`、`train_log.csv`、`metrics.csv`、`history.csv`、`metrics.jsonl`、HuggingFace `trainer_state.json`；没有结构化文件时，自动从控制台 `.log` 提取合法指标。 |
| **严格白名单防噪音 & 防乱列** | 内置超参/系统噪音黑名单（自动拦截 `batch_size`、`lr`、`epochs`、`imgsz`、`workers`、`seed`、`time`、`mem`、`vram`、`speed`、`port`、`pid` 等）+ 合法 ML 指标白名单；多类异构小实验共存时自动切换为「按实验独立展示自身核心指标」，绝不互相污染表格。 |
| **高段分离多实验对比图** | 针对高分平台期曲线挤成一条直线的问题，默认开启 **「聚焦高段 + 跳过热身」**，支持一键切换 **「相对均值差值 Δ」**、滑动平滑（`1/3/5/9`）、右侧防重叠直标标签、悬停聚焦单条、点击隐藏离群线，以及带排名和 `Δ vs #1` 的数值对比汇总表。 |
| **实验备注 & 独立 Test 分离** | 支持给每条实验添加持久化备注（`notes.json`）与快捷标签；自动采集 `eval_best.json` / `test_metrics.json` / `eval_*/summary.json` 及控制台 `[Test]` 输出行，并支持网页手动录入（`test_results.json`）。独立 Test 在 `重点/详细` 模式下单独占一行（`🧪 独立 Test`），也可切到 `测试结果` 专属视图排序对比。 |

---

## 目录结构

```text
train-monitor/
├── server.py           # HTTP 服务入口（支持 CLI 参数覆盖）
├── board.py            # GPU/进程自动发现、多格式与控制台解析引擎
├── console_metrics.py  # 控制台指标严格白名单提取与防噪音模块
├── config.json         # 可选配置文件（监视目录、端口、进程关键字）
├── start.sh            # 前台/后台守护启停脚本（start / daemon / stop / status）
├── static/
│   └── index.html      # 单文件前端页面（无前端构建步骤，离线可用）
├── LICENSE             # MIT 开源协议
└── README.md           # 使用与端口转发说明
```

---

## 30 秒快速上手

### 1. 在目标 Linux GPU 服务器克隆并启动

```bash
git clone https://github.com/songkang688/train-monitor.git
cd train-monitor

# 方式 A：零配置前台试跑（自动监视本机全部显卡与实时训练进程）
bash start.sh

# 方式 B：后台常驻 + 指定要扫描的历史实验输出根目录（推荐日常使用）
bash start.sh daemon --port 8788 --root /你的实验输出目录 --root /另一个项目/runs
```

常用守护运维命令：

```bash
bash start.sh status   # 查看后台运行状态与最新日志
bash start.sh stop     # 停止后台服务
```

### 2. 可选：`config.json` 持久化配置

如果你希望以后直接 `bash start.sh daemon` 就能长期扫描固定项目目录，直接编辑 [config.json](./config.json)：

```json
{
  "roots": [
    "/data/your_project/runs",
    "/home/user/experiments"
  ],
  "log_dirs": [
    "/data/your_project/logs"
  ],
  "script_needles": [
    "train", "finetune", "pretrain", "run_", "torchrun", "accelerate", "deepspeed", "yolo"
  ],
  "max_scan_depth": 4,
  "max_age_days": 30,
  "host": "127.0.0.1",
  "port": 8788
}
```

- `roots`：要递归扫描的历史实验目录列表（可留空，留空时只实时展示正在运行的 GPU/训练进程）。
- `log_dirs`：集中存放 `.log` 的目录（可选，自动扫描近 30 分钟活跃日志）。
- `script_needles`：尚未显存占用（如刚启动、数据预处理、CPU 实验）时用于匹配训练脚本的关键字。
- `host` / `port`：默认 `127.0.0.1:8788`（仅本机监听，配合下方端口转发最安全）。

---

## 端口转发全方法（在本地电脑浏览器访问）

服务默认绑定远程服务器的 `127.0.0.1:8788`，避免无鉴权服务暴露在公网。在**自己的本地电脑**任选以下一种方式转发即可打开 `http://127.0.0.1:8788/`：

### 方法一：SSH 本地端口转发（最通用，Linux / macOS / Windows PowerShell）

```bash
# 1) 前台转发（Ctrl+C 断开）
ssh -N -L 8788:127.0.0.1:8788 用户名@服务器IP -p 22

# 2) 后台静默常驻转发（推荐：不占用终端窗口）
ssh -CNfL 8788:127.0.0.1:8788 用户名@服务器IP -p 22
```

打开本地浏览器访问：**`http://127.0.0.1:8788/`**。

#### 常见场景变体：

- **本地 `8788` 已被占用**（映射到本地 `18788`）：
  ```bash
  ssh -CNfL 18788:127.0.0.1:8788 用户名@服务器IP -p 22
  # 浏览器访问 http://127.0.0.1:18788/
  ```
- **经过跳板机（Bastion）双跳转发到内网计算节点**：
  ```bash
  ssh -CNfL 8788:127.0.0.1:8788 -J 跳板机用户@跳板机IP 用户名@内网节点IP
  ```
- **写入本地 `~/.ssh/config` 永久省事**（Windows 为 `C:\Users\你的用户名\.ssh\config`）：
  ```sshconfig
  Host my-gpu
      HostName 服务器IP
      User 用户名
      Port 22
      LocalForward 8788 127.0.0.1:8788
      ServerAliveInterval 60
      ServerAliveCountMax 3
  ```
  之后每次 `ssh my-gpu`（或后台 `ssh -fN my-gpu`）自动连上转发。
- **关闭后台 SSH 转发隧道**：
  ```bash
  pkill -f "8788:127.0.0.1:8788"
  ```

### 方法二：VS Code / Cursor Remote-SSH 一键转发（最省心）

1. 在远程终端运行 `bash start.sh daemon`。
2. 打开底部 **「端口（PORTS）」** 面板，点击 **「转发端口（Forward a Port）」**，输入 `8788` 回车。
3. 点击端口右侧的 **地球图标** 即可直接在本地浏览器打开。

### 方法三：云算力平台（AutoDL / 恒源云 / 矩池云等）

- **对接平台自带自定义服务端口**（例如 AutoDL 的 `6006`）：
  ```bash
  bash start.sh daemon --host 0.0.0.0 --port 6006
  ```
  启动后直接在云平台控制台点击「自定义服务」HTTPS 链接即可打开。
- **使用平台 SSH 命令叠加 `-L` 参数**：
  ```bash
  ssh -CNfL 8788:127.0.0.1:8788 -p 平台端口 root@平台SSH地址
  ```

### 方法四：实验室同局域网直连（仅限受信任内网）

```bash
bash start.sh daemon --host 0.0.0.0 --port 8788
# 浏览器直接访问 http://服务器局域网IP:8788/
```

> **安全提示**：公网服务器切勿长期使用 `--host 0.0.0.0`，建议保持默认 `127.0.0.1` + SSH / IDE 端口转发。

---

## 让任意项目自动被监控的日志与 Test 规范

### 1. 推荐的训练启动写法

通过 `tee` 或重定向落盘后，监控台会自动穿透管道追踪实时进度、报错堆栈与控制台指标：

```bash
# 终端实时看输出 + 日志自动落盘 + 监控台自动追踪
CUDA_VISIBLE_DEVICES=0 python train.py --epochs 100 2>&1 | tee train.log

# 后台 nohup 同样完全支持
CUDA_VISIBLE_DEVICES=0 nohup python train.py --epochs 100 > train.log 2>&1 &
```

### 2. 控制台指标打印示例（严格防乱白名单）

以下任意常见格式都会被自动识别并绘制曲线：

```text
# 格式 1：常见 key=val / key: val
Epoch 12/100 [Train] loss=0.4123 lr=0.0003 batch_size=32 time=12.4s
Epoch 12/100 [Val] val_loss=0.3510 val_acc=91.45 val_f1=0.9082 mIoU=0.812

# 格式 2：HuggingFace Trainer / JSON / Python dict 行
{'eval_loss': 0.342, 'eval_accuracy': 0.921, 'eval_f1': 0.918, 'epoch': 5.0}

# 格式 3：独立测试输出（含 [Test] 或 test_ 前缀，自动归入「独立 Test」栏）
[Test] acc=92.60 f1=92.15 mAP50=74.80
```

> 上述日志中的 `lr=0.0003`、`batch_size=32`、`time=12.4s` 等超参/耗时字段会被 `console_metrics.py` 自动过滤，只提取合法训练/评估指标。

### 3. 自动采集独立测试 JSON 文件（可选）

实验目录内存在以下 JSON 文件时会自动读取到 **「🧪 独立 Test」** 栏（也可随时在网页点击 `+ Test` 手动录入）：

- `eval_best.json` / `test_eval.json` / `eval_tests.json`（多测试集）：
  ```json
  {
    "ckpt": "best.pt (E85)",
    "tests": {
      "Test-A": {"mAP50": 74.2, "F1": 81.5},
      "Test-B": {"mAP50": 69.8, "F1": 77.4}
    }
  }
  ```
- `test_metrics.json`（单测试集扁平格式）：
  ```json
  {
    "checkpoint_used": "best.pth",
    "checkpoint_epoch": 40,
    "acc": 0.9312,
    "f1": 0.9275
  }
  ```

---

## 常见问题排查

1. **`Address already in use`（端口占用）**：
   - 检查占用：`ss -tulpn | grep 8788` 或 `lsof -i :8788`
   - 换端口启动：`bash start.sh daemon --port 8790`（同步修改转发端口）。
2. **页面打开后没有历史实验？**
   - 默认 `config.json` 中 `roots` 为空，仅实时监视正在运行的 GPU/训练进程。
   - 追加你的输出目录即可：`bash start.sh stop && bash start.sh daemon --root /你的实验目录`。
3. **看到了正在跑的进程，但点开没有终端日志？**
   - 直接输出到交互终端 `/dev/pts/*` 的标准输出无法从外部旁路读取，启动训练时加上 `2>&1 | tee train.log` 或 `> train.log 2>&1` 即可。

---

## License

[MIT](./LICENSE)
