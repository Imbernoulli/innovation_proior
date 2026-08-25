# FrontierSmith 评测栈 @ jiaolab 运行手册

2026-08-25 建立。本机 = **专职评测节点**（gpublaze 专职训练，两边分工固定）。
8×A100-80G **PCIe（无 NVLink）**、128 核 Intel Xeon 8358、1TB RAM、Ubuntu 24.04 /
kernel 6.8、driver 580.126。无 slurm。所有 Princeton 历史脚本（`slurm/`、
`scripts/`）与 gpublaze 的 `scripts/gpublaze/` **一个字节都没改**；本机差异全部
收敛在 **`scripts/jiaolab/` wrapper 层**（env var 覆盖）。

## 0. 机器纪律（先读）

- **GPU 与用户 `druv` 共用**：每卡常驻 2-4G，GPU 0 上 ~20G。**绝不 kill 别人的
  进程**。`env_jiaolab.sh` 的 `fs_guard_gpus` 只接受**空闲显存 ≥ `FS_MIN_FREE_GB`
  （默认 70G）**的卡，`fs_pick_free_gpus N` 自动挑最空的 N 张。
  serve 的 `GPU_MEMORY_UTILIZATION` 默认 **0.85**（gpublaze 是 0.90），给共租者
  留 ~9G 余量；KV cache 大小只影响吞吐，不影响分数。
- **`/` 只剩 ~490G（93% 满）**。大下载前先算空间。
- **docker 需要 sudo（不可用）**，**没有 bwrap**，**apptainer 1.3.1 可用** ——
  ALE-Bench 判分只能走 apptainer（§2.2）。
- **真 go-judge 跑不起来**：Ubuntu 24.04 的
  `apparmor_restrict_unprivileged_userns` 让 `unshare -U true` 通过但
  `/proc/self/uid_map` 写入被拒，go-judge 启动即
  `fork/exec /proc/self/exe: permission denied`（**套进 apptainer 里也一样**）。
  解除需要 sudo → **本机 FCS 判分走认证过的 `gojudge_shim_v2`**（Princeton 认证
  94.3% byte-exact）。见 §2.1。
- **跨节点分数不可同表（EVAL_ROBUSTNESS_zh.md 铁律 1）**。实测节点标定：

  | 节点 | CPU | shim speedFactor | judge backend |
  |---|---|---|---|
  | gpublaze | AMD EPYC 9654 96-Core | **0.7941** | 真 go-judge |
  | jiaolab  | Intel Xeon 8358 2.60GHz | **0.4322** | gojudge_shim_v2 |

  jiaolab 单核约为 gpublaze 的 **54%**，时限边缘题必然翻转。**jiaolab 上的一切
  对照都只能和 jiaolab 的锚点比**（锚点见 §5）。每个 client 都会在输出目录写
  `judge_node_meta.json`（含 `node_speed_calibration` 与
  `ale_container_backend`），对比前先查它。

## 1. 一图流：本机的 serve/eval 分离

| Princeton | gpublaze wrapper | **jiaolab wrapper** |
|---|---|---|
| `sbatch cc_serve_only.sh` | `scripts/gpublaze/serve_local.sh` | `GPUS=3 bash scripts/jiaolab/serve_local.sh <MODEL> <TAG>` |
| `sbatch cc_eval_cpu_client.sh` | `scripts/gpublaze/eval_client_local.sh` | `TAG=.. SOURCE=.. bash scripts/jiaolab/eval_client_local.sh` |
| `cc_eval_split_submit.sh` | `scripts/gpublaze/eval_split_local.sh` | `GPUS=3 bash scripts/jiaolab/eval_split_local.sh <MODEL> [TAG] [KIND] [SHARDS]` |
| （池化 2 卡） | `scripts/gpublaze/launch_pool_eval.sh` | **`bash scripts/jiaolab/launch_pool_eval.sh <MODEL> <TAG> fcsale [gpuA gpuB]`** ← 生产用法 |
| GPFS registry | `.cache/vllm_pool/` | 同（协议逐字节兼容） |
| Apptainer judge SIF | 宿主机直跑 node server | 同（node v18） |
| go-judge / shim | **真 go-judge** | **shim（userns 被 AppArmor 拒）** |
| `.venv`（client） | `.venv-gpublaze` | `.venv-jiaolab`（`.venv` symlink 指向它） |
| serve venv | `/srv/home/bohanlyu/sesl/.venv` | `/home/bohan/venv-vllm-jiaolab` |

公共环境统一在 `scripts/jiaolab/env_jiaolab.sh`（所有 wrapper source 它）。

## 2. 逐评测项状态

### 2.1 FrontierCS algorithm（FCS，C++ judge）✅ 已验收

- 题库：官方 clone 的 `problems`（**188 题**，与 gpublaze 同口径，与 ailab 历史
  172 题不可同表）rsync 到 `.cache/external/Frontier-CS/algorithmic/problems`
  （2.6G），`Frontier-CS/algorithmic/problems` symlink 指过去。
- **官方 python 包（必须，坑）**：driver `eval_qwen35_base_vllm_request.py` 会把
  `.cache/Frontier-CS-official`（指向官方 clone 的 symlink）与其 `src/` 加进
  `sys.path`，import `algorithmic.scripts.generate_solutions`（官方 prompt +
  `extract_cpp_code`）和 `frontier_cs.runner.algorithmic_local`。
  **只搬 `problems` 是不够的**：缺包时 `--frontiercs-score-backend official`
  （默认口径）不会中止，而是对**每一个** frontiercs 样本抛 RuntimeError、记成
  error 且 reward=0.0 —— 整个 FCS 分数悄悄变成地板零。2026-08-25 实测踩到
  （烧了 15 个样本）。因此：
  ```bash
  # gpublaze 上：
  rsync -a --exclude='.git/' --exclude='algorithmic/problems' --exclude='algorithmic/solutions' \
        --exclude='research/' --exclude='datasets/' \
        .cache/external/Frontier-CS/ jiaolab:<FS>/.cache/external/Frontier-CS/
  # jiaolab 上：
  ln -sfn <FS>/.cache/external/Frontier-CS <FS>/.cache/Frontier-CS-official
  ```
  `scripts/jiaolab/eval_client_local.sh` 现在有**开跑前的显式检查**，缺件直接
  拒绝启动（不再靠事后看日志）。
- judge 栈：`start_frontiercs_judge_local.sh` 起 `gojudge_shim_v2` +
  宿主 node v18 跑 `server.js`（stage 目录 `.cache/jiaolab/judge_app`，
  node_modules 从 gpublaze rsync）。
- **`GJ_BACKEND=auto` 的探针改成 `unshare -Ur`**（gpublaze 用的是 `unshare -U`）
  ——本机 `unshare -U` 会假通过，必须探 uid_map 写权限，否则 auto 会选中跑不起来
  的真 go-judge。go-judge 二进制仍留在 `.cache/bin/`：哪天 AppArmor/sudo 变了，
  `GJ_BACKEND=gojudge` 直接可用。
- **双向验收（2026-08-25）**：
  | 用例 | jiaolab | gpublaze 记录 |
  |---|---|---|
  | 空程序（`int main(){return 0;}`）→ 题 1 | **0.0** | 0（真实 WA） |
  | 官方 leaderboard 解 `solutions/1/deepseekreasoner.cpp` | **64.0465** | **64.05** |

- **常驻 judge**：端口 **18082**（不是 gpublaze 的 8082 —— 本机 8082 被别的用户
  占了），go-judge/shim 端口 15050：
  ```bash
  cd /home/bohan/innovation_proior/training/FrontierSmith
  PORT=18082 GJ_PORT=15050 GJ_BACKEND=shim GJ_PARALLELISM=16 JUDGE_WORKERS=16 \
  RUNTIME_DIR=$PWD/.cache/frontiercs-judge/18082 \
    setsid nohup bash scripts/jiaolab/start_frontiercs_judge_local.sh >> logs/judge_18082.log 2>&1 &
  ```
  eval client 会各自起自己的临时 judge（随机端口），**不依赖**这个常驻实例。

### 2.2 ALE-Bench（40 题）✅ 已验收（apptainer 后端）

docker 要 sudo、没有 bwrap → **apptainer 是本机唯一沙箱**。

- SIF（`/home/bohan/sif/`，即 `ALE_BENCH_APPTAINER_DIR`）：
  `ale-bench_cpp17-202301.sif`（742M）、`rust_1.79.0-buster.sif`（431M，
  `apptainer pull docker://rust:1.79.0-buster`）。**两个都必须有**：
  编译/运行走 ale-bench 镜像，input 生成与 tester/vis 走 rust 镜像。
- 数据：`data/alebench/local_data`（40 题 zip，96M）、`val.parquet`、
  `full40.parquet`；rust 工具缓存 `.cache/ale-bench/rust-tool-builds`
  **40/40**（机器无关，从 gpublaze rsync，无需重 build）。
- **后端实现 `scripts/jiaolab/pysite/ale_apptainer_backend.py`**（由
  `pysite/sitecustomize.py` 在 `ALE_BENCH_CONTAINER_BACKEND=apptainer` 时自动
  安装）。**为什么不直接用 ALE-Bench 自带的 apptainer 后端**（
  `ale_bench.utils._ApptainerContainer`，Princeton 集群补丁）：
  1. **fail-loud**：自带的是**惰性**的，`subprocess.run` 在 `wait()` 里才发生；
     而 harness 把 `wait()` 抛出的**任何**异常转成 `COMPILATION_ERROR`
     CaseResult ——即**静默 0 分**。apptainer 不在 PATH、SIF 损坏、/tmp 满，都会
     被记成"模型写的代码编不过"。本后端在 `containers.run()` 里**急切启动**
     （和 docker daemon、和 gpublaze 的 bwrap 后端一致），infra 故障一路抛到
     `compute_score` → **`AleInfraError`**。
  2. **1 核绑定**：harness 向 docker 要 `cpu_quota=100000`（1 CPU），自带后端
     静默丢弃。ALE 分数是**墙钟敏感**的，本机 128 核还与他人共用，不绑核既不
     忠实也不可复现。本后端用 flock 核池（与 gpublaze 同一机制）给每个沙箱租
     一个**不同**的核。
- **已知语义差（记录在案）**：`network_disabled=True` 未强制 ——
  非特权 `apptainer --net` 需要 setuid 安装，本机没有。与 Princeton 自带
  apptainer shim 的差是一样的。`mem_limit` 默认不施加
  （`ALE_APPTAINER_MEM_LIMIT=1` 可开）：apptainer 的 Go starter 会预留巨量虚拟
  地址空间，外层 RLIMIT_AS 会打死 starter；判分相关的内存约束是 harness 自己的
  max-RSS 检查（`parse_profiles`），不受影响。
- **验收套件 `scripts/jiaolab/ale_apptainer_selftest.py`**（gpublaze
  `ale_host_selftest.py` 的等价物，**共用同一份固定样本**
  `scripts/gpublaze/ale_host_test_assets/samples.json`）：
  ```bash
  .venv/bin/python scripts/jiaolab/ale_apptainer_selftest.py all
  ```
  **2026-08-25 结果**：
  - `compile` ✅（SIF 内真 g++ 12.2.0，走 `run_compile_container`）
  - `fault` ✅ 3/3：SIF 目录缺失 → `AleInfraError`；apptainer 不在 PATH →
    `AleInfraError`；健康 → 正常出分（**无静默 0 分**）
  - `concurrency` ✅ 8 样本 / 4 并发 130s，0 error、无 fd 泄漏、无 /tmp 残留、
    无僵尸进程
  - `compare` + `report` vs gpublaze 记录：**6 题 × 2 样本 × 5 字段 全部 OK，
    `ALL EQUAL (tol=1e-06)` / `ALL GREEN`**（含交互题 **ahc008**：
    `abs=3711608 perf=780 rank=625`，与 gpublaze 逐字段一致）

### 2.3 FrontierCS research ⏳ 机制在位、未验收

`eval_client_local.sh` 的 research 分支与 gpublaze 同参
（`FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB=0`、默认 `research_cpu.parquet`），
但 research 的数据集（cant_be_late traces、SIFT1M、pysr/Julia 等）**没有搬到
本机**。本轮 jiaolab 只做 **FCS-alg + ALE40**。

### 2.4 MLS-Bench ❌ 本轮不做（留在 gpublaze）

补丁版 harness 本身只有 2.0G（gpublaze `.cache/mlsbench-eval`），但它跑任务要
gpublaze 的 conda envs：**`/srv/home/bohanlyu/miniconda3/envs` = 212G / 44 个
env**，另加 `vendor/{data,workspace,external_packages}`（symlink 到 dev
checkout，未计）。jiaolab `/` 只剩 **~490G（93% 满）**，而本轮排队要评的 9 个
模型（3 setting + 6 soup）本身就要 ~81G。212G + 81G 会把可用空间压到 ~200G，
在一块与他人共用、已经 93% 满的盘上不可接受，且搬运本身 ~35 分钟。
**结论：jiaolab 只做 FCS+ALE，MLS 留 gpublaze。**

## 3. 环境（venv）

两个 venv 都是从 gpublaze **rsync 过来再重定位**的（同一 uv 管理的 CPython
3.12.13，包集合逐位一致，避免"重装出一个不同的栈"）：

| venv | 路径 | 内容 |
|---|---|---|
| client | `training/FrontierSmith/.venv-jiaolab`（`.venv` symlink 指向它） | torch 2.13.0+**cpu**、verl editable、ALE-Bench editable、transformers 5.7.0、pandas/pyarrow/openai |
| serve | `/home/bohan/venv-vllm-jiaolab` | **vllm 0.21.0+cu129**、torch **2.11.0+cu128**（A100 sm80 OK）、triton 3.6.0、flashinfer 0.6.8、tilelang、transformers 5.8.1 |

重定位做了 4 件事（如果将来再搬一次，照做）：
1. `pyvenv.cfg` 的 `home=` 指向 `/home/bohan/.local/share/uv/python/...`；
2. uv 托管 python 的 `cpython-3.12-linux-x86_64-gnu` symlink 指回本机绝对路径；
3. `bin/*` 的 shebang 与 `activate`、`lib/**/*.pth`（editable 安装指向
   `ALE-Bench/src` 与 `verl`）里的 `/srv/home/bohanlyu` → `/home/bohan`；
4. **vLLM penalty fast path 注入块里的 `_fs_dir`** 也要改成本机的
   `.../FrontierSmith/scripts`，否则 `import vllm_penalty_fastpath` 失败、
   banner 变成 `FAILED, using stock`。

### penalty fast path（**不可省**）

`presence_penalty=1.5` 会让 penalty kernel 每个 decode step 重建
`[batch, max_len]` 张量；不打补丁吞吐只有 **1/6**。serve venv 已打好补丁；
重建 venv 后必须重打：

```bash
RL_VENV=/home/bohan/venv-vllm-jiaolab bash scripts/jiaolab/apply_vllm_penalty_fastpath_jiaolab.sh
```

**每次 serve 都要在日志里确认这一行**：`[fs] vLLM penalty fast path ACTIVE`。

## 4. 跑法

### 4.1 生产用法：2 卡 TP=1 池 + 钉死的 2 个 client

```bash
cd /home/bohan/innovation_proior/training/FrontierSmith
setsid nohup bash scripts/jiaolab/launch_pool_eval.sh <MODEL_DIR_or_HF_id> <TAG> fcsale [gpuA gpuB] \
  > logs/pool_launch_<TAG>.log 2>&1 &
# 省略 gpuA/gpuB 时自动挑两张最空的卡（≥70G 空闲）
```

三条不可动的本机配置（血泪教训）：
1. **每卡一个 TP=1 引擎，不要 TP=2** —— A100 PCIe 无 NVLink，Qwen3.5 混合架构
   （attention + gated-delta-net）跨 PCIe 的 TP 效率很差，两个独立引擎更快；
2. **`--no-enable-prefix-caching`** —— Qwen3.5 的 mamba/GDN 层 + vLLM 0.21 的
   prefix caching 会让引擎卡死（gpublaze 实证）；`serve_local.sh` 里
   `ENABLE_PREFIX_CACHING` 默认已是 0，池化启动器再显式传一次；
3. **`VLLM_RPC_TIMEOUT=600000`** —— 32k 生成 × 并发 64 会打爆默认 RPC 期限。

client 侧同样钉死：`REQUEST_TIMEOUT=7200`、`CONCURRENCY=64`、
`EVAL_SCORE_CONCURRENCY=4`，**每个 shard 用 `VLLM_BASE_URL` 显式钉到不同后端**
（靠 `--least-loaded` 自动挑会撞车：两个 client 同时启动会挑到同一个引擎，
另一张 A100 整轮空转）。

### 4.2 采样协议（不可动）

`n=5` / `max_tokens=32768` / `temperature=1.0` / `top_p=0.95` / `top_k=20` /
`presence_penalty=1.5` / `EVAL_RESEARCHER_YEAR=2026` / thinking on。
`eval_client_local.sh` 强制（`MAX_TOKENS<32768` 直接报错）。

### 4.3 单卡 / 调试

```bash
GPUS=3 bash scripts/jiaolab/eval_split_local.sh <MODEL> <TAG> fcsale 2
bash scripts/jiaolab/serve_stop.sh <TAG>        # 干净停后端（只杀自己的 pid）
bash scripts/jiaolab/serve_stop.sh --list       # 看 registry
```

### 4.4 从 gpublaze 搬模型 → 起池 → 评测（一条命令，**在 gpublaze 上跑**）

```bash
cd /srv/home/bohanlyu/innovation_proior/training/FrontierSmith
bash scripts/jiaolab/eval_model_from_gpublaze.sh v2_multisetting_4b/full_wd01
bash scripts/jiaolab/eval_model_from_gpublaze.sh v2_multisetting_4b/full_wd03
bash scripts/jiaolab/eval_model_from_gpublaze.sh agentic_ablation_4b/soup_withag_a10
# 位置参数： <MODEL_DIR|相对 MODEL_ROOT 的名字> [TAG] [KIND=fcsale|ale|fcs]
# 默认 TAG = ja_<父目录名>_<目录名>，例如 ja_v2_multisetting_4b_full_wd01
```

它做三件事：① `nice -n 19 ionice -c3 rsync` **只搬 serving 载荷**
（config/tokenizer/`*.safetensors`，排除 `checkpoint-*/`、optimizer、
`training_args.bin`、图片）——4B bf16 约 **9G**，而不是整个训练目录的 26G；
② ssh 到 jiaolab 起 `launch_pool_eval.sh`（自动挑 2 张空卡）；③ 打印日志与
输出路径后立刻返回（评测在 jiaolab 上 setsid 常驻）。

聚合仍用历史脚本：
```bash
ssh jiaolab "cd /home/bohan/innovation_proior/training/FrontierSmith && .venv/bin/python scripts/reaggregate_all_summary.py"
```

## 5. 本机锚点（**所有对照的基准**）

跨节点分数不可同表（§0 的 0.79 vs 0.43 标定），因此 jiaolab 上**必须**有自己的
base 锚点：

- TAG **`ja_q35_4b_base`**，模型 `Qwen/Qwen3.5-4B`（HF 缓存，offline），
  KIND=`fcsale`（FCS 188 题 + ALE 40 题），2 卡 TP=1 池，2 shard。
- 输出：`outputs/cc_eval_ja_q35_4b_base_thinking_32k_both_vllm/shard_{0,1}/`
- 日志：`logs/pool_launch_ja_q35_4b_base.log`、
  `logs/serve_pool_ja_q35_4b_base_gpu{3,4}.log`、
  `logs/cli_ja_q35_4b_base_both{0,1}_pool.log`

启动命令（2026-08-25 已启动）：

```bash
cd /home/bohan/innovation_proior/training/FrontierSmith
setsid nohup bash scripts/jiaolab/launch_pool_eval.sh Qwen/Qwen3.5-4B ja_q35_4b_base fcsale 3 4 \
  > logs/pool_launch_ja_q35_4b_base.log 2>&1 &
```

**client 挂了怎么续**（引擎还在时，不要重起引擎，只补 client；`RESUME=1` 是默认，
已完成的样本不会重跑）：

```bash
cd /home/bohan/innovation_proior/training/FrontierSmith
for i in 0 1; do
  P=$(ls .cache/vllm_pool/ja_q35_4b_base__*.json | sed -n "$((i+1))p" | grep -oE "[0-9]+\.json" | tr -d '.json')
  TAG=ja_q35_4b_base MODEL_TAG=ja_q35_4b_base SOURCE=both NUM_SHARDS=2 SHARD_IDX=$i \
    REQUEST_TIMEOUT=7200 CONCURRENCY=64 EVAL_SCORE_CONCURRENCY=4 \
    VLLM_BASE_URL=http://127.0.0.1:$P/v1 setsid nohup bash scripts/jiaolab/eval_client_local.sh \
    > logs/cli_ja_q35_4b_base_both${i}_pool.log 2>&1 &
done
```

**jiaolab 上任何臂 / soup 的数字，只能和这个锚点比。**

## 6. 文件清单（本次新增，全部不改历史、也不改 gpublaze）

```
scripts/jiaolab/
  env_jiaolab.sh                        # 公共 env + druv 共卡守卫 + fs_pick_free_gpus
  serve_local.sh / serve_stop.sh        # GPU serve（registry 协议兼容；prefix-caching 默认关、util 0.85）
  start_frontiercs_judge_local.sh       # judge 栈（shim + 宿主 node18；auto 探针改 unshare -Ur）
  eval_client_local.sh                  # CPU client（协议同 cc_eval_cpu_client.sh；ALE=apptainer）
  eval_split_local.sh                   # 单卡一键 serve+K clients
  launch_pool_eval.sh                   # 2 卡 TP=1 池 + 钉死 client（生产用法）
  eval_model_from_gpublaze.sh           # 在 gpublaze 上跑：搬模型 → 起池 → 评测
  apply_vllm_penalty_fastpath_jiaolab.sh
  ale_apptainer_selftest.py             # ALE apptainer 后端验收套件（compile/fault/concurrency/compare/report）
  pysite/sitecustomize.py               # 按 ALE_BENCH_CONTAINER_BACKEND 装后端（host 直接拒绝）
  pysite/ale_apptainer_backend.py       # 急切启动 + 1 核绑定的 apptainer 沙箱
docs/EVAL_ON_JIAOLAB_zh.md              # 本文
```

运行时资产（gitignored）：`.cache/external/Frontier-CS/algorithmic/{problems,solutions}`、
`.cache/bin/go-judge`、`.cache/ale-bench/rust-tool-builds`、`.cache/jiaolab/judge_app`、
`.venv-jiaolab`（`.venv` symlink）、`/home/bohan/venv-vllm-jiaolab`、
`/home/bohan/sif/*.sif`、`data/alebench/local_data`。

## 7. 待办 / 已知限制

- **真 go-judge 需要 sudo**（AppArmor `apparmor_restrict_unprivileged_userns`
  或给 go-judge 一个带 `userns` 权限的 AppArmor profile）。在此之前 FCS 判分
  = shim。已双向验收到 64.05，但与 gpublaze 的真 go-judge **不是同一 backend**，
  这也是锚点必须留在本机的原因之一。
- **ALE 沙箱无网络隔离**（非特权 apptainer 限制，与 Princeton shim 同）。
- **research 64 题**：数据未搬，本机不跑。
- **MLS-Bench**：conda env 体量 + 磁盘余量，留在 gpublaze。
- **`/` 93% 满**：每评一个模型 +9G。评完的模型目录可以删
  （`/home/bohan/models_from_gpublaze/<name>`），outputs 很小。
