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
- **`/` 只剩 ~360G（95% 满）**。大下载前先算空间。（2026-08-25 MLS-Bench 移植
  又吃掉 ~30G：harness+vendor 2.3G + 10 个包 env + driver env 28G，见 §2.4.4。
  `vendor/workspace` 还会随每轮 MLS run 增长 ~1.7G，要定期清。）
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
| `sbatch cc_eval_mlsbench_cpu_ailab.sh` | `scripts/gpublaze/eval_mlsbench_local.sh` | **`GPUS=2 MODEL_PATH=.. TAG=.. bash scripts/jiaolab/eval_mlsbench_local.sh`**（parser 换 `qwen3_xml`，见 §2.4.1） |
| （MLS N 轮基线） | `scripts/gpublaze/run_4b_base_mls_research_baseline.sh` | `scripts/jiaolab/run_mls_cpu_baseline.sh`（只做 MLS 一半；research 走 `eval_client_local.sh`） |
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

### 2.4 MLS-Bench CPU（22 题）✅ 已移植，冒烟通过（2026-08-25）

**结论先行**：本机现在可以跑 22 题 MLS-Bench CPU eval，协议与 gpublaze 完全一致，
只有两处必要的机器差异：**tool-call parser 必须是 `qwen3_xml`**，
以及 GPU 守卫/显存留白走本机口径。

```bash
# 单卡自起 vLLM（守卫会拒绝空闲 <70G 的卡；绝不碰 druv 的卡）
GPUS=2 MODEL_PATH=<hf_dir> TAG=<tag> bash scripts/jiaolab/eval_mlsbench_local.sh

# 挂到已有 engine（不占卡，TAG 必须等于那个 server 的 --served-model-name）
EXTERNAL_VLLM_URL=http://127.0.0.1:8006/v1 TAG=<served> \
  bash scripts/jiaolab/eval_mlsbench_local.sh

# 发表口径：N>=3 轮 + 固定分母 22 的聚合
GPUS=2 MODEL_PATH=<hf_dir> SERVED=<tag> N_ROLLOUTS=3 \
  bash scripts/jiaolab/run_mls_cpu_baseline.sh
```

#### 2.4.1 `--tool-call-parser qwen3_xml`（**本机与 gpublaze 唯一的协议级差异，不可省**）

MLS agent 全靠 tool call 驱动（edit / test / view / undo）。gpublaze 的
`eval_mlsbench_local.sh` 传的是 `hermes`；**在 Qwen3.5 上 `hermes` 会接受请求
但一个 tool call 都解析不出来**——HTTP 200、`finish_reason=stop`、
`tool_calls=[]`，XML 原样留在 `content` 里，于是 agent 一步都走不了，
**3 分钟内静默拿到 0/22，日志全绿**。本机实测（同一个 Qwen3.5-4B，同一个探针）：

```
# --tool-call-parser hermes
finish_reason= stop
tool_calls= []
content[:400]= 'The user is asking me to run a first experiment ... \n</think>\n\n
               <tool_call>\n<function=test>\n</function>\n</tool_call>'

# --tool-call-parser qwen3_xml
[tools-gate] parsed tool_calls: [{"id": "chatcmpl-tool-9da64054b19bd3ff",
  "type": "function", "function": {"name": "test", "arguments": "{}"}}]
```

所以 `scripts/jiaolab/eval_mlsbench_local.sh`：
- 默认 `TOOL_CALL_PARSER=qwen3_xml`（可用同名 env 覆盖）；
- **起完 vLLM 之后必须过 tool-call 解析闸门 `mls_tools_ok`**：temperature 0
  的确定性探针，要求返回**非空的已解析 `tool_calls` 数组**——只回 200 不算数。
  自己起的 server 解析不出来就直接 abort（探针结果写到
  `$OUTPUT_BASE/tool_call_probe.txt`）；`EXTERNAL_VLLM_URL` 模式下改成轮询等待
  （`WAIT_TOOLS_SEC`，默认 1800s），因为别人的 engine 可能还在热身，
  **绝不去动别人的 server**。

#### 2.4.2 harness：**必须是打过补丁的那棵树**

`.cache/mlsbench-eval` = fresh clone @805adf733 + FrontierSmith 补丁层
（含 view+str_replace edit contract；rewrite 自 2026-08-26 起默认关闭，只作 A/B 臂）。**重新 clone 一份 MLS-Bench 不等价**，
分数不是一个口径。本机是从 gpublaze **rsync** 过来的同一棵树：

```
$ grep -c VIEW_SCHEMA .cache/mlsbench-eval/src/mlsbench/agent/tools.py
1
$ git -C .cache/mlsbench-eval rev-parse --short HEAD
2861229a4          # 与 gpublaze 一致，branch fix/oracle-leakage-pilot
```

`eval_mlsbench_local.sh` 没有这两样就 hard-fail（`VIEW_SCHEMA` 与 `--use-replace`），
`mlsbench_preflight.sh` 会更早地拦住。

搬过来时**排除**了 `logs/`、`paper_assets/`、`.scheduler*`、`.saves/`
（都是 gpublaze 的历史产物，与评测无关），实传 903MB。

#### 2.4.3 `vendor/` 是真目录，不是 symlink

gpublaze 上 `vendor/{data,external_packages,workspace}` 是指向
`/srv/home/bohanlyu/MLS-Bench/` dev checkout 的 symlink（data 508G、
external_packages 12G、workspace 67G）。本机没有那个 checkout，所以这三个做成
**真目录**，只装 22 题 CPU 集真正用到的东西：

| 目录 | 内容 | 大小 |
|---|---|---|
| `vendor/external_packages/` | 10 个包：`badge causal-bnlearn causal-learn deap eplb gplearn naslib scaling-law-lab scikit-learn SMPyBandits` | 1.3G |
| `vendor/data/` | `sklearn`(134M) `badge`(4.9M) `scaling_law`(680K) `adbench`(568K) —— 就是这 10 个包 config 里出现过的全部 data 引用 | 139M |
| `vendor/workspace/` | 空目录；每次 run 的产物落在这里（`<task>/vllm_<tag>_<ts>/`） | 随 run 增长 |

**`vendor/workspace` 会长**：一次 run 会把包整个拷进 run 目录，naslib 单次 ~562M、
SMPyBandits ~470M，一轮 22 题大约 +1.7G。盘只剩 ~360G（95%），**定期清**。

#### 2.4.4 conda envs：`container_runtime: local` 的运行时

`container_runtime: local` 表示每个包在自己的 `mlsbench-<pkg>` conda env 里跑
（docker 在本机要 sudo，用不了）。22 题只用到 **10 个 env**（不是 gpublaze 那 44 个），
从 gpublaze **rsync + 前缀重定位**过来：`/srv/home/bohanlyu/miniconda3` →
`/home/bohan/miniconda3`（文本文件直接替换，二进制文件替换后按原长度补 NUL，
即 conda-unpack 的做法；新前缀更短所以安全）。重定位脚本进了版本库：
`scripts/jiaolab/relocate_conda_prefix.py <旧前缀> <新前缀> <env 目录>`。

> 为什么不在本机重新 `pip install` 一套：install_cmds 大多没锁版本，重装出来的
> 是**另一个包集**，也就是另一个打分口径。搬过来才是同一口径。

driver（跑 `python -m mlsbench agent/score` 的那个解释器）另起一个
`mlsbench-driver` env（python 3.13.15），版本对齐 gpublaze conda base：
`numpy 2.4.4 / scipy 1.17.1 / pandas 2.3.3 / openai 2.24.0 / PyYAML 6.0.3 /
packaging 25.0 / huggingface_hub 1.10.1 / networkx 3.6.1 / tqdm 4.67.3 /
scikit-learn 1.8.0 / statsmodels 0.14.6 / pydot 4.0.1 / momentchi2 0.1.8 /
matplotlib 3.10.9`。

**故意不装 `causal-learn`**：gpublaze 的 driver base 里也没有，所以
`causal-observational-linear-gaussian` 的 `parser.py` 在那边 import 失败、
该题记 `agent_failed`。这边装上就会多得一题分——**那是另一个口径**。

**坑（静默）**：MLS-Bench 的 `_has_conda_support()` 找不到 conda 就会
**不报错地**退化成 `PIP_TARGET` site-packages 模式，跑出来的是完全不同的运行时。
本机 `conda` 不在非登录 shell 的 PATH 上，所以 wrapper 里显式
`export CONDA_EXE=/home/bohan/miniconda3/condabin/conda` 并把 `condabin` 塞进
PATH，preflight 还会断言 `wrap_with_conda()` 真的吐出 `conda run --name ...`。

磁盘占用合计：harness+vendor **2.3G** + 11 个 conda env **28G** ≈ **30G**。
（本文旧版说要 212G，那是把 gpublaze 全部 44 个 env 都算进去了；22 题 CPU 集只要 10 个。）

#### 2.4.5 preflight（**先跑这个，不占卡**）

```bash
bash scripts/jiaolab/mlsbench_preflight.sh                 # 22 题全量检查
TASKS="ml-calibration" bash scripts/jiaolab/mlsbench_preflight.sh
BUILD_LOCAL=1 bash scripts/jiaolab/mlsbench_preflight.sh   # 顺便预热 local runtime
```

检查项：harness 身份 + VIEW_SCHEMA + `--use-replace`；`vendor/*` 是真目录且非空；
driver python 解析到的是**补丁版** mlsbench（不是别的副本）+ openai；
conda-backed local runtime 真的生效；每个要用的 `mlsbench-<pkg>` env 存在且 python 能跑。
`eval_mlsbench_local.sh` 会自动先跑它（`SKIP_PREFLIGHT=1` 可跳）。

`BUILD_LOCAL=1` **必须跑一次**：MLS-Bench 的 "已构建" 指纹里含**绝对路径**
（`pkg_dir` + `data_root`），从 gpublaze 搬过来的树在本机永远算"没建过"，
于是第一次 eval 的第一题要付 `build_local_package` 的钱（在 conda env 里重跑
install_cmds）。绝大多数是 `already satisfied`，但有两处要联网：
scikit-learn 重新抓 adbench 的 4 个 npz；**scaling-law-lab 的 `prepare_data.py`
会无条件重拉 `pkuHaowei/sldbench`**（它不是幂等的，`vendor/data/scaling_law`
已经有文件也拦不住它）。所以 preflight 的 BUILD_LOCAL 段**故意**用
`HF_HUB_OFFLINE=0` 跑（`BUILD_HF_OFFLINE=1` 可强制离线）——
**这是 setup，不是 eval**：eval 本身永远离线（`eval_mlsbench_local.sh` 和
gpublaze 一样 export `HF_HUB_OFFLINE=1`）。该脚本里的 dataset revision 是**钉死**的，
本机重拉的产物与 gpublaze 搬来的文件**逐 md5 一致**（13 个文件全等），
所以口径没变。哪天本机断网，就直接预写
`vendor/images/local/<pkg>.json` 的 stamp。

2026-08-25 实跑：**10 个包全部 `[build] ... ready`**，preflight `ALL CHECKS PASSED`。

#### 2.4.6 冒烟验收（2026-08-25，GPU 2 单卡，Qwen3.5-4B base，TP=1）

3 题、`CONCURRENCY=3`，其余全默认（`max_steps 20 / max_tests 3 /
budget_tokens 10000 / reasoning_effort high / seeds [42] / MAX_MODEL_LEN 40960 /
EVAL_RESEARCHER_YEAR 2026 / --use-replace`）：

| task | jiaolab status | jiaolab score | gpublaze run1 参考 |
|---|---|---|---|
| `ml-selective-deferral` | scored | **0.3597** | 0.3894 |
| `ml-anomaly-detection` | scored | **0.3633** | 0.3633 |
| `optimization-evolution-strategy` | agent_failed+scored | **0.3478** | 0.0226 |

3 题均值 0.3569（`outputs/mls_cpu_smoke_jiaolab_qwen35-4b-base/summary.json`）。
task log 里能看到完整的 `edit / test / view / undo` 步骤，`view` 正是补丁版
edit contract 的工具，说明 harness 补丁层在本机确实生效。

**这三个数不是任何结论**：3 题 ≠ 22 题，单轮 ≠ N>=3 轮的发表口径；
而且按 §0 铁律 1，**jiaolab 的 MLS 数只能和 jiaolab 的 MLS 数同表**
（Xeon 8358 vs EPYC 9654，CPU 任务对算力敏感）。要出可比的数就跑
`run_mls_cpu_baseline.sh`，在本机自己建锚点。

还跑了两个不占卡的旁证：
- **闸门的反向验证**：挂到一个**没带** `--enable-auto-tool-choice` 起的池 engine
  上，闸门吐出服务端原文并拒绝启动（不去动别人的 server）：
  ```
  [tools-gate] no usable response ('choices'); body: {"error":{"message":
    "\"auto\" tool choice requires --enable-auto-tool-choice and
     --tool-call-parser to be set","type":"BadRequestError",...,"code":400}}
  ERROR: http://127.0.0.1:40457/v1 never returned parsed tool_calls after 1s.
    Restart that server with: --enable-auto-tool-choice --tool-call-parser qwen3_xml
  ```
- `run_mls_cpu_baseline.sh` 的 `PHASES=aggregate` 用上面这份 summary 跑通，
  固定分母 22、`agent_failed+scored` 记 0 的语义与 gpublaze chain 完全一致。

**尚未验证（明确列出，别当成跑过了）**：
- 22 题全量跑、`N_ROLLOUTS>=3` 的完整 chain；
- 除 `scikit-learn`/`deap` 之外 8 个包的**出分**（env 与
  `build_local_package` 已预热通过，`causal-bnlearn` 也真的跑起了
  `eval_hailfinder.sh`，但那次 run 为了**还卡**被我提前掐了，没有出分 ——
  见 `outputs/mls_cpu_smoke2_jiaolab_qwen35-4b-base/NOTE_INCOMPLETE.md`）；
- 本机 MLS 的**墙钟**：那次单题跑了 38 分钟还没跑完同一个 test
  （当时全机 load ~180，其他用户的 eval 在跑），gpublaze 上同题 22 并发总共
  310s。**CONCURRENCY=20 + TASK_TIMEOUT=5400 在本机大概率不够**，
  第一次跑全量前先看 load，必要时调大 `TASK_TIMEOUT`。

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

### 5.1 judge 超时残留与 RESUME 收敛（**必读，不是 bug**）

跑到中途会看到：

```
ERROR frontiercs 148 sample=1: RuntimeError('FrontierCS judge infrastructure
failure for problem 148 (status=timeout): Evaluation timed out after 1000s')
```

- 这 1000s 是**官方 runner 写死的常量**
  （`.cache/Frontier-CS-official/src/frontier_cs/runner/base.py:50`
  `DEFAULT_TIMEOUT: int = 1000`），**没有 env 开关**，不要去改它 —— 改了就和
  gpublaze 不同口径。
- **gpublaze 上同样会中招**：历史日志里 143/148/149/153/156/163/169/170 都出现过
  `status=timeout`，其中 148、153 与 jiaolab 命中的是同一批。也就是说这是 FCS
  少数超重题的固有性质，不是 jiaolab 的回归。
- jiaolab 单核只有 gpublaze 的 ~54%（§0 标定），所以**残留率会更高一些**。
  只要每个臂都用同一套收敛流程，锚点内部仍然可比。

**这条链是 fail-loud 的，不会把假零烧进结论**：
1. error 样本在 `samples.jsonl` 里带 `error` 字段（占位 reward 0）；
2. driver 最后按 `--max-errors=0` 判定，**以 rc=2 退出** ——
   `launch_pool_eval.sh` 末尾报"失败"是**预期**的，含义是"需要再跑一遍 resume"；
3. `_record_compatible()` 把带 `error` 的记录视为**未完成**，所以同 TAG 用
   `RESUME=1`（默认）重跑 client 时，**只会重生成 / 重打分这些样本**，不会重跑
   已成功的。

**收敛流程**：反复执行 §5 的「client 挂了怎么续」那段命令，直到 client 打印
`DONE rc=0`。gpublaze 的实证：用当前这套参数（`REQUEST_TIMEOUT=7200` /
`CONCURRENCY=64` / 钉死的 2 引擎池）跑的 `q35_4b_soup_withag_a20`
最终收敛到 **0 / 1 个 error**（那 1 个正是 problem 149 的 1000s 超时）；而更早用
旧参数跑的 `q35_4b_base` 最终态还剩 **330 个 error**（多为
`APITimeoutError`）——所以**出数前一定要核最终态 error 数**：

```bash
python3 - <<'EOF'
import json
for sh in (0,1):
    rec={}
    for l in open(f"outputs/cc_eval_<TAG>_thinking_32k_both_vllm/shard_{sh}/samples.jsonl"):
        r=json.loads(l); rec[(r["data_source"], str(r["ground_truth"]), str(r["sample_idx"]))]=r
    err=[k for k,v in rec.items() if v.get("error")]
    print(f"shard{sh}: keys={len(rec)} errors={len(err)}")
EOF
```
（`samples.jsonl` 是**追加**的，行数会超过 570；必须按 key 取最后一条算最终态，
直接 `grep -c error` 会严重高估。）

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
  eval_mlsbench_local.sh                # MLS-Bench CPU 22 题（parser=qwen3_xml + tool-call 解析闸门；协议同 gpublaze）
  mlsbench_preflight.sh                 # MLS 开跑前的全部非 GPU 检查；BUILD_LOCAL=1 顺便预热 conda 运行时
  run_mls_cpu_baseline.sh               # MLS N 轮 rollout + 固定分母 22 聚合（gpublaze chain 的 MLS 一半）
  relocate_conda_prefix.py              # conda env 跨机前缀重定位（conda-unpack 的等价物）
  pysite/sitecustomize.py               # 按 ALE_BENCH_CONTAINER_BACKEND 装后端（host 直接拒绝）
  pysite/ale_apptainer_backend.py       # 急切启动 + 1 核绑定的 apptainer 沙箱
docs/EVAL_ON_JIAOLAB_zh.md              # 本文
```

运行时资产（gitignored）：`.cache/external/Frontier-CS/algorithmic/{problems,solutions}`、
`.cache/bin/go-judge`、`.cache/ale-bench/rust-tool-builds`、`.cache/jiaolab/judge_app`、
`.venv-jiaolab`（`.venv` symlink）、`/home/bohan/venv-vllm-jiaolab`、
`/home/bohan/sif/*.sif`、`data/alebench/local_data`、
**`.cache/mlsbench-eval`（补丁版 MLS harness + 真 `vendor/`，2.3G）**、
**`/home/bohan/miniconda3/envs/mlsbench-{driver,<10 个包>}`（28G）**、
**`/home/bohan/miniconda3/envs/mlsbench-driver`（MLS driver 解释器）**。

## 7. 待办 / 已知限制

- **真 go-judge 需要 sudo**（AppArmor `apparmor_restrict_unprivileged_userns`
  或给 go-judge 一个带 `userns` 权限的 AppArmor profile）。在此之前 FCS 判分
  = shim。已双向验收到 64.05，但与 gpublaze 的真 go-judge **不是同一 backend**，
  这也是锚点必须留在本机的原因之一。
- **ALE 沙箱无网络隔离**（非特权 apptainer 限制，与 Princeton shim 同）。
- **research 64 题**：数据未搬，本机不跑。
- **MLS-Bench**：已移植并冒烟通过（§2.4）。**仍未验证**：22 题全量、
  `N_ROLLOUTS>=3` 的完整 chain、`scikit-learn`/`deap` 之外 8 个包的出分、
  以及本机 MLS 的真实墙钟（单题实测 >38 min 未完，默认
  `TASK_TIMEOUT=5400` 很可能不够）。本机还没有 MLS 锚点 —— 出数前先用
  `run_mls_cpu_baseline.sh` 建一个。
- **MLS 的 `--tool-call-parser` 绝不能退回 `hermes`**：在 Qwen3.5 上它 200 但
  解析不出 tool call，静默 0/22（§2.4.1 有两边实测输出）。
  `eval_mlsbench_local.sh` 的闸门会拦住，但如果有人绕过 wrapper 手起 server，
  就没有人拦了。
- **`/` 93% 满**：每评一个模型 +9G。评完的模型目录可以删
  （`/home/bohan/models_from_gpublaze/<name>`），outputs 很小。
