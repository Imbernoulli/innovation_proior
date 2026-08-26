# FrontierSmith 评测栈 @ gpublaze 运行手册

2026-08-23 建立。本机:8×H100-80G、384 核、1.5T RAM,无 slurm、无 Apptainer,
**rootless Docker 29.6.1(NVIDIA CDI 已配置)**。所有 Princeton 历史脚本
(`slurm/`、`scripts/`)保持原样;本机差异全部收敛在 **`scripts/gpublaze/`
wrapper 层**(见「复现提醒」:env var 覆盖,不改历史脚本语义)。

## 0. 机器纪律(先读)

- **GPU 6,7 属于其他用户(zzh),任何时候不许调度**。wrapper 层的
  `fs_guard_gpus` 会拒绝 `GPUS` 里出现 6/7(除非显式 `FS_ALLOW_GPU67=1`)。
- GPU 0-5 常被本用户其他任务(eval_one.py / LF-innov)占用,起 serve 前先
  `nvidia-smi` 确认空闲,只用真正空闲的卡。
- `/srv` 99% 满(~850G 可用),`/`(含 /tmp)有 1.3T。大下载(模型权重、
  大镜像)前先算空间。当前本次 setup 新增占用:官方 Frontier-CS clone 3.4G +
  数据集 ~1G、docker 镜像 ~4G(ale-bench + rust)、venv ~2.5G、upstream verl
  clone ~0.4G。
- 本机上判出来的 FCS/ALE 分数与 ailab/pli 历史分数**不可同表比较**
  (EVAL_ROBUSTNESS_zh.md 铁律 1:节点代际会翻转时限边缘题)。每个 client 都会
  在输出目录写 `judge_node_meta.json`(含 gojudge_shim_v2 的节点速度标定),
  跨机对比前先查它。

## 1. 一图流:本机的 serve/eval 分离

原架构(slurm):GPU job 只 serve(`cc_serve_only.sh`),CPU job 拉起
judge+打分(`cc_eval_cpu_client*.sh`),经共享 GPFS registry 发现彼此。
本机等价物(全部在 `scripts/gpublaze/`):

| Princeton | gpublaze wrapper |
|---|---|
| `sbatch cc_serve_only.sh` | `GPUS=4,5 DAEMON=1 bash scripts/gpublaze/serve_local.sh <MODEL> <TAG>` |
| `sbatch cc_eval_cpu_client.sh` | `TAG=.. SOURCE=.. bash scripts/gpublaze/eval_client_local.sh` |
| `cc_eval_split_submit.sh` | `GPUS=4,5 bash scripts/gpublaze/eval_split_local.sh <MODEL> [TAG] [KIND] [SHARDS]` |
| GPFS registry | `$FS/.cache/vllm_pool/`(协议逐字节兼容,`vllm_pool_pick.py` 原样复用) |
| Apptainer judge SIF | 宿主机直跑 node server(`start_frontiercs_judge_local.sh`) |
| go-judge / shim | **真 go-judge 二进制**(本机 userns 可用,`.cache/bin/go-judge` v1.11.1);`GJ_BACKEND=shim` 回退 gojudge_shim_v2 |
| `.venv`(client)| `.venv-gpublaze`(cpu-torch + verl editable + ALE-Bench editable;`.venv` 是指向它的 symlink,历史脚本硬编码的 `.venv` 因此直接可用) |
| `.venv-vllm023`(serve)| 复用本机现成 vllm 环境 `/srv/home/bohanlyu/sesl/.venv`(vllm 0.21.0, torch 2.11 cu128),经 `VLLM_VENV` 注入 `start_vllm_server.sh` |

公共环境统一在 `scripts/gpublaze/env_gpublaze.sh`(所有 wrapper source 它)。

## 2. 逐评测项运行方法与状态

### 2.1 FrontierCS algorithm(FCS,C++ judge)✅ 已验证

资产:官方 repo 已 clone 到 `.cache/external/Frontier-CS`(≈3.4G,
`.cache/Frontier-CS-official` 是它的 symlink);
`Frontier-CS/algorithmic/problems` symlink 到官方 clone 的 problems
(**188 个 numeric 题**;历史口径是 172 题——官方 repo 后来加了题,若要对齐
历史口径需要从 Princeton 拿当时的题单,见 §5 决策项)。
`data/frontiercs/full.parquet` 已由 `prepare_frontiercs_parquet.py --full-for-both` 生成(188 行)。

> **口径声明(2026-08-23 协调者裁定)**:本机 FCS-alg 口径为 **188 题**,与
> ailab 历史数(172 题)**不可同表**;本轮 agentic 消融是本机自含 A/B
> (两臂 + base 同 judge 同题单),内部可比即可。

冒烟已通过(2026-08-23):
- judge 栈:`start_frontiercs_judge_local.sh` 起真 go-judge v1.11.1 + host node
  v25 跑 `server.js`(stage 目录 `.cache/gpublaze/judge_app`,首次自动 `npm ci`);
- 双向打分:空程序 → 真实 Wrong Answer 0 分(checker 输出在案),官方
  leaderboard 解 `solutions/1/deepseekreasoner.cpp` → **64.05 分**。

**常驻 judge 服务(2026-08-23 起已在跑)**:8082 端口起了一个长期 judge
(node API :8082 + 真 go-judge :5050,`logs/judge_8082.log`),外部 driver
(如 `experiments/scripts/eval/eval_qwen35_base_vllm_request.py`)直接用默认
`FRONTIERCS_JUDGE_URL=http://127.0.0.1:8082` 即可;重启命令:

```bash
cd training/FrontierSmith
PORT=8082 GJ_PORT=5050 GJ_BACKEND=gojudge GJ_PARALLELISM=16 JUDGE_WORKERS=16 \
RUNTIME_DIR=$PWD/.cache/frontiercs-judge/8082 \
  setsid nohup bash scripts/gpublaze/start_frontiercs_judge_local.sh >> logs/judge_8082.log 2>&1 &
```

跑法(整条 FCS+ALE):

```bash
cd training/FrontierSmith
GPUS=4,5 bash scripts/gpublaze/eval_split_local.sh <MODEL_DIR_or_HF_id> <TAG> fcsale 2
# 输出: outputs/cc_eval_<TAG>_thinking_32k_both_vllm/shard_*/samples.jsonl + summary_shard.json
# 聚合仍用历史脚本: .venv/bin/python scripts/reaggregate_all_summary.py (或 cc_eval_agg_shards 的逻辑)
```

采样协议与 Princeton 完全一致(32k thinking、presence 1.5、n=5、y26
conditioning),由 `eval_client_local.sh` 强制(MAX_TOKENS<32768 直接报错)。

### 2.2 ALE-Bench(40 题)✅ 基础设施已验证

**Docker 是 ALE harness 的原生后端**(`ALE_BENCH_CONTAINER_BACKEND` 默认
docker;Apptainer 反而是集群补丁),本机反而更忠实。已就位:

- 镜像:`yimjk/ale-bench:cpp17-202301`(2.7G,已 pull)、
  `ale-bench:cpp17-202301`(thin 层,`--build-arg UID=0 GID=0` 已 build)、
  `rust:1.79.0-buster`(1.3G,已 pull)。
- 数据:`data/alebench/local_data`(40 题 zip,95M,来自 HF
  `SakanaAI/ALE-Bench`)、`val.parquet`(10)、`full40.parquet`(40,经
  `scripts/gpublaze/build_alebench_full40_gpublaze.py` 生成——历史脚本硬编码
  /scratch 路径,这是它的 shim)。
- Rust 工具缓存:`.cache/ale-bench/rust-tool-builds` **40/40 已 build**
  (`scripts/gpublaze/prepare_alebench_tool_cache_docker.py`,历史版硬编码
  apptainer 后端,docker 版是逐字节等价拷贝)。
- **rootless 坑与解法**:harness 以 `user=os.getuid()` 起容器,rootless 下该
  uid 映射为无关 subuid → `/workdir` 与 rw bind 文件不可写
  (`ld: cannot open output file a.out: Permission denied`)。解法在 wrapper 层:
  `scripts/gpublaze/pysite/sitecustomize.py` 在
  `ALE_BENCH_DOCKER_ROOT_USER=1` 时把 ALE 相关镜像的容器用户改为 root
  (rootless 下容器 root == 宿主本用户,语义与原设计一致)。
  `eval_client_local.sh` 已自动启用。
- 冒烟已通过:`ale_compile_selftest`(编译自检)+ `alebench.compute_score`
  对 ahc008 完整链(输入生成 → 编译 → 批量评测 → performance/rank)。

跑法:`GPUS=4,5 bash scripts/gpublaze/eval_split_local.sh <MODEL> <TAG> ale 1`
(或 KIND=both 与 FCS 同跑)。

注意:历史 driver 预检要求 `ALE_BENCH_APPTAINER_DIR` 路径**存在**(不使用),
wrapper 会 `mkdir -p` 一个空目录满足它。

### 2.3 FrontierCS research(64 题:43 CPU + 21 GPU)✅ CPU 侧机制已验证 / ⚠️ 差部分数据 + GPU overlay

机制已验证(2026-08-23):`frontiercs_research_cpu_eval.evaluate_cpu_research_solution`
在本机 venv 下正常 spawn 官方 evaluator.py(sky_spot 等 vendored 依赖可import),
语法错解 → model-side 0,基础设施错 → 显式 status/message(无静默零分)。

- parquet 已生成:`data/frontiercs/research.parquet`(64)/
  `research_cpu.parquet`(43)/ `research_gpu.parquet`(21)。
- evaluator 解释器:`FRONTIERCS_RESEARCH_PYTHON` 默认指向 `.venv-gpublaze`
  (含 faiss-cpu、coverage、torch-cpu、numpy、openai)。
- 数据集状态(clone 不带大文件,各题 `download_datasets.sh` 已全部跑过):
  - ✅ vdb_pareto(SIFT1M 已下)、llm_router、llm_sql、cloudcast、
    symbolic_regression、nbody_simulation、grammar_fuzzing、imagenet_pareto(无需下载)
  - ❌ **cant_be_late / cant_be_late_multi(20 题)**:需要
    `resources/real_traces.tar.gz`,不在 git、脚本只会解包不下载 →
    需从 Princeton `/scratch/gpfs/CHIJ/bohan/fs/...` 拷贝(见 §5)。
- ⚠️ **symbolic_regression 还需 pysr + juliacall + Julia depot**(未装,重且
  首跑要 JIT 编译分钟级;这 3 题本来就是 lottery 题)。装法:venv 内
  `pip install pysr`,首次 import 自动装 Julia 依赖,depot 用
  `JULIA_DEPOT_PATH` 固定到 `.cache/julia_depot`。
- ❌ **GPU 21 题(Triton)**:需要 Princeton 的 research_overlay 环境
  (torch 2.10 + triton 3.6 + flashinfer + CUDA)。本机 sesl venv 的
  torch2.11/triton 未验证兼容(官方还是 triton-tlx 定制 build);重建 overlay
  是独立工作项(§5)。`eval_client_local.sh` 因此默认
  `RESEARCH_DATA=data/frontiercs/research_cpu.parquet`(43 CPU 题);
  GPU 题的正确入口是 `frontiercs_research_gpu_rescore.py`(有 preflight +
  control gate,不会输出假零分)。

**首轮 4B base 实跑暴露的三个 evaluator 坑(已修,2026-08-23)**:
1. cloudcast:evaluator 需要 `colorama`(已装进 `.venv-gpublaze`);
2. symbolic_regression:Julia/PySR 启动即预留巨量地址空间,任何 RLIMIT_AS 都
   让它 rc=-6(SIGABRT)→ 25 个样本假 infra 零分。`eval_client_local.sh` 的
   research 分支现默认 `FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB=0`(本箱 research
   eval 独跑、1.5T RAM,该护栏本是 RL 共存场景的;RL 训练侧保留 64G 默认);
3. vdb_pareto:faiss `DatasetSIFT1M` 读 CWD 相对 `data/sift1M/*`,官方 clone
   的下载脚本却落在 `research/datasets/vdb_design/` → 已在 5 个 leaf 下建
   `data/sift1M/` symlink(`_mirror_leaf` 会把它们带进私有 workdir,正是
   Princeton 布局)。
   有 error 行的 samples.jsonl 修复流程:备份 → 删 error 行 → 同 TAG
   `RESUME=1` 重跑 client 自动补齐。

**Runnable 子集口径(2026-08-23 协调者裁定)**:缺 cant_be_late 数据的 20 题
跳过,按可跑子集出数并注明分母。`data/frontiercs/research_cpu_runnable.parquet`
= 23 题(cloudcast 1 / grammar_fuzzing 2 / imagenet_pareto 5 / llm_router 1 /
llm_sql 2 / nbody_simulation 2 / symbolic_regression 5 / vdb_pareto 5);pysr/Julia
不可用时再去掉 symbolic_regression 5 题(`research_cpu_runnable_nosr.parquet`,
18 题)。基线链会在输出目录写 `DENOMINATOR_NOTE.json` 注明实际分母。

跑法(CPU 43 题):

```bash
GPUS=4 bash scripts/gpublaze/eval_split_local.sh <MODEL> <TAG> research 2
# lottery 题 median-of-3(推荐口径):
FRONTIERCS_RESEARCH_SCORE_REPS=3 \
FRONTIERCS_RESEARCH_REPS_ONLY=$PWD/data/research_scoring_variance.json \
GPUS=4 bash scripts/gpublaze/eval_split_local.sh <MODEL> <TAG> research 2
```

### 2.4 MLS-Bench(22 CPU 任务)

> **用户裁定(2026-08-23):`/srv/home/bohanlyu/MLS-Bench` dev checkout 不是
> 合法评测 harness——它没有 view+str_replace 编辑工具契约**(其
> tools.py grep 不到 view)。正确 harness = fresh `Imbernoulli/MLS-Bench-dev`
> clone + 本仓库补丁层,已建在 **`.cache/mlsbench-eval`**:
> 基底 `fix/oracle-leakage-pilot@805adf733`(含 132 个任务加固 commit),
> 依次应用 `scripts/mlsbench_harness_fix.diff`(3-way;models.py hunk 上游已
> 含)→ `scripts/mlsbench_edit_contract.diff`(在该序下干净应用)→
> `scripts/mlsbench_score_json_fix.diff`,本地 commit 2861229a4 留档。
> 默认态已核:宽容 matcher(`MLSBENCH_STRICT_STR_REPLACE` 未设)+ view 工具开
> (`MLSBENCH_VIEW_TOOL` 默认 1)= Princeton 最终协议。**`MLSBENCH_REWRITE_OP` 默认 0**
> (2026-08-26 用户裁定:rewrite 是 FrontierSmith 自家 patch 的实验臂,团队共用的
> MLS-Bench-dev 没有;SFT 语料 v2 已按无 rewrite 契约重建,eval/RL 默认同步关闭,
> 本地 checkout commit 见 `.cache/mlsbench-eval` git log)。
> `apply_mlsbench_cpu_fixes.sh` 甄别:**C 基底已含**(HEAD 即 issue #83 官方
> 版)、**D 在 wrapper env 已有**(BLAS 钉 1)、**A 已手工补**(两个
> budget_check.py 不再 import mid_edit)、**B(fairness sidecar)为 apptainer
> SIF 特有**——local runtime 下 aif360 0.6.1 已在 conda env、compas/lsac/openml
> 数据在 vendor/data/sklearn,判断为已被 dev 分支正式化,重跑日志复核。
> 大资产不复制:`vendor/{data,workspace,external_packages}` symlink 自 dev
> checkout(数据非 harness);conda env 全局共享;`.scheduler` 等运行态全新。
> wrapper 默认 `MLSBENCH_ROOT=.cache/mlsbench-eval` 并带守卫(root 无 view
> 契约直接拒跑)。**此后所有臂/soup 的 MLS 评测(和 RL 的 mlsbench_agent 行)
> 都必须走这个 checkout**;research 不受影响(不走 MLS harness)。

wrapper:

```bash
GPUS=4 MODEL_PATH=<MODEL> TAG=<TAG> bash scripts/gpublaze/eval_mlsbench_local.sh
# 冒烟: 加 SMOKE_TASK=scikit-learn-cpu-... 或 LIMIT=2
```

相对历史脚本的差异(其余语义一致,LAPACK 线程钉 1、协议 env 全保留):
- `MLSBENCH_ROOT=/srv/home/bohanlyu/MLS-Bench`、
  `MLSBENCH_DATA_ROOT=$root/vendor/data`;
- 生成 config 的 `container_runtime: local`(conda envs;
  `MLSBENCH_CONTAINER_RUNTIME=docker` 可切)。**root/runtime 选择 = 评分口径**,
  每次运行会写 `mlsbench_provenance.json`(root、branch、commit、runtime),
  不同口径的数字不要同表。
- `MLSBENCH_PY` 默认 `/srv/home/bohanlyu/miniconda3/bin/python`
  (系统 python3 没有 openai,conda base 验证过 `import mlsbench, openai, yaml`)。

已验证:mlsbench 包 import、22 任务与 conda env 存在、headless agent init
(经 RL parquet build,见 §3.4);**完整 episode 链已通**(2026-08-23):
`EXTERNAL_VLLM_URL` 模式 + 脚本化 mock agent(test→submit)在
ml-clustering-algorithm 上走完 config→agent→conda 真实验→score→summary,
30s,mean_score 0.3875(模板基线分,非静默零分)。

**外接已有 vLLM**(不自起服务、不占 GPU):

```bash
EXTERNAL_VLLM_URL=http://127.0.0.1:8006/v1 TAG=<served-model-name> \
  bash scripts/gpublaze/eval_mlsbench_local.sh
```

**大坑:MLS agent 需要服务端 tool-call parser。** vLLM 不带
`--enable-auto-tool-choice --tool-call-parser hermes` 时,带 `tools` 的请求直接
400(`tool_choice=required` 与默认 auto 都被拒),episode 全灭且形似 0/22。
该 parser 只解析 `<tool_call>` 块,**与 reasoning parser 无关,不影响原始
think 文本口径**。FCS/ALE/research 生成不带 tools,不受影响。

**4B base 基线链(2026-08-23 已挂起等待)**:
`scripts/gpublaze/run_4b_base_mls_research_baseline.sh` 以 nohup 常驻
(`logs/base4b_mls_research_chain.log`),每 60s 探测 8006 的 tools 支持,
服务端带 parser 重启后自动依次执行:MLS 22 任务 ×3 rollouts
(`outputs/mls_cpu_base_qwen35-4b-base/run{1,2,3}` + `aggregate.json`,固定分母
22)→ research runnable 子集(见 §2.3)。

### 2.5 ThetaEvolve / TTT ❌ 缺外部仓库(阻塞)

`cc_eval_theta_openevolve_ailab.sh` 硬编码
`THETA_ROOT=/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve`(无 env 覆盖),需要其
`openevolve_adapted/`(fork 特有 `openevolve.modular_utils.file_io_controller`、
`examples/<task>/{initial_programs,evaluators,configs}` 布局、
`config_*_it_XL.yaml`)。本机三个 openevolve checkout
(sesl / exp2 / MLS-Bench fork)布局全不匹配、都没有该模块。
`cc_eval_ttt_ahc_cpu.sh` 同理需要 `/scratch/.../TTT-Discover`(evaluate_ahc_released.py
+ AHC 测例缓存)。**必须从 Princeton 拷贝这两个仓库**,拷来后 wrapper 化是小活
(该脚本本身 CPU-only)。`prepare_thetaevolve_ttt_parquet.py` 是唯一可直接跑的
(纯本地,已不阻塞)。

### 2.6 NatureBench ❌ 缺资产 / ❓ 建议改走官方 Docker 路线

`naturebench/harness/` 代码本身无 /scratch 硬编码,但缺:base 镜像
(9.1G,della 上是 apptainer build 的)、HF 任务数据(12 任务子集 0.5G,全量
1.25T)、eval conda env(~6G)、per-task overlays。本机有 docker,**建议直接用
`naturebench/repo/` 的官方 Docker 路径**(`scripts/ensure_naturebench_base.sh`
build `naturebench-base:v3`),整个 apptainer overlay 方案可以不搬。另注意
SUBSET.md 结论:该子集当前对 9B checkpoint 排序几乎无分辨力,优先级最低。

### 2.7 GPU serve 冒烟 ⏳(见 §4 最新状态)

`serve_local.sh` 已在 GPU 空闲窗口验证到「vLLM 启动+权重加载」阶段;完整
serve→registry→client 链已用 **mock backend** 全链路打通(见 §4)。本机现成
可 serve 的模型:HF cache 里的 `Qwen/Qwen3.6-27B`、`Qwen/Qwen3.8-27B`
(各 52G,TP=2 两张卡)。**Qwen3.5-9B 权重不在本机**(tokenizer/config 已拉,
权重 ~18G 待决策,见 §5)。

## 3. RL 训练 reward 栈(题目 verification)

入口 `slurm/cc_rl_multisource.sh` 的三路 reward,全部在本机完成冒烟或定位:

### 3.1 frontiersmith_synth reward ✅ 双向验证

- 沙箱:本机 `unprivileged_userns_clone=1`、**bwrap 原生可用**(`bwrap
  --ro-bind / / true` 通过)——Princeton 后期的 Apptainer surrogate 在本机
  不需要,原生 bwrap 路径即历史语义。
- 语料:`FRONTIERSMITH_SYNTH_ROOT=/srv/home/bohanlyu/innovation_proior/frontiersmith_synth`
  (reward 模块默认路径按 Princeton 目录层级推导,本机必须显式 export)。
- 冒烟(FAIL_SOFT=0 fail-loud,双向):fsx_A_0082
  strong.py→63.75 / greedy.py→44.70 / invalid.py→0;routing 检查里 fsx_G_0455
  strong→49.15 / trivial→10.0。**无静默零分**。

### 3.2 FrontierCS algorithm reward ✅(与 §2.1 同一 judge 栈)

真 go-judge 二进制直跑(userns OK);shim(gojudge_shim.py / _v2,Princeton
认证 94.3% byte-exact)都在 `scripts/` 里已随 PR 带来,`GJ_BACKEND=shim` 可切。
双向验证见 §2.1。RL 侧 reward(`verl/utils/reward_score/frontiercs.py`)与评测
共用 `FRONTIERCS_JUDGE_URL`。

### 3.3 FrontierCS research reward ⚠️

机制通(§2.3);`FRONTIERCS_RESEARCH_PYTHON` 指 `.venv-gpublaze` 可跑 CPU 43
题里除 cant_be_late(缺 traces)与 symbolic_regression(缺 pysr/julia)外的
全部。**GPU 题 reward 需要重建 research_overlay env**,缺料清单:
torch 2.10 + triton 3.6(官方 triton-tlx build)+ flashinfer 0.6.12 +
julia depot(离线预装)。注意 2026-08-04 用户裁定:research 64 题是 eval-only,
**不进训练 mix**(prepare_multisource_rl_parquet.py 默认已 disable)。

### 3.4 MLS-Bench episode reward ✅ 静态链验证 / ⏳ 待 GPU episode 冒烟

- worker 的 `container_runtime` 在历史文件里硬编码 "apptainer" →
  gpublaze 拷贝 `scripts/gpublaze/mlsbench_rl_episode_worker_local.py`
  (除该行外逐字节相同,`MLS_RL_CONTAINER_RUNTIME` 默认 local),通过 launcher
  已有的 `MLS_RL_WORKER_SCRIPT` env 指过去,不改历史文件。
- 三层内存护栏原样保留(全是 env:`ulimit -s 8192`、
  `FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB=64`、`MLS_RL_EPISODE_MEM_MB=16384`)。
- 已验证:`mlsbench_rl_episode_worker` 对本机 MLS root 的 init/prompt 构建
  (RL parquet build 成功产出 4 行)+ conda python worker import。
- 本机 RL 训练一套 env 覆盖(未来 wrapper 用):

```bash
export FRONTIERSMITH_SYNTH_ROOT=/srv/home/bohanlyu/innovation_proior/frontiersmith_synth
# 2026-08-23 裁定:RL 的 mlsbench_agent 行同样必须用补丁版 harness
export MLS_RL_MLSBENCH_ROOT=$PWD/.cache/mlsbench-eval
export MLS_RL_DATA_ROOT=$PWD/.cache/mlsbench-eval/vendor/data
export MLS_RL_WORKER_PYTHON=/srv/home/bohanlyu/miniconda3/bin/python
export MLS_RL_WORKER_SCRIPT=$PWD/scripts/gpublaze/mlsbench_rl_episode_worker_local.py
export FRONTIERCS_RESEARCH_PYTHON=$PWD/.venv-gpublaze/bin/python
```

  注意本机 MLS checkout 没有 `_tv*`/`_train` 训练任务变体(Princeton 在
  MLS-Bench-train root,由 `scripts/mlsbench_build_train_tasks.py` 生成)——
  全量 MLS RL 数据要么拷 MLS-Bench-train,要么本机重新生成变体。

### 3.5a RL 训练环境(gpublaze,2026-08-23)

目标:Qwen3.5-4B 起点、2×H100 multisource GRPO。全部件在 `scripts/gpublaze/`:

- **venv**:`.venv-rl-gpublaze` —— torch 2.11.0+cu128(必须先于 vllm 安装,
  否则 PyPI vllm 拉 cu130 torch、与本机 nvcc 12.8 冲突导致所有 CUDA 扩展编译
  失败)+ vllm 0.21.0 + transformers 5.7.0 + fla-core/flash-linear-attention
  0.5.2 + causal_conv1d 1.7.0 + tilelang 0.1.13(LF-innov SFT venv 实证版本组
  合;fla #640 Triton 3.6 Hopper 反向 bug 的处方)+ flash-attn(源码编译;
  runner 强制 flash_attention_2,sdpa 需显式 ALLOW_SDPA=1)+ verl fork
  editable。**坑**:vllm 带的 torchvision 是 cu130 版 → 需重装
  `torchvision==0.26.0+cu128`(操作见下)。
- **数据**:`scripts/gpublaze/prepare_synth_fcs_rl_parquet.py` →
  `data/multisource_rl/train_synth_fcs.parquet`(1488 = synth 1300 + FCS 188,
  全部 single_turn_agent;**y26 纯时间句** system prefix "It is now year 2026."
  ——历史 `add_year_prefix_to_rl_parquet.py` 还带 08-18 裁定已删的人设句,故意
  不用)+ 8 行 `train_synth_fcs_smoke.parquet`。prompt 预算实测:synth max
  2282 / FCS max 8539 tokens < 10240,零溢出。routing check:[1] 注册表 ✓
  [2] synth 双向 ✓(strong 16.6 > trivial 10.0);[3] 设计上跳过(mix 无
  research 行);FCS 行经 default_compute_score dispatch 双向 ✓
  (WA→0 / 官方解→64.05,打到常驻 8082 judge)。
- **launcher**:`scripts/gpublaze/rl_multisource_local.sh` ——
  `slurm/cc_rl_multisource.sh` 的本机移植:同一 runner
  (`run_verl_grpo_frontiercs_qwen35_9b.sh` snapshot 机制保留)、三层内存护栏
  保留(`ulimit -s 8192` / `FSX_CHILD_MEM_MB=8192` / `MLS_RL_EPISODE_MEM_MB`)、
  judge 不在则自动拉起 :8082。4B/2×H100-80G 折算:TP=1、
  MAX_PROMPT_LENGTH=10240(去 MLS 后从 26624 缩)、MAX_RESPONSE_LENGTH=32768
  (底线)、MAX_MODEL_LEN=43008、PPO_MAX_TOKEN_LEN_PER_GPU=43008×SP1、
  GPU_MEMORY_UTILIZATION=0.5(80G 卡与 141G H200 的差)、
  ACTOR_OPTIMIZER_OFFLOAD=True;采样协议 top_p0.95/top_k20/presence1.5
  (ROLLOUT_PRESENCE_PENALTY/ROLLOUT_MIN_P 经 env 进 agent_loop.py,已核)。
- **penalty fastpath**:`apply_vllm_penalty_fastpath_gpublaze.sh` 把历史 patch
  块注入本 venv 的 vllm 0.21(`_convert_to_tensors` 签名与 0.23 完全一致,已
  核);selftest 必须在 patch 安装后跑(它对比 `_fs_original_convert`)。
- **恢复件风险面**(训练路径实际执行的 upstream 恢复代码):
  `verl/models/transformers/monkey_patch.py` qwen3_5 分支 +
  `verl/models/transformers/qwen3_5.py`(替换 Model/DecoderLayer/GatedDeltaNet
  forward——**训练 forward 本体**)+ `verl/models/registry.py`(import 面)。
  upstream@2026-08-22 的 qwen3_5 类名(ForConditionalGeneration/VisionModel/
  TextModel)与 transformers 5.7/5.8 匹配已核;与 Princeton fork 的潜在 diff
  无法本机验证,冒烟通过为准、正式跑前建议对 Princeton 树 diff。
  `verl/utils/qat/core.py` stub:默认配置(qat.enable=false,generated yaml
  已核)下 `apply_qat` 原样返回模型、不 raise;enable 时才 raise(防静默)。

**启动命令**(4B/2 卡正式跑;GPU 对号入座,4,5 释放后用 4,5):

```bash
cd training/FrontierSmith
# 1-step 冒烟(单卡即可;GPU 7 需 FS_ALLOW_GPU67=1,协调者 2026-08-23 已清场):
GPUS=7 FS_ALLOW_GPU67=1 NGPU=1 TOTAL_TRAINING_STEPS=1 ROLLOUT_N=2 \
  TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 SAVE_FREQ=100000 \
  ACTOR_PARAM_OFFLOAD=True GPU_MEMORY_UTILIZATION=0.35 \
  TRAIN_DATA=$PWD/data/multisource_rl/train_synth_fcs_smoke.parquet \
  EXPERIMENT_NAME=ms_smoke_gpublaze bash scripts/gpublaze/rl_multisource_local.sh
# reward 非退化验证:
.venv-gpublaze/bin/python scripts/gpublaze/check_smoke_rewards.py \
  outputs/rl_multisource_rollout/ms_smoke_gpublaze/1.jsonl

# 正式 40 步(2×H100,默认 batch 64 x rollout 16、上下文 10240+32768):
GPUS=4,5 EXPERIMENT_NAME=ms_qwen35_4b_grpo_v1 \
  setsid nohup bash scripts/gpublaze/rl_multisource_local.sh \
  >> logs/rl_ms_4b_v1.log 2>&1 &
```

**单卡 rollout+reward 干验证已通过(2026-08-23 02:37)**:GPU 7、RL venv 的
patched vLLM(fastpath ACTIVE)、8 prompts×n=2、完整协议采样 → 双源打分:
batch=16、nonzero=3、spread=0.32、fsx_B_0156 组内 0 vs 32.03(真实 GRPO 梯度
组)、frontiercs #8 两样本 0.998(judge 部分分)→ **reward 非退化 ✅**。
样本落盘 `outputs/rl_dry_rollout_reward/dry_samples.jsonl`。

**单卡完整 trainer 不可行(2026-08-23 实测,两次 OOM)**:world_size=1 时
FSDP 自动退化 `NO_SHARD`,param/optimizer offload 旋钮全部失效 → actor 栈常驻
~62G,与 43k 上下文的 rollout engine 无法共存于 80G(0.5/no-offload 和
0.35/offload 两组配置都在 weight-sync 阶段 OOM)。**单卡口径 = rollout+reward
干验证**(`scripts/gpublaze/rl_dry_rollout_reward_check.py`,对 RL venv 的
patched vLLM 直采 + default_compute_score 双源打分 + 非退化判定);完整 1-step
冒烟必须 ≥2 卡(FULL_SHARD + offload 生效)。上面的 1-step 命令把 `GPUS=7
NGPU=1` 换成 `GPUS=4,5`(或 5,7)即可。

### 3.5 multisource 冒烟结果

- parquet 链:`prepare_frontiersmith_synth_parquet.py`(1300 行)→
  `prepare_mlsbench_rl_parquet.py`(本机 root,2 任务 ×2 tier)→
  `prepare_multisource_rl_parquet.py --smoke`(1304 行 train + 8 行 smoke,
  read-back OK)全部本机跑通。
- `check_multisource_reward_routing.py`:[1] agent 注册表 ✅、[2] synth 双向
  ✅、[4] pertask norm ✅、[5] MLS agent-loop 布线 + worker import ✅;
  [3](research replay)❌ 阻塞——它要重放 Princeton `outputs/` 里存的历史
  samples.jsonl(未提交)且题目是 cant_be_late(缺 traces)。
- **PR 完整性发现(重要)**:vendored verl fork 缺文件,任何
  `import verl.workers.config` 都 crash:
  1. `verl/verl/utils/qat/core.py` **确实丢了**(fork-only 模块,upstream 没有)
     → 已放入**接口 stub**(QAT enable 即 raise;FrontierSmith 从不启用 qat,
     grep 过全部 launcher)。真文件在 Princeton
     `fs/FrontierSmith/verl/verl/utils/qat/core.py`,拿到后覆盖 stub。
  2. `verl/verl/models/` 整个包丢了 → 从 upstream volcengine/verl@1dda039b
     恢复(31 个文件,`RESTORED_FROM_UPSTREAM_NOTE.txt` 在目录里)。**注意
     fork 可能改过这个包,回 Princeton 树 diff 一次再信任它做训练**。
  3. 对 upstream 的文件盘点还有 ~95 个文件差异(experimental/trainer/workers
     等),多数疑似 upstream 新增而非 fork 丢失;训练前建议对 Princeton 树做
     一次目录级 diff。

## 4. 冒烟测试记录(2026-08-23)

| 冒烟 | 结果 |
|---|---|
| rootless docker GPU 直通 | ✅ `docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L` 与 `--device nvidia.com/gpu=all` 都列出 8×H100(CDI 已配好,**无需 sudo**) |
| FCS judge(真 go-judge) | ✅ WA 0 / 官方解 64.05(§2.1) |
| ALE 编译自检 + ahc008 全链 | ✅(§2.2;需 sitecustomize root-user patch) |
| synth reward 双向 | ✅ 63.75/44.70/0(§3.1) |
| research CPU evaluator | ✅ 机制通;cant_be_late 缺数据在 status/message 里显式报告 |
| serve→registry→client 全链 | ✅ 用 `scripts/gpublaze/mock_vllm_backend.py`(无 GPU):pick→心跳→judge→生成→解耦打分→samples.jsonl/summary_shard.json,rc=0 零 error |
| 真模型 GPU serve | ⏳ GPU 忙(本用户 eval_one.py / LF-innov 占 0-3,6-7 禁用);Qwen3.6-27B TP=2 serve 在 4,5 卡已发起,状态见 `logs/serve_q36-27b-smoke_*.log` |
| MLS episode(真模型) | ⏳ 待 GPU |
| routing check [1][2][4][5] | ✅;[3] 缺 Princeton replay 样本 |

mock 全链冒烟复现:

```bash
cd training/FrontierSmith
.venv-gpublaze/bin/python scripts/gpublaze/mock_vllm_backend.py --tag mocksmoke --port 18999 &
printf '{"tag":"mocksmoke","host":"127.0.0.1","port":18999,"url":"http://127.0.0.1:18999/v1","model":"mock","job_id":"mock"}\n' \
  > .cache/vllm_pool/mocksmoke__127.0.0.1__18999.json
while :; do touch .cache/vllm_pool/mocksmoke__127.0.0.1__18999.json; sleep 30; done &  # 心跳(entry 5min 过期)
TAG=mocksmoke SOURCE=both FRONTIERCS_LIMIT=1 ALEBENCH_LIMIT=1 N_SAMPLES=1 \
  bash scripts/gpublaze/eval_client_local.sh
```

## 5. 缺资产 / 待决策清单

**需要从 Princeton scratch 拷贝**(本机无替代):
1. `cant_be_late*` 的 `resources/real_traces.tar.gz`(20 个 research 题的数据);
2. `verl/verl/utils/qat/core.py` 真身(现为 stub);
3. ThetaEvolve 仓库(`fs/ThetaEvolve`,含 openevolve_adapted)与
   TTT-Discover 仓库(含 `examples/ahc/lib/cache` 测例缓存);
4. Princeton 的历史 `outputs/`(如需 routing check [3] 的 replay、以及与历史
   数字对表);
5. (对表用)当时 172 题的 FCS 题单(本机官方 clone 是 188 题)。
6. (可选)MLS-Bench-train root(训练任务变体 _tv*),或本机用
   `scripts/mlsbench_build_train_tasks.py` 重新生成。

> 2026-08-23 协调者裁定:ThetaEvolve/TTT、NatureBench、research GPU 21 题
> overlay 重建**本轮挂起**(不在四件套 FCS-alg / FCS-research / ALE40 / MLS CPU
> 内);cant_be_late 20 题按 runnable 子集口径跳过。

**待决策(不擅自下载/构建)**:
- Qwen3.5-9B 权重(~18G,HF 有;tokenizer/config 已拉)——要跑 9B 基线/
  checkpoint 评测就需要;
- research GPU overlay 重建(torch2.10+triton3.6+flashinfer,≥15G 盘 + 编译
  时间;或验证 sesl venv 的 triton 是否满足 `triton.set_allocator`);
- pysr + Julia depot(symbolic_regression 3 题,均为 lottery 题);
- NatureBench 官方 docker 路线(base 镜像 build + 0.5G 子集数据);
- ALE cpp20 镜像(可选:parquet prompt 说 C++20 但 judge 是 cpp17 的历史
  divergence,build `ale-bench:cpp20-202301` + `ALEBENCH_CODE_LANGUAGE=cpp20` 可闭合)。

**无需 sudo 的事实**:CDI、bwrap、userns 本机全可用;整个 setup 未用 sudo。

## 6. 文件清单(本次新增,全部不改历史)

```
scripts/gpublaze/
  env_gpublaze.sh                        # 公共 env + GPU 6/7 守卫
  serve_local.sh / serve_stop.sh         # GPU serve(registry 协议兼容)
  start_frontiercs_judge_local.sh        # judge 栈(go-judge/shim + host node)
  eval_client_local.sh                   # CPU client(协议与 cc_eval_cpu_client.sh 一致)
  eval_split_local.sh                    # 一键 serve+K clients
  eval_mlsbench_local.sh                 # MLS 22 任务(container_runtime: local)
  mock_vllm_backend.py                   # 无 GPU 全链冒烟用
  build_alebench_full40_gpublaze.py      # full40 parquet(历史脚本的 /scratch shim)
  prepare_alebench_tool_cache_docker.py  # rust 工具缓存(docker 后端)
  mlsbench_rl_episode_worker_local.py    # RL episode worker(runtime env 化)
  pysite/sitecustomize.py                # ALE rootless-docker root-user patch
  run_4b_base_mls_research_baseline.sh   # 4B base MLS x3 + research 子集基线链(qwen3_xml 门检)
  rl_multisource_local.sh                # RL GRPO 本机 launcher(4B/2 卡折算)
  prepare_synth_fcs_rl_parquet.py        # synth+FCS y26 训练 parquet
  apply_vllm_penalty_fastpath_gpublaze.sh # fastpath 注入本机 RL venv
  check_smoke_rewards.py                 # 冒烟 reward 非退化验证
  mlsbench_rl_episode_worker_local.py    # RL episode worker(runtime env 化)
verl/verl/utils/qat/core.py              # 接口 stub(PR 丢失文件,见 §3.5)
verl/verl/models/                        # 从 upstream 恢复(见 §3.5)
docs/EVAL_ON_GPUBLAZE_zh.md              # 本文
```

运行时资产(gitignored):`.cache/external/Frontier-CS`(官方 clone)、
`.cache/Frontier-CS-official`(symlink)、`.cache/bin/go-judge`、
`.cache/ale-bench/`(HF snapshot + rust-tool-builds)、`.cache/gpublaze/judge_app`、
`.venv-gpublaze`(`.venv` symlink)、`data/{frontiercs,alebench,frontiersmith_synth,mlsbench_rl,multisource_rl}/*.parquet`、
`Frontier-CS/algorithmic/problems`(symlink)、docker 镜像 ale-bench/rust。
