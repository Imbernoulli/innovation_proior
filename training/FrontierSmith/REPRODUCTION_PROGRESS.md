# FrontierSmith 复现进度总结

## 1. 环境与基础设施摸底

- **计算资源**：使用 Slurm `ailab` partition，节点为 H200（每节点 8×H200）。已验证可正常提交 GPU/CPU job。
- **容器**：README 默认使用 Docker，但登录节点无法 build Docker。改用 **Apptainer**，将 judge 镜像转为 `.sif` 后离线跑通 Frontier-CS judge。
- **依赖**：项目 `.venv` 已配置 `vllm 0.8.5.post1`、`transformers 5.7.0`、`torch 2.6.0`、`ray 2.55.1`，且 HF 权重已下载到 `models/` 下，可离线运行。

## 2. 代码兼容性修复

为了让现有代码在当前环境跑通，修改了以下关键文件：

| 文件 | 修复内容 |
|------|----------|
| `verl/verl/experimental/agent_loop/agent_loop.py` | Transformers 5 的 `apply_chat_template` 默认返回 `BatchEncoding`，强制改为返回 list |
| `verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py` | 避免给 vLLM 传空 `multi_modal_data`，否则纯文本 Qwen3 也会触发多模态路径 |
| `verl/verl/workers/rollout/vllm_rollout/utils.py` | 适配 vLLM 0.8.x 中 `process_weights_after_loading` 的 import 位置 |
| `.venv/lib/.../prometheus_fastapi_instrumentator/routing.py` | 修复 vLLM API server `/v1/models` 500 错误（`_IncludedRouter` 无 `path` 属性） |
| `scripts/merge_fsdp_to_hf.py` | 原脚本导出 `.bin` 且未处理 FSDP2 的 `DTensor`；改为先 `to_local()` 再保存为 `safetensors` |
| `slurm/eval_qwen3_model_smoke_ailab.sh` | 修复 sbatch 与手动执行的 `SLURM_SUBMIT_DIR` 兼容问题，改用绝对路径激活 venv |
| `slurm/train_qwen3_8b_smoke_1gpu_ailab.sh` | 由 1 GPU 改为 **2 GPU**（H200 单卡 optimizer state 爆显存），TP=2，mem=360G，save_freq=1 |

## 3. Merge 模型生成

已使用 `scripts/merge_qwen3_8b_linear.py` 生成：

- `models/Qwen3-8B-linear-alpha-0p25`
- `models/Qwen3-8B-linear-alpha-0p50`
- `models/Qwen3-8B-linear-alpha-0p75`

均为 `Qwen3-8B` 与 `Qwen3-8B-Base` 的线性插值模型。

## 4. 基线 Qwen3-8B（instruct）跑通

### 4.1 Base model 评测

- 模型：`models/Qwen3-8B`
- 结果：`score@1 = 0.00`（在仓库自带的 10 道 synthetic 题上）
- 输出：`outputs/eval/Qwen3-8B_smoke.csv`

### 4.2 GRPO 训练

- 使用 2 张 H200 成功跑完 5 step smoke training，保存 checkpoint：
  - `checkpoints/Qwen3-8B_baseline/global_step_5/actor/...`
- Slurm job：`9899819`，状态 `COMPLETED`

### 4.3 导出 HuggingFace 格式

- 使用更新后的 `scripts/merge_fsdp_to_hf.py` 合并 FSDP shards，导出至：
  - `models/Qwen3-8B_baseline_trained_smoke_hf`

### 4.4 训练后模型评测

- 模型：`models/Qwen3-8B_baseline_trained_smoke_hf`
- 结果：`score@1 = 0.00`
- 输出：`outputs/eval/Qwen3-8B_trained_smoke.csv`
- 说明：仅 10 道 synthetic 题且训练步数很少，分数未提升属正常。

## 5. Merge model pipeline

已使用 `slurm/submit_merge_model_pipeline.sh` 提交 3 条链（每个 α 一条）：

| α | base eval `score@1` | 当前状态 |
|---|---------------------|----------|
| 0.25 | 0.00 | 训练 job 9901852 正在运行 |
| 0.50 | **1.92** | 训练已提交，排队中 |
| 0.75 | 0.00 | 训练已提交，排队中 |

Base eval 输出文件：

- `outputs/eval/Qwen3-8B-linear-alpha-0p25_base_smoke.csv`
- `outputs/eval/Qwen3-8B-linear-alpha-0p50_base_smoke.csv`
- `outputs/eval/Qwen3-8B-linear-alpha-0p75_base_smoke.csv`

另外已额外提交 `Qwen3-8B-Base` 的 standalone base eval（job `9902471`），用于与 instruct 模型对比。

## 6. 关于“数据对不上”的说明

根据 `README.md`：

- 本仓库**仅附带 10 道 synthetic 题**（FrontierSmith 306–315）。
- 论文中完整的 **Frontier-CS algorithmic track 172 道题是公开的**，但未一起分发，需要从 `FrontierCS/Frontier-CS` 官方仓库下载并放到 `Frontier-CS/algorithmic/problems/<numeric_id>/`。
- **问题生成器（orchestrator / LLM-driven generator / checker generator）是故意 withheld 的**，不公开。

因此，目前所有分数都是在仅有的 10 道题上评测的，无法直接与论文 full benchmark 对比。若要对齐论文数字，下一步需要补齐 172 道题并重新生成 `train.parquet` / `full.parquet` 后训练。

## 7. 当前状态一句话

**基线（Qwen3-8B）的 RL 训练 + 评测链路已完全跑通**，merge model 的 base eval 已完成、训练正在跑。若目标是复现论文数字，下一步应下载完整 172 道 Frontier-CS 问题。

---

## 8. 2026-06-19 更新：Qwen3.5 / FrontierCS 官方评测口径

- 公开 `FrontierCS/FrontierSmith` 仓库未发布完整 FrontierSmith(200) 训练集；GitHub releases 为空，HF 上只找到模型权重 `runyuanhe/qwen35-9b-frontiersmith` / `runyuanhe/qwen35-9b-frontiersmith-nofilter`，未找到 dataset。
- 本地已准备：
  - FrontierCS eval/train parquet：`data/frontiercs/full.parquet`、`data/frontiercs/train.parquet`，各 172 题。
  - FrontierSmith sample parquet：`data/frontiercs/train_synthetic.parquet`，10 题。
  - ALE-Bench lite eval：`data/alebench/val.parquet`，10 题。
  - ALE-Bench full training candidate：`data/alebench_full/val.parquet`，40 题，`data_source=alebench_full`。
  - Mixed training candidate：`data/mixed/train_frontiercs172_frontiersmith10_alebench40.parquet`，共 222 条。
- 评测已切到官方 FrontierCS 口径：
  - Prompt 来自 `.cache/Frontier-CS-official/algorithmic/scripts/generate_solutions.py` 的 `CPP_SYSTEM_PROMPT`，拼接格式为 `Problem:\n\n{statement}\n\nGenerate solution code:`。
  - C++ 提取使用官方 `extract_cpp_code`。
  - Scoring 使用官方 `frontier_cs.runner.algorithmic_local.AlgorithmicLocalRunner`。
- FrontierCS-only Qwen3.5-9B base eval 已完成：
  - 初始 job `9953413` 跑到约 `823/860` 后因写完整 `samples.jsonl` 触发 `Disk quota exceeded` 失败；vLLM 与 FrontierCS judge 本身正常。
  - 已把输出改为 compact JSONL，并跳过损坏尾行 resume。
  - 续跑 job：`9955889`，`COMPLETED 0:0`，资源为 `ailab` 1×H200、8 CPU、220G，耗时 `00:06:43`。
  - 输出：`outputs/frontiercs_eval_qwen35_9b_official_vllm/summary.json` 与 `samples_compact.jsonl`。
  - 样本完整性：172 题 × 5 samples = 860 条，0 条 JSONL 坏行，0 条 sample error。
  - 结果：`reward mean@5 = 0.069767`，`reward best@5/mean = 0.115233`，`reward oracle_best@5 = 0.116279`。
- 后续直接解析结果：
  - `python scripts/summarize_vllm_eval_summary.py outputs/frontiercs_eval_qwen35_9b_official_vllm/summary.json`

## 9. 2026-06-19 更新：FrontierCS reasoning 复现、混合训练、Theta/TTT 扩展

### FrontierCS / Qwen3.5-9B reasoning eval

- 官方 no-thinking eval 分数明显低于文章，原因不是 scorer/prompt，而是未开 Qwen3.5 thinking。
- 已查 Qwen3.5 README 推荐参数，并按 reasoning/general 口径跑过 thinking-general：
  - `temperature=1.0`
  - `top_p=0.95`
  - `top_k=20`
  - `min_p=0.0`
  - `presence_penalty=1.5`
  - `repetition_penalty=1.0`
  - `enable_thinking=True`
- 长输出诊断显示 32K/64K/82K 都有部分样本长思考或 API timeout；保存原文诊断确认不少 capped 样本确实是重复/循环，不是正常持续推理。
- 关键中间结果：
  - 32K diagnostic `qwen_coding_thinking`：32 条样本，3/32 正分，样本均值约 `5.23`，最高 `90.41`，但 22/32 hit 32K cap。
  - 64K thinking-general job `9959698` 跑到 2h24m 后取消；前缀结果已到文章量级：约 15-18 个完整题时 `mean@5 ~= 1.55-1.76`，文章表中 Qwen3.5-9B Base FrontierCS `Avg@5=1.80`、`Best@5=5.00`。
  - 82K best-practice job `9959785` 前缀分数更高，但这是更宽的输出预算，不作为严格文章口径。
- 因用户判断“差不多复现出来”，已取消后续完整 32K eval job `9967509`，不再继续烧卡。
- 当前推荐复现口径：官方 FrontierCS prompt/scorer + Qwen3.5 thinking-general sampling + 32K response cap。超长样本截断接受为 0/低分，不再为追求完整思考开 64K/82K。

### 公开数据混合 GRPO training

- 完整 FrontierSmith 训练集仍未发现公开 release；`FrontierCS/FrontierSmith` 只含 10 条 synthetic，README 明确 generator/checker orchestrator withheld。
- 已使用公开可得数据挂混合训练：
  - `data/frontiercs/train.parquet`：FrontierCS 172
  - `data/frontiercs/train_synthetic.parquet`：FrontierSmith synthetic 10
  - `data/alebench_full/val.parquet`：ALE full public candidate 40，`data_source=alebench_full`
  - mixed parquet：`data/mixed/train_frontiercs172_frontiersmith10_alebench40.parquet`，共 222 条。
- 新增训练脚本：`slurm/train_qwen35_9b_mixed_public_ailab.sh`
  - 4×H200，4.5h，`TOTAL_TRAINING_STEPS=100`
  - model：`models/Qwen3.5-9B`
  - train：mixed 222 条
  - val：FrontierCS full + ALE lite
  - project/ckpt：`checkpoints/verl_frontiercs_qwen35_9b_mixed_public/qwen35_9b_grpo_mixed_public`
  - rollout：`outputs/rollout_data_qwen35_9b_mixed_public`
- Slurm job：`9967723`，提交时状态 `PENDING (Priority)`。

### ThetaEvolve

- 官方 repo 已 clone：`/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve`，commit `7c12898`。
- 官方公开 evaluator/task：
  - `circle_packing_modular`
  - `first_autocorr_inequality`
  - `second_autocorr_inequality`
  - `third_autocorr_inequality`
  - `hadamard_matrix`
- 官方 results 在 `ThetaEvolve/Results/`，但没有单独 GitHub release/tag，也没发现 HF dataset/model release。
- no-RL/evaluator smoke：circle packing initial program 已本地跑通，输出 `Score: 0.9597642169962064`。
- first autocorr initial evaluator 单次手动 smoke 超过 60s 未结束；因此新增 CPU Slurm smoke 脚本：
  - `ThetaEvolve/slurm/eval_thetaevolve_initial_programs_ailab.sh`
  - 实际改用 `cpu` partition，16 CPU，逐 task 300s timeout。
  - Slurm job：`9967995`。
- 完整 ThetaEvolve RL：官方 README/脚本走 Docker/slime/OpenEvolve，8B 官方说明至少 8×80G；当前先不直接抢 H200 跑完整 RL。后续更现实路径是 1.5B no-RL / rollout-only smoke，再逐步扩到 RL。

### TTT-Discover

- 官方 repo 已 clone：`/scratch/gpfs/CHIJ/bohan/fs/TTT-Discover`，commit `6c40e82`。
- 公开 examples 包括：
  - `examples/circle_packing`
  - `examples/ac_inequalities`
  - `examples/erdos_min_overlap`
  - `examples/ahc`
  - `examples/denoising`
  - `examples/gpu_mode`（暂时忽略硬件依赖项）
- 官方 TTT-RL 入口 `python -m examples.<task>.env` 会调用 `ttt_discover.discover()`，依赖 `tinker`、`chz`、`TINKER_API_KEY` 和 gpt-oss 模型服务；本地当前没有 `tinker/chz`，且没有 Tinker API key，因此不能声称能完整本地复现 TTT-RL training。
- 可执行优先级：先安装/隔离 TTT math requirements，跑本地 reward/evaluator smoke；有 Tinker key 后再提交官方 `discover()` 训练。AHC 还额外需要 ALE-Bench C++ container/cache。

### TTT-Discover released math result verification

- 新增本地离线 verifier：`/scratch/gpfs/CHIJ/bohan/fs/TTT-Discover/scripts/verify_released_math_results.py`
- 该脚本不依赖 `tinker/chz`，只用 `numpy` 复算 `results/mathematics/ttt_*.json` 的 released sequences。
- 本地验证结果：
  - Erdős minimum overlap：`0.380875323218`，lower-is-better，报告量级 `0.380876`。
  - AC1：`1.502862898256`，lower-is-better，报告量级 `1.50287`。
  - AC2：`0.959179771148`，higher-is-better，与 JSON 内 `C_lower_bound` 一致；该项论文中没有超过 previous best。
- 结论：TTT-Discover 的数学 released artifacts/evaluator 级结果可以离线复现；TTT-RL training 仍需要 Tinker API。

## 10. 2026-06-19 11:40 EDT 当前 Slurm 状态

### FrontierSmith / Qwen3.5 mixed public training

- 主训练 job：`9967723`，脚本 `slurm/train_qwen35_9b_mixed_public_ailab.sh`
- 当前状态：`PENDING (Priority)`
- 资源：`ailab`，4×H200，32 CPU，480G，time limit 4.5h
- Slurm 预计开始时间：`2026-06-19T12:40:38`

### Qwen3-8B merge closure jobs

这些 job 都设置为依赖 `9967723`，避免和 Qwen3.5 mixed training 抢 H200：

- `9968346`：eval `Qwen3-8B-linear-alpha-0p50_trained_smoke_hf`，`PENDING (Dependency)`，1×H200
- `9968347`：train `Qwen3-8B-linear-alpha-0p75` smoke，`PENDING (Dependency)`，2×H200
- `9968348`：export alpha 0.75 checkpoint to HF，`PENDING (Dependency)`，CPU
- `9968349`：eval alpha 0.75 trained smoke HF，`PENDING (Dependency)`，1×H200

### ThetaEvolve evaluator smoke

- Slurm job：`9967995`，`COMPLETED 0:0`，耗时 `00:06:18`
- 输出目录：`/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve/outputs/thetaevolve_initial_eval`
- 结果：
  - `circle_packing_modular`：`Score: 0.9597642169962064`
  - `second_autocorr_inequality`：`Score: 0.909791354104255`
  - `third_autocorr_inequality`：`Score: 0.3166009247179958`
  - `hadamard_matrix`：`Score: 0.14327485380116958`
  - `first_autocorr_inequality`：300s timeout，`status=124`
- 结论：官方 evaluator/task 基本可直接运行；first autocorr initial program 需要单独调 timeout 或改从官方 released/evolved result 做验证。

### ThetaEvolve released-result verification

- 新增 verifier：`/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve/scripts/verify_released_results.py`
- 新增 CPU Slurm 脚本：`/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve/slurm/verify_thetaevolve_released_results_cpu.sh`
- 本地复算已通过，并提交 CPU job：`9968515`
- 本地复算 best：
  - CirclePacking：`8B-w_RL@65`，`2.63598566124`，higher-is-better
  - FirstAutoCorrIneq：`8B-wo_RL@46-w_SOTA`，`1.50313243598`，lower-is-better
  - SecondAutoCorrIneq：`8B-w_RL@65`，`0.946897013157`，higher-is-better
  - ThirdAutoCorrIneq：`8B-w_RL@65`，`1.49300120219`，lower-is-better
- 输出：`/scratch/gpfs/CHIJ/bohan/fs/ThetaEvolve/outputs/thetaevolve_released_verify/summary.local.json`
- 完整 ThetaEvolve RL 目前未提交 GPU job，原因：
  - 本地 Python 环境缺 `sglang`、`megatron`。
  - rootless Docker CLI 存在，但 daemon 未运行：`Cannot connect to the Docker daemon at unix:///run/user/.../docker.sock`。
  - 官方 RL 路径依赖 slime/Ray/SGLang/Megatron/Docker；直接 sbatch 会浪费排队和 GPU。
  - 当前已先完成官方 evaluator smoke 与 released-result verification，后续若要跑 RL，需要先准备官方 slime 容器或等价 conda 环境。

### ThetaEvolve no-RL OpenEvolve local search

- 新增 local smoke config：
  - `ThetaEvolve/openevolve_adapted/examples/circle_packing_modular/configs/config_circle_packing_modular_qwen35_local_smoke.yaml`
  - 使用本地 OpenAI-compatible vLLM：`qwen35-9b`，24 iterations，population 256，2 islands，parallel eval 2。
- 新增 Slurm 脚本：
  - `ThetaEvolve/slurm/run_openevolve_circle_qwen35_local_smoke_ailab.sh`
  - 1×H200，8 CPU，2.5h；先启动本地 vLLM server，再调用官方 `openevolve.cli` 跑 circle packing no-RL evolution/search。
- 静态检查通过，initial program evaluator score：`0.9597642169962064`。
- Slurm job：`9969376`，状态待 Slurm 调度。

### TTT-Discover released math verification

- Slurm job：`9968241`，`COMPLETED 0:0`，耗时 `00:00:01`
- 输出：
  - `erdos_min_overlap raw_score=0.380875323218`
  - `ac1 raw_score=1.502862898256`
  - `ac2 raw_score=0.959179771148`
- 结论：TTT-Discover 数学 released artifacts 可以离线复算；官方 test-time RL training 仍需要 Tinker API / `TINKER_API_KEY`。

### TTT-Discover AHC public-cache eval

- 官方 AHC cache 已从 repo 指定 Google Drive 链接下载并解压：
  - `TTT-Discover/examples/ahc/lib/cache`
  - 包含 `tester_binaries/{ahc039_tester,ahc058_tester}` 和 150 条 public inputs。
- 本机无 `g++-12`，但 `module load gcc-toolset/14` 后可用 `g++` 编译；已在 evaluator 脚本里自动生成 `g++-12` wrapper。
- 新增 evaluator：`/scratch/gpfs/CHIJ/bohan/fs/TTT-Discover/scripts/evaluate_ahc_released.py`
- 新增 CPU Slurm 脚本：`/scratch/gpfs/CHIJ/bohan/fs/TTT-Discover/slurm/evaluate_ahc_released_cpu.sh`
- 首次 Slurm job：`9968727`，64 CPU，因官方 AHC local-cache 路径在 import 阶段硬依赖缺失的可选包 `modal` 失败。
- 已 patch 本地离线路径：
  - `examples/ahc/lib/data.py`：`modal`/`polars` 改为可选，CSV 读取有标准库 fallback。
  - `examples/ahc/lib/utils.py`：`cairosvg`/`ahocorapy` 改为用到时才报错；AHC released public eval 不需要题面 SVG/statement parsing。
- 离线 cache smoke 已通过：`ahc039` 和 `ahc058` 均能读到 150 条 public inputs 与 cached tester。
- 第二次 Slurm job：`9969020`，`COMPLETED` 但 evaluator 返回编译失败：Ray worker 的 `g++-12` wrapper 链接阶段找不到 `-lgmpxx -lgmp`，导致没有生成 `a.out`。
- 已更新 wrapper 生成逻辑：使用 module-loaded `g++` 绝对路径，并显式设置 `/usr/lib64` 到 `LD_LIBRARY_PATH`/`LIBRARY_PATH`；手工编译 released C++ 通过。
- 第三次 Slurm job：`9969286` 仍在 CPU worker 上报同样 GMP 链接问题；判断为 CPU 节点环境与登录节点库/链接配置不一致。
- 已新增环境开关 `ALE_BENCH_STRIP_GMP_LINK=1`：仅在本地 released-code eval 中去掉未使用的 `-lgmpxx -lgmp`，不改官方 evaluator/scoring。
- 第四次重交 Slurm job：`9970053`，64 CPU，评测 `results/algorithm-design/{ahc039.cpp,ahc058.cpp}` 在官方 cached public inputs 上的分数。
- 输出目标：`/scratch/gpfs/CHIJ/bohan/fs/TTT-Discover/outputs/ahc_released_eval/summary.json`

## 11. 2026-06-19 12:10 EDT 当前状态

- FrontierSmith/Qwen3.5 mixed public training：job `9967723` 仍为 `PENDING (Priority)`，4×H200，4.5h。
- 依赖的 Qwen3 merge closure jobs `9968346/9968347/9968348/9968349` 仍为 `PENDING (Dependency)`。
- ThetaEvolve no-RL OpenEvolve local search：job `9969376` 已 `RUNNING`，1×H200；vLLM server 正常启动，已完成 checkpoint 8，circle packing best 从 initial `0.9598` 提到 `1.0117`。
- TTT-Discover AHC public-cache eval：前几次失败原因已定位到本地 CPU 节点 GMP 链接环境；已启用 `ALE_BENCH_STRIP_GMP_LINK=1` 并重交 job `9970053`。

## 12. 2026-06-19 12:53 EDT 当前状态

### FrontierSmith / FrontierCS mixed training

- 主训练 job `9967723` 已开始运行：
  - 状态：`RUNNING`
  - 已运行：`00:22:27`
  - 节点：`della-i21g2`
  - 资源：4×H200，32 CPU，480G，time limit 4.5h
- 依赖 job `9968346/9968347/9968348/9968349` 仍为 `PENDING (Dependency)`，等待 `9967723` 完成后导出/评测。
- 训练尚未产出新的 `global_step_*` checkpoint；日志显示仍在首批 rollout/reward 阶段。
- 重要 caveat：当前 mixed 训练刚开始时 ALE full reward 对未缓存 problem 触发了计算节点联网构建 Rust tools，失败样例包括 `ahc030`、`ahc007`，这些早期 ALE 样本 reward 已被 `alebench_full.py` 捕获为 0。FrontierCS/FrontierSmith reward 不受这个问题影响。

### ALE-Bench full tool cache 修复

- 已确认 compute node 不能访问 `index.crates.io`，但登录节点可以。
- 原先只缓存了 ALE lite 10 个 problem 的 Rust tools；full 40 中缺 30 个，导致 mixed training 的 ALE full reward 在 compute node 上失败。
- 已更新：
  - `scripts/prepare_alebench_tool_cache.py`：支持 `--no-lite --check-only`、`--problem-id`，并强制使用传入的 ALE cache/data/apptainer 路径。
  - `ALE-Bench/src/ale_bench/utils.py`：Apptainer rust build 额外设置可写 `RUSTUP_HOME`，修复 `ahc045` 的 `rust-toolchain.toml` 在只读 `/usr/local/rustup` 下失败。
- 已在登录节点补齐 full 40 cache：
  - `python scripts/prepare_alebench_tool_cache.py --no-lite --check-only`
  - 结果：`Prepared Rust tool cache for 40/40 problems`
  - cache：`.cache/ale-bench/rust-tool-builds`，40 个 key，119 个二进制文件。
- 后续新的 ALE mixed training/eval 不应再因 Rust tool build 访问外网失败；当前正在跑的 `9967723` 对已经失败过的早期 ALE 样本无法 retroactively 修正。

### ThetaEvolve

- no-RL OpenEvolve circle packing local search job `9969376` 已 `COMPLETED 0:0`，耗时 `00:10:38`。
- 输出：
  - `ThetaEvolve/outputs/openevolve_circle_qwen35_local_smoke/job_9969376/best/best_program.py`
  - `ThetaEvolve/outputs/openevolve_circle_qwen35_local_smoke/job_9969376/best/best_program_info.json`
  - `ThetaEvolve/outputs/openevolve_circle_qwen35_local_smoke/job_9969376/checkpoints/checkpoint_24`
- best metrics：
  - `objective_value = 2.112576846895433`
  - `combined_score = 2.112576846895433`
  - `validity = 1.0`
  - initial score 是 `0.9597642169962064`，说明官方 OpenEvolve no-RL search 路径能用本地 Qwen3.5/vLLM 跑起来。

### TTT-Discover

- AHC released public-cache eval job `9970053` 已 `COMPLETED 0:0`，耗时 `00:00:34`。
- 输出：`TTT-Discover/outputs/ahc_released_eval/summary.json`
- 结果：
  - `ahc039`：150 public cases，8 accepted，mean absolute score `207.35333333333332`，total `31103.0`。
  - `ahc058`：150 public cases，150 accepted，mean absolute score `5668121.946666666`，total `850218292.0`。
- 结论：TTT-Discover 的 AHC released-code evaluator 路径已经能在本地 CPU/Slurm 环境跑通；官方 TTT-RL training 仍受 `tinker/chz/TINKER_API_KEY` 限制。

### 后续框架合并方向

- 短期先以 VERL 为统一训练入口，因为 FrontierCS/FrontierSmith/ALE 的数据与 reward 已经接在 VERL 格式上；Slime 留给 Theta 官方 RL 路径或后续需要 GRPO test-time RL 时再接。
- 统一 task adapter 应覆盖：
  - FrontierCS/FrontierSmith：C++ generation + official FrontierCS judge reward。
  - ALE-Bench：C++ generation + cached Apptainer public eval reward。
  - ThetaEvolve：program-search/evaluator task，先 no-RL rollout/search，RL 再接 Slime/VERL wrapper。
  - TTT-Discover：math/AHC released evaluator 可离线复算；真正 TTT-RL 需要 Tinker 或替代本地 LoRA/RL runner。

## 13. 2026-06-19 13:24 EDT 低频训练检查

- `9967723` 仍为 `RUNNING`，已运行 `00:54:03`，节点 `della-i21g2`，资源 4×H200/32 CPU/480G，ExitCode `0:0`。
- 训练已实际推进到 `training/global_step:3`。
- rollout 已写出：
  - `outputs/rollout_data_qwen35_9b_mixed_public/1.jsonl`
  - `outputs/rollout_data_qwen35_9b_mixed_public/2.jsonl`
  - `outputs/rollout_data_qwen35_9b_mixed_public/3.jsonl`
- checkpoint 还未出现；当前 `SAVE_FREQ=10`，因此要到 step 10 附近才应看到 `global_step_*`。
- `.err` 中仍有 12:41 左右旧的 ALE `index.crates.io` / Rust tool build 失败记录；12:53 补齐 ALE full tool cache 后，本次尾部检查没有看到新的 ALE/crates.io 失败。

## 14. 2026-06-19 13:30 EDT 训练/评测链补充

### Qwen3.5 mixed training 速度与后续 eval

- `9967723` 仍在运行，13:25 EDT 已运行约 `00:55:11`。
- 当前 step timing：
  - step 1：`1057s`
  - step 2：`850s`
  - step 3：`896s`
- 因此 `TOTAL_TRAINING_STEPS=100` 在 4.5h time limit 内不可能完整跑完；预期更现实的是先拿到 step 10 附近 checkpoint，再导出/评测。
- 已存在训练后导出+eval job：
  - `9972042` / `cc-fs-q35mix-eval`
  - dependency：`afterany:9967723`
  - command：`slurm/export_and_eval_qwen35_ckpt_vllm_ailab.sh`
  - checkpoint root：`checkpoints/verl_frontiercs_qwen35_9b_mixed_public/qwen35_9b_grpo_mixed_public`
  - model output：`models/cc_qwen35_9b_mixed_hf`
  - eval output：`outputs/cc_eval_qwen35_9b_mixed_vllm`
  - setting：1×H200，6h，`SOURCE=both` 默认，`ENABLE_THINKING=0`，16K 默认输出 cap。
- 新增 trained FrontierCS thinking-general eval job：
  - `9972762`
  - dependency：`afterok:9972042`
  - model：`models/cc_qwen35_9b_mixed_hf`
  - output：`outputs/frontiercs_eval_cc_qwen35_9b_mixed_thinking_32k_c64_vllm`
  - setting：1×H200，4h，FrontierCS only，官方 prompt/scorer，32K cap，Qwen3.5 thinking-general sampling。

### Qwen3-8B / merge-model smoke pipeline evidence

- 已存在 base models：
  - `models/Qwen3-8B`
  - `models/Qwen3-8B-Base`
  - `models/Qwen3-8B-linear-alpha-0p25`
  - `models/Qwen3-8B-linear-alpha-0p50`
  - `models/Qwen3-8B-linear-alpha-0p75`
- 已完成 smoke checkpoints：
  - `checkpoints/Qwen3-8B_baseline/global_step_1..5`
  - `checkpoints/Qwen3-8B-linear-alpha-0p25_smoke/global_step_1..5`
  - `checkpoints/Qwen3-8B-linear-alpha-0p50_smoke/global_step_1..5`
- 已导出 HF trained smoke models：
  - `models/Qwen3-8B_baseline_trained_smoke_hf`
  - `models/Qwen3-8B-linear-alpha-0p25_trained_smoke_hf`
  - `models/Qwen3-8B-linear-alpha-0p50_trained_smoke_hf`
- 当前 available smoke eval CSV：
  - `Qwen3-8B-Base_smoke.csv`：`score_at_1=0.0`
  - `Qwen3-8B_smoke.csv`：`score_at_1=0.0`
  - `Qwen3-8B_trained_smoke.csv`：`score_at_1=0.0`
  - `Qwen3-8B-linear-alpha-0p25_base_smoke.csv`：`score_at_1=0.0`
  - `Qwen3-8B-linear-alpha-0p25_trained_smoke.csv`：`score_at_1=6.9026`
  - `Qwen3-8B-linear-alpha-0p50_base_smoke.csv`：`score_at_1=1.9181`
  - `Qwen3-8B-linear-alpha-0p75_base_smoke.csv`：`score_at_1=0.0`
- 仍待依赖 job 完成的 Qwen3-8B closure：
  - `9968346`：alpha 0.50 trained smoke eval，`afterany:9967723`
  - `9968347`：alpha 0.75 smoke train，`afterany:9967723`
  - `9968348`：alpha 0.75 export HF，`afterok:9968347`
  - `9968349`：alpha 0.75 trained smoke eval，`afterok:9968348`

## 15. 2026-06-19 14:00 EDT 低频训练检查

- `9967723` 仍为 `RUNNING`，已运行 `01:29:52`，节点 `della-i21g2`，ExitCode `0:0`。
- 最新训练进度：`training/global_step:5`。
- rollout 已写出 `1.jsonl` 到 `5.jsonl`：
  - 新增 `4.jsonl`
  - 新增 `5.jsonl`
- checkpoint root 仍未出现：`checkpoints/verl_frontiercs_qwen35_9b_mixed_public/qwen35_9b_grpo_mixed_public` 不存在。当前 `SAVE_FREQ=10`，所以预期 step 10 才会有 checkpoint。
- `9972042`（导出+no-thinking eval）和 `9972762`（trained FrontierCS thinking eval）仍为 `PENDING (Dependency)`。
- `.err` 末尾仍有旧的 12:41 ALE/crates.io/Rust build 失败记录；没有看到 12:53 补齐 ALE cache 之后的新 ALE/crates.io/Rust build 失败。

## 16. 2026-06-19 14:31 EDT 低频训练检查

- `9967723` 仍为 `RUNNING`，已运行 `02:00:55`，ExitCode `0:0`。
- 最新训练进度：`training/global_step:7`。
- rollout 已写出 `1.jsonl` 到 `7.jsonl`：
  - 新增 `6.jsonl`
  - 新增 `7.jsonl`
- checkpoint root 仍未出现：没有 `global_step_*` 或 `latest_checkpointed_iteration.txt`。
- `9972042` 和 `9972762` 仍为 `PENDING (Dependency)`。
- `.err` 中仍只看到旧的 12:41 ALE/crates.io/Rust build 失败；没有看到 12:53 补齐 ALE cache 之后的新 ALE/crates.io/Rust build 失败。

## 17. 2026-06-19 15:03 EDT checkpoint/export 检查

- `9967723` 仍为 `RUNNING`，已运行约 `02:32`。
- 最新训练进度：`training/global_step:10`。
- rollout 已写出 `1.jsonl` 到 `10.jsonl`。
- checkpoint 已出现：
  - `checkpoints/verl_frontiercs_qwen35_9b_mixed_public/qwen35_9b_grpo_mixed_public/global_step_10`
  - `latest_checkpointed_iteration.txt = 10`
  - actor shards：`model_world_size_4_rank_0..3.pt`
- `9972042` 已从 dependency pending 进入 `RUNNING`：
  - node：`della-i20g2`
  - 资源：1×H200，8 CPU，300G
  - 已开始从 `global_step_10` 导出 HF 模型到 `models/cc_qwen35_9b_mixed_hf`
  - 日志显示 FSDP world_size=4，760 个 checkpoint keys，426 个 key remap，shape check passed，已开始写 `model-00001.safetensors`
- `9972762` 仍为 `PENDING (Dependency)`，等待 `9972042` 完成。
- 15:03 检查时未看到新的 ALE/crates.io/Rust build 失败。

## 18. 2026-06-19 15:42 EDT Qwen3.5 trained eval 修复与重提

- `9972042` 已失败，根因不是 checkpoint 本身，而是 HF export key 命名错误：
  - 原始 Qwen3.5 HF 权重：`model.language_model.layers.*`
  - 失败导出权重：`model.language_model.model.layers.*`
  - vLLM 因此报 `Following weights were not initialized from checkpoint`。
- 已修复 `scripts/merge_fsdp_to_hf.py`：
  - 去掉错误的 `model.language_model.* -> model.language_model.model.*` remap。
  - 增加 missing/extra key 检查，避免之后“shape 看似通过但 key 不匹配”的假阳性。
- 已有 fixed HF 目录可被 vLLM 正常加载：
  - `models/cc_qwen35_9b_mixed_hf_fixed`
  - 当前 eval job：`9977353` / `cc-fs-q35mix-eval2`
  - 资源：1×H200，8 CPU，300G
  - 配置：`SOURCE=both`，FrontierCS official scorer + ALE-Bench，`N_SAMPLES=5`，`ENABLE_THINKING=0`，`MAX_TOKENS=16000`
  - vLLM 已成功启动并加载 fixed model；日志显示 `Loaded 182 problems`，其中 `frontiercs=172`、`alebench=10`，总 `910` samples。
- 旧的 trained thinking eval `9972762` 因依赖失败进入 `DependencyNeverSatisfied`，已取消。
- 新增并提交 trained thinking-general sharded eval：
  - 脚本：`slurm/eval_qwen35_9b_mixed_thinking_both_array_ailab.sh`
  - job：`9978858_[0-3]`
  - dependency：`afterany:9977353`
  - 每个 shard：1×H200，6h，`SOURCE=both`
  - 参数：Qwen3.5 thinking-general，官方 FrontierCS prompt/scorer，`MAX_TOKENS=32768`，`MAX_MODEL_LEN=49152`，`CONCURRENCY=64`，`N_SAMPLES=5`
  - 输出：`outputs/cc_eval_qwen35_9b_mixed_thinking_general_32k_both_vllm/shards/shard_*`
- 新增并提交 sharded summary job：
  - 脚本：`slurm/summarize_qwen35_9b_mixed_thinking_both_cpu.sh`
  - job：`9978859`
  - dependency：`afterok:9978858`
  - 汇总输出：`outputs/cc_eval_qwen35_9b_mixed_thinking_general_32k_both_vllm/summary.json`
- 当前训练 `9967723` 仍在跑，已到 `training/global_step=12`；由于 `SAVE_FREQ=10` 且 4.5h time limit，`global_step_10` 大概率就是可用的最终 checkpoint。

## 19. 2026-06-19 15:48 EDT Qwen3-8B mixed-public 训练链排队

- 之前 Qwen3-8B / merge-model 路径主要还是 smoke 级：
  - 已有 merge 模型：`alpha=0.25/0.50/0.75`
  - 已有 smoke checkpoint：baseline、alpha 0.25、alpha 0.50
  - 已有 smoke eval CSV；alpha 0.50 trained eval 与 alpha 0.75 train/export/eval 仍在等 `9967723`。
- 为了把原始 Qwen3-8B 复现从 smoke 推到同一套 public mixed 数据，新增脚本：
  - `slurm/train_qwen3_8b_mixed_public_ailab.sh`
  - 数据：`data/mixed/train_frontiercs172_frontiersmith10_alebench40.parquet`
  - 资源：2×H200，16 CPU，360G，4h
  - 默认模型：`models/Qwen3-8B`
  - checkpoint：`checkpoints/verl_frontiercs_qwen3_8b_mixed_public/qwen3_8b_grpo_mixed_public`
  - rollout：`outputs/rollout_data_qwen3_8b_mixed_public`
  - 默认训练：`TOTAL_TRAINING_STEPS=30`，`SAVE_FREQ=10`，`ROLLOUT_N=4`，`MAX_RESPONSE_LENGTH=8192`
- 已提交 Qwen3-8B mixed-public train/export/eval/summary 链：
  - `9978978` / `fs-qwen3-mix`：2×H200 train，dependency `afterany:9967723`
  - `9978979` / `fs-export-hf`：CPU export，dependency `afterok:9978978`
  - `9978980_[0-3]` / sharded thinking eval：dependency `afterok:9978979`
    - model：`models/qwen3_8b_mixed_public_hf`
    - output：`outputs/eval_qwen3_8b_mixed_public_thinking_general_32k_both_vllm/shards/shard_*`
    - eval：FrontierCS + ALE-Bench，official FrontierCS prompt/scorer，Qwen thinking-general params，32K output cap
  - `9978981` / summary：dependency `afterok:9978980`
    - output：`outputs/eval_qwen3_8b_mixed_public_thinking_general_32k_both_vllm/summary.json`
- 当前 Qwen3.5 fixed no-thinking eval `9977353` 仍在跑；最新轻量检查到约 `458/910` samples，仍在 FrontierCS 部分，尚未进入 ALE-Bench。

## 20. 2026-06-19 15:50 EDT Qwen3-8B eval context cap 修复

- 检查模型配置发现：
  - `models/Qwen3-8B/config.json`：`max_position_embeddings=40960`
  - `models/Qwen3-8B-Base/config.json`：`max_position_embeddings=32768`
  - Qwen3.5 HF config 没有顶层 `max_position_embeddings`
- 之前 sharded thinking eval wrapper 默认 `MAX_MODEL_LEN=49152`，对 Qwen3.5 可用，但 Qwen3-8B eval array `9978980_[0-3]` 复用该 wrapper 时可能导致 vLLM 启动阶段超过模型上下文上限。
- 已修复 `slurm/eval_qwen35_9b_mixed_thinking_both_array_ailab.sh`：
  - 如果用户没有显式设置 `MAX_MODEL_LEN`，脚本会读取 `${MODEL_PATH}/config.json`。
  - 若存在 `max_position_embeddings`，默认使用 `min(49152, max_position_embeddings)`。
  - 因此 Qwen3-8B 会自动使用 `40960`，Qwen3-8B-Base 会自动使用 `32768`，Qwen3.5/fixed Qwen3.5 仍使用 `49152`。
- 已用 `bash -n` 和独立 config 读取验证默认值；排队 job `9978980_[0-3]` 未显式导出 `MAX_MODEL_LEN`，运行时会读取修复后的脚本默认值。

## 21. 2026-06-19 15:54 EDT Qwen3/Qwen3.5 GDN 参数与 sharded 端口隔离修复

- 继续静态排查发现两个会影响后续 queued jobs 的风险：
  - `scripts/run_verl_grpo_frontiercs_qwen35_9b.sh` 无条件传 `+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=triton`；这是 Qwen3.5/GDN 相关参数，Qwen3-8B smoke 训练日志里没有该参数。
  - `scripts/start_vllm_server.sh` 也无条件传 `--gdn-prefill-backend triton`；Qwen3-8B sharded eval 会复用该启动脚本。
  - sharded eval array 默认共用 `8000/8082/5050`，如果多个 array task 被调度到同一节点会端口冲突。
- 已修复：
  - `scripts/run_verl_grpo_frontiercs_qwen35_9b.sh`：未显式设置 `GDN_PREFILL_BACKEND` 时读取 `${MODEL_PATH}/config.json`；`model_type in {qwen3_5,qwen3_5_moe}` 才传 `gdn_prefill_backend=triton`，普通 `qwen3` 不传。
  - `scripts/start_vllm_server.sh`：同样只对 Qwen3.5/Qwen3.5-MoE 自动加 `--gdn-prefill-backend triton`。
  - `slurm/train_qwen3_8b_mixed_public_ailab.sh`：默认 `GDN_PREFILL_BACKEND=""`，避免覆盖自动判断。
  - `slurm/eval_qwen35_9b_mixed_thinking_both_array_ailab.sh`：按 `SHARD_IDX` 自动设置端口：
    - `PORT=8082 + 10*SHARD_IDX`
    - `VLLM_PORT=8000 + 10*SHARD_IDX`
    - `GJ_PORT=5050 + 10*SHARD_IDX`
    - `RUNTIME_DIR` 也带 array/shard 标识。
- 因为 Slurm 会在 `sbatch` 时复制 batch script，旧的 pending jobs 不会自动拿到 wrapper 修复；已取消旧 pending jobs：
  - `9978858`, `9978859`, `9978978`, `9978979`, `9978980`, `9978981`
- 已重交修复后的链：
  - Qwen3.5 trained thinking eval：`9979101_[0-3]`，dependency `afterany:9977353`
  - Qwen3.5 trained thinking summary：`9979102`
  - Qwen3-8B mixed-public train：`9979103`，dependency `afterany:9967723`
  - Qwen3-8B mixed-public export：`9979104`
  - Qwen3-8B mixed-public sharded eval：`9979105_[0-3]`
  - Qwen3-8B mixed-public summary：`9979106`
- 已通过 `bash -n` 验证相关脚本；独立 config 检查结果：
  - Qwen3-8B / Qwen3-8B-Base：auto `GDN_PREFILL_BACKEND=''`
  - Qwen3.5 / fixed Qwen3.5：auto `GDN_PREFILL_BACKEND='triton'`

## 22. 2026-06-19 16:00 EDT 当前运行状态与后续顺序

- 当前 `squeue/sacct` 状态：
  - `9967723` / `fs-q35-mixtrain` 仍在运行，已运行约 `03:29/04:30`，node=`della-i21g2`。
  - `9977353` / `cc-fs-q35mix-eval2` 仍在运行，已运行约 `00:51/06:00`，node=`della-i20g2`。
  - `9979101_[0-3]`、`9979102`、`9979103`、`9979104`、`9979105_[0-3]`、`9979106` 均仍在等依赖。
- 训练日志最新已到 `training/global_step=14`，已写出 checkpoint：
  - `global_step_10`
  - 当前 4.5h walltime 大概率不能跑满原先的 `TOTAL_TRAINING_STEPS=100`，所以这条 Qwen3.5 训练更像可用 checkpoint 产出，不应记为完整 100-step 训练闭环。
- Qwen3.5 fixed no-thinking eval 最新落盘：
  - `outputs/cc_eval_qwen35_9b_mixed_fixed_vllm/samples.jsonl`
  - `634/910` samples
  - 仍在 FrontierCS 部分，尚未汇总。
- 训练 `.err` 里仍有 12:41 的 ALE/crates.io 离线构建失败；这是当前 job 早期 reward worker 触发的旧错误。
- 已做只读 ALE cache 完整性检查：
  - ALE full problems：`40`
  - 缺失 cached Rust tools：`0`
  - 后续新启动训练不应再因为 compute node 无网络访问 `crates.io` 而失败。
- 后续工作顺序：
  - 先完成 FrontierCS/FrontierSmith 这条线：base eval、trained eval、mixed-public train、export、thinking eval、summary。
  - 框架合并放在下一阶段：把 FrontierCS/FrontierSmith、ALE、ThetaEvolve/TTT-Discover 中非硬件依赖的数学/packing/inequality/AtCoder 类任务统一成同一个 train/eval 数据与 reward 接口。
  - VERL/Slime 选择先按能跑通的路径推进；当前本仓库已经有 VERL 训练链，优先不切换框架，等 FrontierCS/Smith 闭环后再设计统一任务层。

## 23. 2026-06-19 16:08 EDT Qwen3/Qwen3-Base merge full eval 脚本与排队

- 现有 Qwen3-8B / merge 比例证据仍主要是 smoke：
  - `outputs/eval/*_smoke.csv`
  - `checkpoints/Qwen3-8B*_smoke/global_step_*`
  - 这不能当作完整 FrontierCS + ALE-Bench 复现结果。
- 新增完整单卡评测脚本：
  - `slurm/eval_qwen3_both_thinking_1gpu_ailab.sh`
  - 资源：1×H200，8 CPU，240G，6h
  - eval：`SOURCE=both`，FrontierCS official prompt/scorer + ALE-Bench
  - sampling：Qwen thinking-general defaults，`ENABLE_THINKING=1`，`temperature=1.0`，`top_p=0.95`，`top_k=20`，`presence_penalty=1.5`
  - output：默认 `outputs/eval_${MODEL_TAG}_thinking_general_both_vllm`
  - 保存 `text_preview`，不默认保存完整 32K 文本，方便检查 repetition 且避免输出过大。
- 新增 submitter：
  - `slurm/submit_qwen3_base_merge_full_eval.sh`
  - 覆盖 5 个 base model：
    - `models/Qwen3-8B`
    - `models/Qwen3-8B-Base`
    - `models/Qwen3-8B-linear-alpha-0p25`
    - `models/Qwen3-8B-linear-alpha-0p50`
    - `models/Qwen3-8B-linear-alpha-0p75`
- 静态检查：
  - `bash -n` 通过。
  - 自动 context/token cap：
    - Qwen3-8B：`MAX_MODEL_LEN=40960`, request `max_tokens=32768`
    - Qwen3-8B-Base：`MAX_MODEL_LEN=32768`, request `max_tokens=28672`
    - alpha 0.25/0.50/0.75：`MAX_MODEL_LEN=40960`, request `max_tokens=32768`
- 已将 full eval jobs 排到 Qwen3-8B baseline summary `9979106` 之后，当前不会占 GPU：
  - `9979417`：`qwen3_8b`
  - `9979418`：`qwen3_8b_base`
  - `9979419`：`qwen3_8b_alpha0p25`
  - `9979420`：`qwen3_8b_alpha0p50`
  - `9979421`：`qwen3_8b_alpha0p75`
  - `squeue` 确认均为 `PENDING (Dependency)`。

## 24. 2026-06-19 16:12 EDT Qwen3-Base/merge mixed-public 训练链排队

- 新增训练 submitter：
  - `slurm/submit_qwen3_base_merge_mixed_public_train_pipeline.sh`
  - Qwen3-8B baseline 本身已由 `9979103 -> 9979104 -> 9979105 -> 9979106` 处理，因此此 submitter 覆盖剩余 endpoint/ratio：
    - `qwen3_8b_base`
    - `qwen3_8b_alpha0p25`
    - `qwen3_8b_alpha0p50`
    - `qwen3_8b_alpha0p75`
- 每个模型的 pipeline：
  - 2×H200 mixed-public GRPO training (`data/mixed/train_frontiercs172_frontiersmith10_alebench40.parquet`)
  - CPU export to HF
  - 1×H200 trained-model FrontierCS + ALE-Bench thinking eval
- 该 submitter 默认顺序串行，避免 4 个 2-GPU 训练同时启动。
- 已将训练链挂到 5 个 base/full eval jobs `9979417:9979418:9979419:9979420:9979421` 全部 afterok 之后：
  - `qwen3_8b_base`: train `9979479`, export `9979480`, trained eval `9979481`
  - `qwen3_8b_alpha0p25`: train `9979482`, export `9979483`, trained eval `9979484`
  - `qwen3_8b_alpha0p50`: train `9979485`, export `9979486`, trained eval `9979487`
  - `qwen3_8b_alpha0p75`: train `9979488`, export `9979489`, trained eval `9979490`
- `squeue` 确认 `9979479-9979490` 均为 `PENDING (Dependency)`，当前不会占用 GPU。

## 25. 2026-06-19 16:24 EDT Qwen3 full eval context cap 修复与重交

- 对 Qwen3-8B-Base / Qwen3-8B / merge 的 full eval 做了静态 prompt 长度检查：
  - 使用评测实际 `.venv` 和 Qwen3-8B-Base tokenizer。
  - official FrontierCS prompt 最长约 `8180` tokens，ALE prompt 最长约 `8078` tokens。
  - 之前 Qwen3-8B-Base 的 `MAX_MODEL_LEN=32768` + `max_tokens=28672` 会在长 prompt 上超过 context。
  - Qwen3-8B / merge 的 `MAX_MODEL_LEN=40960` + `max_tokens=32768` 也只剩约 12 token 余量，不稳。
- 已修复 `slurm/eval_qwen3_both_thinking_1gpu_ailab.sh`：
  - 默认 `MAX_PROMPT_RESERVE=8704`。
  - request cap 现在是 `min(MAX_TOKENS, MAX_MODEL_LEN - reserve)`，底线 `1024`。
  - 静态验证：
    - ctx `32768` -> request `24064`，最长 prompt 总长约 `32244`，余量 `524`
    - ctx `40960` -> request `32256`，最长 prompt 总长约 `40436`，余量 `524`
- 因 Slurm 会复制 batch script，已取消旧 pending 链：
  - full eval：`9979417-9979421`
  - downstream train/export/eval：`9979479-9979490`
- 已重交 full eval，仍依赖 `9979106`：
  - `9979611`：`qwen3_8b`
  - `9979612`：`qwen3_8b_base`
  - `9979613`：`qwen3_8b_alpha0p25`
  - `9979614`：`qwen3_8b_alpha0p50`
  - `9979615`：`qwen3_8b_alpha0p75`
- 已重交 downstream mixed-public train/export/trained-eval，依赖上述 5 个 full eval 全部 afterok：
  - `qwen3_8b_base`: train `9979618`, export `9979619`, trained eval `9979620`
  - `qwen3_8b_alpha0p25`: train `9979621`, export `9979622`, trained eval `9979623`
  - `qwen3_8b_alpha0p50`: train `9979624`, export `9979625`, trained eval `9979626`
  - `qwen3_8b_alpha0p75`: train `9979627`, export `9979628`, trained eval `9979629`
- `squeue` 确认新链 `9979611-9979629` 均为 `PENDING (Dependency)`，当前不会占用 GPU。

## 26. 2026-06-19 16:31 EDT vLLM eval prefill 参数修复

- 静态检查发现 `scripts/start_vllm_server.sh` 没有显式传 `--max-num-batched-tokens`，vLLM 会使用较保守默认值。
- 对 FrontierCS/ALE 的 8K 级 prompt，这会让 prefill 偏慢，尤其是高并发 full eval。
- 已修复 `scripts/start_vllm_server.sh`：
  - 新增 `MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}`。
  - 启动 vLLM 时传 `--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"`。
  - 启动日志打印该值。
- 该文件是 batch job 运行时调用的 helper；已排队但未启动的 vLLM eval jobs 会自动使用这个修复，不需要取消重交。
- 已通过：
  - `bash -n scripts/start_vllm_server.sh`
  - `git diff --check` relevant files

## 27. 2026-06-19 16:36 EDT mixed-public 训练数据完整性检查

- 检查 `data/mixed/train_frontiercs172_frontiersmith10_alebench40.parquet`：
  - 总行数：`222`
  - `mix_source` 分布：
    - `frontiercs172`: `172`
    - `frontiersmith10`: `10`
    - `alebench40`: `40`
  - `data_source` 分布：
    - `frontiercs`: `182`
    - `alebench_full`: `40`
- 结论：
  - `frontiersmith10` 被标成 `data_source=frontiercs`，用同一套 FrontierCS judge/reward。
  - 这 10 条的 ground truth 是 `frontiersmith_1..frontiersmith_10`。
  - 本地 judge 目录 `Frontier-CS/algorithmic/problems/frontiersmith_1..10` 都存在，包含 `statement.txt` 和 `gen.cpp`。
  - official cache `.cache/Frontier-CS-official` 没有这些 `frontiersmith_*`，但训练 reward 走本地 FrontierCS judge，因此可评分。
- 当前完整 eval 仍按复现主线评估：
  - FrontierCS official problems `172`
  - ALE-Bench val `10`
  - FrontierSmith synthetic `10` 当前作为训练数据，不作为 official eval 集。

## 28. 2026-06-19 16:39 EDT mixed-public 训练 prompt 长度检查

- 用 Qwen3-8B tokenizer 统计 `data/mixed/train_frontiercs172_frontiersmith10_alebench40.parquet` 的 chat-template 后 prompt 长度。
- 当前 `slurm/train_qwen3_8b_mixed_public_ailab.sh` 默认 `MAX_PROMPT_LENGTH=8192`。
- 统计结果：
  - `frontiercs172`: max `8131`, over8192 `0/172`
  - `frontiersmith10`: max `1841`, over8192 `0/10`
  - `alebench40`: max `8154`, over8192 `0/40`
  - overall max `8154`, over8192 `0/222`
- 结论：当前 Qwen3 mixed-public training 不会因为 `data.filter_overlong_prompts=True` 丢掉训练样本。

## 29. 2026-06-19 16:43 EDT Docker/Apptainer 依赖复核

- FrontierCS judge：
  - `scripts/start_frontiercs_judge_hybrid.sh` 使用本地 `.cache/bin/go-judge`。
  - Node judge API 用 `apptainer exec --cleanenv` 运行 `.cache/apptainer/frontiercs-judge.sif`。
  - 脚本只检查 SIF/二进制是否存在，不在 compute node 上 build。
- ALE-Bench：
  - `ALE-Bench/src/ale_bench/utils.py` 的 `docker_client()` 已支持 `ALE_BENCH_CONTAINER_BACKEND=apptainer`。
  - 在 apptainer 模式下，ALE 的原 docker client calls 会被 `_ApptainerClient` 接管，实际执行 `apptainer exec --cleanenv --no-home`。
  - Slurm train/eval scripts 均导出：
    - `ALE_BENCH_CONTAINER_BACKEND=apptainer`
    - `ALE_BENCH_APPTAINER_DIR=$PROJECT_ROOT/.cache/apptainer/alebench`
    - `ALE_BENCH_TOOL_CACHE=$PROJECT_ROOT/.cache/ale-bench/rust-tool-builds`
- ALE Rust tool cache 此前已验证 40 个 full problems 缺失数为 `0`。
- 结论：当前训练/评测链不依赖 compute node Docker build；FrontierCS 和 ALE 都走 Apptainer + 预构建/预缓存资产。

## 30. 2026-06-19 16:47 EDT submitter 环境变量显式导出

- 静态检查了 Qwen3 base/merge eval 与 downstream train/export/eval submitter。
- 之前 submitter 使用 `VAR=value sbatch ...`，依赖 Slurm 默认 `--export=ALL` 行为。
- 该写法此前已经被 smoke pipeline 使用并产出结果，说明当前集群默认环境导出可用；但为了之后重交更稳，已改成显式 `--export=ALL,...`：
  - `slurm/submit_qwen3_base_merge_full_eval.sh`
  - `slurm/submit_qwen3_base_merge_mixed_public_train_pipeline.sh`
- 显式导出的关键变量包括：
  - eval: `MODEL_PATH`, `MODEL_TAG`, `SERVED_MODEL_NAME`, `PORT_OFFSET`, `OUTPUT_DIR`
  - train: `MODEL_PATH`, `CKPT_DIR`, `ROLLOUT_DIR`, `PROJECT_NAME`, `EXPERIMENT_NAME`, `TOTAL_TRAINING_STEPS`, `SAVE_FREQ`, `TEST_FREQ`, `ROLLOUT_N`
  - export: `CHECKPOINT_ROOT`, `OUTPUT_DIR`
- 已通过 `bash -n` 和 `git diff --check` relevant files。
- 已排队的 `9979611-9979629` 不受这个 submitter 修改影响；它们仍依赖提交时 Slurm 默认导出。鉴于之前 smoke submitter 已成功使用同一机制，先不取消重交，除非后续状态/日志证明环境未传入。

## 31. 2026-06-19 16:50 EDT FrontierCS official prompt statement 一致性检查

- `scripts/eval_qwen35_base_vllm_request.py` 的 official prompt 路径：
  - system prompt / extractor / scorer 来自 `.cache/Frontier-CS-official`。
  - statement 文件读取顺序是本地 `Frontier-CS` 优先，然后 official cache。
- 检查 `data/frontiercs/full.parquet` 中 172 个 ground truth 对应的 `statement.txt`：
  - 本地 `Frontier-CS/algorithmic/problems/<id>/statement.txt`
  - official `.cache/Frontier-CS-official/algorithmic/problems/<id>/statement.txt`
- SHA256 比较结果：
  - same `172`
  - diff `0`
  - missing `0`
- 结论：当前 eval 使用本地 statement 不会改变 FrontierCS official prompt 内容。

## 32. 2026-06-19 16:54 EDT FrontierCS official problem 目录一致性检查

- 进一步比较了 172 个 FrontierCS official problem 的完整目录文件哈希：
  - local: `Frontier-CS/algorithmic/problems/<id>`
  - official cache: `.cache/Frontier-CS-official/algorithmic/problems/<id>`
  - 忽略：`node_modules`, `__pycache__`
- 结果：
  - `problem_dir_same=172`
  - `diff=0`
  - `missing=0`
- 结论：当前 local judge 对 172 个 official FrontierCS eval problems 的 generator/config/statement 与 official cache 一致；本地额外的 `frontiersmith_*` 只影响训练集 synthetic problems。

## 33. 2026-06-19 16:58 EDT Qwen3 base/merge full eval 并发限制与重交

- 资源策略复核：
  - 旧的 base/merge full eval `9979611-9979615` 会在同一依赖满足后同时启动，最多占 5 张 H200。
  - 用户明确要求 2-4 卡即可，不要浪费显存；因此不应让这 5 个 1-GPU eval 同时启动。
- 已修改 `slurm/submit_qwen3_base_merge_full_eval.sh`：
  - 新增 `MAX_PARALLEL=${MAX_PARALLEL:-2}`。
  - 使用滑动窗口依赖，默认最多 2 个 full eval 同时运行。
  - 同时保留外部 `DEPENDENCY`，即仍等 `9979106` 后再开始。
  - 已改为显式 `--export=ALL,...`。
- 已取消旧 pending 链：
  - full eval `9979611-9979615`
  - downstream train/export/eval `9979618-9979629`
- 已重交 full eval，依赖 `9979106` 且默认最多 2 并发：
  - `9979840`: `qwen3_8b`
  - `9979841`: `qwen3_8b_base`
  - `9979842`: `qwen3_8b_alpha0p25`
  - `9979843`: `qwen3_8b_alpha0p50`
  - `9979844`: `qwen3_8b_alpha0p75`
- 已重交 downstream mixed-public train/export/trained-eval，依赖上述 5 个 full eval 全部 afterok：
  - `qwen3_8b_base`: train `9979846`, export `9979847`, trained eval `9979848`
  - `qwen3_8b_alpha0p25`: train `9979849`, export `9979850`, trained eval `9979851`
  - `qwen3_8b_alpha0p50`: train `9979852`, export `9979853`, trained eval `9979854`
  - `qwen3_8b_alpha0p75`: train `9979855`, export `9979856`, trained eval `9979857`
- `squeue` 提交后校验：`9979840-9979857` 均为 `PENDING (Dependency)`。

## 34. 2026-06-19 17:01 EDT reward data_source 映射检查

- 检查 `verl/verl/utils/reward_score/__init__.py::default_compute_score`：
  - `data_source == "frontiercs"` -> `frontiercs.compute_score`
  - `data_source == "alebench"` -> `alebench.compute_score`
  - `data_source == "alebench_full"` -> `alebench_full.compute_score`
- 结论：
  - mixed training parquet 中 `frontiercs172` 和 `frontiersmith10` 都标为 `frontiercs`，会走 FrontierCS judge。
  - mixed training parquet 中 `alebench40` 标为 `alebench_full`，会走 full public ALE reward。
  - eval parquet 中 `alebench` 仍走 eval 版 `alebench.compute_score`，与训练 `alebench_full` 区分清楚。

## 35. 2026-06-19 17:06 EDT 结果汇总脚本与 Qwen3.5 no-thinking eval summary

- 新增结果汇总脚本：
  - `scripts/collect_reproduction_results.py`
  - 输出：`outputs/reproduction_results.csv`
  - 特点：expected matrix 中缺失的 summary 也会列出 `exists=False`，方便区分“没跑完”和“分数为 0”。
- 已运行：
  - `source .venv/bin/activate && python scripts/collect_reproduction_results.py`
- 当前 collector 读到的已完成结果：
  - `qwen35_9b_base`: `complete_problem_count=182`, FrontierCS best@5 `2.6976037`, ALE performance best@5 `400.8188`
  - `qwen35_9b_base_model`: `complete_problem_count=182`, FrontierCS best@5 `0`, ALE performance best@5 `286.5`
  - `qwen35_9b_mixed_no_thinking`: `complete_problem_count=182`, FrontierCS best@5 `0`, ALE performance best@5 `354.9387`
- `outputs/cc_eval_qwen35_9b_mixed_fixed_vllm/summary.json` 已落盘，说明 fixed Qwen3.5 mixed no-thinking eval 已完成。
- 进一步检查 `samples.jsonl`：
  - counts: `frontiercs=860`, `alebench=50`
  - errors: none
  - FrontierCS reward: `860/860` 为 `0`
  - ALE reward: `39/50` 为正，`11/50` 为 `0`
- 结论：Qwen3.5 mixed no-thinking 的 FrontierCS 全 0 不是 evaluator error，而是当前 no-thinking generations 没拿到分；thinking eval 仍是关键结果。
- 尚缺 summary：
  - Qwen3.5 mixed thinking eval
  - Qwen3-8B mixed-public trained eval
  - Qwen3/Qwen3-Base/merge base eval
  - Qwen3-Base/merge mixed-public trained eval

## 36. 2026-06-19 17:10 EDT Qwen3.5 mixed thinking eval partial file check

- 不查 Slurm，仅检查输出文件系统。
- `outputs/cc_eval_qwen35_9b_mixed_thinking_general_32k_both_vllm/shards/` 已开始写入 4 个 shard 的 `samples.jsonl`：
  - `shard_0`: 10 samples
  - `shard_1`: 8 samples
  - `shard_2`: 15 samples
  - `shard_3`: 14 samples
  - total: `47/910`
- 当前 partial samples：
  - 全部仍是 `frontiercs`
  - errors: `0`
  - positive rewards: `0`
- 结论：thinking eval 已开始写结果，但样本数太少，不能据此判断最终分数；等待 summary job 产出。

## 37. 2026-06-19 17:14 EDT collector partial sample 支持

- 更新 `scripts/collect_reproduction_results.py`：
  - 对 `summary.json` 尚未存在的 expected result，自动扫描：
    - `<output_dir>/samples.jsonl`
    - `<output_dir>/shards/shard_*/samples.jsonl`
  - 输出 `partial_sample_count` 和 `partial_error_count`。
- 已通过：
  - `python -m py_compile scripts/collect_reproduction_results.py`
  - `python scripts/collect_reproduction_results.py`
  - `git diff --check`
- 当前 collector 结果：
  - `qwen35_9b_mixed_thinking`: `exists=False`, `partial_sample_count=56`, `partial_error_count=0`
  - Qwen3/Qwen3-Base/merge 相关 rows 仍 `partial_sample_count=0`，说明尚未开始写对应输出文件。

## 38. 2026-06-19 17:18 EDT sharded summary metadata 兼容

- 发现 sharded eval summary 由 `scripts/summarize_base_eval_hf.py` 生成，历史版本只有 `samples_jsonl_files`，没有完整 `config`。
- 已修复 `scripts/summarize_base_eval_hf.py`：
  - 新增 `summary["config"]`，包含 `n_samples`, `seed`, `source`, `sources`。
  - 后续 pending summary jobs 运行时会自动带上这些 metadata。
- 已增强 `scripts/collect_reproduction_results.py`：
  - 兼容旧 sharded summary 的 `samples_jsonl_files`。
  - 如果缺 `config.n_samples`，从 metric key（如 `mean@5`/`best@5`）推断。
  - 如果缺 `config.source`，从 summary `metrics` 的 source keys 推断。
- 已通过：
  - `python -m py_compile scripts/summarize_base_eval_hf.py scripts/collect_reproduction_results.py`
  - `python scripts/collect_reproduction_results.py`
  - `git diff --check`
- 当前 collector 读到：
  - `qwen35_9b_mixed_thinking`: `exists=False`, `partial_sample_count=64`, `partial_error_count=0`

## 39. 2026-06-19 16:45 EDT 旧训练链取消与离线 ALE cache 保护

- 一次性状态脚本：
  - 新增 `scripts/reproduction_status_once.sh`
  - 只打印一次 `squeue`/`sacct`/collector，不循环、不 sleep。
  - 已通过 `bash -n`。
- 检查 `9967723` Qwen3.5 mixed training 日志：
  - 旧脚本实际是 `Total steps: 100`，4.5h 到取消前只到 step17，只有 `global_step_10` checkpoint。
  - 早期 ALE-Bench full reward 在 compute node 尝试 `cargo build`，因无法解析 `index.crates.io` 失败；FrontierCS reward 仍在跑，但混合训练里的 ALE reward 早期样本不可靠。
  - 本地 `scripts/prepare_alebench_tool_cache.py --no-lite --check-only` 当前确认 full ALE 40/40 Rust tool cache 都已存在。
- 已修复/加固：
  - `ALE-Bench/src/ale_bench/data.py` 支持 `ALE_BENCH_REQUIRE_TOOL_CACHE=1`：cache 未命中时直接 fail-fast，不再在 compute node 尝试联网 build。
  - `scripts/run_verl_grpo_frontiercs_qwen35_9b.sh`、`scripts/eval_base_model_qwen35_9b_vllm_request.sh`、`scripts/eval_base_model_qwen35_9b_hf.sh` 默认设置该保护。
  - `slurm/train_qwen35_9b_mixed_public_ailab.sh` 和 `slurm/train_qwen3_8b_mixed_public_ailab.sh` 启动前运行 `prepare_alebench_tool_cache.py --no-lite --check-only`。
- 训练默认值调整：
  - Qwen3.5 mixed：默认 `FRESH_START=0`，从 `global_step_10` 续训；`TOTAL_TRAINING_STEPS=20`，`SAVE_FREQ=5`，`ROLLOUT_N=4`，`MAX_RESPONSE_LENGTH=8192`，`MAX_MODEL_LEN=20480`。
  - Qwen3 mixed：默认 `FRESH_START=0`，`SAVE_FREQ=5`。
- 已取消旧的会被 `9967723` 拖住的链：
  - running/canceled：`9967723`
  - pending canceled：`9979103-9979106`, `9979840-9979844`, `9979846-9979857`
  - 保留正在运行的 step10 thinking eval：`9979101_[0-3]` 和 summary `9979102`。

## 40. 2026-06-19 16:45 EDT 新提交链

- Qwen3.5 mixed step20 continuation/eval：
  - `9980304`: continue train from latest checkpoint (`global_step_10` -> target step20), 4x H200, 2.5h limit.
  - `9980305`: CPU export latest checkpoint to `models/cc_qwen35_9b_mixed_hf_step20`.
  - `9980306_[0-3]`: 1-GPU sharded thinking eval on FrontierCS + ALE-Bench.
  - `9980307`: CPU summary.
- Qwen3/Qwen3-Base/linear-merge base eval:
  - `9980310`: `qwen3_8b`
  - `9980311`: `qwen3_8b_base`
  - `9980312`: `qwen3_8b_alpha0p25`
  - `9980313`: `qwen3_8b_alpha0p50`
  - `9980314`: `qwen3_8b_alpha0p75`
  - submitter still limits max concurrent evals to 2; each eval uses 1 H200.
- Qwen3-8B mixed-public train/export/eval:
  - `9980315 -> 9980316 -> 9980317`
  - train waits for base eval `9980310` to finish successfully.
- Qwen3-Base/linear-merge mixed-public train/export/eval:
  - `qwen3_8b_base`: `9980319 -> 9980320 -> 9980321`
  - `alpha0p25`: `9980322 -> 9980323 -> 9980324`
  - `alpha0p50`: `9980325 -> 9980326 -> 9980327`
  - `alpha0p75`: `9980328 -> 9980329 -> 9980330`
  - all wait for base eval jobs `9980310-9980314` to finish successfully; then train pipelines are serialized by the submitter.
- One-shot status after submission:
  - `9979101_[0-3]` running, at 108/910 partial samples, 0 partial errors.
  - `9980304`, `9980310`, `9980311` pending with reason `(None)`; remaining jobs are dependency-pending as expected.
- Verification:
  - `bash -n` passed for updated shell scripts.
  - `python -m py_compile` passed for `scripts/collect_reproduction_results.py` and `ALE-Bench/src/ale_bench/data.py`.
  - `python scripts/collect_reproduction_results.py --no-table` wrote `outputs/reproduction_results.csv`.
  - `git diff --check` passed.

## 41. 2026-06-19 16:50 EDT task framework inventory

- 新增 `docs/task_framework_inventory.md`：
  - 记录当前 `data_source` -> reward backend：
    - `frontiercs` -> FrontierCS judge
    - `alebench` -> ALE-Bench private eval/lite eval
    - `alebench_full` -> ALE-Bench full public eval for training
  - 记录当前 parquet inventory：
    - `frontiercs/full.parquet`: 172
    - `frontiercs/train.parquet`: 172
    - `frontiercs/train_synthetic.parquet`: 10
    - `alebench/val.parquet`: 10
    - `alebench_full/val.parquet`: 40
    - `mixed/train_frontiercs172_frontiersmith10_alebench40.parquet`: 222 (`frontiercs=182`, `alebench_full=40`)
  - 记录后续 ThetaEvolve/TTT-Discover task 接入接口：parquet builder、reward module、`default_compute_score` routing、offline preflight、summary schema。
- 修正 `slurm/submit_qwen3_base_merge_mixed_public_train_pipeline.sh`：
  - 默认 `SAVE_FREQ` 从 10 改为 5，避免短训练时只保存太少 checkpoint。
- 已通过：
  - `bash -n slurm/submit_qwen3_base_merge_mixed_public_train_pipeline.sh slurm/submit_qwen3_mixed_public_train_pipeline.sh slurm/submit_qwen35_mixed_public_continue_eval.sh`
  - `git diff --check`

## 42. 2026-06-19 17:16 EDT one-shot status after 30 min

- 按半小时等待后只运行一次 `scripts/reproduction_status_once.sh`。
- 当前运行：
  - `9979101_[0-3]`: Qwen3.5 mixed step10 thinking eval，运行约 51 min。
  - `9980310`: Qwen3-8B base eval，运行约 28 min，partial `259/910`，errors `0`。
  - `9980313`: Qwen3-8B alpha0p50 base eval，运行约 8 min，partial `143/910`，errors `0`。
- 当前完成：
  - `9980311`: Qwen3-8B-Base base eval 完成。
- 当前等待：
  - `9980304`: Qwen3.5 mixed step20 continuation train，pending `(Resources)`。
  - 其他 Qwen3/merge train/export/eval 均按依赖 pending。
- collector 当前读到：
  - `qwen35_9b_mixed_thinking`: partial `551/910`, errors `0`。
  - `qwen3_8b_base`: summary exists, complete `182`, scored samples `910`。
  - `qwen3_8b_alpha0p50`: partial `143/910`, errors `0`。
- `qwen3_8b_base` base eval 分数：
  - FrontierCS best@5 mean: `0`
  - ALE performance best@5 mean: `442.9527`
  - ALE performance oracle best@5: `517.0`
  - config: official FrontierCS prompt/scorer, thinking enabled, `max_tokens=24064`, `concurrency=64`, 1 H200。

## 43. 2026-06-19 17:50 EDT Qwen3.5 thinking eval shard failures handled

- 第二次半小时状态检查：
  - `9980304` Qwen3.5 step20 continuation train 已开始运行。
  - `9979101_2` 和 `9979101_3` 显示 `FAILED 2:0`。
  - collector partial: `qwen35_9b_mixed_thinking` 已到 `882/910`, errors `0`。
- 本地日志确认：
  - shard 2 写满 `225/225` samples。
  - shard 3 写满 `225/225` samples。
  - 失败发生在 eval 完成之后，shell 尾部报 `scripts/eval_base_model_qwen35_9b_vllm_request.sh: line 123: syntax error near unexpected token ')'`。
  - 当前工作树中的 `scripts/eval_base_model_qwen35_9b_vllm_request.sh` 已通过 `bash -n`，因此这是旧运行脚本版本造成的尾部 exit 2；样本文件本身没有 error。
- 当前 shard sample counts：
  - shard 0: `225/225`
  - shard 1: `224/225`（当时仍有任务运行）
  - shard 2: `225/225`
  - shard 3: `225/225`
- 已处理：
  - 取消旧的 `afterok` summary `9979102`。
  - 新提交 summary `9983981`，依赖 `afterany:9979101`，路径仍为 `outputs/cc_eval_qwen35_9b_mixed_thinking_general_32k_both_vllm`。
  - 更新 `scripts/reproduction_status_once.sh` 使用 `9983981`。

## 44. 2026-06-19 18:22 EDT FrontierCS/Smith status and next framework direction

- 一次性状态脚本：
  - `9983981` summary 已完成，`outputs/cc_eval_qwen35_9b_mixed_thinking_general_32k_both_vllm/summary.json` 已落盘。
  - Qwen3.5 mixed step10 thinking eval：`complete_problem_count=182`, `scored_sample_count=910`。
  - 分数：FrontierCS reward best@5 mean `7.4127665`; ALE performance best@5 mean `606.1506`。
- Qwen3.5 step20 continuation train：
  - `9980304` 正在运行，日志确认从 `global_step_10` 正确 resume。
  - 当前日志至少到 `training/global_step:16`；已保存 `global_step_15`，`latest_checkpointed_iteration.txt=15`。
  - 当前环境设置包含 `ALE_BENCH_REQUIRE_TOOL_CACHE=1`、本地 ALE data/cache、Apptainer backend；本轮日志未出现 cargo/index.crates.io 之类 compute-node 联网 build 失败。
  - 训练输出仍显示较高 response clipping（例如 step11 `0.6875`, step16 `0.46875`），但这是当前 8K training response cap 下的运行状态；eval 仍用 32K。
- Qwen3/Qwen3-Base/merge base eval：
  - `9980310` Qwen3-8B 完成：FrontierCS best@5 mean `0`, ALE best@5 mean `639.4286`。
  - `9980311` Qwen3-8B-Base 完成：FrontierCS best@5 mean `0`, ALE best@5 mean `442.9527`。
  - `9980313` alpha0p50 完成：FrontierCS best@5 mean `0`, ALE best@5 mean `577.8954`。
  - `9980312` alpha0p25 已启动，当前 `samples.jsonl` 已开始写入；`9980314` alpha0p75 仍待依赖/资源。
- 待完成链：
  - `9980304 -> 9980305 -> 9980306_[0-3] -> 9980307`: Qwen3.5 step20 train/export/eval/summary。
  - `9980315 -> 9980316 -> 9980317`: Qwen3-8B mixed-public train/export/eval。
  - `9980319/9980322/9980325/9980328` 等 Qwen3-Base/merge mixed-public train pipelines 等 base eval 全部完成后串行启动。
- 后续统一训练框架方向：
  - 先保持 VERL 路线，因为 FrontierCS + ALE-Bench 的 parquet/data_source/reward routing 已经跑通。
  - `docs/task_framework_inventory.md` 已记录当前 `frontiercs`、`alebench`、`alebench_full` 接口和后续 ThetaEvolve/TTT-Discover task 接入要求。
  - 等 FrontierCS/Smith 当前训练评测闭环完成后，再把 ThetaEvolve/TTT-Discover 的非硬件依赖任务按同一 parquet builder + reward module + offline preflight 接进来。

## 45. 2026-06-19 18:52 EDT one-shot low-frequency monitor

- 按要求先 sleep 1800 秒，然后只运行一次 `bash scripts/reproduction_status_once.sh`。
- Slurm 当前状态：
  - 旧 Qwen3.5 mixed train `9967723` 已被取消：`CANCELLED by 372967`，运行 `04:12:59`。
  - 当前 Qwen3.5 step20 continuation train `9980304` 正在运行，检查时 elapsed `01:08:42`。
  - Qwen3.5 step20 export/eval/summary 链 `9980305 -> 9980306_[0-3] -> 9980307` 仍按依赖 pending。
  - Qwen3-8B mixed-public train `9980315` 正在运行，检查时 elapsed `00:11:46`；后续 `9980316/9980317` pending。
  - Qwen3-Base/merge downstream trains `9980319/9980322/9980325/9980328` 及对应 export/eval 仍 pending。
- Qwen3.5 step20 checkpoint/log：
  - checkpoint root 已有 `global_step_20`。
  - `latest_checkpointed_iteration.txt=20`。
  - continuation log `logs/fs-q35-mixtrain-9980304.out` 记录 `local_global_step_folder: .../global_step_20`，并已写 `outputs/rollout_data_qwen35_9b_mixed_public/20.jsonl`。
  - log 里最新完整 train metric 行到 `training/global_step:19`，随后开始 step20 validation：`test_gen_batch ... global_steps: 20`。
- `scripts/reproduction_status_once.sh` 写出 `outputs/reproduction_results.csv`，当前 collector 表：
  - `qwen35_9b_mixed_thinking`: complete `182`, FrontierCS best@5 mean `7.4127665`, ALE performance best@5 mean `606.1506`。
  - `qwen35_9b_mixed_step20_thinking`: summary 尚不存在。
  - `qwen3_8b`: complete `182`, FrontierCS best@5 mean `0`, ALE performance best@5 mean `639.4286`。
  - `qwen3_8b_base`: complete `182`, FrontierCS best@5 mean `0`, ALE performance best@5 mean `442.9527`。
  - `qwen3_8b_alpha0p25`: partial `909/910`, errors `0`，summary 尚未完成。
  - `qwen3_8b_alpha0p50`: complete `182`, FrontierCS best@5 mean `0`, ALE performance best@5 mean `577.8954`。
  - `qwen3_8b_alpha0p75`: 尚无样本/summary。
- 新落盘/已确认的 Qwen3 merge/base eval summaries：
  - `outputs/eval_qwen3_8b_base_thinking_general_both_vllm/summary.json`
  - `outputs/eval_qwen3_8b_alpha0p50_thinking_general_both_vllm/summary.json`
  - `outputs/eval_qwen3_8b_thinking_general_both_vllm/summary.json`
  - `outputs/cc_eval_qwen35_9b_base_samepipe_vllm/summary.json`
- Qwen3.5 mixed thinking combined summary 已存在：
  - `outputs/cc_eval_qwen35_9b_mixed_thinking_general_32k_both_vllm/summary.json`
  - FrontierCS best@5 mean `7.4127665`; ALE performance best@5 mean `606.1506`。

## 46. 2026-06-19 18:56 EDT local confirmation after monitor

- 本地文件确认：
  - Qwen3.5 step20 continuation checkpoint root 已有 `global_step_10`, `global_step_15`, `global_step_20`。
  - `checkpoints/verl_frontiercs_qwen35_9b_mixed_public/qwen35_9b_grpo_mixed_public/latest_checkpointed_iteration.txt=20`。
  - `outputs/cc_eval_qwen35_9b_mixed_step20_thinking_general_32k_both_vllm/` 尚未出现；原因是 Slurm job `9980304` 仍在 validation/收尾，`9980305` export 还未触发。
- Qwen3-8B mixed-public training `9980315`：
  - 日志确认从 scratch 启动，`Total steps: 30`。
  - 已完成 `training/global_step:1`，写出 `outputs/rollout_data_qwen3_8b_mixed_public/1.jsonl`。
  - step1 `response_length/clip_ratio=0.9375`，说明当前 8K training response cap 下大量样本截断；eval 仍然独立用 32K。
- Qwen3 alpha0p25 base eval：
  - `outputs/eval_qwen3_8b_alpha0p25_thinking_general_both_vllm/samples.jsonl` 当前 `909` 行。
  - `summary.json` 尚未落盘；等待最后一个 sample/summary 收尾。

## 47. 2026-06-19 19:26 EDT Qwen3.5 step20 export failure fixed and resubmitted

- 19:24 低频状态快照：
  - Qwen3.5 step20 continuation train `9980304` 已完成：`COMPLETED`, elapsed `01:25:05`。
  - export job `9980305` 失败：`FAILED 1:0`，所以旧 eval array `9980306_[0-3]` 进入 `DependencyNeverSatisfied`。
  - Qwen3-8B mixed-public train `9980315` 正在运行，elapsed `00:44:04`。
  - Qwen3 alpha0p25 base eval `9980312` 已完成，collector：FrontierCS best@5 mean `0`, ALE best@5 mean `458.6261`。
  - Qwen3 alpha0p75 base eval `9980314` 正在运行，collector partial `486/910`, errors `0`。
- export failure root cause：
  - `logs/fs-export-hf-9980305.err` 报 `ModuleNotFoundError: No module named 'torch.distributed._mesh_layout'`。
  - 原因是 `slurm/export_verl_ckpt_to_hf_cpu.sh` 使用旧 `.venv` 的 torch `2.6.0+cu124`；当前训练环境 `.venv-vllm023` 的 torch `2.11.0+cu130` 可以 import `torch.distributed._mesh_layout`。
- 已修复：
  - `slurm/export_verl_ckpt_to_hf_cpu.sh` 默认改用 `VENV_DIR=$PROJECT_ROOT/.venv-vllm023`，并允许环境变量覆盖。
  - 新增 `slurm/submit_qwen35_step20_export_eval_only.sh`，只从现有 `global_step_20` 做 export/eval/summary，不重跑训练。
  - `scripts/reproduction_status_once.sh` job list 更新到新链。
  - `bash -n` 已通过：
    - `slurm/export_verl_ckpt_to_hf_cpu.sh`
    - `slurm/submit_qwen35_step20_export_eval_only.sh`
    - `slurm/eval_qwen35_9b_mixed_thinking_both_array_ailab.sh`
    - `slurm/summarize_qwen35_9b_mixed_thinking_both_cpu.sh`
- 已处理旧链并重新提交：
  - 取消旧 pending jobs：`9980306`, `9980307`。
  - 新链：`9988286 -> 9988287 -> 9988288`。
  - `9988286`: step20 HF export to `models/cc_qwen35_9b_mixed_hf_step20`。
  - `9988287`: step20 thinking eval array on FrontierCS + ALE-Bench。
  - `9988288`: summary, dependency uses `afterany` on eval array so completed samples can still be summarized if shard tail exits nonzero。

## 48. 2026-06-19 19:57 EDT new step20 eval running and Qwen3 base eval complete

- 按 30 分钟 sleep 后只运行一次状态脚本。
- Qwen3.5 step20 chain：
  - New export `9988286` 已完成：`COMPLETED`, elapsed `00:01:11`。
  - HF model 已落盘：`models/cc_qwen35_9b_mixed_hf_step20`, size about `18G`，包含 4 个 safetensors shards。
  - New eval array `9988287_[0-3]` 正在运行。
  - collector 当前 partial：`qwen35_9b_mixed_step20_thinking` `335/910`, errors `0`。
  - 本地 shard counts：
    - shard0: `144`
    - shard1: `92`
    - shard2: `109`
    - shard3: `13`
- Qwen3/merge base eval 全部完成：
  - `qwen3_8b`: FrontierCS best@5 `0`, ALE best@5 `639.4286`。
  - `qwen3_8b_base`: FrontierCS best@5 `0`, ALE best@5 `442.9527`。
  - `alpha0p25`: FrontierCS best@5 `0`, ALE best@5 `458.6261`。
  - `alpha0p50`: FrontierCS best@5 `0`, ALE best@5 `577.8954`。
  - `alpha0p75`: FrontierCS best@5 `0`, ALE best@5 `568.3454`。
- Qwen3-8B mixed-public train `9980315`：
  - 正在运行，状态脚本时 elapsed `01:16:47`。
  - 本地日志至少到 `training/global_step:13`。
  - 已保存 `global_step_5` 和 `global_step_10`; `latest_checkpointed_iteration.txt` 已存在。
  - 训练仍表现为高 response clipping，例如 step10 `response_length/clip_ratio=1.0`，step13 `0.6875`；当前先让 30-step run 完整跑完，再看 eval 结果决定是否调大 training response cap。
- Downstream：
  - `9980319` Qwen3-Base mixed train 已因全部 base eval 完成而解除依赖，目前 pending `(Resources)`。
  - 其他 merge train/export/eval 仍按串行依赖 pending。

## 49. 2026-06-19 20:28 EDT step20 eval near complete; Qwen3 and Qwen3-Base trainings running

- 按 30 分钟 sleep 后只运行一次状态脚本。
- Qwen3.5 step20 thinking eval:
  - `9988287_0`, `9988287_1`, `9988287_2` 已完成。
  - `9988287_3` 仍在运行。
  - collector 状态：`qwen35_9b_mixed_step20_thinking` partial `907/910`, errors `0`。
  - 本地 shard count 稍后确认到 `908/910`：
    - shard0: `230`
    - shard1: `230`
    - shard2: `225`
    - shard3: `223`
  - `9988288` summary 仍等 eval array 完成。
- Qwen3 trainings:
  - `9980315` Qwen3-8B mixed-public train 正在运行，status elapsed `01:47:37`。
  - 本地日志至少到 `training/global_step:13`，稍后本地 checkpoint tracker 已到 `15`；已保存 `global_step_5`, `global_step_10`, `global_step_15`。
  - `9980319` Qwen3-8B-Base mixed-public train 已启动，status elapsed `00:28:45`。
  - Qwen3-Base 训练确认使用 `models/Qwen3-8B-Base`，2 GPUs，从 scratch 启动，日志至少到 `training/global_step:6`。
  - Qwen3-Base 日志出现一次 ALE `ahc016` full public eval failure：`FileNotFoundError: .../a.out`。训练未中断；这是候选程序编译/产物失败路径，不是 compute-node 网络 build 问题。
- Resource usage:
  - `slurm/train_qwen3_8b_mixed_public_ailab.sh` 使用 `#SBATCH --gres=gpu:2`。
  - 当前两个 Qwen3 train jobs 在同一 H200 node 上运行，各 2 GPU，符合 2-4 卡约束。

## 50. 2026-06-19 20:59 EDT Qwen3.5 step20 eval complete; checkpoint appears worse than step10

- 按 30 分钟 sleep 后只运行一次状态脚本。
- Qwen3.5 step20 eval/export:
  - `9988287_[0-3]` 全部完成，exit `0:0`。
  - `9988288` summary 完成，exit `0:0`。
  - `outputs/cc_eval_qwen35_9b_mixed_step20_thinking_general_32k_both_vllm/summary.json` 已落盘。
  - complete `182`, scored samples `910`。
- Qwen3.5 step20 scores:
  - FrontierCS best@5 mean: `0.0`。
  - ALE performance best@5 mean: `383.1549`。
  - FrontierCS oracle best@5: `0.0`。
  - ALE oracle best@5: `423.0`。
- Compared with step10 thinking:
  - step10 FrontierCS best@5 mean `7.4128`, ALE best@5 mean `606.1506`。
  - step20 is much worse; treat step10 as the currently usable FrontierCS/Smith reproduction checkpoint.
- Export/model sanity checks:
  - step20 HF keys match canonical Qwen3.5 format (`model.language_model.layers...`), same as base and step10 fixed export.
  - vLLM loaded the step20 model without missing/unexpected parameter messages; model loading memory about `17.66 GiB` per shard job.
  - vLLM `EngineDeadError` messages appeared at server shutdown after completion; Slurm exit codes are all `0:0` and summary is complete.
- Output length diagnostics:
  - Step10 completion tokens: mean `23709.8`, median `32309.5`, max `32768`, `461/910` samples at `>=32000` tokens.
  - Step20 completion tokens: mean `11284.8`, median `3381`, max `32768`, `133/910` samples at `>=32000` tokens.
  - Current jsonl was run with `SAVE_TEXT=0`, so raw response text/repetition cannot be inspected from this eval; token statistics do not support a simple "all responses hit max due to repetition" explanation.
- Current Qwen3 train state from 20:59 status:
  - `9980315` Qwen3-8B mixed-public train running, elapsed `02:18:39`。
  - `9980319` Qwen3-8B-Base mixed-public train running, elapsed `00:59:47`。
  - Downstream export/eval chains still pending on these train jobs.

## 51. 2026-06-19 21:32 EDT training still running; unified-task direction

- 按低频检查节奏运行一次 `scripts/reproduction_status_once.sh`。
- Queue state:
  - `9980315` Qwen3-8B mixed-public train 仍在跑，elapsed `02:51:55`，2 GPU on `della-i21g1`。
  - `9980319` Qwen3-8B-Base mixed-public train 仍在跑，elapsed `01:33:03`，2 GPU on `della-i21g1`。
  - `9980316/9980317` 和 `9980320/9980321` export/eval 仍在 dependency pending。
  - alpha0p25/0p50/0p75 merge train/export/eval chains 仍在 dependency pending。
- Local checkpoint/log confirmation:
  - Qwen3-8B mixed-public latest checkpoint tracker: `25`，已有 `global_step_5/10/15/20/25`。
  - Qwen3-Base mixed-public latest checkpoint tracker: `20`，已有 `global_step_10/20`；日志已经到 step24。
  - Qwen3-8B train logs show high 8K response clipping for instruct model; Qwen3-Base logs mostly shorter responses with low clipping.
  - Qwen3-Base 日志里出现 ALE candidate compile artifact missing (`/tmp/.../a.out`)；reward wrapper catches it and training continues, not a compute-node network/cache failure.
- Current result matrix unchanged for completed evals:
  - Best usable FrontierSmith/CS reproduction remains Qwen3.5 mixed step10 thinking: FrontierCS best@5 `7.4128`, ALE best@5 `606.1506`。
  - Qwen3.5 mixed step20 thinking completed but degraded: FrontierCS `0.0`, ALE `383.1549`。
  - Qwen3/Qwen3-Base/merge base evals completed; all FrontierCS `0`, ALE varies, with Qwen3-8B base `639.4286` highest among those base evals.
- Framework direction:
  - Use VERL first for unified training because the current FrontierCS + ALE-Bench mixed parquet/data_source/reward routing is already working and checkpoint/export/eval scripts exist.
  - Slime remains a candidate later if ThetaEvolve-style GRPO/test-time RL integration needs it, but current priority is finishing FrontierSmith/FrontierCS train and eval end to end before expanding tasks.

## 52. 2026-06-19 22:03 EDT Qwen3-8B step25 fallback export/eval submitted

- 按 30 分钟 sleep 后只运行一次状态脚本。
- Queue state:
  - `9980315` Qwen3-8B mixed-public train 仍在跑，elapsed `03:22:50 / 04:00:00`。
  - `9980319` Qwen3-8B-Base mixed-public train 仍在跑，elapsed `02:03:58 / 04:00:00`。
  - Original downstream jobs `9980316/9980317` 和 `9980320/9980321` 仍在 dependency pending。
- Local train state:
  - Qwen3-8B latest tracker remains `25`；日志显示 step25 checkpoint 已写出，并进入 `test_gen_batch` validation。
  - Qwen3-8B 30-step job 在 4h walltime 内自然完成的风险较高，因为 step25 validation 已经占用较长时间，且还有 step26-30。
  - Qwen3-Base latest tracker remains `20`，但日志已推进到 step28；因为 `SAVE_FREQ=10`，预计 step30 才会更新 tracker。
- Added and submitted fallback:
  - 新增 `slurm/submit_qwen3_existing_ckpt_export_eval.sh`，用于对任意已写完的 VERL `global_step_N` 直接 export + standard 1-GPU Qwen3 eval。
  - `scripts/collect_reproduction_results.py` 增加 `qwen3_8b_mixed_public_step25` 结果行。
  - `scripts/reproduction_status_once.sh` 增加 fallback job ids。
  - Submitted Qwen3-8B step25 fallback chain:
    - export `9992316`: `global_step_25` -> `models/qwen3_8b_mixed_public_step25_hf`。
    - eval `9992317`: output `outputs/eval_qwen3_8b_mixed_public_step25_thinking_general_32k_both_vllm`。
- Rationale:
  - Keep original 30-step train/export/eval chain alive; if it completes, it gives the nominal result.
  - Step25 fallback prevents losing the already completed training progress if `9980315` hits walltime before step30.

## 53. 2026-06-19 22:35 EDT Qwen3 train fallback active; Qwen3-Base export resubmitted

- 按 30 分钟 sleep 后只运行一次状态脚本。
- Qwen3-8B mixed-public:
  - Original train `9980315` failed at `03:43:29`, exit `1:0`。
  - Failure happened during step25 validation: FrontierCS judge apptainer/node process aborted, Ray actor became unavailable.
  - It had already saved `global_step_25` before validation, so fallback chain is valid.
  - Fallback export `9992316` completed in `00:00:41` using current `.venv-vllm023` export path.
  - Fallback eval `9992317` is running on 1 GPU; collector partial at status time: `12` samples, errors `0`。
  - Original downstream `9980316/9980317` became dependency-never-satisfied and was cancelled.
- Qwen3-8B-Base mixed-public:
  - Train `9980319` completed at `02:24:38`, checkpoint tracker now `30`; `global_step_30` exists.
  - Old export `9980320` failed because that already-submitted Slurm script used old `.venv`, giving `ModuleNotFoundError: torch.distributed._mesh_layout`。
  - Submitted fallback step30 export/eval using current script:
    - export `9993214`: `global_step_30` -> `models/qwen3_8b_base_mixed_public_hf`。
    - eval `9993215`: output `outputs/eval_qwen3_8b_base_mixed_public_thinking_general_both_vllm`。
  - Old eval `9980321` became dependency-never-satisfied and was cancelled.
- Merge chain:
  - Old alpha merge chain `9980322-9980330` was blocked by old `9980321` and was cancelled.
  - `slurm/submit_qwen3_base_merge_mixed_public_train_pipeline.sh` now supports `SKIP_QWEN3_BASE=1`。
  - New merge-only chain depends on new Qwen3-Base eval `9993215`:
    - alpha0p25: `9993217 -> 9993218 -> 9993219`
    - alpha0p50: `9993220 -> 9993221 -> 9993222`
    - alpha0p75: `9993223 -> 9993224 -> 9993225`

## 54. 2026-06-19 23:07 EDT fallback evals running

- 按 30 分钟 sleep 后只运行一次状态脚本。
- Qwen3-8B mixed-public step25 fallback:
  - `9992316` export completed, elapsed `00:00:41`。
  - `9992317` eval running, elapsed `00:53:55`。
  - Collector partial: `425/910`, errors `0`。
- Qwen3-8B-Base mixed-public step30 fallback:
  - `9993214` export completed, elapsed `00:00:36`。
  - `9993215` eval running, elapsed `00:16:59`。
  - Collector partial: `909/910`, errors `0`。
- Merge chain:
  - `9993217` alpha0p25 train remains pending on successful completion of `9993215`。
  - alpha0p50/alpha0p75 chains remain pending behind alpha0p25 serial chain.

## 55. 2026-06-19 23:38 EDT Qwen3-Base trained eval complete; alpha0p25 training started

- 按 30 分钟 sleep 后只运行一次状态脚本。
- Qwen3-8B-Base mixed-public:
  - `9993215` eval completed, elapsed `00:19:44`。
  - `outputs/eval_qwen3_8b_base_mixed_public_thinking_general_both_vllm/summary.json` exists。
  - Complete problems `182`。
  - FrontierCS best@5 mean `0`。
  - ALE performance best@5 mean `374.9041`。
  - This is worse than Qwen3-8B-Base base eval ALE `442.9527`; no FrontierCS improvement.
- Qwen3-8B mixed-public step25 fallback:
  - `9992317` still running, elapsed `01:24:19`。
  - Collector partial: `766/910`, errors `0`。
- Merge chain:
  - `9993217` alpha0p25 mixed-public train started at `23:23:10`, elapsed `00:15:00` at snapshot.
  - alpha0p50 and alpha0p75 remain pending behind alpha0p25 train/export/eval.

## 56. 2026-06-20 00:08 EDT Qwen3 step25 eval complete; alpha0p25 at step10

- 按 30 分钟 sleep 后只运行一次状态脚本。
- Qwen3-8B mixed-public step25 fallback:
  - `9992317` eval completed, elapsed `01:42:43`。
  - `outputs/eval_qwen3_8b_mixed_public_step25_thinking_general_32k_both_vllm/summary.json` exists。
  - Complete problems `182`。
  - FrontierCS best@5 mean `0`。
  - ALE performance best@5 mean `736.5447`。
  - This improves over Qwen3-8B base eval ALE `639.4286`, but still gives no FrontierCS reward.
- Alpha0p25 mixed-public train:
  - `9993217` running, elapsed `00:45:24` at status snapshot.
  - Local checkpoint tracker: `10`，with `global_step_5` and `global_step_10` written.
  - Logs show normal training progress. One ALE candidate compile artifact miss (`/tmp/.../a.out`) was caught by reward code and did not stop training.
- Downstream:
  - `9993218/9993219` alpha0p25 export/eval pending on train completion.
  - alpha0p50 and alpha0p75 remain serially pending.

## 57. 2026-06-20 00:39 EDT alpha0p25 training progressing

- 按 30 分钟 sleep 后只运行一次状态脚本。
- Queue state:
  - `9993217` alpha0p25 train running, elapsed `01:16:00 / 04:00:00`。
  - `9993218/9993219` alpha0p25 export/eval pending。
  - alpha0p50 and alpha0p75 remain serially pending.
- Local train state:
  - alpha0p25 checkpoint tracker: `15`。
  - Existing checkpoints: `global_step_5`, `global_step_10`, `global_step_15`。
  - Log progress has reached step18, so at current pace step30 should fit within walltime.
- Result matrix unchanged since section 56.

## 58. 2026-06-20 01:09 EDT alpha0p25 at step20, logs near step25

- 按 30 分钟 sleep 后只运行一次状态脚本。
- Queue state:
  - `9993217` alpha0p25 train running, elapsed `01:46:39 / 04:00:00`。
  - Downstream alpha0p25 export/eval and alpha0p50/alpha0p75 chains remain pending.
- Local train state:
  - alpha0p25 checkpoint tracker: `20`。
  - Existing checkpoints: `global_step_5`, `global_step_10`, `global_step_15`, `global_step_20`。
  - Log progress has reached step24; next expected expensive point is step25 validation.
- Result matrix unchanged since section 56.

## 59. 2026-06-20 01:41 EDT alpha0p25 step25 fallback submitted

- Local post-status check:
  - alpha0p25 checkpoint tracker: `25`。
  - `global_step_25` exists.
  - Logs show step25 validation took a long time; this is the same failure-prone point where Qwen3-8B instruct train later failed.
- Action:
  - Added collector row `qwen3_8b_alpha0p25_mixed_public_step25`。
  - Submitted alpha0p25 step25 fallback export/eval:
    - export `9997403`: `global_step_25` -> `models/qwen3_8b_alpha0p25_mixed_public_step25_hf`。
    - eval `9997404`: output `outputs/eval_qwen3_8b_alpha0p25_mixed_public_step25_thinking_general_32k_both_vllm`。
  - Original step30 train/export/eval chain `9993217 -> 9993218 -> 9993219` remains alive.
- Checks:
  - `bash -n` passed for status/fallback scripts.
  - `python -m py_compile scripts/collect_reproduction_results.py` passed.
  - `git diff --check` passed for modified reproduction files.

## 60. 2026-06-20 02:13 EDT alpha0p25 step30 checkpoint protected

- Queue state:
  - `9993217` alpha0p25 train still running at `02:49:59 / 04:00:00`。
  - `9993218/9993219` nominal export/eval remain pending behind successful train exit.
  - alpha0p50 and alpha0p75 chains remain serially pending behind alpha0p25 eval.
- Local state:
  - `latest_checkpointed_iteration.txt` now reports `30`。
  - `global_step_30` exists under `checkpoints/verl_frontiercs_qwen3_8b_alpha0p25_mixed_public/qwen3_8b_alpha0p25_grpo_mixed_public/`。
- Completed fallback result since section 59:
  - alpha0p25 step25 fallback eval `9997404` completed, complete problems `182`。
  - FrontierCS best@5 mean `0`。
  - ALE performance best@5 mean `454.3255`。
- Action:
  - Added collector row `qwen3_8b_alpha0p25_mixed_public_step30`。
  - Submitted alpha0p25 step30 fallback export/eval:
    - export `9997748`: `global_step_30` -> `models/qwen3_8b_alpha0p25_mixed_public_step30_hf`。
    - eval `9997749`: output `outputs/eval_qwen3_8b_alpha0p25_mixed_public_step30_thinking_general_32k_both_vllm`。
  - Rationale: keep the nominal `9993217 -> 9993218 -> 9993219` chain alive, but avoid losing the final checkpoint if final validation exits nonzero.

## 61. 2026-06-20 02:44 EDT alpha0p25 complete; alpha0p50/0p75 resubmitted with port isolation

- After 30-minute sleep, one-shot status showed:
  - alpha0p25 train `9993217` completed, elapsed `03:04:40`。
  - nominal export `9993218` completed.
  - nominal eval `9993219` failed in `00:00:32`。
  - step30 fallback export/eval `9997748 -> 9997749` completed.
- alpha0p25 step30 fallback result:
  - Complete problems `182`。
  - FrontierCS best@5 mean `0`。
  - ALE performance best@5 mean `485.4449`。
  - This is the valid alpha0p25 step30 result; the nominal `9993219` result is invalid.
- Root cause for nominal eval failure:
  - `9993219` landed on the same node/ports as fallback eval `9997749`。
  - Its `/v1/models` check hit the fallback vLLM server, which served `qwen3_8b_alpha0p25_mixed_public_step30`。
  - Requests using `qwen3_8b_alpha0p25_mixed_public` all returned 404, so the summary was all-zero and should not be used.
- Fix:
  - `slurm/eval_qwen3_both_thinking_1gpu_ailab.sh` now defaults `PORT_OFFSET` from `SLURM_JOB_ID % 10000` and verifies that `/v1/models` contains the requested `SERVED_MODEL_NAME` before evaluation.
  - `slurm/train_qwen3_8b_mixed_public_ailab.sh` and `slurm/train_qwen35_9b_mixed_public_ailab.sh` now also default judge ports from the Slurm job id.
  - Collector marks the nominal alpha0p25 row as `trained_eval_invalid_port_collision`; the step30 fallback row is the valid result.
- Action:
  - Cancelled old blocked alpha0p50/0p75 jobs `9993220-9993225`。
  - Resubmitted remaining merge-chain jobs behind successful `9997749`:
    - alpha0p50: `9998167 -> 9998168 -> 9998169`
    - alpha0p75: `9998170 -> 9998171 -> 9998172`

## 62. 2026-06-20 03:17 EDT alpha0p50 waiting for resources

- After 30-minute sleep, one-shot status showed:
  - `9998167` alpha0p50 train pending with reason `Resources`。
  - `9998168/9998169` pending on alpha0p50 train.
  - `9998170/9998171/9998172` pending serially behind alpha0p50 eval.
- Result matrix unchanged:
  - Best FrontierCS reproduction remains Qwen3.5 mixed step10 thinking: FrontierCS best@5 `7.4127665`, ALE best@5 `606.1506`。
  - Qwen3-8B mixed-public step25 fallback remains best ALE among Qwen3 trained runs so far: ALE best@5 `736.5447`, FrontierCS `0`。
  - alpha0p25 valid step30 fallback: FrontierCS `0`, ALE best@5 `485.4449`。

## 63. 2026-06-20 03:48 EDT alpha0p50 training running

- After 30-minute sleep, one-shot status showed:
  - `9998167` alpha0p50 train running on `della-i20g2`, elapsed `00:25:08 / 04:00:00`。
  - `9998168/9998169` pending behind alpha0p50 train.
  - `9998170/9998171/9998172` pending serially behind alpha0p50 eval.
- Result matrix unchanged since section 62.

## 64. 2026-06-20 04:18 EDT alpha0p50 training still running

- After 30-minute sleep, one-shot status showed:
  - `9998167` alpha0p50 train running on `della-i20g2`, elapsed `00:55:26 / 04:00:00`。
  - `9998168/9998169` pending behind alpha0p50 train.
  - `9998170/9998171/9998172` pending serially behind alpha0p50 eval.
- No new eval results since section 63.

## 65. 2026-06-20 04:48 EDT alpha0p50 still training

- After 30-minute sleep, one-shot status showed:
  - `9998167` alpha0p50 train running on `della-i20g2`, elapsed `01:25:47 / 04:00:00`。
  - `9998168/9998169` pending behind alpha0p50 train.
  - `9998170/9998171/9998172` pending serially behind alpha0p50 eval.
- No new eval results since section 64.

## 66. 2026-06-20 05:19 EDT alpha0p50 still training

- After 30-minute sleep, one-shot status showed:
  - `9998167` alpha0p50 train running on `della-i20g2`, elapsed `01:56:07 / 04:00:00`。
  - `9998168/9998169` pending behind alpha0p50 train.
  - `9998170/9998171/9998172` pending serially behind alpha0p50 eval.
- No new eval results since section 65.

## 67. 2026-06-20 05:49 EDT alpha0p50 step25 fallback submitted

- After 30-minute sleep, one-shot status showed:
  - `9998167` alpha0p50 train running on `della-i20g2`, elapsed `02:26:24 / 04:00:00`。
  - `9998168/9998169` pending behind alpha0p50 train.
  - `9998170/9998171/9998172` pending serially behind alpha0p50 eval.
- Local checkpoint/log check:
  - `global_step_25` exists under `checkpoints/verl_frontiercs_qwen3_8b_alpha0p50_mixed_public/qwen3_8b_alpha0p50_grpo_mixed_public/`。
  - Logs show `test_gen_batch ... validate ... global_steps: 25`, so train is in the known failure-prone step25 validation point.
- Action:
  - Added collector row `qwen3_8b_alpha0p50_mixed_public_step25`。
  - Submitted alpha0p50 step25 fallback export/eval:
    - export `10002286`: `global_step_25` -> `models/qwen3_8b_alpha0p50_mixed_public_step25_hf`。
    - eval `10002287`: output `outputs/eval_qwen3_8b_alpha0p50_mixed_public_step25_thinking_general_32k_both_vllm`。
  - Original alpha0p50 train/export/eval chain `9998167 -> 9998168 -> 9998169` remains alive.

## 68. 2026-06-20 06:20 EDT alpha0p50 step25 fallback eval running

- After 30-minute sleep, one-shot status showed:
  - `9998167` alpha0p50 train still running on `della-i20g2`, elapsed `02:57:27 / 04:00:00`。
  - `10002286` alpha0p50 step25 fallback export completed in `00:00:38`。
  - `10002287` alpha0p50 step25 fallback eval running on `della-i20g2`, elapsed `00:29:06`。
  - Collector partial for `qwen3_8b_alpha0p50_mixed_public_step25`: `502/910`, errors `0`。
- The port-isolated eval script is working: fallback eval and train are on the same node without the previous wrong-model 404 failure.

## 69. 2026-06-20 06:51 EDT alpha0p50 fallback almost complete

- After 30-minute sleep, one-shot status showed:
  - `9998167` alpha0p50 train still running on `della-i20g2`, elapsed `03:27:50 / 04:00:00`。
  - `10002287` alpha0p50 step25 fallback eval running on `della-i20g2`, elapsed `00:59:29`。
  - Collector partial for `qwen3_8b_alpha0p50_mixed_public_step25`: `909/910`, errors `0`。
- No final alpha0p50 score yet at this snapshot.

## 70. 2026-06-20 07:24 EDT alpha0p50 step25 done; step30 and alpha0p75 resubmitted

- One-shot status showed:
  - `9998167` alpha0p50 train completed in `03:54:49`。
  - Nominal export `9998168` failed in `00:00:43`; `9998169/9998170/9998171/9998172` were stuck behind the failed dependency.
  - Step25 fallback export/eval `10002286 -> 10002287` completed.
- alpha0p50 step25 fallback result:
  - Complete problems `182`。
  - FrontierCS best@5 mean `0`。
  - ALE performance best@5 mean `437.82`。
- Root cause for nominal export failure:
  - `scripts/merge_fsdp_to_hf.py` failed while writing safetensors with `Disk quota exceeded (os error 122)`。
  - The checkpoint itself is usable; `global_step_30` exists.
- Action:
  - Cleaned only generated smoke/failed-export artifacts to recover quota:
    - `checkpoints/*_smoke` and old `checkpoints/Qwen3-8B_baseline` smoke checkpoints.
    - smoke HF exports and the partial `models/qwen3_8b_alpha0p50_mixed_public_hf` directory.
  - Cancelled blocked old jobs `9998169 9998170 9998171 9998172`。
  - Added collector row `qwen3_8b_alpha0p50_mixed_public_step30`。
  - Submitted alpha0p50 step30 fallback:
    - export `10004498`: `global_step_30` -> `models/qwen3_8b_alpha0p50_mixed_public_step30_hf`。
    - eval `10004499`: output `outputs/eval_qwen3_8b_alpha0p50_mixed_public_step30_thinking_general_32k_both_vllm`。
  - Resubmitted alpha0p75 full train/export/eval:
    - train `10004506`。
    - export `10004507`。
    - eval `10004508`。
- Framework direction:
  - Keep using VERL first for the merged training harness because the current parquet `data_source` and reward routing already support FrontierCS/Smith + ALE.
  - Next integration target is to wrap additional evaluator-style tasks behind the same data/reward contract before considering Slime.

## 71. 2026-06-20 07:55 EDT alpha0p50 step30 eval running; alpha0p75 training running

- After 30-minute sleep, one-shot status showed:
  - `10004498` alpha0p50 step30 fallback export completed in `00:00:42`。
  - `10004499` alpha0p50 step30 fallback eval running on `della-i21g2`, elapsed `00:30:17`。
  - Collector partial for `qwen3_8b_alpha0p50_mixed_public_step30`: `565/910`, errors `0`。
  - `10004506` alpha0p75 train running on `della-i23g1`, elapsed `00:26:41 / 04:00:00`。
  - `10004507/10004508` pending behind alpha0p75 train.
- No new final score yet; step30 eval is healthy so far.

## 72. 2026-06-20 08:26 EDT alpha0p50 step30 eval complete

- After 30-minute sleep, one-shot status showed:
  - `10004499` alpha0p50 step30 fallback eval completed in `00:51:44`。
  - `10004506` alpha0p75 train running on `della-i23g1`, elapsed `00:57:02 / 04:00:00`。
  - `10004507/10004508` pending behind alpha0p75 train.
- alpha0p50 step30 fallback result:
  - Complete problems `182`。
  - FrontierCS best@5 mean `0`。
  - ALE performance best@5 mean `438.9584`。
- Comparison:
  - alpha0p50 base ALE was `577.8954`。
  - alpha0p50 step25 fallback ALE was `437.82`。
  - step30 did not recover FrontierCS and did not improve ALE meaningfully over step25.

## 73. 2026-06-20 08:56 EDT alpha0p75 training still running

- After 30-minute sleep, one-shot status showed:
  - `10004506` alpha0p75 train running on `della-i23g1`, elapsed `01:27:28 / 04:00:00`。
  - `10004507/10004508` remain pending behind the training job.
- Result matrix unchanged since section 72.

## 74. 2026-06-20 09:26 EDT alpha0p75 training still running

- After 30-minute sleep, one-shot status showed:
  - `10004506` alpha0p75 train running on `della-i23g1`, elapsed `01:57:50 / 04:00:00`。
  - `10004507/10004508` remain pending behind the training job.
- Result matrix unchanged since section 72.

## 75. 2026-06-20 09:57 EDT alpha0p75 step25 fallback submitted

- After 30-minute sleep, one-shot status showed:
  - `10004506` alpha0p75 train running on `della-i23g1`, elapsed `02:28:10 / 04:00:00`。
  - `10004507/10004508` remain pending behind the training job.
- Local checkpoint/log check:
  - `global_step_25` exists under `checkpoints/verl_frontiercs_qwen3_8b_alpha0p75_mixed_public/qwen3_8b_alpha0p75_grpo_mixed_public/`。
  - Logs show `test_gen_batch ... validate ... global_steps: 25`, so the train is in the same step25 validation point seen in previous runs.
- Action:
  - Added collector row `qwen3_8b_alpha0p75_mixed_public_step25`。
  - Submitted alpha0p75 step25 fallback export/eval:
    - export `10007245`: `global_step_25` -> `models/qwen3_8b_alpha0p75_mixed_public_step25_hf`。
    - eval `10007246`: output `outputs/eval_qwen3_8b_alpha0p75_mixed_public_step25_thinking_general_32k_both_vllm`。
  - Original alpha0p75 train/export/eval chain `10004506 -> 10004507 -> 10004508` remains alive.

## 76. 2026-06-20 10:28 EDT alpha0p75 step25 fallback eval running

- After 30-minute sleep, one-shot status showed:
  - `10004506` alpha0p75 train running on `della-i23g1`, elapsed `02:59:35 / 04:00:00`。
  - `10007245` alpha0p75 step25 fallback export completed in `00:00:49`。
  - `10007246` alpha0p75 step25 fallback eval running on `della-i21g2`, elapsed `00:29:22`。
  - Collector partial for `qwen3_8b_alpha0p75_mixed_public_step25`: `514/910`, errors `0`。
  - Nominal alpha0p75 export/eval `10004507/10004508` remain pending behind the train.
- No new final alpha0p75 score yet.

## 77. 2026-06-20 10:59 EDT alpha0p75 step25 fallback complete

- After 30-minute sleep, one-shot status showed:
  - `10007246` alpha0p75 step25 fallback eval completed in `00:54:00`。
  - `10004506` alpha0p75 train still running on `della-i23g1`, elapsed `03:30:22 / 04:00:00`。
  - Nominal alpha0p75 export/eval `10004507/10004508` remain pending behind the train.
- alpha0p75 step25 fallback result:
  - Complete problems `182`。
  - FrontierCS best@5 mean `0`。
  - ALE performance best@5 mean `507.5497`。
- Current risk:
  - The nominal train is close to the 4-hour walltime; if it exits nonzero after writing `global_step_30`, submit a step30 fallback export/eval.

## 78. 2026-06-20 11:29 EDT alpha0p75 step30 fallback submitted

- User interrupted the local sleep to ask for status; this did not affect Slurm jobs.
- One-shot status showed:
  - `10004506` alpha0p75 train still listed as running on `della-i23g1`, elapsed `04:00:06 / 04:00:00`。
  - Nominal alpha0p75 export/eval `10004507/10004508` still pending behind the train.
  - `10007246` alpha0p75 step25 fallback eval completed: FrontierCS best@5 `0`, ALE best@5 `507.5497`。
- Local checkpoint/log check:
  - `global_step_30` exists.
  - Logs show the job wrote `global_step_30` and entered validation at `global_steps: 30` near the walltime.
- Action:
  - Killed the stale local `sleep 1800` process left from the interrupted turn.
  - Added collector row `qwen3_8b_alpha0p75_mixed_public_step30`。
  - Submitted alpha0p75 step30 fallback export/eval:
    - export `10009711`: `global_step_30` -> `models/qwen3_8b_alpha0p75_mixed_public_step30_hf`。
    - eval `10009712`: output `outputs/eval_qwen3_8b_alpha0p75_mixed_public_step30_thinking_general_32k_both_vllm`。

## 79. 2026-06-20 11:31 EDT alpha0p75 duplicate nominal chain cancelled

- Follow-up status showed:
  - `10009711` alpha0p75 step30 fallback export completed in `00:00:44`。
  - `10009712` alpha0p75 step30 fallback eval pending for resources.
  - Original train `10004506` was still listed as running at `04:02:03 / 04:00:00` after writing `global_step_30`。
- Action:
  - Cancelled original nominal chain `10004506 10004507 10004508` to avoid wasting H200 time.
  - Kept the step30 fallback eval `10009712` as the valid final alpha0p75 step30 path.

## 80. 2026-06-20 11:32 EDT alpha0p75 step30 eval running

- One-shot status showed:
  - `10004506/10004507/10004508` cancelled successfully.
  - `10009711` step30 fallback export completed.
  - `10009712` step30 fallback eval started on `della-i19g3`。
- Result pending; next check should wait approximately 30 minutes.

## 81. 2026-06-20 12:36 EDT alpha0p75 step30 eval complete

- After two 30-minute sleep intervals, one-shot status showed:
  - `10009712` alpha0p75 step30 fallback eval completed in `00:58:01` on `della-i19g3`。
  - No active Slurm jobs remain for the current Qwen3/merge reproduction chain.
- alpha0p75 step30 fallback result:
  - Output: `outputs/eval_qwen3_8b_alpha0p75_mixed_public_step30_thinking_general_32k_both_vllm`。
  - Complete problems `182`。
  - Scored samples `910`。
  - FrontierCS best@5 mean `0`。
  - ALE performance best@5 mean `454.3301`。
  - ALE oracle best@5 `463.1`。
- Updated collector output:
  - `outputs/reproduction_results.csv` now includes `qwen3_8b_alpha0p75_mixed_public_step30` as a completed fallback eval row.
- Current Qwen3/merge conclusion:
  - FrontierCS remains `0` for Qwen3-8B, Qwen3-8B-Base, and alpha `0.25/0.50/0.75` merge variants, before and after public mixed RL training.
  - ALE shows the clearest training gain for Qwen3-8B instruct: `639.4286 -> 736.5447` best@5.
  - Merge-trained variants do not beat their corresponding base evals on ALE in the current public-data run.

## 82. 2026-06-20 12:39 EDT cancelled invalid Qwen3.5 base-thinking reference eval

- A separate Qwen3.5 base-thinking reference array was found running after the Qwen3/merge chain completed:
  - array `10010302` using `slurm/eval_qwen35_9b_mixed_thinking_both_array_ailab.sh`。
  - dependent summary `10010303`。
  - model path `models/Qwen3.5-9B`。
  - output base `outputs/cc_eval_qwen35_9b_BASE_thinking_general_32k_both_vllm`。
- This run was not valid enough to keep consuming H200 time:
  - shard 1 exited immediately because go-judge failed to create `gojudge.scope`。
  - remaining shards were still running with many samples hitting `32768` generated tokens and FrontierCS reward `0`。
- Action:
  - Cancelled `10010302` and `10010303`。
  - Follow-up `squeue` showed no remaining Qwen/Frontier Slurm jobs.
  - Existing partial logs/samples were left in place for diagnosis, but this output should not be treated as a completed eval.
