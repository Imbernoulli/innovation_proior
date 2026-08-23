# FrontierSmith 训练与评测集成交接

本分支把 `bl3615` 在 `/scratch/gpfs/CHIJ/bohan/fs` 下的最终训练/评测工作整理进
`innovation_proior`，但没有把整个 scratch 工作目录原样上传。完整 `fs/` 里有模型、虚拟环境、
checkpoint、W&B、Slurm 日志、评测逐样本输出、parquet/raw jsonl 和若干无权限 runtime workspace；
这些都不适合进入 GitHub。

## 路径推断

- `/home/bl3615` 不可读，且 `/scratch/gpfs/CHIJ/bl3615` 下没有目标 `fs/`。
- `/scratch/gpfs/CHIJ/bohan` 属于 `bl3615:chij`，可读可执行；其中
  `/scratch/gpfs/CHIJ/bohan/fs` 是实际工作目录。
- 本次使用的两个源工作树：
  - `/scratch/gpfs/CHIJ/bohan/fs/innovation_prior`：目标仓库的本地副本，包含网站、SFT 数据和
    `frontiersmith_synth`。
  - `/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith`：训练、RL、VERL 修改、serve/eval 分离评测代码。

## 已纳入本分支

- `training/FrontierSmith/`：清理后的 FrontierSmith 源码、脚本、slurm 入口、文档、VERL fork、
  ALE-Bench 适配、Frontier-CS judge/src 代码、NatureBench harness/repo 代码和小型 metadata。
- `training/FrontierSmith/results/reproduction_results.csv`：从 `outputs/` 中抽出的轻量结果矩阵。
- `sft/split_r1.py`、`sft/split_r2.py`：源仓库中未跟踪的小型 SFT 拆分脚本。
- `frontiersmith_synth/scratch_tune.py`：源仓库中未跟踪的小型合成题调参工具。

未纳入：`models*`、`checkpoints/`、`.venv*`、`.cache/`、`outputs/`、`logs/`、`wandb/`、
`sft/*.jsonl` 原始大文件、`sft/*.bak*`、`sft/r1/`、`sft/r2/`、parquet、逐样本 JSONL、core dump、
Claude 原始 session/history。

## Claude Code 历史使用情况

原始 Claude Code transcript 不可直接读取：`/scratch/gpfs/CHIJ/bohan/.claude/projects`、
`.claude/sessions` 和 `history.jsonl` 对当前用户没有读权限。因此本分支没有、也不应该上传原始
Claude session。

可读并用于理解项目的来源是：

- `training/FrontierSmith/docs/INNOVATION_DATA_POLISH_HANDOFF_zh.md`：一次大型 session 留下的中文
  handoff，概括了数据问题诊断、wave2/wave3 的补救方向和最终 system prompt 决策。
- `/scratch/gpfs/CHIJ/bohan/.claude/file-history/` 中可读的 file snapshots：用于核对
  `ray_trainer.py`、`agent_loop.py`、`slurm/cc_rl_multisource.sh` 等关键文件的演化脉络；未上传原始快照。
- `FrontierSmith` 本地 git 历史：最近提交集中在 vLLM penalty fast path、adaptive deepening、
  overlong filtering、rlv10 watchdog、eval pipeline offload/pinning 等最终修改。

该 handoff 的核心结论：rlv12 四臂 RL 后，两个 soup 臂在 FCS 上落后 base；问题不是缺长思考，而是
缺“遇到困难后果断交付”的真实示范。后续处方是恢复 gated 清理误删的验证语言、从 wave2/wave3
筛长且能交卷的真实切片、清理 expert-CP 代码语域噪声、继续保留/回放 maintain 数据中的代码能力。
2026-08-18 用户裁定 system prompt 只保留时间句：训练用方法真实年份，RL/评测用
`It is now year 2026.`；去掉人设和交付条款。2026-08-19 还发现 6 个新轨迹未登记年份，导致下游曾出现
`It is now year None.`，训练侧已在临时下游文件修正，源注册表仍需避免重建复现。

## SFT 最终状态

SFT 详情以 `sft/README.md` 为准。最终可训练包是 gzipped 数据：

- `sft/innovation_sft.jsonl.gz`
- `sft/innovation_wave2_sft.jsonl.gz`
- `sft/innovation_v4_sft.jsonl.gz`
- `sft/innovation_wave2_raw_keepers.jsonl.gz`
- `sft/innovation_wave3_sft.jsonl.gz`

`innovation_wave3_sft.jsonl.gz` 对应 2026-08-20 final snapshot：5,291 条 verified keeper，覆盖 wave2
之后所有新解出的 query，并携带 `pass_rate`。原始 `innovation_wave3_sft.jsonl` 是 246MB，不提交；gzip
版本约 90.5MB，低于 GitHub 单文件限制。

## RL 训练入口

最终 multisource GRPO 入口：

- `training/FrontierSmith/slurm/cc_rl_multisource.sh`

它把 `frontiersmith_synth`、FrontierCS research、MLS-Bench RL 合到同一训练 parquet；按每行的
`agent_name` 路由：

- `single_turn_agent`：synth + research，reward 走 RewardLoopWorker 和 `data_source` dispatch。
- `mlsbench_agent`：MLS 行，reward 在 agent loop 内直接生成。
- `FS_PERTASK_REWARD_NORM=1`：把 0-100 单轮 scorer 归一到 MLS 的 `[0,1]` 尺度。

关键代码和脚本：

- `training/FrontierSmith/scripts/prepare_multisource_rl_parquet.py`
- `training/FrontierSmith/scripts/check_multisource_reward_routing.py`
- `training/FrontierSmith/verl/verl/trainer/ppo/adaptive_sampling.py`
- `training/FrontierSmith/verl/verl/utils/reward_score/overlong_penalty.py`
- `training/FrontierSmith/verl/verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- `training/FrontierSmith/scripts/vllm_penalty_fastpath.py`
- `training/FrontierSmith/scripts/apply_vllm_penalty_fastpath.sh`

vLLM penalty fast path 是本轮 RL 性能关键点：presence penalty 让 vLLM stock path 每步重建
`[batch, max_len]` int64 tensor；本地测到 B=128、L=32768 时约 162ms/step。fast path 只增量写新 token，
设计上保持 bit-identical，并通过 `FS_VLLM_PENALTY_FASTPATH=1` 开启。

overlong/adaptive 相关默认值已经在 launcher 中写清楚：DAPO soft overlong penalty、可选
`FS_OVERLONG_FILTER`、`LOSS_AGG_MODE=seq-mean-token-mean`、adaptive group deepening/requeue 等。
这些修改主要为了解决 flat/dead GRPO groups、truncation-driven gradients 和长序列梯度放大。

## 评测 serve/eval 分离

最终评测架构把 GPU serve 和 CPU scoring/client 拆开：

- GPU serve：
  - `training/FrontierSmith/scripts/vllm_pool_serve.sh`
  - `training/FrontierSmith/slurm/cc_serve_only.sh`
- CPU client：
  - `training/FrontierSmith/scripts/submit_pool_clients.sh`
  - `training/FrontierSmith/slurm/cc_eval_cpu_client.sh`
  - `training/FrontierSmith/slurm/cc_eval_cpu_client_pinned.sh`
- 一键拆分提交：
  - `training/FrontierSmith/slurm/cc_eval_split_submit.sh`
- 重提交/补齐：
  - `training/FrontierSmith/scripts/resubmit_decoupled_evals.sh`

服务端在共享 GPFS registry 写入 `<node>:<port>`；client 轮询 registry 并直连 compute node。这样避免每个评测
job 各自加载 vLLM，也让打分/评测驱动跑在 CPU partition。`cc_eval_split_submit.sh` 的默认策略是
`client_first`：client 先排队，serve job 等第一个 client 开始后再拿 GPU，减少 GPU 空等。

评测覆盖入口包括 FrontierCS algorithm/research、ALE-Bench、MLS-Bench、ThetaEvolve/TTT、NatureBench
相关 harness。大型逐样本 `samples.jsonl` 和原始 `outputs/` 没有提交，只保留可读的矩阵汇总。

## 结果整理

轻量矩阵：

- `training/FrontierSmith/results/reproduction_results.csv`
- `training/FrontierSmith/results/README.md`

该 CSV 有 22 行，其中 19 行已有完整结果。注意 `trained_eval_invalid_port_collision` 和 `exists=False`
的行不应作为最终数值引用；它们保留是为了说明哪些评测未完成或结果无效。

## 复现提醒

这些脚本保留了 Princeton/GPFS 环境下的真实最终版本，部分默认路径仍指向
`/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith`、`models_rl`、`models_sft` 或外部官方数据集。迁移到新机器时，
优先通过脚本中的 env vars 覆盖路径，不要直接改历史脚本语义；若要提交 portability patch，应单独做一层
wrapper。
