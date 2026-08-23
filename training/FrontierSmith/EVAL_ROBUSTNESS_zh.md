# 评测鲁棒性守则(EVAL_ROBUSTNESS)

2026-08-03 定稿。针对两类系统性噪声:**评测器自身随机性**(evaluator-side
randomness)与**硬件/墙钟敏感性**(node-speed / wall-time sensitivity)。
相关实现:`scripts/measure_eval_variance.py`、`scripts/frontiercs_research_eval.py`、
`scripts/frontiercs_research_cpu_eval.py`、`scripts/gojudge_shim.py`。
官方 evaluator.py 与 Frontier-CS judge 内部**一律未改**。

## 1. 铁律(引用任何分数前先过这一节)

1. **FCS / ALE 评测钉死在 ailab 分区**(H200 宿主,EPYC)。不同 CPU 代际会把
   时限边缘的解在 TLE 两侧翻转(实测:同一份代码旧 Xeon 1.44s vs ailab EPYC
   <1.0s 过题)。跨分区跑出的 FCS/ALE 分数不可与 ailab 基线同表比较。
2. **低于噪声底的差值一律不许引用为结论**:
   - FCS(172 题):Δ < **1.0** 不可区分;
   - ALE(10 题):Δ < **40** 不可区分;
   - Research(64 题):没有全局常数噪声底——按题查
     `data/research_scoring_variance.json` 的 per-problem 跨 rep std / range,
     Δ 小于所涉题目的合成 CI 时视为并列。
   (FCS/ALE 噪声底来源见 memory `eval-noise-floor-fcs-ale`。)
3. **lottery 题单一律以 `data/research_scoring_variance.json` 为准**
   (由 `scripts/measure_eval_variance.py` 生成/更新;判据:某 sample 跨 rep
   std > 5)。已证实的 lottery 家族:`symbolic_regression/*`(PySR 随机搜索,
   同一解可在 0.04 与 100 之间抖动);`vdb_pareto/*`、`imagenet_pareto/*`
   部分按实测运行时打分 → 节点速度敏感。
4. **决策表(decision tables)必须同时给出 with-lottery 和 without-lottery
   两套聚合**。只有两套聚合结论一致时才可下"模型 A 优于 B"的判断;不一致时
   结论悬置,先加 REPS(见 §2)复评 lottery 题。
5. 引用任何 samples.jsonl 分数前先查 `error` 数(memory
   `eval-resume-error-zero-bug`):infra 0 分不是成绩。

## 2. Research 多次评分(median-of-K)

两个评分入口(`frontiercs_research_eval.py` / `frontiercs_research_cpu_eval.py`)
新增环境开关(调用时读取,sbatch `--export` 直接透传):

- `FRONTIERCS_RESEARCH_SCORE_REPS`(默认 1):>1 时对同一份解跑 N 次
  evaluator,**记 median 分**;每次的分数/状态存进记录的 `raw` 字段
  (`[multi-rep] {...}` 头)。全部 rep 都 infra-fail 才抛 `ResearchInfraError`。
- `FRONTIERCS_RESEARCH_REPS_ONLY`:限定多次评分的题集,控制成本。
  取值:逗号分隔的题 id / 家族前缀(如 `symbolic_regression,vdb_pareto`),
  **或**指向 variance JSON 的路径(自动取其 `lottery_problems` +
  per-problem `lottery: true`)。空 = 全部题。

推荐口径:`REPS=3` + `REPS_ONLY=data/research_scoring_variance.json`,
即只对 lottery 题付 3 倍成本。

## 3. FCS shim 计时鲁棒性(scripts/gojudge_shim.py)

- **边缘 TLE 重跑**:`SHIM_TLE_RETRY=1`(默认开)。单 cmd 请求(不含交互
  pipeMapping 对)以 TLE 结束且 cpu_ns < `SHIM_TLE_RETRY_FACTOR`(1.25)×
  cpuLimit 时,在全新 workdir 重跑一次,重跑不再 TLE 则采用重跑结果
  (对应真实 judge 的 rerun 政策)。重跑事件打到 stderr 日志
  (`[gojudge-shim] borderline TLE ... retrying once`)。`SHIM_TLE_RETRY=0` 关闭。
- **节点速度标定**:启动时跑 ~1s 确定性整数循环,测 ops/s,与硬编码的
  ailab-EPYC 参考常数 `AILAB_EPYC_REF_OPS_PER_SEC` 相除得 `speedFactor`
  (>1 = 比参考快),打印在启动日志并暴露于 `GET /version` 的
  `nodeSpeedCalibration` 字段 → 每份 eval 日志自带节点速度等级。
  **只测不改**:默认不按 factor 缩放任何时限。
  - factor 含义:解释跨节点分差用。两次 eval 的 factor 差 >~10% 时,
    时限边缘题的 TLE 翻转应先归因于节点速度,不是模型回退。
  - 当前参考常数是 della 登录节点 EPYC 7H12 的**临时基线**(3.12e6 ops/s,
    CPython 3.12)。在 ailab 节点跑
    `python3 scripts/gojudge_shim.py -calibrate-only` 后应把打印的
    `opsPerSec` 回填该常数(换解释器版本也要重标)。

## 4. 墙钟(wall-time)注意事项

- `vdb_pareto/*`、`imagenet_pareto/*`(以及所有按实测 latency/runtime 打分的
  evaluator)的分数随宿主负载与 CPU 代际漂移:共享节点上的邻居负载、
  OMP 线程数(adapter 默认 `OMP_NUM_THREADS=8`)、冷/热缓存都进分数。
  对这些题:同节点同 cgroup 配额下比较,或用 §2 的 median-of-K。
- `imagenet_pareto` 单次 eval 可达 2400s(超时上限),多 rep 前先看
  variance JSON 里该题的 rep_seconds,别把 CPU 预算烧在必然超时的解上。
- 登录节点**不许**跑成规模的评测(PySR 首跑还要 Julia 预编译,分钟级)。
  smoke 限 2-3 道便宜题。

## 5. 方差测量(生成/更新 lottery 题单)

```bash
# 全量 CPU 家族测量(2 shard,cpu 分区;GPU/Triton 题需另起 GPU 任务 --include-gpu)
sbatch --job-name=fs-eval-var-s0 --partition=cpu --cpus-per-task=16 --mem=64G \
  --time=12:00:00 --chdir=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith \
  --output=logs/%x-%j.out \
  --wrap='.venv/bin/python scripts/measure_eval_variance.py \
    --samples outputs/cc_eval_q36_35bA3b_base_research_thinking_32k_vllm/merged/samples.jsonl \
    --reps 3 --shard 0/2 --out data/research_scoring_variance.shard0.json'
# 同上 --shard 1/2 -> shard1.json;两个都完成后合并:
.venv/bin/python scripts/measure_eval_variance.py \
  --merge data/research_scoring_variance.shard0.json data/research_scoring_variance.shard1.json \
  --out data/research_scoring_variance.json
```

## 6. 带 REPS 的 research eval 示例

```bash
sbatch --job-name=cc-research-<TAG>-reps3 --time=08:00:00 \
  --export=ALL,MODEL_PATH=<MODEL_DIR>,TAG=<TAG>_reps3,FRONTIERCS_RESEARCH_SCORE_REPS=3,FRONTIERCS_RESEARCH_REPS_ONLY=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/data/research_scoring_variance.json \
  slurm/cc_eval_research_ailab.sh
```
(REPS 只作用于 lottery 题,预算约 +2×(lottery 题 eval 时长);
variance JSON 未生成前可先用家族列表
`FRONTIERCS_RESEARCH_REPS_ONLY=symbolic_regression,vdb_pareto,imagenet_pareto`。)
