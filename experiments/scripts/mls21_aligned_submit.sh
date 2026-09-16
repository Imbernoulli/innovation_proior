#!/usr/bin/env bash
# mls21_aligned_submit.sh -- MLS-21 主表,采样协议与 FCS/ALE/research 对齐后重跑。
#
# 为什么(2026-09-16 核实,不是猜的):
#   FrontierCS / ALE / research 都走 cc_eval_cpu_client*.sh,每个作业日志里都印
#     [cpu-client] protocol: max_tokens=32768 temp=1.0 top_p=0.95 top_k=20 pp=1.5 n=5
#   MLS-Bench 走的是另一条代码路径 mlsbench/agent/models.py,那条路对本地 vLLM
#   **一个采样参数都不发**(只发 model/messages/tools/tool_choice),全部用服务端默认。
#   所以此前每一个 MLS 数字都不是在评测协议下产生的。
#   后果不是装饰性的:419 个 (臂,年份,题) 格子里 34% 是空提交,其中 65% 以
#   "[agent] No action returned after 3 attempts" 收尾,模型生成约 2.6 万 token
#   却没吐出可解析的 tool call —— 正是 presence_penalty=1.5 用来压的那种复读失控。
#
# 本脚本的唯一变量就是采样协议:其余 env 与 p1 那批**逐字相同**
#   (MLSBENCH_PY / MLSBENCH_SYS_PREFIX / MAX_MODEL_LEN=40960 / TASK_TIMEOUT=7200 /
#    CONCURRENCY=7 / MLSBENCH_DATA_ROOT),所以 al1 对 p1 是一个干净的 A/B。
# TAG 带后缀 al1:TAG 同时是 --served-model-name,也是排行榜选行的键;
# 复用同名会让两批互相顶替(见 innov-prior-mls-leaderboard-contamination-and-template-scores)。
# 原有 p1 / 年份目录一个字节都不动。
#
# max_tokens 故意不设成 32768:MLS 是多轮、prompt 每步变长,单轮 32768 的上限
# 会在任务中途撑爆 MAX_MODEL_LEN。留空 = 服务端允许到上下文上限,与 p1 同。
#
#   bash scripts/mls21_aligned_submit.sh <ARM> [<ARM> ...]
#   SUF=al1  TASKS_OVERRIDE="<一道题>"(冒烟)  WALL=20:00:00
set -uo pipefail
D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
declare -A MP=(
  [base9b_v2c]="$FS/models/Qwen3.5-9B-bf16"
  [ft01mix_a10]="$D/models/full_wd01_withag_v2b_soup10"
  [rlv5_base_s20]="$D/models/rlv5_base_s20"
  [rlv5_ft01mix_a10_s20]="$D/models/rlv5_ft01mix_a10_s20"
  [base4b]="$D/models/Qwen3.5-4B"
  [4b_ft01mix_a10]="$D/models/4b_full_wd01_withag_soup10"
  [rlv5_4b_base_s20]="$D/models/rlv5_4b_base_s20"
  [rlv5_4b_ft01mix_a10_s20]="$D/models/rlv5_4b_ft01mix_a10_s20"
)
TASKS21="causal-discovery-discrete causal-observational-linear-gaussian causal-observational-linear-non-gaussian causal-observational-nonlinear causal-treatment-effect ml-active-learning ml-anomaly-detection ml-calibration ml-clustering-algorithm ml-dimensionality-reduction ml-ensemble-boosting ml-missing-data-imputation ml-selective-deferral ml-subgroup-calibration-shift ml-symbolic-regression mlsys-moe-load-balance optimization-evolution-strategy optimization-hyperparameter-search optimization-multi-objective optimization-nas optimization-online-bandit"
SUF="${SUF:-al1}"
MML="${MML:-40960}"        # 与 p1 同
TTO="${TTO:-7200}"         # 与 p1 同
CONC="${CONC:-7}"          # 与 p1 同
PREFIX="${PREFIX:-It is now year 2026.}"
WALL="${WALL:-20:00:00}"   # 必须 < 24h
TASKS_RUN="${TASKS_OVERRIDE:-$TASKS21}"
JOBSUF="${JOBSUF:-}"
# 对齐协议,数值取自 cc_eval_cpu_client_pinned.sh 的 export 默认值
# ---------------------------------------------------------------------------
# 2026-09-16 投递前静态核查:MLSBENCH_* 到底能不能走到真正读它的那行代码。
# (r2 那次丢 MLSBENCH_SYS_PREFIX 的教训;而且「已审计」不等于审计对了。)
# 逐跳查过,没有任何一跳会过滤环境:
#   1. sbatch --export=ALL,<env>,${ALIGN},...   与 mls21_rerun_submit.sh(产出 p1,已知好)
#                                               同一构造;r2 丢前缀是因为那个脚本压根没传,
#                                               不是 --export 的引号问题。
#   2. slurm_overlay/cc_dual_wrap.sh:16         `exec bash "$REAL_SCRIPT"`,全量继承。
#   3. $FS/slurm/cc_eval_mlsbench_cpu_ailab.sh  全文不出现 MLSBENCH_SAMPLING_ALIGN /
#                                               MLSBENCH_SYS_PREFIX,原样透传。
#   4. $FS/scripts/mlsbench_run_cpu_tasks.py    :235 env_base=dict(os.environ)
#                                               :105 env=dict(env_base)
#                                               :126 subprocess.run(..., env=env)
#   5. mlsroot/src/mlsbench/agent/models.py     读 MLSBENCH_SAMPLING_ALIGN(采样对齐分支)
#      mlsroot/src/mlsbench/agent/interactive.py:85 读 MLSBENCH_SYS_PREFIX(年份条件句)
# 注意 agent 客户端跑在宿主 python($D/envs/client/bin/python),不在 Apptainer 里,
# 所以不需要 APPTAINERENV_ 前缀;容器只用来跑任务自己的代码。
# 2026-09-16 19:07 经验确认(14004044 base9b_v2c_al1 / 14004046 ft01mix_a10_al1):
#   a) 16/16 已开工的 task_log 全部打出
#      `[mlsbench] sampling aligned: temp=1.0 top_p=0.95 pp=1.5 top_k=20 min_p=0.0 rep=1.0 max_tokens=unset`
#   b) 带空格的值会不会被 --export 的逗号列表截断,日志看不出来,所以直接读了运行中
#      进程的环境:`ssh della-i24g3` 进 job cgroup,`tr "\0" "\n" < /proc/<agent_pid>/environ`。
#      MLSBENCH_SYS_PREFIX=It is now year 2026.  ← 空格完整,没被截断
#      MLSBENCH_SAMPLING_ALIGN=1 / MLSBENCH_PRESENCE_PENALTY=1.5 / MLSBENCH_TOP_K=20
#      EVAL_RESEARCHER_YEAR=2026 / MAX_MODEL_LEN=40960 / TASK_TIMEOUT=7200 / CONCURRENCY=7
#      —— 与 p1 逐项一致,al1 与 p1 之差确实只有采样。
#   /proc/<pid>/environ 这招比翻日志可靠:它读的是进程真正在用的 env,不是谁打印了什么。
# ---------------------------------------------------------------------------
ALIGN="MLSBENCH_SAMPLING_ALIGN=1,MLSBENCH_TEMPERATURE=1.0,MLSBENCH_TOP_P=0.95,MLSBENCH_TOP_K=20,MLSBENCH_MIN_P=0.0,MLSBENCH_PRESENCE_PENALTY=1.5,MLSBENCH_REPETITION_PENALTY=1.0"
LOG=$D/logs/mls21_aligned_jobids.txt
mkdir -p $D/logs/locks

dual() {  # name walltime env-string real-script
  local name=$1 t=$2 env=$3 script=$4 a g
  [ -d "$D/logs/locks/${name}.lock" ] && { echo "SKIP $name: lock exists (already ran)"; return; }
  a=$(sbatch --parsable --partition=ailab --account=chij --qos=short --gres=gpu:1 -c 8 --mem=200G --time="$t" \
      -J "$name" -o "$D/logs/${name}-%j.out" \
      "--export=ALL,${env},DUAL_NAME=${name},REAL_SCRIPT=${script}" "$D/slurm_overlay/cc_dual_wrap.sh") || { echo "sbatch ailab failed: $name"; return; }
  g=$(sbatch --parsable --account=chij --qos=gpu-short --constraint=gpu80 --gres=gpu:1 -c 8 --mem=200G --time="$t" \
      -J "$name" -o "$D/logs/${name}-%j.out" \
      "--export=ALL,${env},DUAL_NAME=${name},REAL_SCRIPT=${script}" "$D/slurm_overlay/cc_dual_wrap.sh") || g=none
  echo "$a $g" > "$D/logs/locks/${name}.ids"
  echo "$name ailab=$a gpu=$g" | tee -a "$LOG"
}

for ARM in "$@"; do
  M="${MP[$ARM]:?unknown arm $ARM}"; [ -e "$M/config.json" ] || { echo "no model $M"; continue; }
  TAG="${ARM}_${SUF}"
  cd "$FS"   # mlsbench 脚本从 SLURM_SUBMIT_DIR 取 PROJECT_ROOT
  dual "mls21-$TAG$JOBSUF" "$WALL" "MODEL_PATH=$M,TAG=$TAG,OUTPUT_BASE=$D/outputs/cc_mls21_$TAG$JOBSUF,MLSBENCH_ROOT=$D/mlsroot,MLSBENCH_DATA_ROOT=$D/mlsvendor/data,MLSBENCH_PY=$D/envs/client/bin/python,MLSBENCH_SYS_PREFIX=$PREFIX,EVAL_RESEARCHER_YEAR=2026,MAX_MODEL_LEN=$MML,TASK_TIMEOUT=$TTO,HF_HOME=$D/.hf,VLLM_VENV=$D/envs/vllm023,VLLM_CACHE_DIR=$D/.cache/vllm,CONCURRENCY=$CONC,VLLM_PORT=$((41000 + RANDOM % 20000)),${ALIGN},TASKS=$TASKS_RUN" "$FS/slurm/cc_eval_mlsbench_cpu_ailab.sh"
done
