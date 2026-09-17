#!/usr/bin/env bash
# =============================================================================
# MLS-Bench 补跑:一条臂一个作业,起一次 vLLM,按 setting 依次补该 setting 缺的题。
#
# 为什么要有这个脚本(2026-09-16):
#   用户定了「MLS 的分母必须是 21」。审计 49 个 (臂,setting) 格之后,缺的题共 85 个,
#   死因只有三种,全是基础设施而非模型答得差:
#     ctxlen  35 —— openai.BadRequestError 400,上下文撑爆。**是我们自己把 MAX_MODEL_LEN
#                   卡在 40960,而模型 config 的 text_config.max_position_embeddings=262144。**
#                   这里放到 65536。
#     serve   30 —— openai.APIConnectionError,那一格 vLLM 当时是死的。
#     timeout 20 —— 撞 TASK_TIMEOUT。这里放到 9000s。
#   注意归因要看**最后一段 traceback 的异常行**:日志正文里满是模型自己代码的
#   SyntaxError/AttributeError,那些是 mlsbench 回灌给模型继续改的,不是失败。
#
# 用法:
#   PLANFILE=/path/plan.tsv ARM=... MODEL_PATH=... sbatch 本脚本
#   plan.tsv 每行 = <setting>\t<year>\t<逗号分隔的题>。setting=al1 时自动加对齐采样。
#   **计划必须走文件,不能走 --export**:sbatch 的 --export 按逗号切 KEY=VAL,
#   题目列表里的逗号会把 PLAN 切碎(2026-09-17 踩过,整批作业只跑了一道题)。
#   结果写 $OUT_ROOT/cc_mls21_<ARM>_<setting>-fix/,**不动原目录**,读数时按题后写覆盖。
#
#SBATCH --job-name=cc-mls-fix
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

# PROJECT_ROOT 必须是 FrontierSmith(scripts/start_vllm_server.sh 在那儿)。
# SLURM_SUBMIT_DIR 是「执行 sbatch 时所在的目录」,不是 -D 给的工作目录 ——
# 2026-09-17 我用 -D 传,结果 SLURM_SUBMIT_DIR 指到 repo,serve 脚本找不到,整批作业空转。
# 提交器现在会先 cd 到 FS;这里再兜一层 FS_ROOT。
PROJECT_ROOT="${FS_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
[ -x "$PROJECT_ROOT/scripts/start_vllm_server.sh" ] || {
  echo "ERROR: no scripts/start_vllm_server.sh under PROJECT_ROOT=$PROJECT_ROOT" >&2; exit 1; }
cd "$PROJECT_ROOT"

: "${ARM:?ARM required}"; : "${MODEL_PATH:?MODEL_PATH required}"; : "${PLANFILE:?PLANFILE required}"
[ -f "$PLANFILE" ] || { echo "ERROR: no PLANFILE $PLANFILE" >&2; exit 1; }
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
MLSBENCH_ROOT="${MLSBENCH_ROOT:?}"
[ -d "$MLSBENCH_ROOT/src/mlsbench" ] || { echo "ERROR: bad MLSBENCH_ROOT" >&2; exit 1; }

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR="${TMPDIR:-/tmp}"
export MLSBENCH_NO_PREBUILT=1 MLSBENCH_SCHEDULER_MANAGED=1
export MLSBENCH_USE_REPLACE="${MLSBENCH_USE_REPLACE:-1}"
export HF_HOME="${HF_HOME:?}" ; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

JOBU=$(( ${SLURM_JOB_ID:-$$} % 9000 ))
export VLLM_PORT="${VLLM_PORT:-$(( 34000 + JOBU ))}"
export HOST=127.0.0.1
SERVE_TAG="${ARM}_fix"
AGENT_MODEL="vllm/${SERVE_TAG}"
SERVED_MODEL_NAME="${SERVE_TAG} vllm/${SERVE_TAG}"

export TP="${TP:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"   # 原 40960 是 35 题的死因;模型原生 262144
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export DTYPE="${DTYPE:-bfloat16}"
CONC="${CONCURRENCY:-7}"
TTO="${TASK_TIMEOUT:-9000}"

echo "[mls-fix] ARM=$ARM MODEL=$MODEL_PATH"
echo "[mls-fix] MAX_MODEL_LEN=$MAX_MODEL_LEN (was 40960) TASK_TIMEOUT=${TTO}s CONCURRENCY=$CONC"
echo "[mls-fix] PLANFILE=$PLANFILE"
cat "$PLANFILE" | sed 's/^/    /'

export MODEL_PATH
PORT="$VLLM_PORT" SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
  scripts/start_vllm_server.sh --enable-auto-tool-choice --tool-call-parser hermes &
VLLM_PID="$!"
cleanup(){ kill "${KEEPALIVE_PID:-0}" >/dev/null 2>&1 || true
           kill "$VLLM_PID" >/dev/null 2>&1 || true
           wait "$VLLM_PID" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

echo "[mls-fix] waiting for vLLM /health on ${VLLM_PORT} ..."
for _ in $(seq 1 900); do
  curl -fsS "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1 && break
  sleep 2
  kill -0 "$VLLM_PID" >/dev/null 2>&1 || { echo "vLLM exited early" >&2; exit 1; }
done
curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1 \
  || { echo "ERROR: vLLM never served /v1/models" >&2; exit 1; }
echo "[mls-fix] vLLM ready (served='$SERVED_MODEL_NAME')"

# 空转 GPU 会被巡检杀掉(90 分钟 0% util)。原来是每 8 分钟 8 个 token —— 那点活
# 在采样窗口里就是 0%,14024214 跑了 3h41m 还是被 "CANCELLED by 123" 扫掉了。
# 改成每 60 秒 256 个 token:agent 在跑 CPU 容器测试时 GPU 才是真闲着,这个量
# 对它的推理几乎没有影响,但足以让利用率不为零。
( while true; do sleep 60; \
    curl -s -m 120 "http://127.0.0.1:${VLLM_PORT}/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"${SERVE_TAG}\",\"prompt\":\"keepalive\",\"max_tokens\":256,\"temperature\":1.0}" >/dev/null 2>&1 || true; done ) &
KEEPALIVE_PID="$!"

RC_ALL=0
while IFS=$'\t' read -r SETTING YEAR TASKS; do
  [ -n "${SETTING:-}" ] || continue
  OB="$OUT_ROOT/cc_mls21_${ARM}_${SETTING}-fix"
  mkdir -p "$OB/saves"
  GEN="$OB/config_${SLURM_JOB_ID:-manual}.yaml"
  cat > "$GEN" <<YAML
max_steps: ${MLSBENCH_MAX_STEPS:-20}
max_tests: ${MLSBENCH_MAX_TESTS:-3}
save_path: ${OB}/saves
data_root: ${MLSBENCH_DATA_ROOT:?}
seeds: [42]
container_runtime: apptainer
thinking:
  enabled: true
  reasoning_effort: "high"
  budget_tokens: ${MLSBENCH_BUDGET_TOKENS:-10000}
providers:
  vllm:
    api_key: "EMPTY"
    base_url: "http://127.0.0.1:${VLLM_PORT}/v1"
YAML
  # 世代主键:al1/p1 走 client venv,年份点走 conda —— 必须和被补的那一格一致。
  case "$SETTING" in
    al1|p1) PY="${CLIENT_PY:?}" ;;
    *)      PY="${CONDA_PY:?}"  ;;
  esac
  ALIGN_ENV=()
  if [ "$SETTING" = "al1" ]; then
    ALIGN_ENV=(MLSBENCH_SAMPLING_ALIGN=1 MLSBENCH_TEMPERATURE=1.0 MLSBENCH_TOP_P=0.95
               MLSBENCH_TOP_K=20 MLSBENCH_MIN_P=0.0 MLSBENCH_PRESENCE_PENALTY=1.5
               MLSBENCH_REPETITION_PENALTY=1.0)
  fi
  echo ""
  echo "===== [mls-fix] $ARM / $SETTING (year=$YEAR, py=$(basename $(dirname $(dirname $PY)))) ====="
  echo "[mls-fix] tasks: $TASKS"
  set +e
  env "${ALIGN_ENV[@]}" \
    MODEL="$AGENT_MODEL" MLSBENCH_ROOT="$MLSBENCH_ROOT" \
    EVAL_RESEARCHER_YEAR="$YEAR" MLSBENCH_SYS_PREFIX="It is now year ${YEAR}." \
    "$PY" "$PROJECT_ROOT/scripts/mlsbench_run_cpu_tasks.py" \
      --config "$GEN" --model "$AGENT_MODEL" --root "$MLSBENCH_ROOT" \
      --out "$OB/summary.json" --concurrency "$CONC" --timeout "$TTO" --python "$PY" \
      --tasks ${TASKS//,/ }
  rc=$?
  set -e
  echo "[mls-fix] $ARM/$SETTING rc=$rc"
  [ $rc -eq 0 ] || RC_ALL=$rc
done < "$PLANFILE"

echo "[mls-fix] ALL DONE rc=$RC_ALL"
exit 0
