#!/bin/bash
# Re-judge the cells that have a stored generation but no score, for one bench and a list of arms.
#   sbatch --export=ALL,BENCH=frontiercs,ARMS="a b c" ... rejudge_job2.sh
# Everything except the judging environment is held fixed: the texts come from the existing
# samples.jsonl, so the repaired cells belong to the SAME generation as the rest of the run.
set -uo pipefail
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
: "${BENCH:?set BENCH}"; : "${ARMS:?set ARMS}"
KEYS="${KEYS:-$D/rejudge/rejudge_keys.json}"
cd "$FS"

export PATH="$D/envs/client/bin:$PATH"
export PYTHONPATH="$FS:$FS/verl:$FS/ALE-Bench/src:$FS/Frontier-CS/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$D/.hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# node-local TMPDIR: the login .bashrc points TMPDIR at a 50G /home dir, which is what produced
# the `llm_router: No such file or directory: '/home/zy7019/.tmp/resources/...'` failures and the
# ALE `OSError(122, Disk quota exceeded)` ones.
export TMPDIR="/tmp/rejudge-${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"
echo "[rejudge] node=$(hostname -s) part=${SLURM_JOB_PARTITION:-} bench=$BENCH arms=$ARMS"
echo "[rejudge] TMPDIR=$TMPDIR ($(df -h /tmp | tail -1 | awk '{print $4}') free)"

# keep the allocated card busy: an idle GPU gets the whole job cancelled after 90 min
if [ -f "$HOME/gpu_hold/run.py" ]; then
  CUDA_VISIBLE_DEVICES=0 python3 "$HOME/gpu_hold/run.py" --sleep 20 &
  HOLD=$!; echo "[rejudge] gpu keep-alive pid $HOLD"
fi
cleanup() { [ -n "${HOLD:-}" ] && kill "$HOLD" 2>/dev/null; [ -n "${JUDGE_PID:-}" ] && kill "$JUDGE_PID" 2>/dev/null; rm -rf "$TMPDIR"; }
trap cleanup EXIT

case "$BENCH" in
  frontiercs)
    export FS_INFRA_CORES="${FS_INFRA_CORES:-4}"
    read -r INFRA_SET POOL_SET <<< "$(python3 - <<'PY'
import os
c = sorted(os.sched_getaffinity(0)); n = max(2, min(int(os.environ.get("FS_INFRA_CORES","4")), len(c)-4))
print(",".join(map(str, c[:n])), ",".join(map(str, c[n:])))
PY
)"
    export SHIM_PIN_CORES="$POOL_SET" SHIM_BIN="$FS/scripts/gojudge_shim_v2.py"
    read -r _P1 _P2 <<< "$(python3 -c '
import socket
s=[]
for _ in range(2):
    x=socket.socket(); x.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1); x.bind(("127.0.0.1",0)); s.append(x)
print(*[y.getsockname()[1] for y in s])
for y in s: y.close()')"
    # shim, not auto: on ailab `auto` now picks the real go-judge, which puts itself in its own
    # systemd scope outside the SLURM cgroup and gets SIGKILLed. The shim is also what the runs
    # that produced the existing scores actually used, so it keeps the judge topology comparable.
    export PORT="$_P1" GJ_PORT="$_P2" GJ_BACKEND="${GJ_BACKEND:-shim}"
    export GJ_CGROUP_PREFIX="gojudge-${SLURM_JOB_ID:-$$}"
    export RUNTIME_DIR="$D/.cache/frontiercs-judge-rejudge-${SLURM_JOB_ID:-manual}"
    export FRONTIERCS_JUDGE_URL="http://127.0.0.1:${PORT}"
    taskset -c "$INFRA_SET" bash scripts/start_frontiercs_judge_hybrid_v2.sh &
    JUDGE_PID=$!
    ok=0
    for _ in $(seq 1 240); do
      curl -fsS "http://127.0.0.1:${GJ_PORT}/version" >/dev/null 2>&1 && \
      curl -fsS "http://127.0.0.1:${PORT}/health"     >/dev/null 2>&1 && { ok=1; break; }
      sleep 0.5; kill -0 "$JUDGE_PID" 2>/dev/null || { echo "judge exited early" >&2; exit 1; }
    done
    [ "$ok" = 1 ] || { echo "judge never healthy" >&2; exit 1; }
    echo "[rejudge] judge ready on $PORT"
    export REJUDGE_WORKERS="${REJUDGE_WORKERS:-6}" REJUDGE_FCS_TIMEOUT="${REJUDGE_FCS_TIMEOUT:-3000}"
    ;;
  alebench)
    export ALE_BENCH_DATA="$FS/data/alebench/local_data"
    export ALE_BENCH_CACHE="$D/.cache/ale-bench"
    export ALE_BENCH_TOOL_CACHE="$D/.cache/ale-bench/rust-tool-builds"
    export ALE_BENCH_REQUIRE_TOOL_CACHE=1 ALE_BENCH_CONTAINER_BACKEND=apptainer
    export ALE_BENCH_APPTAINER_DIR="$FS/.cache/apptainer/alebench"
    export ALEBENCH_JUDGE_VERSION=202301 ALEBENCH_NUM_WORKERS="${ALEBENCH_NUM_WORKERS:-2}"
    export ALEBENCH_LITE=false
    export REJUDGE_WORKERS="${REJUDGE_WORKERS:-3}"
    ;;
  frontiercs_research)
    # writable copies: the read-only tree is exactly what broke these cells
    #   julia_env/lock.pid          -> symbolic_regression/* (174 cells)
    #   problems/.../sql/output_ans -> grammar_fuzzing/fuzzer/sql (41 cells)
    export REJUDGE_ENV="$D/rejudge_env"
    export FRONTIERCS_RESEARCH_PYTHON="/scratch/gpfs/CHIJ/bohan/fs/envs/research_overlay/bin/python"
    export JULIA_DEPOT_PATH="$REJUDGE_ENV/julia_depot"
    export PYTHON_JULIAPKG_PROJECT="$REJUDGE_ENV/julia_env"
    export FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB="${FRONTIERCS_RESEARCH_EVAL_RLIMIT_GB:-64}"
    export FRONTIERCS_RESEARCH_TIMEOUT="${FRONTIERCS_RESEARCH_TIMEOUT:-1800}"
    export FRONTIERCS_RESEARCH_CPU_TIMEOUT="${FRONTIERCS_RESEARCH_CPU_TIMEOUT:-2400}"
    export REJUDGE_WORKERS="${REJUDGE_WORKERS:-3}"
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1
    ;;
  *) echo "unknown BENCH=$BENCH" >&2; exit 2;;
esac

# Record where the repaired cells were judged. ALE is wall-clock scored and FrontierCS TLE
# verdicts are speed-sensitive, so a repaired shard whose topology is not written down is a
# confound waiting to happen -- dump2's judge_meta() globs shard_*/judge_node_meta.json.
write_meta() {
  local arm="$1" sub="$2" dir="$D/outputs/cc_eval_${arm}_${sub}/shard_rejudge"
  mkdir -p "$dir"
  python3 - "$dir/judge_node_meta.json" <<'PY2'
import json, os, socket, subprocess, sys
speed = None
try:
    out = subprocess.run(["python3", os.environ["SHIM_BIN"], "-calibrate-only"],
                         capture_output=True, text=True, timeout=90).stdout
    speed = (json.loads(out[out.index("{"):]) or {}).get("speedFactor")
except Exception:
    pass
json.dump({"node": socket.gethostname().split(".")[0],
           "partition": os.environ.get("SLURM_JOB_PARTITION"),
           "node_speed_calibration": {"speedFactor": speed},
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "rejudge": True}, open(sys.argv[1], "w"), indent=1)
PY2
}

rc=0
for ARM in $ARMS; do
  echo "=== [rejudge] $ARM / $BENCH"
  case "$BENCH" in
    frontiercs|alebench)     write_meta "$ARM" thinking_32k_both_vllm ;;
    frontiercs_research)     write_meta "$ARM" research_thinking_32k_vllm ;;
  esac
  python3 "$D/rejudge/rejudge_missing.py" --arm "$ARM" --bench "$BENCH" --keys "$KEYS" \
      ${FRONTIERCS_JUDGE_URL:+--judge-url "$FRONTIERCS_JUDGE_URL"} ${REJUDGE_LIMIT:+--limit $REJUDGE_LIMIT} ${REJUDGE_CONTROL:+--control $REJUDGE_CONTROL} || rc=1
done
echo "[rejudge] ALLDONE rc=$rc"
exit $rc
