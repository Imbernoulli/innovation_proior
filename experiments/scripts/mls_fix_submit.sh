#!/usr/bin/env bash
# 提交 MLS 补跑:一条臂一个作业,起一次 serve,按 setting 依次补缺的题。
# 计划由 experiments/innov_quant/mls_audit21.py 审出来,写在 $PLANFILE(JSON)。
set -euo pipefail
D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
FS=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
PLANFILE="${1:?usage: mls_fix_submit.sh <fixplan.json>}"
WALL="${WALL:-23:00:00}"          # <24h
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
mkdir -p "$D/logs/locks"
# 计划走文件,绝不走 --export:sbatch 的 --export 按逗号切 KEY=VAL,
# 题目列表里的逗号会把它切碎(2026-09-17 踩过)。
PLANDIR="$D/logs/mlsfix_plans"; mkdir -p "$PLANDIR"
sub_one() {  # $1=arm $2=plan-file
  local ARM="$1" PF="$2" M="${MP[$1]:?unknown arm $1}"
  [ -e "$M/config.json" ] || { echo "no model $M"; return; }
  local LK="$D/logs/locks/mlsfix-$ARM.lock"
  mkdir "$LK" 2>/dev/null || { echo "SKIP $ARM (lock exists)"; return; }
  local ENVS="ARM=$ARM,MODEL_PATH=$M,PLANFILE=$PF,OUT_ROOT=$D/outputs,FS_ROOT=$FS"
  ENVS="$ENVS,MLSBENCH_ROOT=$D/mlsroot,MLSBENCH_DATA_ROOT=$D/mlsvendor/data"
  ENVS="$ENVS,HF_HOME=$D/.hf,VLLM_VENV=$D/envs/vllm023,VLLM_CACHE_DIR=$D/.cache/vllm"
  ENVS="$ENVS,CLIENT_PY=$D/envs/client/bin/python,CONDA_PY=/home/zy7019/miniconda3/bin/python3"
  ENVS="$ENVS,MAX_MODEL_LEN=65536,TASK_TIMEOUT=9000,CONCURRENCY=7"
  ENVS="$ENVS,VLLM_PORT=$((41000 + RANDOM % 20000))"
  local A G
  # $FS/logs 是 bl3615 的、不可写 —— 日志必须落到 $D/logs
  A=$(sbatch --parsable -J "mlsfix-$ARM" -t "$WALL" \
        --partition=ailab --account=chij --qos=short --gres=gpu:1 \
        -o "$D/logs/mlsfix-$ARM-%j.out" -e "$D/logs/mlsfix-$ARM-%j.err" \
        --export=ALL,"$ENVS" "$D/slurm_overlay/cc_mls_fix_multi.sh")
  G=$(sbatch --parsable -J "mlsfix-$ARM" -t "$WALL" \
        --account=chij --qos=gpu-short --constraint=gpu80 --gres=gpu:1 \
        -o "$D/logs/mlsfix-$ARM-%j.out" -e "$D/logs/mlsfix-$ARM-%j.err" \
        --dependency=afterany:"$A" --kill-on-invalid-dep=yes \
        --export=ALL,"$ENVS" "$D/slurm_overlay/cc_mls_fix_multi.sh")
  echo "$A $G" > "$LK/ids"
  echo "mlsfix-$ARM ailab=$A gpu=$G"
}
# 每条臂写一个 tsv 计划文件
python3 - "$PLANFILE" "$PLANDIR" <<'PYGEN'
import json, sys, os
plan = json.load(open(sys.argv[1])); outdir = sys.argv[2]
for arm, ent in plan.items():
    with open(os.path.join(outdir, arm + ".tsv"), "w") as f:
        for setting, year, tasks in ent:
            f.write(f"{setting}\t{year}\t{','.join(tasks)}\n")
    print(arm)
PYGEN
cd "$FS"   # SLURM_SUBMIT_DIR 必须是 FS,作业脚本从那里取 scripts/start_vllm_server.sh
for arm in $(python3 -c "import json;print(' '.join(json.load(open('$PLANFILE'))))"); do
  sub_one "$arm" "$PLANDIR/$arm.tsv"
done
