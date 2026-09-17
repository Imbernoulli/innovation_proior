#!/bin/bash
# 按「最终文件状态」重评一个 (臂, 题) 格子。纯 CPU:这些 test_cmds 的 pkg_config
# 全是 use_cuda=false,申请 GPU 反而会被空转巡检杀掉,所以这里不要 --gres。
set -uo pipefail
D=/scratch/gpfs/CHIJ/ziran/innov_v2_multi
PLAN="$D/scripts/rescore_plan.tsv"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PLAN")
TAG=$(printf '%s' "$LINE" | cut -f1)
TASK=$(printf '%s' "$LINE" | cut -f2)
if [ -z "${TAG:-}" ] || [ -z "${TASK:-}" ]; then
  echo "[rescore] 空计划行 ${SLURM_ARRAY_TASK_ID}"; exit 1
fi
echo "[rescore] ${SLURM_ARRAY_TASK_ID}: $TAG / $TASK"
cd "$D/mlsroot" || exit 1
exec "$D/envs/client/bin/python" "$D/scripts/rescore_cell.py" "$TASK" "$TAG"
