#!/bin/bash
set -e

# Resolve LIBERO source: prefer the workspace copy (under MLSBENCH_PKG_DIR),
# fall back to /workspace/LIBERO if mlsbench mounts it there.
if [ -n "${MLSBENCH_PKG_DIR:-}" ] && [ -d "${MLSBENCH_PKG_DIR}" ]; then
  cd "${MLSBENCH_PKG_DIR}"
elif [ -d /workspace/LIBERO ]; then
  cd /workspace/LIBERO
else
  echo "Cannot find LIBERO source directory" >&2; exit 1
fi

# LIBERO's libero/libero/__init__.py prompts on first import if no
# ~/.libero/config.yaml exists. Container is launched with --no-home so
# /root/.libero baked at build time is not visible — re-create it here.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/tmp/.libero}"
mkdir -p "$LIBERO_CONFIG_PATH"
python3 - <<PYEOF
import os, yaml
b = '/workspace/LIBERO/libero/libero'
cfg = {
    'benchmark_root': b,
    'bddl_files': b + '/bddl_files',
    'init_states': b + '/init_files',
    'datasets': '/data/libero/libero_spatial',
    'assets': b + '/assets',
}
yaml.safe_dump(cfg, open(os.path.join(os.environ['LIBERO_CONFIG_PATH'], 'config.yaml'), 'w'))
PYEOF

python3 -m libero.lifelong.main \
    seed=${SEED:-42} \
    benchmark_name=LIBERO_SPATIAL \
    lifelong=custom \
    use_wandb=false \
    eval.eval=true \
    train.num_workers=0 \
    eval.num_workers=0
