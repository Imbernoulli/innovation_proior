#!/usr/bin/env bash
#SBATCH --job-name=build_flashattn
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/logs/%x-%j.err
# Build flash-attn from source against the RL venv's torch 2.11+cu130 (no prebuilt
# wheel exists for that ABI). CPU-only cross-compile for H200 (sm90). If it fails,
# no harm: RL still runs on sdpa. This is the biggest RL speedup + restores verl's
# verified Ulysses/rmpad flash-attention path.
set -uo pipefail
cd /scratch/gpfs/CHIJ/bohan/fs/FrontierSmith
# `module` is a shell function — must init it in a non-login batch shell first.
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/local/share/Modules/init/bash 2>/dev/null || true
module load cudatoolkit/13.0 2>/dev/null || true
# Derive CUDA_HOME from whatever the module put on PATH (don't hardcode).
NVCC="$(command -v nvcc || true)"
if [ -z "$NVCC" ]; then echo "FATAL: nvcc not on PATH after module load — CUDA not available on this node"; exit 2; fi
export CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
export PATH="$CUDA_HOME/bin:$PATH"
echo "[build] using CUDA_HOME=$CUDA_HOME nvcc=$NVCC"
export TORCH_CUDA_ARCH_LIST="9.0"      # H200 = sm90; cross-compile, no GPU needed
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=32
PY=.venv-vllm023/bin/python
PIP=.venv-vllm023/bin/pip
echo "[build] host=$(hostname) nvcc=$($CUDA_HOME/bin/nvcc --version | tail -1)"
echo "[build] torch: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
echo "[build] ninja: $($PY -c 'import ninja;print(ninja.__version__)' 2>/dev/null || echo missing)"
# Compute nodes are OFFLINE — build from the cached source tarball, --no-index.
SRC=/scratch/gpfs/CHIJ/bohan/fs/wheelhouse/flash_attn-2.8.3.post1.tar.gz
echo "===== building from local source: $SRC ====="
$PIP install "$SRC" --no-build-isolation --no-index 2>&1 | tail -40
if $PY -c "import flash_attn; print('flash_attn OK', flash_attn.__version__)" 2>/dev/null; then
  echo "SUCCESS: flash_attn imports"
  $PY -c "import flash_attn, torch; from flash_attn import flash_attn_varlen_func; print('varlen func OK')" 2>&1 | tail -3
  exit 0
fi
echo "BUILD FAILED (likely torch2.11/cu130 ABI) — RL stays on sdpa (no harm)"
exit 1
