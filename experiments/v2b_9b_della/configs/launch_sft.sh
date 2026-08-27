#!/usr/bin/env bash
#SBATCH --job-name=innov-sft
#SBATCH --partition=ailab
#SBATCH --account=chij
#SBATCH --qos=short
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/gpfs/CHIJ/ziran/innov_v2_multi/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/CHIJ/ziran/innov_v2_multi/logs/%x-%j.err
set -uo pipefail
TRAIN_CFG="$(readlink -f "${1:?need LF train yaml}")"
[ -f "$TRAIN_CFG" ] || { echo "config not found: $TRAIN_CFG"; exit 2; }
LF=/scratch/gpfs/CHIJ/bohan/fs/LF-innov
ENV=/scratch/gpfs/CHIJ/bohan/fs/envs/sft_lf
cd "$LF"
export PATH="$ENV/bin:$PATH"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cudatoolkit/12.8 2>/dev/null || true
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
[ -x "$CUDA_HOME/bin/nvcc" ] && export PATH="$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="/scratch/gpfs/CHIJ/bohan/fs/cudafix/lib:${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/scratch/gpfs/CHIJ/bohan/fs/cudafix/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
# Per-arm HF_HOME when set on the sbatch line: LlamaFactory keys its tokenized-dataset
# cache on the dataset combo, so two concurrent arms sharing a combo (full-FT and LoRA
# both on innov_v2,maintain_w2w3) would race each other under overwrite_cache:true.
export HF_HOME="${HF_HOME:-/scratch/gpfs/CHIJ/ziran/innov_v2_multi/.hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false WANDB_DISABLED=true WANDB_MODE=offline
export DISABLE_VERSION_CHECK=1 TMPDIR=/tmp PYTHONUNBUFFERED=1 FORCE_TORCHRUN=1
NGPU=$(nvidia-smi -L | wc -l)
echo "node=$(hostname) env=$ENV NGPU=$NGPU cfg=$TRAIN_CFG start=$(date -Is)"
python -c "import llamafactory,os;print('LF source:',os.path.dirname(llamafactory.__file__))"
python -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__)"
llamafactory-cli train "$TRAIN_CFG"
echo "DONE rc=$? $(date -Is)"
