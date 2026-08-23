#!/usr/bin/env bash
# =============================================================================
# NATIVE ThetaEvolve runner STUB  --  CURRENTLY BLOCKED IN THIS ENVIRONMENT
# =============================================================================
# This is the faithful entrypoint for the OFFICIAL ThetaEvolve loop
# (ThetaEvolve/run.sh -> scripts_evolve/<MODEL>/general.sh -> train.py). It is a
# STUB: it documents exactly what the native loop does and the exact blocker, and
# refuses to run rather than silently degrading. Use the PROXY runner
# (cc_eval_theta_openevolve_ailab.sh) for actual evals here; it is protocol-
# faithful on prompt + scoring (see that script and make_theta_faithful_configs.py).
#
# -----------------------------------------------------------------------------
# WHAT THE NATIVE LOOP DOES (and how it differs from our proxy)
# -----------------------------------------------------------------------------
# The native loop is a slime-based test-time RL harness that wraps the SAME
# OpenEvolve evolutionary search as an "evolving-gym" rollout environment:
#
#   ThetaEvolve/train.py
#     -> slime.ray.placement_group.create_rollout_manager  (sglang engines in Ray)
#     -> slime.rollout.rm_hub.evolving_gym_rm.evolving_gym_rm  (scores each rollout
#        via _GYM.response_scorer -> OpenEvolve evaluator -> combined_score)
#     -> GRPO update of the ACTOR (advantage-estimator grpo; general.sh GRPO_ARGS)
#
#   Two official modes (ThetaEvolve/run.sh IS_TRAINING):
#     IS_TRAINING=True  -> full test-time RL: 300 rollouts x 32 batch x 16 samples,
#                          rollout_temperature 1.0, response_len 16384, GRPO update
#                          (eps_clip 0.2/0.28, lr 1e-6) -> the model WEIGHTS evolve
#                          across the search (the paper's "test-time RL" result,
#                          e.g. circle packing 2.63598308 @ ~65 iters with R1-Qwen3-8B).
#     IS_TRAINING=False -> "--debug-rollout-only": the SAME evolving-gym OpenEvolve
#                          loop with a FROZEN model (in-context evolution only).
#
#   Difference vs our proxy (cc_eval_theta_openevolve_ailab.sh):
#     * Our proxy runs the OpenEvolve controller directly (openevolve.cli) with a
#       frozen model served by vLLM. This is functionally the native IS_TRAINING=False
#       inference-only path (same population/MAP-Elites/islands/mutation/selection,
#       same evaluator, same combined_score) -- NOT a different search.
#     * What the proxy CANNOT do is the IS_TRAINING=True weight update (GRPO). So the
#       proxy measures "in-context discovery with a frozen model"; the native RL mode
#       measures "in-context discovery WHILE fine-tuning the model on its own rollouts".
#       These are different experiments; the proxy is faithful to the former only.
#
# -----------------------------------------------------------------------------
# THE EXACT BLOCKER (verified 2026-07-05 on this cluster)
# -----------------------------------------------------------------------------
#   1. DEPENDENCIES: train.py imports `slime`, which pulls `sglang` + NVIDIA
#      `megatron`. None are installed in ANY conda env here:
#        $ python -c "import sglang"   -> ModuleNotFoundError
#        $ python -c "import megatron" -> ModuleNotFoundError
#        $ python -c "import slime"    -> import chain fails
#      The official setup (README.md) is a Docker image
#        slimerl/slime@sha256:704eb14e1b02ef229e4ab440981aa81b543716c335e2af32cb32ffdc030e3008
#      built with flash-attn for Hopper (`TORCH_CUDA_ARCH_LIST=9.0;9.0a` in
#      build_conda.sh / docker/Dockerfile) + apex + transformer-engine + Megatron.
#      Docker is NOT available on ailab compute nodes, and even the inference-only
#      path (--debug-rollout-only) still goes train.py -> create_rollout_manager
#      -> sglang engines, so it needs the SAME stack.
#   2. HARDWARE: general.sh hardcodes `ray start --num-gpus 8` +
#      `--actor-num-gpus-per-node 8`, tensor+context parallel = 2x2, sglang
#      mem-fraction 0.7. README: "for 8B model, we need at least 8x80G GPUs".
#      This node/partition provides 1x A100-40GB (see `nvidia-smi`); the ailab QOS
#      caps a user at 16 GPUs of that class. 8x80G (A100-80G/H100/H200) with NVLink
#      is not available to us.
#   3. NETWORK: the RL mode logs to wandb (WANDB_API_KEY) and general.sh downloads
#      the HF model; ailab compute nodes have NO internet. (Model can be pre-staged,
#      but the wandb + Ray dashboard assumptions add friction.)
#
# -----------------------------------------------------------------------------
# CONCRETE PLAN TO UNBLOCK (if/when infra allows)
# -----------------------------------------------------------------------------
#   A. Obtain an 8x80GB node with NVLink (della-h200 / a dedicated H100 reservation)
#      and either (i) Apptainer/Singularity conversion of the pinned slimerl/slime
#      image, or (ii) build_conda.sh's micromamba env rebuilt for the local CUDA/arch
#      (flash-attn + megatron + apex + transformer-engine + sglang @ the pinned commits).
#   B. Pre-stage the HF model into $SAVE_SHM_DIR and run tools/convert_hf_to_torch_dist.py
#      once (needs Megatron on PYTHONPATH).
#   C. Point ThetaEvolve/run.sh at the local model dir (bypass the `hf download`),
#      set SAVE_PATH to scratch with >200GB free, set IS_TRAINING per experiment.
#   D. For a faithful FROZEN-model comparison without the RL update, run
#      IS_TRAINING=False -- but note this STILL needs the slime/sglang stack; the
#      proxy runner is the only frozen-model path that works without it.
#
# Until A-D are satisfied, this script exits non-zero on purpose.
# =============================================================================
set -euo pipefail

cat >&2 <<'MSG'
[BLOCKED] Native ThetaEvolve slime/GRPO loop cannot run in this environment.
  Missing deps : sglang, megatron, slime (need the slimerl/slime Docker image /
                 build_conda.sh env; Hopper flash-attn; apex; transformer-engine).
  Hardware     : native loop hardcodes 8 GPUs (>=8x80G w/ NVLink per README);
                 this partition provides 1x A100-40GB.
  Even IS_TRAINING=False (--debug-rollout-only) routes through sglang engines.

  USE INSTEAD: FrontierSmith/slurm/cc_eval_theta_openevolve_ailab.sh
    -> runs the SAME OpenEvolve search (MAP-Elites / islands / diff-mutation) with a
       FROZEN local model, protocol-faithful on prompt + scoring
       (config_<task>_qwen35_faithful.yaml). It matches the native IS_TRAINING=False
       (inference-only) path; it does NOT perform the GRPO weight update.

  To unblock the native RL loop, see the "CONCRETE PLAN TO UNBLOCK" section above.
MSG
exit 3
