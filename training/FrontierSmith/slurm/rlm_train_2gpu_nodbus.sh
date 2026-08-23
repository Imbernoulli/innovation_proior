#!/usr/bin/env bash
# rlm_* wrapper: stop the FrontierCS go-judge from being SIGKILLed during
# co-located vLLM/Ray startup, then exec the validated 2-GPU per-task-norm
# launcher (forwarding all extra args). Used ONLY by the rlm_* amplify runs.
#
# ROOT CAUSE (diagnosed + reproduced 3x): go-judge v1.11.1 on this cgroup-v2 node
# creates its cgroup as a systemd transient scope INSIDE the per-user slice
# (user.slice/user-372967.slice/.../gojudge-<jobid>.scope). That user slice has a
# shared MemoryMax of ~151 GB (systemctl show user-372967.slice -p MemoryMax =>
# 162126266368), independent of the Slurm job's --mem cgroup. Under co-located
# load the USER-SLICE cgroup OOM-killer reaps go-judge right after it serves
# /version -> the apptainer node judge on :8092 never comes up -> RL training
# crashes at first reward with JudgeInfraError (Connection refused :8092). This is
# why bumping job --mem to 700G did NOT help (different cgroup hierarchy).
#
# FIX (validated on the login node against the actual binary): point
# DBUS_SESSION_BUS_ADDRESS at a dead socket. go-judge then logs
#   "connecting to systemd dbus failed, assuming running in container, enable
#    cgroup v2 nesting support and take control of the whole cgroupfs"
# and nests its cgroup at the unified root /sys/fs/cgroup instead of the capped
# user slice -- escaping the ~151 GB user-slice cap, so the OOM-killer no longer
# reaps it. (Merely unsetting DBUS_SESSION_BUS_ADDRESS/XDG_RUNTIME_DIR does NOT
# work: go-judge still reaches the well-known socket /run/user/<uid>/bus.)
# go-judge v1.11.1 has NO CLI flag to disable systemd-dbus cgroup creation, so
# this dead-socket redirect is the supported lever. The unchanged shared starter
# scripts/start_frontiercs_judge_hybrid.sh inherits this env.
#SBATCH --job-name=rlm-train-2g
#SBATCH --partition=ailab
#SBATCH --qos=gpu-short
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=750G
#SBATCH --time=23:59:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# CO-LOCATION FIX: --mem=750G. Two of these 2-GPU verl runs co-located on ONE
# ailab node (1500 GB RealMemory, 8 GPUs) crash each other: combined host-RAM
# (~370 GB peak each) + two Ray clusters competing -> raylet SIGKILL/OOM and
# FrontierCS judge per-submission timeouts (the run-killers seen on della-i19g1
# where soupa00+soupa100 both died). At 750G, two jobs (=1500G) cannot fit on one
# node, so Slurm SPREADS one rlm run per node -- removing the co-location AND
# giving each verl run ample host RAM. Still 2 GPUs/run (the right size); well
# under the QOS GPU cap. A single run per node was rock-solid in the smoke.

set -euo pipefail
# Sever go-judge's systemd dbus path so it nests at /sys/fs/cgroup (root), NOT
# the capped user slice. Dead unix socket -> systemd connect refused -> cgroupfs
# nesting fallback. Also drop XDG_RUNTIME_DIR for good measure.
export DBUS_SESSION_BUS_ADDRESS="unix:path=/dev/null"
unset XDG_RUNTIME_DIR || true
echo "[rlm wrapper] DBUS_SESSION_BUS_ADDRESS=unix:path=/dev/null -> go-judge nests cgroup at /sys/fs/cgroup root (escapes user-slice ~151G cap)"

# JOB-UNIQUE judge ports so two co-located rlm_* runs on the SAME node do not
# collide on the recipe's fixed node:8092 / go-judge:5060 ports (the second run
# otherwise dies with "bind: address already in use" / "judge launcher exited
# early"). The training chain honors inherited PORT/GJ_PORT (${PORT:-8092} /
# ${GJ_PORT:-5060}); FRONTIERCS_JUDGE_URL is derived from PORT downstream.
# RUNTIME_DIR and GJ_DIR are already job-id-unique in the chain.
JOBU=$(( ${SLURM_JOB_ID:-$$} % 9000 ))
export PORT="${PORT:-$(( 18000 + JOBU ))}"
export GJ_PORT="${GJ_PORT:-$(( 41000 + JOBU ))}"
export GJ_CGROUP_PREFIX="${GJ_CGROUP_PREFIX:-gojudge-rlm-${SLURM_JOB_ID:-$$}}"
echo "[rlm wrapper] unique judge ports: node PORT=$PORT go-judge GJ_PORT=$GJ_PORT"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/scratch/gpfs/CHIJ/bohan/fs/FrontierSmith}"
cd "$PROJECT_ROOT"
exec bash slurm/cc_tt_train_mixed_pertask_norm_2gpu_ailab.sh "$@"
