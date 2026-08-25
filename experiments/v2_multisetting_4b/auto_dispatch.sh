#!/usr/bin/env bash
# Waits for GPU pairs to go idle, then launches the three v2 settings.
D=/srv/home/bohanlyu/innovation_proior/experiments/v2_multisetting_4b
free_pair() {  # both GPUs < 1000 MiB used and no compute procs
  local a=$1 b=$2
  local ua ub
  ua=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $a)
  ub=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $b)
  [ "$ua" -lt 1000 ] && [ "$ub" -lt 1000 ]
}
declare -A LAUNCHED
while true; do
  if [ -z "${LAUNCHED[wd01]:-}" ] && free_pair 6 7; then
    FS_ALLOW_GPU67=1 bash "$D/launch_setting.sh" full_wd01 6,7 && LAUNCHED[wd01]=1 && echo "[dispatch] full_wd01 -> 6,7"
  fi
  if [ -z "${LAUNCHED[wd03]:-}" ] && free_pair 0 1; then
    bash "$D/launch_setting.sh" full_wd03 0,1 && LAUNCHED[wd03]=1 && echo "[dispatch] full_wd03 -> 0,1"
  fi
  if [ -z "${LAUNCHED[lora]:-}" ] && free_pair 2 3; then
    bash "$D/launch_setting.sh" lora_r32 2,3 && LAUNCHED[lora]=1 && echo "[dispatch] lora_r32 -> 2,3"
  fi
  [ -n "${LAUNCHED[wd01]:-}" ] && [ -n "${LAUNCHED[wd03]:-}" ] && [ -n "${LAUNCHED[lora]:-}" ] && { echo "[dispatch] all three launched"; exit 0; }
  sleep 120
done
