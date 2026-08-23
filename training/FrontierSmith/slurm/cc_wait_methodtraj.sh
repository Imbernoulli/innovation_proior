#!/usr/bin/env bash
# Single-shot: wait until the 5 methodtraj r1 SFT jobs leave the queue, then run the
# idempotent post-SFT submitter once (handles their soup/merge + FCS eval). Robust (no submit-in-loop).
set -uo pipefail
J="sft_q35_methodtraj_r1,sft_q35_methodtraj_v4_r1,lora_q35_methodtraj_r1_r32,lora_q35_methodtraj_v4_r1_r32,lora_q35_methodtraj_v4_r1_r64"
for i in $(seq 1 240); do   # up to 8h
  squeue -u "$USER" -h -n "$J" -t RUNNING,PENDING 2>/dev/null | grep -q . || break
  sleep 120
done
echo "methodtraj jobs left the queue; running post-SFT submitter"
bash /scratch/gpfs/CHIJ/bohan/fs/FrontierSmith/slurm/cc_post_sft_submit.sh
