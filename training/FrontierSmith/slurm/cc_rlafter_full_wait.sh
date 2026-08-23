#!/usr/bin/env bash
# Wait for soup+RL FCS re-run + the 3 RL-after MLS, then print full RL before->after (FCS/ALE/MLS).
set -uo pipefail
cd /scratch/gpfs/CHIJ/bohan/fs
for i in $(seq 1 360); do
  squeue -u "$USER" -h -n cc-eval-rlafter_rl_soup_mtv4,cc-mlsdevfix-rlafter_rl_start,cc-mlsdevfix-rlafter_rl_soup_mtv4,cc-mlsdevfix-rlafter_rl_sft_mtv4 -t RUNNING,PENDING 2>/dev/null | grep -q . || break
  sleep 120
done
python3 - <<'PY'
import json,os
def fcsale(tag):
    o=f"FrontierSmith/outputs/cc_eval_{tag}_thinking_32k_both_vllm/summary.json"
    if not os.path.exists(o): return None,None
    d=json.load(open(o)); m=d['metrics']; f=m['frontiercs']['reward']; f=f.get('mean@5') if isinstance(f,dict) else f
    return round(f,3), round(m['alebench']['performance'].get('mean@5'),1)
def mls(tag):
    o=f"FrontierSmith/outputs/cc_mlsbench_cpu_{tag}/summary.json"
    return round(json.load(open(o))['mean_score'],4) if os.path.exists(o) else None
# before: FCS/ALE from cc_eval_thinking; MLS from the r1 devfix batch
before={'rl_start':(3.139,359.7,0.0752),'rl_soup_mtv4':(2.501,313.2,0.1112),'rl_sft_mtv4':(0.738,292.6,0.1023)}
print("\n===== RL before -> after (40 steps), MATCHED harness =====")
print(f"{'model':14s}{'FCS':>16}{'ALE':>16}{'MLS':>18}")
for tag in ['rl_start','rl_soup_mtv4','rl_sft_mtv4']:
    fa,aa=fcsale(f"rlafter_{tag}"); ma=mls(f"rlafter_{tag}_devfix"); fb,ab,mb=before[tag]
    print(f"{tag:14s}{f'{fb}->{fa}':>16}{f'{ab}->{aa}':>16}{f'{mb}->{ma}':>18}")
print("\nthesis: a soup/sft model whose RL-after beats start's RL-after on SOME benchmark")
PY
