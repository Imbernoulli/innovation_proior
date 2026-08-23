#!/usr/bin/env bash
# Single-shot: wait for the 3 RL-checkpoint evals, then print RL-before vs RL-after (FCS/ALE).
set -uo pipefail
cd /scratch/gpfs/CHIJ/bohan/fs
for i in $(seq 1 240); do
  squeue -u "$USER" -h -n cc-rleval-rl_start,cc-rleval-rl_soup_mtv4,cc-rleval-rl_sft_mtv4 -t RUNNING,PENDING 2>/dev/null | grep -q . || break
  sleep 120
done
python3 - <<'PY'
import json,glob,os
def rd(o):
    s=glob.glob(o+"/**/summary.json",recursive=True) or ([o+"/summary.json"] if os.path.exists(o+"/summary.json") else [])
    if not s: return None,None
    d=json.load(open(s[0])); m=d.get('metrics',d)
    f=(m.get('frontiercs') or {}).get('reward'); f=f.get('mean@5') if isinstance(f,dict) else f
    a=(m.get('alebench') or {}).get('performance',{}).get('mean@5')
    return f,a
before={'rl_start':(3.139,359.7),'rl_soup_mtv4':(2.501,313.2),'rl_sft_mtv4':(0.738,292.6)}
print("\n===== RL-vs-start: FrontierCS / ALE (before -> after 40 steps) =====")
print(f"{'model':16s}{'FCS_before':>11}{'FCS_after':>11}{'ALE_before':>11}{'ALE_after':>11}")
for tag in ['rl_start','rl_soup_mtv4','rl_sft_mtv4']:
    fa,aa=rd(f"FrontierSmith/outputs/eval_{tag}_step40_vllm")
    fb,ab=before[tag]
    print(f"{tag:16s}{fb:>11}{str(round(fa,3) if fa is not None else '—'):>11}{ab:>11}{str(round(aa,1) if aa is not None else '—'):>11}")
print("\nThesis: soup+RL after  >  start+RL after  => 'SFT/soup+RL > start+RL'")
PY
