#!/usr/bin/env bash
# Wait for the 3 matched RL-after evals, then print clean before->after (same 32k harness).
set -uo pipefail
cd /scratch/gpfs/CHIJ/bohan/fs
for i in $(seq 1 300); do
  squeue -u "$USER" -h -n cc-eval-rlafter_rl_start,cc-eval-rlafter_rl_soup_mtv4,cc-eval-rlafter_rl_sft_mtv4 -t RUNNING,PENDING 2>/dev/null | grep -q . || break
  sleep 120
done
python3 - <<'PY'
import json,glob,os
def fcsale(tag):
    o=f"FrontierSmith/outputs/cc_eval_{tag}_thinking_32k_both_vllm/summary.json"
    if not os.path.exists(o): return None,None
    d=json.load(open(o)); m=d['metrics']
    f=m['frontiercs']['reward']; f=f.get('mean@5') if isinstance(f,dict) else f
    a=m['alebench']['performance'].get('mean@5')
    return round(f,3),round(a,1)
before={'rl_start':(3.139,359.7),'rl_soup_mtv4':(2.501,313.2),'rl_sft_mtv4':(0.738,292.6)}
print("\n===== MATCHED (32k) before -> after 40 RL steps =====")
print(f"{'model':14s}{'FCS_bef':>9}{'FCS_aft':>9}{'dFCS':>8}   {'ALE_bef':>8}{'ALE_aft':>8}{'dALE':>8}")
for tag in ['rl_start','rl_soup_mtv4','rl_sft_mtv4']:
    fa,aa=fcsale(f"rlafter_{tag}"); fb,ab=before[tag]
    df=f"{fa-fb:+.3f}" if fa is not None else "—"; da=f"{aa-ab:+.1f}" if aa is not None else "—"
    print(f"{tag:14s}{fb:>9}{str(fa):>9}{df:>8}   {ab:>8}{str(aa):>8}{da:>8}")
print("\nthesis holds iff: soup+RL gain (dFCS/dALE) > start+RL gain")
PY
