#!/usr/bin/env bash
# Single-shot: wait until the r1 FCS eval/merge jobs drain, then print the full FrontierCS leaderboard.
set -uo pipefail
for i in $(seq 1 300); do
  squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qE "cc-eval-(sftr1|soupr1|lorar1)|cc-merge|cc-soup" || break
  sleep 180
done
cd /scratch/gpfs/CHIJ/bohan/fs
python3 - <<'PY'
import json, glob, re
rows=[]
for o in sorted(glob.glob("FrontierSmith/outputs/cc_eval_*r1*_thinking_32k_both_vllm/summary.json")):
    tag=re.search(r'cc_eval_(.+?)_thinking',o).group(1)
    try:
        d=json.load(open(o)); fcs=d['metrics']['frontiercs']['reward']; fcs=fcs.get('mean@5') if isinstance(fcs,dict) else fcs
        ale=d['metrics']['alebench']['performance'].get('mean@5')
        rows.append((round(fcs,3), tag, round(ale,1) if ale else None))
    except: pass
rows.sort(reverse=True)
print(f"\n===== FrontierCS leaderboard ({len(rows)} models) — baselines: start 3.139 / old full-FT method 0.015 =====")
for f,t,a in rows: print(f"  FCS={f:>7}  ALE={str(a):>6}  {t}")
PY
