#!/usr/bin/env python3
"""Print the current SFT/RL results matrix from whatever evals have landed.

Reads outputs/cc_eval_<tag>_{thinking_32k_both_vllm,research_thinking_32k_vllm}
and reports FCS-172, ALE-40 and Research-64 per tag. Partial runs are shown with
their sample counts so a half-finished eval can never be mistaken for a final
number (that mistake cost this campaign a retracted table once already).
"""
import json, glob, os, sys
from collections import defaultdict

def score(pat, src=None):
    per=defaultdict(dict); errs=0
    for f in glob.glob(pat):
        for line in open(f):
            line=line.strip()
            if not line: continue
            try: r=json.loads(line)
            except Exception: continue
            if src and r.get("data_source")!=src: continue
            if r.get("error"): errs+=1; continue
            s=(r.get("metrics") or {}).get("score")
            if s is not None: per[r["ground_truth"]][int(r["sample_idx"])]=float(s)
    if not per: return None
    n=len(per); tot=sum(len(v) for v in per.values())
    return (sum(sum(v.values())/len(v) for v in per.values())/n, n, tot, errs)

def main():
    pref = sys.argv[1] if len(sys.argv)>1 else ""
    tags=set()
    for d in glob.glob("outputs/cc_eval_*"):
        b=os.path.basename(d)
        for suf in ("_thinking_32k_both_vllm","_research_thinking_32k_vllm","_ale40_thinking_32k_vllm"):
            if b.endswith(suf): tags.add(b[len("cc_eval_"):-len(suf)])
    rows=[]
    for t in sorted(tags):
        if pref and pref not in t: continue
        f=score(f"outputs/cc_eval_{t}_thinking_32k_both_vllm/shard_*/samples.jsonl","frontiercs")
        a=score(f"outputs/cc_eval_{t}_thinking_32k_both_vllm/shard_*/samples.jsonl","alebench")
        r=score(f"outputs/cc_eval_{t}_research_thinking_32k_vllm/shard_*/samples.jsonl")
        if not any([f,a,r]): continue
        rows.append((t,f,a,r))
    fmt=lambda x,d=2: f"{x[0]:.{d}f}({x[1]})" if x else "--"
    print(f"{'tag':46s} {'FCS(n)':>14s} {'ALE(n)':>12s} {'Research(n)':>14s}")
    for t,f,a,r in rows:
        print(f"{t:46s} {fmt(f):>14s} {fmt(a,1):>12s} {fmt(r):>14s}")

if __name__=="__main__":
    main()
