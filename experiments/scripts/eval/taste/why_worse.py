#!/usr/bin/env python3
"""Where does the 4B SFT arm's SciJudge deficit actually come from?

Splits the swap-consistent result three ways:
  (a) by how well-known the papers are (citation magnitude) -- tests whether the
      SFT eroded literature RECALL rather than judgement,
  (b) by how much the arm thought -- tests the under-reasoning hypothesis,
  (c) position bias.
"""
import json, os, sys, statistics
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benches import extract_ab

SJ = {}
for i, l in enumerate(open(".cache/taste_eval/scijudge_test.jsonl")):
    r = json.loads(l)
    SJ[f"{i:05d}_{r.get('paper_a_arxiv_id')}"] = r

def arm(path):
    rows = {}
    for l in open(path):
        r = json.loads(l); rows[r["id"]] = r
    by = defaultdict(dict)
    for r in rows.values():
        by[r["meta"]["pair"]]["swap" if r["meta"]["swap"] else "plain"] = r
    out = {}
    for pid, d in by.items():
        if "plain" not in d or "swap" not in d: continue
        ok = all(extract_ab(d[k].get("output") or "") == d[k]["meta"]["gold"] for k in ("plain","swap"))
        tok = sum((d[k].get("completion_tokens") or 0) for k in ("plain","swap"))/2
        picks = [extract_ab(d[k].get("output") or "") for k in ("plain","swap")]
        out[pid] = (1.0 if ok else 0.0, tok, picks)
    return out

A = {n: arm(f"outputs_taste/run4b_redo/cc_eval_pp_{n}_scijudge.jsonl")
     for n in ["base","wd01","soup_a10"]}
common = set.intersection(*(set(v) for v in A.values()))
print(f"共有 {len(common)} 对\n")

print("(a) 按 winner 的引用量分层  —— 检验是不是「文献记忆」被破坏")
def maxcit(pid):
    r = SJ.get(pid)
    if not r: return None
    return max(r["paper_a_citations"], r["paper_b_citations"])
bins = [(0,20),(20,100),(100,500),(500,10**9)]
print(f"{'winner 引用量':>16} {'n':>5} " + " ".join(f"{n:>9}" for n in A))
for lo,hi in bins:
    ids=[p for p in common if (c:=maxcit(p)) is not None and lo<=c<hi]
    if len(ids)<30: continue
    row=" ".join(f"{statistics.mean(A[n][p][0] for p in ids):9.3f}" for n in A)
    print(f"{f'{lo}-{hi if hi<10**9 else '+'}':>16} {len(ids):5d} {row}")

print("\n(b) 按该臂自己的思考长度分层 —— 检验「想得太短」")
for n in A:
    toks=sorted(A[n][p][1] for p in common)
    q=[toks[int(f*len(toks))] for f in (0.25,0.5,0.75)]
    print(f"  {n:9s} 思考 token 中位={q[1]:6.0f}  四分位 {q[0]:.0f}/{q[2]:.0f}")
    for lo,hi,lbl in [(0,q[0],"最短25%"),(q[0],q[2],"中间50%"),(q[2],10**9,"最长25%")]:
        ids=[p for p in common if lo<=A[n][p][1]<hi]
        if ids: print(f"      {lbl:8s} n={len(ids):4d}  准确率={statistics.mean(A[n][p][0] for p in ids):.3f}")

print("\n(c) 位置偏置 picked-A（理想 0.50）与作答率")
for n in A:
    picks=[x for p in common for x in A[n][p][2]]
    valid=[x for x in picks if x]
    print(f"  {n:9s} picked-A={sum(1 for x in valid if x=='A')/len(valid):.3f}  作答率={len(valid)/len(picks):.3f}")
