"""ideastats.py -- per-item @5 accuracy on the research-taste tasks + paired test.

Same unit discipline as allstats.py: reduce the 5 draws of an item to one number,
then bootstrap over ITEMS. Accuracy is exact match against third-party ground truth,
so there is no judge anywhere in the loop.

Exactly two tags also prints the paired test between them.

The 8192-budget run is kept on disk under the cc_idea_ prefix and the correct-budget
run under cc_idea32k_. They are NOT interchangeable: at 8192 the ranking is essentially
the reverse ranking of the truncation rate (Spearman 0.79 between the 8192 unanswered
rate and the gain from re-running at 32768), so anything read off the short-budget run
is an artifact. PREFIX exists only so that comparison can be reproduced.

  python3 ideastats.py <tagA> <tagB> ...
  PREFIX=cc_idea_ python3 ideastats.py <tagA> <tagB> ...   # the void 8192 run
"""
import json, glob, os, random, sys
from collections import defaultdict

D = os.environ.get("INNOV_OUTPUTS", "/scratch/gpfs/CHIJ/ziran/innov_v2_multi/outputs")
PREFIX = os.environ.get("PREFIX", "cc_idea32k_")

def load(tag):
    per = defaultdict(dict)
    p = f"{D}/{PREFIX}{tag}/samples.jsonl"
    if not os.path.exists(p):
        return {}
    for line in open(p):
        try: r = json.loads(line)
        except Exception: continue
        per[(r["task"], r["id"])][r["sample_idx"]] = r
    out = {}
    for k, v in per.items():
        if len(v) < 5: continue
        ks = sorted(v)[:5]
        out[k] = {"acc": sum(v[i]["correct"] for i in ks) / 5,
                  "any": float(any(v[i]["correct"] for i in ks)),
                  "all": float(all(v[i]["correct"] for i in ks)),
                  "unp": sum(bool(v[i].get("unparsed")) for i in ks) / 5}
    return out

def boot(vals, n=10000, seed=0):
    random.seed(seed); N=len(vals)
    b=sorted(sum(vals[random.randrange(N)] for _ in range(N))/N for _ in range(n))
    return b[int(.025*n)], b[int(.975*n)]

def pairboot(d, n=10000, seed=0):
    random.seed(seed); N=len(d)
    b=sorted(sum(d[random.randrange(N)] for _ in range(N))/N for _ in range(n))
    return sum(d)/N, b[int(.025*n)], b[int(.975*n)], sum(1 for x in b if x>0)/n

tags=sys.argv[1:]
data={t:load(t) for t in tags}
tags=[t for t in tags if data[t]]
tasks=sorted({k[0] for t in tags for k in data[t]})
for task in tasks:
    common=sorted(set.intersection(*[{k for k in data[t] if k[0]==task} for t in tags]))
    print(f"\n=== {task}  n={len(common)} items, {len(tags)} arms ===")
    print(f"  {'arm':30s} {'mean@5':>8s} {'[95% CI]':>18s} {'pass@5':>8s} {'all5':>7s} {'unparsed':>9s}")
    rows=[]
    for t in tags:
        a=[data[t][k]["acc"] for k in common]
        rows.append((sum(a)/len(a), t, boot(a),
                     sum(data[t][k]["any"] for k in common)/len(common),
                     sum(data[t][k]["all"] for k in common)/len(common),
                     sum(data[t][k]["unp"] for k in common)/len(common)))
    for m,t,(lo,hi),pa,al,un in sorted(rows, reverse=True):
        print(f"  {t:30s} {m:8.4f} [{lo:.3f},{hi:.3f}] {pa:8.1%} {al:7.1%} {un:9.1%}")
    if len(tags)==2:
        A,B=tags
        d=[data[A][k]["acc"]-data[B][k]["acc"] for k in common]
        m,lo,hi,p=pairboot(d)
        w=sum(1 for x in d if x>0); l=sum(1 for x in d if x<0)
        print(f"  paired {A} - {B}: {m:+.4f} CI[{lo:+.4f},{hi:+.4f}] P(>0)={p:.3f}  {w}W/{l}L/{len(d)-w-l}T")
