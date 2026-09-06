"""pairprob.py -- paired bootstrap at the PROBLEM level, off allstats.py's JSON dump.

allstats already reduced each arm to one number per problem (mean@5 / best@5 /
worst@5), which is the unit that is actually independent. Pairing on that unit is
the right test for "did arm A beat arm B": the two arms saw the same problems, so
the per-problem difference removes problem difficulty entirely, and resampling
problems (not samples) keeps the interval honest.

  python3 pairprob.py stats/<bench>.json <armA> <armB> [metric]
"""
import json, random, sys

f, A, B = sys.argv[1], sys.argv[2], sys.argv[3]
metric = sys.argv[4] if len(sys.argv) > 4 else "mean@5"
d = json.load(open(f))
pa, pb = d["per_problem"][A][metric], d["per_problem"][B][metric]
probs = [p for p in d["problems"] if p in pa and p in pb]
diff = [pa[p] - pb[p] for p in probs]
n = len(diff)
random.seed(0)
boot = sorted(sum(diff[random.randrange(n)] for _ in range(n)) / n for _ in range(10000))
m = sum(diff) / n
win = sum(1 for x in diff if x > 0); lose = sum(1 for x in diff if x < 0)
print(f"{d['bench']:22s} {metric:8s} {A} - {B}: n={n}  diff={m:+.4f} "
      f"CI[{boot[250]:+.4f},{boot[9750]:+.4f}]  P(>0)={sum(1 for x in boot if x>0)/10000:.3f} "
      f"  per-problem {win}W/{lose}L/{n-win-lose}T")
