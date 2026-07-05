# TIER: trivial
# First-come-first-served dispatch: give the cuts priority strictly in order of
# arrival (release time), ties by index.  This reproduces the evaluator's FCFS
# reference exactly, so every instance scores ~0.1 -- a naive yard default that
# ignores weights and cut-offs entirely.
import sys, json

inst = json.load(sys.stdin)
cuts = inst["cuts"]
order = sorted(range(len(cuts)), key=lambda i: (cuts[i]["r"], i))
print(json.dumps({"order": order}))
