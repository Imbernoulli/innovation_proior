# TIER: trivial
# Equal, never-adapting split: every round, divide the A ants as evenly as
# possible across all K sources and never change it. Ignores rate, stock,
# regen, decay -- everything. This is the evaluator's own "do nothing clever"
# reference, so it anchors at ~0.1.
import sys, json

inst = json.load(sys.stdin)
K = inst["K"]
T = inst["T"]
A = inst["A"]

base = A // K
rem = A - base * K
row = [base] * K
for i in range(rem):
    row[i] += 1

print(json.dumps({"alloc": [row[:] for _ in range(T)]}))
