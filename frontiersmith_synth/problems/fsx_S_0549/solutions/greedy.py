# TIER: greedy
# The obvious approach: LONGEST-PROCESSING-TIME on the estimated MEAN load.
# Estimate each job's mean processing time from the probe columns, sort jobs
# big-to-small, and drop each onto the machine with the smallest running MEAN
# load.  This is the textbook makespan heuristic -- it balances first-order load
# well, but it is BLIND to covariance, so it happily splits anti-correlated jobs
# across machines and leaves them volatile.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
k = inst["k"]
probe = inst["probe"]
S = len(probe)

mean = [0.0] * n
for row in probe:
    for j in range(n):
        mean[j] += row[j]
for j in range(n):
    mean[j] /= max(1, S)

order = sorted(range(n), key=lambda j: (-mean[j], j))
load = [0.0] * k
assign = [0] * n
for j in order:
    best = min(range(k), key=lambda m: (load[m], m))
    assign[j] = best
    load[best] += mean[j]

print(json.dumps({"assign": assign}))
