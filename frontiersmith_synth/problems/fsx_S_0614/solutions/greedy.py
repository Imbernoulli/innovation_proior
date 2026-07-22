# TIER: greedy
# THE OBVIOUS RECIPE (the trap): ratio-greedy first-fit.  Sort items by standalone
# value/weight descending and drop each into the first bin that still has room.  This is
# what an average strong coder writes first for a multi-bin knapsack.  It NEVER looks at
# the synergy table, so it fills bins with high-ratio decoy singletons, fragments the
# capacity, and only accidentally co-locates a synergy pair.  On trap instances (where the
# value is hidden in mediocre-ratio synergy pairs) it collects almost none of the bonus.
import sys, json

inst = json.load(sys.stdin)
N, M = inst["N"], inst["M"]
C = list(inst["C"])
w, v = inst["w"], inst["v"]

order = sorted(range(N), key=lambda i: (v[i] / w[i] if w[i] > 0 else 0.0), reverse=True)
assign = [-1] * N
rem = list(C)
for i in order:
    for b in range(M):
        if rem[b] >= w[i]:
            assign[i] = b
            rem[b] -= w[i]
            break

print(json.dumps({"assign": assign}))
