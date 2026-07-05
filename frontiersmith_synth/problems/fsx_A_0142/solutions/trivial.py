# TIER: trivial
# First-fit in catalogue order, mass- and slot-aware.  This reproduces the
# evaluator's weak first-fit reference exactly, so it anchors at ~0.1.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["slots"]
masses = inst["masses"]

loads = []            # (mass_used, count)
assign = []
for m in masses:
    placed = -1
    for j in range(len(loads)):
        mu, cnt = loads[j]
        if mu + m <= C and cnt + 1 <= K:
            loads[j] = (mu + m, cnt + 1)
            placed = j
            break
    if placed < 0:
        loads.append((m, 1))
        placed = len(loads) - 1
    assign.append(placed)

print(json.dumps({"assign": assign}))
