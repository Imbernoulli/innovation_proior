# TIER: greedy
# First-fit-decreasing: seal the heaviest artifacts first, each into the lowest-index
# crate with room in BOTH mass and slots.  Packing big items first leaves small ones
# to top off partly-filled crates, so it beats catalogue-order first-fit.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["slots"]
masses = inst["masses"]
N = len(masses)

order = sorted(range(N), key=lambda i: (-masses[i], i))

loads = []                       # (mass_used, count)
assign = [0] * N
for i in order:
    m = masses[i]
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
    assign[i] = placed

print(json.dumps({"assign": assign}))
