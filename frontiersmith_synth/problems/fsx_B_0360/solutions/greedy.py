# TIER: greedy
# Density greedy that upgrades each cut STRAIGHT to its single highest-value option,
# ranked by (value_gain / fuel_gain). Ignores the diminishing-returns / multiple-choice
# structure, so it spends budget on inefficient heavy units and leaves value behind.
import sys, json
inst = json.load(sys.stdin)
N = inst["n_cuts"]
fuel = inst["fuel"]
value = inst["value"]
B = inst["budget"]

assign = [0] * N
used = sum(fuel[i][0] for i in range(N))

cand = []
for i in range(N):
    bj = max(range(len(value[i])), key=lambda j: value[i][j])
    df = fuel[i][bj] - fuel[i][0]
    dv = value[i][bj] - value[i][0]
    if df > 0 and dv > 0:
        cand.append((dv / df, i, bj, df))
cand.sort(reverse=True)

for _dens, i, bj, df in cand:
    if used + df <= B:
        assign[i] = bj
        used += df

print(json.dumps({"assign": assign}))
