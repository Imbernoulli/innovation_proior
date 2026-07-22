# TIER: greedy
# The obvious recipe: a plain saturating tally. Use the whole state budget as
# states 0..s-1; a hit symbol advances the state by one (clamped at s-1), a
# non-hit symbol leaves it unchanged; report the state itself as the count.
# This is exact as long as the true count stays under the budget -- but once
# a stream's true hit count blows past s-1 (which happens on the long, dense
# instances) the state freezes at s-1 forever and the reported estimate stops
# tracking reality completely.
import sys, json

inst = json.load(sys.stdin)
m = inst["m"]
s = inst["s"]
k = inst["k"]
residues = set(inst["target_residues"])

n_states = s
hit = [(x % k) in residues for x in range(m)]

trans = []
for state in range(n_states):
    row = []
    for x in range(m):
        if hit[x]:
            row.append(min(state + 1, n_states - 1))
        else:
            row.append(state)
    trans.append(row)

out = [float(state) for state in range(n_states)]

print(json.dumps({"n_states": n_states, "start": 0, "trans": trans, "out": out}))
