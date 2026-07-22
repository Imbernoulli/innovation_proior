# TIER: strong
# INSIGHT: minimize the EXPECTED makespan, not the mean load.  The probe scenarios
# are an empirical sample of the hidden joint distribution, so the empirical mean
# of max-machine-load over the probe rows is a COVARIANCE-AWARE objective -- it
# already "sees" that co-locating anti-correlated jobs flattens a machine's peaks.
#
# Method (covariance-sampling-probe + LPT-seed + pairwise-swap-refine):
#   1. Seed with LPT on the estimated means (a balanced first-order start).
#   2. Refine by hill-climbing the empirical expected makespan over the probe:
#      single-job RELOCATIONS (fix gross imbalance) and pairwise SWAPS between
#      machines (regroup jobs while preserving mean balance).  Swaps are the key:
#      exchanging a +peak job for a -peak job on another machine drives each
#      machine's net factor loading toward zero -- co-locating anti-correlated
#      jobs so their peaks cancel, which pure mean-balancing never rewards.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
k = inst["k"]
probe = inst["probe"]
S = len(probe)

# ---- LPT seed on estimated means ----
mean = [0.0] * n
for row in probe:
    for j in range(n):
        mean[j] += row[j]
for j in range(n):
    mean[j] /= max(1, S)

order = sorted(range(n), key=lambda j: (-mean[j], j))
mload = [0.0] * k
assign = [0] * n
for j in order:
    best = min(range(k), key=lambda m: (mload[m], m))
    assign[j] = best
    mload[best] += mean[j]

# ---- per-machine, per-scenario load table (incremental empirical makespan) ----
loads = [[0.0] * S for _ in range(k)]
for j in range(n):
    m = assign[j]
    Lm = loads[m]
    for s in range(S):
        Lm[s] += probe[s][j]


def obj():
    t = 0.0
    for s in range(S):
        mx = loads[0][s]
        for m in range(1, k):
            if loads[m][s] > mx:
                mx = loads[m][s]
        t += mx
    return t / S


def mv(j, f, to):
    Lf = loads[f]; Lt = loads[to]
    for s in range(S):
        v = probe[s][j]
        Lf[s] -= v
        Lt[s] += v
    assign[j] = to


cur = obj()
for _ in range(80):
    improved = False
    # single-job relocations
    for j in range(n):
        f = assign[j]
        best_to = f
        best_val = cur
        for to in range(k):
            if to == f:
                continue
            mv(j, f, to)
            v = obj()
            if v < best_val - 1e-9:
                best_val = v
                best_to = to
            mv(j, to, f)          # revert
        if best_to != f:
            mv(j, f, best_to)
            cur = best_val
            improved = True
    # pairwise swaps between machines (exchange argument)
    for x in range(n):
        for y in range(x + 1, n):
            if assign[x] == assign[y]:
                continue
            mx_, my = assign[x], assign[y]
            mv(x, mx_, my)
            mv(y, my, mx_)
            v = obj()
            if v < cur - 1e-9:
                cur = v
                improved = True
            else:
                mv(x, my, mx_)     # revert
                mv(y, mx_, my)
    if not improved:
        break

print(json.dumps({"assign": assign}))
