# TIER: strong
# THE INSIGHT: reinvestment always compounds fastest the earlier it happens
# (an extra unit of capital at turn t is worth at least as much by the end
# as the same unit added at any later turn, since capital differences never
# shrink under continued reinvestment). So instead of searching the huge
# space of per-turn fractions, or restricting to one constant fraction, we
# search the much smaller and much more expressive space of SINGLE-SWITCH
# bang-bang policies: reinvest 100% of output for the first tau turns (front-
# loading every dollar of compounding as early as possible), then harvest
# 100% for the remaining N-tau turns (converting to the bank before the
# endgame's salvage discount can strand it). We try EVERY switch turn
# tau = 0..N (using the instance's own thresholds, multipliers and horizon --
# no fixed ratio of N) and keep whichever simulated final score is largest.
# This naturally discovers the right answer to "is the next unlock worth
# chasing with the turns I have left?" per instance, instead of assuming a
# fixed reinvest ratio -- exactly the switch a constant-fraction policy
# cannot express.
import sys, json

inst = json.load(sys.stdin)
N = inst["n_turns"]
K0 = inst["capital0"]
base_rate = inst["base_rate"]
thresholds = inst["thresholds"]
multipliers = inst["multipliers"]
salvage = inst["salvage"]


def tier_of(K):
    t = 0
    for T in thresholds:
        if K >= T:
            t += 1
        else:
            break
    return t


def simulate(invest_seq):
    K = K0
    B = 0.0
    for t in range(N):
        tier = tier_of(K)
        rate = base_rate * multipliers[tier]
        output = K * rate
        f = invest_seq[t]
        inv = f * output
        K += inv
        B += (output - inv)
    return B + salvage * K


best_val = None
best_tau = 0
for tau in range(N + 1):
    seq = [1.0] * tau + [0.0] * (N - tau)
    val = simulate(seq)
    if best_val is None or val > best_val:
        best_val = val
        best_tau = tau

policy = [1.0] * best_tau + [0.0] * (N - best_tau)
print(json.dumps({"invest": policy}))
