# TIER: greedy
# The natural first attempt: parameterize the whole policy by a SINGLE fixed
# reinvest fraction f, applied identically every turn, and grid-search f to
# find the best constant. This is a very reasonable-looking approach -- more
# reinvestment grows capital faster, less reinvestment banks more directly --
# and the grid search even reads the exact thresholds/multipliers/horizon
# from the instance to pick its best constant. But a CONSTANT fraction is
# structurally the wrong shape: it always leaves some of every turn's output
# stranded in capital and un-harvested near the end (unless f=0), AND it
# always harvests some of every turn's output even during the early phase
# where 100% reinvestment would compound fastest toward the next unlock
# (unless f=1). It cannot express "sprint to the next threshold, then dump
# everything to harvest" -- the shape the endgame actually rewards -- so it
# caps out well below a policy that can pick a regime switch.
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
best_f = 0.0
STEPS = 51
for i in range(STEPS):
    f = i / (STEPS - 1)
    val = simulate([f] * N)
    if best_val is None or val > best_val:
        best_val = val
        best_f = f

print(json.dumps({"invest": [best_f] * N}))
