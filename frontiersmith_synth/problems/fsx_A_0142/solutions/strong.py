# TIER: strong
# Best-fit-decreasing (mass- and slot-aware) followed by a deterministic
# "empty-the-weakest-crate" local search: repeatedly try to dissolve the crate
# with the fewest artifacts by re-homing its contents (best-fit) into the other
# crates.  Whenever a crate is fully emptied, the crate count drops by one.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["slots"]
masses = inst["masses"]
N = len(masses)


def bfd(order):
    """Best-fit-decreasing over the given item order. Returns list of crates,
    each a list of item indices."""
    crates = []          # list of [item_idx,...]
    mu = []              # mass used per crate
    for i in order:
        m = masses[i]
        best = -1
        best_slack = None
        for j in range(len(crates)):
            if mu[j] + m <= C and len(crates[j]) + 1 <= K:
                slack = C - (mu[j] + m)        # tightest fit -> smallest leftover
                if best_slack is None or slack < best_slack:
                    best_slack = slack
                    best = j
        if best < 0:
            crates.append([i])
            mu.append(m)
        else:
            crates[best].append(i)
            mu[best] += m
    return crates, mu


order = sorted(range(N), key=lambda i: (-masses[i], i))
crates, mu = bfd(order)

# ---- local search: dissolve the weakest (fewest-item) crate repeatedly ----
def total_mass(idxs):
    return sum(masses[i] for i in idxs)

improved = True
passes = 0
while improved and passes < 40:
    improved = False
    passes += 1
    if len(crates) <= 1:
        break
    # pick the crate with the fewest artifacts (ties: least total mass, then index)
    victim = min(range(len(crates)),
                 key=lambda j: (len(crates[j]), mu[j], j))
    items = sorted(crates[victim], key=lambda i: (-masses[i], i))
    # try to place every item of the victim into some OTHER crate (best-fit)
    trial = {j: (mu[j], len(crates[j])) for j in range(len(crates)) if j != victim}
    placement = {}
    ok = True
    for i in items:
        m = masses[i]
        best = -1
        best_slack = None
        for j in trial:
            cm, cc = trial[j]
            if cm + m <= C and cc + 1 <= K:
                slack = C - (cm + m)
                if best_slack is None or slack < best_slack:
                    best_slack = slack
                    best = j
        if best < 0:
            ok = False
            break
        cm, cc = trial[best]
        trial[best] = (cm + m, cc + 1)
        placement[i] = best
    if ok:
        # commit: move victim's items, drop the victim crate
        for i, j in placement.items():
            crates[j].append(i)
        for j in trial:
            mu[j] = trial[j][0]
        del crates[victim]
        del mu[victim]
        improved = True

assign = [0] * N
for cidx, items in enumerate(crates):
    for i in items:
        assign[i] = cidx

print(json.dumps({"assign": assign}))
