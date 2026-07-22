# TIER: strong
# The insight: posted rates are only a ROUTING DEVICE over the arrival order.
# The real decision is WHICH worker should end up on WHICH field -- an
# assignment problem -- and the rate menu is just how you steer the sequential,
# self-interested arrival process toward that target assignment as cheaply as
# possible.
#
# Step 1 (reformulate as matching): a field i is only reachable by a worker j
# if j arrives at or before the field's deadline (j <= deadline_i).  For every
# eligible (field, worker) pair compute the TRUE surplus value_i - cost(i,j)
# (what the pair is actually worth, ignoring price -- price is just a
# transfer).  Build a target matching by a descending-surplus greedy exchange:
# take pairs in order of decreasing surplus, keep a pair if both the field and
# the worker are still free.  This assigns each valuable field to a cheap,
# eligible worker instead of "whoever can afford the going rate."
#
# Step 2 (price only what is needed): for a field matched to worker j*, post
# the MINIMUM rate that clears j*'s reservation: cost(i,j*) + wage_j* + 1.
# Because j* was chosen as (one of) the cheapest eligible workers for that
# field, this minimal rate is usually NOT enough to tempt an earlier, more
# expensive (mismatched) worker to cherry-pick it first -- unlike a rate
# proportional to difficulty, which overshoots every worker's cost by a wide,
# uniform margin and is therefore always grabbed by whoever shows up first.
# Unmatched fields are posted at rate 0 (no wage risk, no false hope).
import sys


def ceil_div(a, b):
    return -(-a // b)


def cost(c_unit, d, s):
    return ceil_div(c_unit * d, s)


def main():
    it = sys.stdin.read().split()
    p = 0
    N = int(it[p]); M = int(it[p + 1]); C_UNIT = int(it[p + 2]); RMAX = int(it[p + 3]); p += 4
    fields = []
    for _ in range(N):
        v = int(it[p]); d = int(it[p + 1]); dl = int(it[p + 2]); p += 3
        fields.append((v, d, dl))
    workers = []
    for _ in range(M):
        s = int(it[p]); w = int(it[p + 1]); p += 2
        workers.append((s, w))

    # ---- Step 1: build candidate (surplus, field, worker) pairs -----------
    pairs = []
    for i, (v, d, dl) in enumerate(fields):
        eligible = min(dl, M)
        for j in range(eligible):
            s, w = workers[j]
            c = cost(C_UNIT, d, s)
            surplus = v - c
            if surplus > 0:
                pairs.append((surplus, i, j))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))

    used_field = [False] * N
    used_worker = [False] * M
    assign = {}
    for surplus, i, j in pairs:
        if used_field[i] or used_worker[j]:
            continue
        used_field[i] = True
        used_worker[j] = True
        assign[i] = j

    # ---- Step 2: minimal clearing price for the assigned worker -----------
    rates = [0] * N
    for i, (v, d, dl) in enumerate(fields):
        if i in assign:
            j = assign[i]
            s, w = workers[j]
            c = cost(C_UNIT, d, s)
            r = c + w + 1
            rates[i] = max(0, min(RMAX, r))

    sys.stdout.write(" ".join(str(r) for r in rates) + "\n")


if __name__ == "__main__":
    main()
