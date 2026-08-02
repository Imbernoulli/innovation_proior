# TIER: strong
"""The insight has two layers.

(1) DECOMPOSITION: a step change (edge) in the aggregate is always exactly
one appliance's own ON/OFF power delta (by the family's construction), so
its magnitude alone tells you which POWER GROUP produced it -- appliances
whose P differs from every other appliance are then fully solved instantly
(matching P uniquely fixes the appliance). Only within a group of
appliances that share the SAME power level is there real identity
ambiguity, and edges from different groups never interact -- so the joint
reconstruction problem decomposes into small, INDEPENDENT per-power-group
subproblems instead of one coupled A-appliance guess.

(2) DWELL ELIGIBILITY: within a shared-power group, an edge can only be
attributed to a member that is currently in the matching state AND has
already dwelt at least its own legal minimum -- an instantaneous power
reading cannot tell two same-power appliances apart, but "who is legally
allowed to be transitioning right now" usually can. Ties (several members
simultaneously eligible) are broken toward whichever is closest to being
FORCED to transition (least room left before its own dwell max) -- the
statistically more likely true source; because within-group dwell windows
are disjoint by design, such ties are rare rather than the norm.

Each group's reconstructed sequence is finally pushed through a legality
repair pass as a safety net (guarantees a feasible output even on a
genuinely ambiguous tie)."""
import sys


def repair(raw, mon, mxon, moff, mxoff):
    T = len(raw)
    out = []
    state = 0
    elapsed = 0
    for t in range(T):
        want = raw[t]
        maxd = mxoff if state == 0 else mxon
        mind = moff if state == 0 else mon
        if elapsed >= maxd:
            state = 1 - state
            elapsed = 0
        elif want != state and elapsed >= mind:
            state = 1 - state
            elapsed = 0
        out.append(state)
        elapsed += 1
    return out


def solve_group(T, members, diffs):
    """members: list of (min_on,max_on,min_off,max_off) for appliances sharing
    one power level. diffs: list of (t, sign) events at this power magnitude
    (sign=+1 turn-on, -1 turn-off), in time order. Walk the events, tracking
    each member's (state, elapsed-in-state); assign each event to the
    eligible member closest to being forced (least dwell room left), falling
    back to the closest-matching member if none is strictly eligible (should
    only occur on a genuine boundary-of-eligibility tie)."""
    m = len(members)
    state = [0] * m
    elapsed = [0] * m
    raw = [[0] * T for _ in range(m)]
    di = 0
    for t in range(T):
        if di < len(diffs) and diffs[di][0] == t:
            sign = diffs[di][1]
            cands = []
            for i, (mon, mxon, moff, mxoff) in enumerate(members):
                st = state[i]
                if st == 0 and sign > 0:
                    lo, hi = moff, mxoff
                    cands.append((i, elapsed[i] >= lo, hi - elapsed[i]))
                elif st == 1 and sign < 0:
                    lo, hi = mon, mxon
                    cands.append((i, elapsed[i] >= lo, hi - elapsed[i]))
            if cands:
                elig = [c for c in cands if c[1]]
                pool = elig if elig else cands
                pool.sort(key=lambda c: (c[2], c[0]))
                pick = pool[0][0]
                state[pick] = 1 - state[pick]
                elapsed[pick] = -1
            di += 1
        for i in range(m):
            elapsed[i] += 1
            raw[i][t] = state[i]
    return raw


def main():
    data = sys.stdin.read().split()
    ptr = 0
    T = int(data[ptr]); A = int(data[ptr + 1]); ptr += 2
    ptr += 2  # wT wA -- objective SHAPE only, not needed by the reconstruction itself
    archs = []
    for _ in range(A):
        P, mon, mxon, moff, mxoff = (int(x) for x in data[ptr:ptr + 5]); ptr += 5
        archs.append((P, mon, mxon, moff, mxoff))
    aggregate = [int(x) for x in data[ptr:ptr + T]]; ptr += T

    # group appliance indices by power level -- edges of different magnitudes
    # never interact, so each group is an independent subproblem.
    groups = {}
    for i, (P, mon, mxon, moff, mxoff) in enumerate(archs):
        groups.setdefault(P, []).append(i)

    prev = 0
    events = {}  # P -> list of (t, sign)
    for t in range(T):
        cur = aggregate[t]
        diff = cur - prev
        if diff != 0:
            P = abs(diff)
            sign = 1 if diff > 0 else -1
            events.setdefault(P, []).append((t, sign))
        prev = cur

    raw = [None] * A
    for P, idxs in groups.items():
        members = [(archs[i][1], archs[i][2], archs[i][3], archs[i][4]) for i in idxs]
        grp_events = events.get(P, [])
        grp_raw = solve_group(T, members, grp_events)
        for local_i, global_i in enumerate(idxs):
            raw[global_i] = grp_raw[local_i]

    out = [str(A)]
    for i, (P, mon, mxon, moff, mxoff) in enumerate(archs):
        seq = repair(raw[i], mon, mxon, moff, mxoff)
        out.append(" ".join(str(v) for v in seq))
    print("\n".join(out))


if __name__ == "__main__":
    main()
