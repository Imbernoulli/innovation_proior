#!/usr/bin/env python3
"""verify.py -- deterministic checker for fsx_A_1101 (Night Depot LIFO).

CLI: python3 verify.py <in> <out> <ans>     (ans ignored)

All quantities are integers; scoring is exact integer arithmetic.
Prints the final score on its own last line as `Ratio: <float in [0,1]>`.
Any infeasibility / parse problem -> Ratio: 0.0 and exit 0.
"""
import sys


def die(msg=""):
    if msg:
        sys.stderr.write("infeasible: %s\n" % msg)
    print("Ratio: 0.000000")
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
    pos = [0]

    def nxt():
        if pos[0] >= len(toks):
            raise ValueError("unexpected eof in instance")
        v = int(toks[pos[0]])
        pos[0] += 1
        return v

    n = nxt(); R = nxt(); W = nxt(); T = nxt(); P = nxt(); SH = nxt()
    prc = [nxt() for _ in range(T)]
    cap = [nxt() for _ in range(T)]
    a = []; d = []; E = []
    for _ in range(n):
        a.append(nxt()); d.append(nxt()); E.append(nxt())
    if pos[0] != len(toks):
        raise ValueError("trailing tokens in instance")
    return n, R, W, T, P, SH, prc, cap, a, d, E


# ---------------------------------------------------------------------------
# artifact scoring (shared by participant output and the internal baseline)
# ---------------------------------------------------------------------------

def score_artifact(park, exit_order, recs, n, R, W, T, P, SH, prc, cap, a, d, E):
    """park: list of (bus,row) in parking order; exit_order: permutation;
    recs: list of (bus,hour,kwh).  Raises ValueError on any infeasibility.
    Returns exact integer cost F."""
    if len(park) != n:
        raise ValueError("need exactly n parking lines")
    row_of = [-1] * n
    rows = [[] for _ in range(R)]
    last_a = -1
    seen = [False] * n
    for (i, r) in park:
        if not (0 <= i < n and 0 <= r < R):
            raise ValueError("bad parking line")
        if seen[i]:
            raise ValueError("bus parked twice")
        seen[i] = True
        if a[i] < last_a:
            raise ValueError("parking order must be nondecreasing in arrival hour")
        last_a = a[i]
        rows[r].append(i)
        row_of[i] = r
    if len(exit_order) != n or sorted(exit_order) != list(range(n)):
        raise ValueError("exit order is not a permutation")
    for k in range(n - 1):
        if d[exit_order[k]] > d[exit_order[k + 1]]:
            raise ValueError("exit order must be nondecreasing in departure hour")
    # shunt simulation
    rowsim = [lst[:] for lst in rows]
    shunts = 0
    for b in exit_order:
        r = row_of[b]
        idx = rowsim[r].index(b)          # buses still ahead of b
        shunts += idx
        rowsim[r].pop(idx)
    # charging records
    if len(recs) > n * T:
        raise ValueError("too many charging records")
    tot = [0] * T
    got = [0] * n
    rowhour = [[0] * T for _ in range(R)]
    for (i, h, k) in recs:
        if not (0 <= i < n and 0 <= h < T):
            raise ValueError("bad charging record")
        if not (a[i] <= h < d[i]):
            raise ValueError("charging outside the parked window")
        if k < 1:
            raise ValueError("nonpositive charge amount")
        r = row_of[i]
        pos = 0
        for j in rows[r]:
            if j == i:
                break
            if a[j] <= h < d[j]:
                pos += 1
        if pos >= W:
            raise ValueError("bus beyond cable reach while charging")
        got[i] += k
        rowhour[r][h] += k
        tot[h] += k
        if rowhour[r][h] > P:
            raise ValueError("row charger rate exceeded")
        if tot[h] > cap[h]:
            raise ValueError("transformer capacity exceeded")
        if got[i] > E[i]:
            raise ValueError("overcharge")
    for i in range(n):
        if got[i] != E[i]:
            raise ValueError("bus %d not fully charged" % i)
    F = sum(prc[h] * tot[h] for h in range(T)) + SH * shunts
    return F


# ---------------------------------------------------------------------------
# reference (baseline) construction: balanced parking + earliest-possible charge
# ---------------------------------------------------------------------------

def trivial_construct(n, R, W, T, P, cap, a, d, E):
    order = sorted(range(n), key=lambda i: (a[i], i))
    rows = [[] for _ in range(R)]
    row_of = [-1] * n
    for i in order:
        r = min(range(R), key=lambda r: (len(rows[r]), r))
        rows[r].append(i)
        row_of[i] = r
    park = [(i, row_of[i]) for i in order]
    exit_order = sorted(range(n), key=lambda i: (d[i], i))
    need = E[:]
    rowhour = [[0] * T for _ in range(R)]
    capleft = cap[:]
    recs = []
    for h in range(T):
        for r in range(R):
            for i in rows[r]:
                if need[i] > 0 and a[i] <= h < d[i]:
                    give = min(need[i], P - rowhour[r][h], capleft[h])
                    if give > 0:
                        need[i] -= give
                        rowhour[r][h] += give
                        capleft[h] -= give
                        recs.append((i, h, give))
    # repair pass: deadline order, earliest hours first
    for i in sorted(range(n), key=lambda i: (d[i], i)):
        if need[i] == 0:
            continue
        r = row_of[i]
        for h in range(a[i], d[i]):
            give = min(need[i], P - rowhour[r][h], capleft[h])
            if give > 0:
                need[i] -= give
                rowhour[r][h] += give
                capleft[h] -= give
                recs.append((i, h, give))
            if need[i] == 0:
                break
    if any(x > 0 for x in need):
        return None
    return park, exit_order, recs


def baseline_cost(n, R, W, T, P, SH, prc, cap, a, d, E):
    art = trivial_construct(n, R, W, T, P, cap, a, d, E)
    if art is None:
        # should never happen on generated instances; keep B positive
        return sum(E) * max(prc) + SH * n * n
    park, exit_order, recs = art
    return score_artifact(park, exit_order, recs,
                          n, R, W, T, P, SH, prc, cap, a, d, E)


def evaluate(path, n, R, W, T, P, SH, prc, cap, a, d, E):
    toks = open(path).read().split()
    if len(toks) > 4 * n + 3 * n * T + 16:
        raise ValueError("output too large")
    pos = [0]

    def gi():
        if pos[0] >= len(toks):
            raise ValueError("unexpected eof")
        v = int(toks[pos[0]])          # rejects nan/inf/garbage (ValueError)
        pos[0] += 1
        return v

    park = [(gi(), gi()) for _ in range(n)]
    exit_order = [gi() for _ in range(n)]
    C = gi()
    if C < 0 or C > n * T:
        raise ValueError("bad record count")
    recs = [(gi(), gi(), gi()) for _ in range(C)]
    if pos[0] != len(toks):
        raise ValueError("trailing tokens")
    return score_artifact(park, exit_order, recs,
                          n, R, W, T, P, SH, prc, cap, a, d, E)


def main():
    try:
        n, R, W, T, P, SH, prc, cap, a, d, E = read_instance(sys.argv[1])
    except Exception as e:
        die("bad instance: %s" % e)
    B = baseline_cost(n, R, W, T, P, SH, prc, cap, a, d, E)
    try:
        F = evaluate(sys.argv[2], n, R, W, T, P, SH, prc, cap, a, d, E)
    except Exception as e:
        die(str(e))
    ratio = min(1.0, 0.1 * B / max(1, F))
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
