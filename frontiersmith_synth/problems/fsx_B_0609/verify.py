#!/usr/bin/env python3
# verify.py <in> <out> <ans>   (ans ignored)
#
# Deterministic scorer for the gap-seeded packed-array problem.
#
# The solver ships an INITIAL physical layout of the N sorted keys inside a
# capacity-C cell array (keys in cells, everything else a gap; keys must appear
# in value order left-to-right) plus an optional list of REBUILD points. We then
# replay the fully-visible op trace and charge the exact cost:
#   * INSERT v : v is placed keeping sorted order. If a gap already sits between
#                its value-neighbours it drops in for free; otherwise the block of
#                keys between the landing slot and the NEAREST gap is shifted one
#                cell toward that gap and we charge one unit PER key moved.
#   * SCAN l r : charge every physical cell touched, first-key>=l .. last-key<=r,
#                gaps included (gap dilution).
#   * REBUILD  : re-spread all current keys uniformly across the C cells; charge C.
# Objective: minimize total cost. Score = 100 * B / F capped at 10x, where B is
# the cost of a naive packed-at-the-front layout the checker builds itself.
import sys, bisect


def _fail(reason):
    print("infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def parse_input(path):
    with open(path) as f:
        d = f.read().split()
    it = iter(d)
    N = int(next(it)); M = int(next(it)); Q = int(next(it))
    C = int(next(it)); VMAX = int(next(it))
    keys = [int(next(it)) for _ in range(N)]
    ops = []
    for _ in range(Q):
        typ = next(it)
        if typ == 'I':
            ops.append((1, int(next(it))))
        else:
            l = int(next(it)); r = int(next(it))
            ops.append((0, l, r))
    return N, M, Q, C, VMAX, keys, ops


def simulate(keys, ops, C, phys0, rebuild_set):
    """Return total cost (int). Assumes phys0 is a valid layout (checked by caller)."""
    vals = list(keys)
    phys = list(phys0)
    cost = 0
    for j, op in enumerate(ops):
        if j in rebuild_set:
            K = len(vals)
            phys = [(i * C) // K for i in range(K)]
            cost += C
        if op[0] == 1:                     # INSERT
            v = op[1]
            i = bisect.bisect_left(vals, v)
            n = len(phys)
            pred = phys[i - 1] if i > 0 else -1
            succ = phys[i] if i < n else C
            if succ - pred >= 2:           # a gap already sits between the neighbours
                gp = (pred + succ) // 2
                vals.insert(i, v); phys.insert(i, gp)
            else:                          # neighbours are physically adjacent -> shift
                cost_left = None; k = i
                if i > 0:
                    k = i - 1
                    while k > 0 and phys[k - 1] == phys[k] - 1:
                        k -= 1
                    if phys[k] - 1 >= 0:
                        cost_left = pred - (phys[k] - 1)
                cost_right = None; m = i
                if i < n:
                    m = i
                    while m + 1 < n and phys[m + 1] == phys[m] + 1:
                        m += 1
                    if phys[m] + 1 < C:
                        cost_right = (phys[m] + 1) - succ
                if cost_left is None and cost_right is None:
                    _fail("no gap available for insert (capacity exhausted)")
                use_left = (cost_right is None) or \
                           (cost_left is not None and cost_left <= cost_right)
                if use_left:
                    cost += cost_left
                    for tt in range(k, i):
                        phys[tt] -= 1
                    vals.insert(i, v); phys.insert(i, pred)
                else:
                    cost += cost_right
                    for tt in range(i, m + 1):
                        phys[tt] += 1
                    vals.insert(i, v); phys.insert(i, succ)
        else:                              # SCAN
            l = op[1]; r = op[2]
            a = bisect.bisect_left(vals, l)
            b = bisect.bisect_right(vals, r) - 1
            if a <= b:
                cost += phys[b] - phys[a] + 1
    return cost


def parse_output(path, N, C, Q):
    """Strictly parse + validate the solver artifact. On ANY problem -> feasibility fail."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception:
        _fail("cannot read output")
    if not toks:
        _fail("empty output")

    def as_int(s):
        # reject nan/inf/floats/garbage; only plain integers allowed
        try:
            return int(s)
        except (ValueError, TypeError):
            _fail("non-integer token %r" % s)

    idx = 0
    if len(toks) < 2:
        _fail("missing header")
    K = as_int(toks[0]); R = as_int(toks[1]); idx = 2
    if K != N:
        _fail("declared %d keys, expected %d" % (K, N))
    if R < 0 or R > Q:
        _fail("bad rebuild count %d" % R)
    if len(toks) < idx + K + R:
        _fail("truncated: not enough tokens")
    phys = [as_int(toks[idx + i]) for i in range(K)]
    idx += K
    rebuilds = [as_int(toks[idx + i]) for i in range(R)]

    # layout validity: strictly increasing positions within [0, C)
    prev = -1
    for p in phys:
        if p < 0 or p >= C:
            _fail("position %d out of [0,%d)" % (p, C))
        if p <= prev:
            _fail("positions not strictly increasing")
        prev = p
    # rebuild indices: strictly increasing within [0, Q)
    prev = -1
    for x in rebuilds:
        if x < 0 or x >= Q:
            _fail("rebuild index %d out of [0,%d)" % (x, Q))
        if x <= prev:
            _fail("rebuild indices not strictly increasing")
        prev = x
    return phys, set(rebuilds)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    N, M, Q, C, VMAX, keys, ops = parse_input(inf)

    phys, rebuild_set = parse_output(outf, N, C, Q)

    F = simulate(keys, ops, C, phys, rebuild_set)
    # internal baseline: naive "packed at the front" layout, no rebuilds
    B = simulate(keys, ops, C, list(range(N)), set())

    F = max(1, F)
    B = max(1, B)
    sc = min(1000.0, 100.0 * B / F)
    print("cost=%d baseline=%d" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
