import sys, math
from collections import defaultdict

SCALE = 1 << 16


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def nearest_anchor_col(anchors, c):
    best = None; bd = None
    for a in anchors:
        d = abs(a - c)
        if bd is None or d < bd or (d == bd and a < best):
            bd = d; best = a
    return best


def lpath(anchor_col, r_l, c_l):
    cells = []
    for r in range(1, r_l + 1):
        cells.append((r, anchor_col))
    lo, hi = min(anchor_col, c_l), max(anchor_col, c_l)
    for c in range(lo, hi + 1):
        cells.append((r_l, c))
    return cells


def baseline_cells(anchors, loads):
    s = set()
    for (r_l, c_l, f_l) in loads:
        a = nearest_anchor_col(anchors, c_l)
        for cell in lpath(a, r_l, c_l):
            s.add(cell)
    return s


def relax(nodes, anchor_set, force_at, K):
    """Synchronous (Jacobi) fixed-point displacement relaxation, fixed K iterations,
    pure-integer fixed-point arithmetic (u stored as true_displacement * SCALE).
    Anchor nodes pinned at 0. A non-anchor node with `d` face-adjacent material
    neighbours updates to the average of its neighbours' displacements plus its own
    injected load, divided by d. A node with d==0 (no face-adjacent neighbour at
    all -- e.g. reached only via a corner/diagonal touch) has no averaging
    reference and simply DRIFTS by its injected load every iteration: this is what
    makes corner-only connections catastrophic under the relaxation."""
    neigh = defaultdict(list)
    for (r, c) in nodes:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in nodes:
                neigh[(r, c)].append(nb)
    non_anchor = [n for n in nodes if n not in anchor_set]
    u = {n: 0 for n in nodes}
    for _ in range(K):
        newu = {}
        for n in non_anchor:
            nbrs = neigh[n]
            f_n = force_at.get(n, 0) * SCALE
            if nbrs:
                s = 0
                for x in nbrs:
                    s += u[x]
                newu[n] = (s + f_n) // len(nbrs)
            else:
                newu[n] = u[n] + f_n
        for n in non_anchor:
            u[n] = newu[n]
    return u


def compliance(anchors, loads, K, material):
    anchor_set = set((0, a) for a in anchors)
    nodes = set(material) | anchor_set
    force_at = {}
    for (r, c, f) in loads:
        force_at[(r, c)] = force_at.get((r, c), 0) + f
    u = relax(nodes, anchor_set, force_at, K)
    F_num = 0
    for (r, c, f) in loads:
        F_num += f * u[(r, c)]
    return F_num


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    it = iter(itoks)
    try:
        W = int(next(it)); H = int(next(it))
        A = int(next(it))
        anchors = [int(next(it)) for _ in range(A)]
        L = int(next(it))
        loads = []
        for _ in range(L):
            r = int(next(it)); c = int(next(it)); f = int(next(it))
            loads.append((r, c, f))
        M = int(next(it))
        K = int(next(it))
    except Exception:
        fail("malformed input (setter bug)")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    oit = iter(otoks)

    try:
        k_raw = next(oit)
        k = int(k_raw)
    except StopIteration:
        fail("empty output")
    except Exception:
        fail("k not an integer")

    if not (0 <= k <= M):
        fail("k out of range [0,%d]" % M)

    cells = []
    try:
        for _ in range(k):
            r = int(next(oit))
            c = int(next(oit))
            if not (1 <= r <= H - 1 and 0 <= c <= W - 1):
                fail("cell (%d,%d) out of bounds" % (r, c))
            cells.append((r, c))
    except StopIteration:
        fail("output truncated before k cells were read")
    except Exception:
        fail("non-integer cell token")

    if len(set(cells)) != len(cells):
        fail("duplicate material cell")

    material = set(cells)
    load_set = set((r, c) for (r, c, f) in loads)
    if not load_set.issubset(material):
        fail("a load mound is not covered by material")

    B = compliance(anchors, loads, K, baseline_cells(anchors, loads))
    B = max(1, B)

    F = compliance(anchors, loads, K, material)
    if F <= 0:
        fail("non-positive compliance (degenerate)")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d cells=%d Ratio: %.6f" % (F, B, k, sc / 1000.0))


if __name__ == "__main__":
    main()
