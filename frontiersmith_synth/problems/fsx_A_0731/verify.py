#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the levee-budget
storm-sweep problem. Prints 'Ratio: <float in [0,1]>' on its own final line.

Physics: each storm k drops a fixed total rain volume V_k, spread evenly
over cells [a_k,b_k]. Water then finds its equilibrium ("fill-and-spill"):
it pools in the lowest basin(s) that catch it, and once a basin's water
reaches the height of its lowest bounding ridge, the excess spills over
into whichever basin lies beyond that ridge (which may itself then fill and
spill further, or may be too small to fully absorb the overflow, in which
case only part of it fills and the rest keeps searching for headroom).
Cells 0 and N-1 sit at an open boundary: water reaching them at their own
natural height simply drains away and never pools above it.

This is computed exactly and deterministically via a Kruskal-order
union-find sweep over the N+1 adjacent-cell edges (weighted by the higher
of each pair's raised elevation). Each component tracks a small sorted list
of "layers" -- sub-regions of its footprint already uniformly filled to
different levels -- so that a merge which floods a receiving basin only
part-way (not all the way up to the shared ridge) is handled exactly,
instead of being incorrectly diluted across cells that never actually got
wet.
"""
import sys
import math

NEG = -10 ** 9
EPS = 1e-9


def _cascade(lyr, pending, W):
    """Consume `pending` raising the lowest layer(s) of lyr, capped at W.
    Mutates lyr in place; returns (leftover_pending, score_added)."""
    score_added = 0.0
    while len(lyr) > 1 and pending > 0:
        lvl0, w0, sv0 = lyr[0]
        lvl1, w1, sv1 = lyr[1]
        room = lvl1 - lvl0
        if room < 0:
            room = 0.0
        cap = w0 * room
        if pending >= cap:
            score_added += sv0 * room
            pending -= cap
            lyr[0] = [lvl1, w0 + w1, sv0 + sv1]
            del lyr[1]
        else:
            rise = pending / w0 if w0 > 0 else 0.0
            score_added += sv0 * rise
            lyr[0][0] += rise
            pending = 0.0
    if len(lyr) == 1 and pending > 0:
        lvl0, w0, sv0 = lyr[0]
        room = W - lvl0
        if room > 0:
            cap = w0 * room
            if pending >= cap:
                score_added += sv0 * room
                pending -= cap
                lyr[0] = [W, w0, sv0]
            else:
                rise = pending / w0 if w0 > 0 else 0.0
                score_added += sv0 * rise
                lyr[0][0] += rise
                pending = 0.0
    return pending, score_added


def objective(H, v, storms, N):
    """Worst-case (max over storms) value-weighted flooded volume."""
    best = 0.0
    for (a, b, V) in storms:
        span = b - a + 1
        base = V // span
        rem = V - base * span
        inflow = [0] * N
        for i in range(a, b + 1):
            inflow[i] = base
        for i in range(a, a + rem):
            inflow[i] += 1

        M = N + 2  # 0 = left sentinel, 1..N = cells, N+1 = right sentinel
        parent = list(range(M))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        pending = [0.0] * M
        is_sea = [False] * M
        layers = [None] * M
        for p in range(1, N + 1):
            layers[p] = [[float(H[p - 1]), 1, float(v[p - 1])]]
            pending[p] = float(inflow[p - 1])
        is_sea[0] = True
        is_sea[N + 1] = True
        layers[0] = []
        layers[N + 1] = []

        def ext_h(p):
            if p == 0 or p == N + 1:
                return NEG
            return H[p - 1]

        edges = []
        for p in range(0, M - 1):
            w = ext_h(p)
            w2 = ext_h(p + 1)
            if w2 > w:
                w = w2
            edges.append((w, p))
        edges.sort(key=lambda t: (t[0], t[1]))

        score = 0.0
        for (W, p) in edges:
            ri, rj = find(p), find(p + 1)
            if is_sea[ri] and is_sea[rj]:
                new_is_sea = True
                new_layers = []
                new_pending = 0.0
            else:
                combined_layers = layers[ri] + layers[rj]
                combined_layers.sort()
                combined_pending = pending[ri] + pending[rj]
                new_pending, added = _cascade(combined_layers, combined_pending, W)
                score += added
                new_layers = combined_layers
                was_sea = is_sea[ri] or is_sea[rj]
                if was_sea and len(new_layers) == 1 and new_layers[0][0] >= W - EPS:
                    new_is_sea = True
                    new_layers = []
                    new_pending = 0.0
                else:
                    new_is_sea = False

            parent[ri] = rj
            is_sea[rj] = new_is_sea
            layers[rj] = new_layers
            pending[rj] = new_pending

        if score > best:
            best = score
    return best


def fail(msg):
    print("Ratio: 0.0  (%s)" % msg)
    return 0


def main():
    if len(sys.argv) < 3:
        return fail("bad args")
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        itoks = f.read().split()
    it = iter(itoks)
    try:
        N = int(next(it))
        Budget = int(next(it))
        K = int(next(it))
        e = [int(next(it)) for _ in range(N)]
        v = [int(next(it)) for _ in range(N)]
        storms = []
        for _ in range(K):
            a = int(next(it))
            b = int(next(it))
            V = int(next(it))
            storms.append((a, b, V))
    except StopIteration:
        return fail("truncated input")

    try:
        with open(out_path) as f:
            otoks = f.read().split()
    except FileNotFoundError:
        return fail("no output file")

    if len(otoks) != N:
        return fail("expected %d tokens, got %d" % (N, len(otoks)))

    h = []
    for tok in otoks:
        try:
            val = int(tok)
        except ValueError:
            return fail("non-integer token %r" % tok)
        if not math.isfinite(val):
            return fail("non-finite token %r" % tok)
        if val < 0:
            return fail("negative height %d" % val)
        h.append(val)

    if sum(h) > Budget:
        return fail("budget exceeded: %d > %d" % (sum(h), Budget))

    H = [e[i] + h[i] for i in range(N)]
    F = objective(H, v, storms, N)

    # internal baseline: spread the budget evenly over all cells
    base_h = [Budget // N] * N
    rem = Budget - sum(base_h)
    for i in range(rem):
        base_h[i] += 1
    Hb = [e[i] + base_h[i] for i in range(N)]
    B = objective(Hb, v, storms, N)

    if B <= 0:
        return fail("degenerate instance: baseline achieves zero damage")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
