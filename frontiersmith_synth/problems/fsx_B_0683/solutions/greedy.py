# TIER: greedy
# The obvious recipe: myopic best-first tree growth (CART-style). At every step, look at
# every currently-open leaf across every channel, compute the SSE reduction of splitting
# THAT leaf right now (and only that leaf), and commit whichever single split looks best
# this instant. Repeat until the budget is spent or nothing helps. This never looks past
# one move, so it can never see that a cheap-looking split is the *only* way to reach a
# much richer split hiding behind it.
import sys


def leaf_err(pts):
    sw = sum(w for _, w in pts)
    if sw <= 0:
        return 0.0
    mean = sum(v * w for v, w in pts) / sw
    return sum(w * (v - mean) ** 2 for v, w in pts)


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    C = int(next(it)); S = int(next(it)); D = int(next(it))
    channels = []
    for _ in range(C):
        P = int(next(it))
        pts = []
        for _ in range(P):
            v = float(next(it)); w = int(next(it))
            pts.append((v, w))
        channels.append(pts)

    active = [set([(0, 0)]) for _ in range(C)]
    ops = []
    budget = S
    while budget > 0:
        best_gain = 1e-9
        best = None
        for c in range(C):
            for (depth, pos) in active[c]:
                if depth >= D:
                    continue
                lo = pos / (1 << depth); hi = (pos + 1) / (1 << depth)
                sel = [(v, w) for v, w in channels[c] if lo <= v < hi]
                if len(sel) < 2:
                    continue
                mid = (lo + hi) / 2.0
                left = [(v, w) for v, w in sel if v < mid]
                right = [(v, w) for v, w in sel if v >= mid]
                gain = leaf_err(sel) - leaf_err(left) - leaf_err(right)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (c, depth, pos)
        if best is None:
            break
        c, depth, pos = best
        active[c].discard((depth, pos))
        active[c].add((depth + 1, 2 * pos))
        active[c].add((depth + 1, 2 * pos + 1))
        ops.append((c, depth, pos))
        budget -= 1

    out = [str(len(ops))]
    for c, depth, pos in ops:
        out.append("%d %d %d" % (c, depth, pos))

    leaves = []
    for c in range(C):
        for (depth, pos) in active[c]:
            lo = pos / (1 << depth); hi = (pos + 1) / (1 << depth)
            sel = [(v, w) for v, w in channels[c] if lo <= v < hi]
            sw = sum(w for _, w in sel)
            val = (sum(v * w for v, w in sel) / sw) if sw > 0 else (lo + hi) / 2.0
            leaves.append((c, depth, pos, val))
    out.append(str(len(leaves)))
    for c, depth, pos, val in leaves:
        out.append("%d %d %d %.9f" % (c, depth, pos, val))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
