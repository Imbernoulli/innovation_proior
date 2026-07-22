# TIER: strong
# The insight: reformulate as a TREE KNAPSACK. For each channel, compute -- by DP over its
# dyadic tree, bottom-up -- the exact minimum SSE achievable using every possible local
# split budget k=0..S (a node's value is min(leave-it-alone, split-it-and-optimally-share
# k-1 between the two children)). This exposes each channel's TRUE per-split return curve,
# including cases where an early split looks worthless in isolation but is the only gate to
# a much larger downstream gain. A second DP then allocates the single shared budget across
# channels against these exact curves. Unlike greedy's one-step lookahead, this sees the
# whole subtree before deciding whether the gate is worth opening.
import sys


def leaf_stats(pts):
    sw = sum(w for _, w in pts)
    if sw <= 0:
        return 0.0
    mean = sum(v * w for v, w in pts) / sw
    return sum(w * (v - mean) ** 2 for v, w in pts)


class Node:
    __slots__ = ("depth", "pos", "pts", "err", "left", "right", "f", "dec", "jsplit")

    def __init__(self, depth, pos, pts, D):
        self.depth = depth
        self.pos = pos
        self.pts = pts
        self.err = leaf_stats(pts)
        self.left = None
        self.right = None
        if depth < D and len(pts) >= 1:
            lo = pos / (1 << depth); hi = (pos + 1) / (1 << depth)
            mid = (lo + hi) / 2.0
            lp = [(v, w) for v, w in pts if v < mid]
            rp = [(v, w) for v, w in pts if v >= mid]
            self.left = Node(depth + 1, 2 * pos, lp, D)
            self.right = Node(depth + 1, 2 * pos + 1, rp, D)


def compute_dp(node, S):
    f = [node.err] * (S + 1)
    dec = [False] * (S + 1)
    jsplit = [0] * (S + 1)
    if node.left is not None:
        compute_dp(node.left, S)
        compute_dp(node.right, S)
        fl, fr = node.left.f, node.right.f
        for k in range(1, S + 1):
            bestj, bestval = 0, fl[0] + fr[k - 1]
            for j in range(1, k):
                val = fl[j] + fr[k - 1 - j]
                if val < bestval - 1e-15:
                    bestval, bestj = val, j
            if bestval < f[k] - 1e-12:
                f[k], dec[k], jsplit[k] = bestval, True, bestj
    node.f, node.dec, node.jsplit = f, dec, jsplit


def reconstruct(node, k, ops, leaves):
    if node.dec[k]:
        ops.append((node.depth, node.pos))
        j = node.jsplit[k]
        reconstruct(node.left, j, ops, leaves)
        reconstruct(node.right, k - 1 - j, ops, leaves)
    else:
        sw = sum(w for _, w in node.pts)
        val = (sum(v * w for v, w in node.pts) / sw) if sw > 0 else \
              ((node.pos / (1 << node.depth) + (node.pos + 1) / (1 << node.depth)) / 2.0)
        leaves.append((node.depth, node.pos, val))


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    C = int(next(it)); S = int(next(it)); D = int(next(it))
    roots = []
    for _ in range(C):
        P = int(next(it))
        pts = []
        for _ in range(P):
            v = float(next(it)); w = int(next(it))
            pts.append((v, w))
        root = Node(0, 0, pts, D)
        compute_dp(root, S)
        roots.append(root)

    # global knapsack: allocate the shared budget S across channels against their exact
    # per-split return curves f_c[0..S].
    INF = float("inf")
    h = [INF] * (S + 1); h[0] = 0.0
    allocs = []
    for c in range(C):
        fc = roots[c].f
        newh = [INF] * (S + 1)
        alloc = [0] * (S + 1)
        for k in range(S + 1):
            bestj, bestval = 0, INF
            for j in range(k + 1):
                if h[k - j] == INF:
                    continue
                val = h[k - j] + fc[j]
                if val < bestval - 1e-15:
                    bestval, bestj = val, j
            newh[k] = bestval
            alloc[k] = bestj
        h = newh
        allocs.append(alloc)

    bestk = min(range(S + 1), key=lambda k: h[k])
    remaining = bestk
    kc = [0] * C
    for c in range(C - 1, -1, -1):
        j = allocs[c][remaining]
        kc[c] = j
        remaining -= j

    all_ops, all_leaves = [], []
    for c in range(C):
        ops, leaves = [], []
        reconstruct(roots[c], kc[c], ops, leaves)
        for depth, pos in ops:
            all_ops.append((c, depth, pos))
        for depth, pos, val in leaves:
            all_leaves.append((c, depth, pos, val))

    out = [str(len(all_ops))]
    for c, depth, pos in all_ops:
        out.append("%d %d %d" % (c, depth, pos))
    out.append(str(len(all_leaves)))
    for c, depth, pos, val in all_leaves:
        out.append("%d %d %d %.9f" % (c, depth, pos, val))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
