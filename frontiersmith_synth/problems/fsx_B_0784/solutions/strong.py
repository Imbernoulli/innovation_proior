# TIER: strong
# The insight: the score is really a joint space-time territory division PLUS a
# color-run schedule inside each territory. First split the rail into H contiguous
# territories (by position) and hand each to the head that starts nearest it, so
# heads rarely come within the throttle distance of one another (little/no
# proximity slowdown). Then, INSIDE a territory, don't just sweep by raw position
# (which pays a swap almost every stroke when colors are interleaved) -- group the
# territory's strokes into color runs (all same-color strokes back to back), and
# order the runs by their mean position so the route still drifts left-to-right.
# This trades a modest amount of extra local movement for far fewer swaps, while
# the territory split keeps the throttle penalty (paid mostly by ignoring it) off
# the table entirely. Both trap regimes (partition-by-color / partition-by-region)
# are dominated by this joint reformulation.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    L = int(next(it)); H = int(next(it)); D = int(next(it)); S = int(next(it))
    K = int(next(it)); N = int(next(it))
    starts = [int(next(it)) for _ in range(H)]
    strokes = []
    for _ in range(N):
        p = int(next(it)); c = int(next(it))
        strokes.append((p, c))

    strokes_sorted = sorted(strokes)  # by position
    # split into H contiguous territories, as equal in count as possible
    sizes = [N // H + (1 if k < N % H else 0) for k in range(H)]
    territories = []
    idx = 0
    for sz in sizes:
        territories.append(strokes_sorted[idx:idx + sz])
        idx += sz

    # assign territory i (0 = leftmost) to the head with the i-th smallest start pos
    head_order = sorted(range(H), key=lambda h: starts[h])
    territory_for_head = {}
    for i, h in enumerate(head_order):
        territory_for_head[h] = territories[i]

    all_ops = []
    for h in range(H):
        terr = territory_for_head.get(h, [])
        groups = {}
        for (p, c) in terr:
            groups.setdefault(c, []).append(p)
        color_order = sorted(groups.keys(), key=lambda c: sum(groups[c]) / len(groups[c]))
        route = []
        for c in color_order:
            for p in sorted(groups[c]):
                route.append((p, c))

        pos = starts[h]
        cur_color = 0
        ops = []
        for (p, c) in route:
            dx = 1 if p >= pos else -1
            while pos != p:
                ops.append("M %d" % dx)
                pos += dx
            if cur_color != c:
                ops.append("S %d" % c)
                cur_color = c
            ops.append("P")
        all_ops.append(ops)

    out = [str(H)]
    for h in range(H):
        out.append(str(len(all_ops[h])))
        out.extend(all_ops[h])
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
