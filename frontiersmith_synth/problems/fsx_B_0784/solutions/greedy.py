# TIER: greedy
# The obvious "minimize setups" recipe: hand each color to one head (round robin),
# so every head only ever swaps once per color it owns. Within a head, visit its
# assigned strokes color-block by color-block, sorted by position inside a block.
# This minimizes total swap count but completely ignores the proximity/throttle
# rule -- when colors are spatially interleaved across the whole rail, every head
# ends up sweeping the ENTIRE rail and constantly overlapping with the others,
# so most of its moves get throttled to half speed.
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

    per_head = [[] for _ in range(H)]
    for (p, c) in strokes:
        h = (c - 1) % H
        per_head[h].append((c, p))

    all_ops = []
    for h in range(H):
        items = sorted(per_head[h])  # by (color, position)
        pos = starts[h]
        cur_color = 0
        ops = []
        for (c, p) in items:
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
