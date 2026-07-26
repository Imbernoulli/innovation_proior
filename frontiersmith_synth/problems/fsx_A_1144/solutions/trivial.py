# TIER: trivial
"""
Naive per-voxel placement: ONE shared UNIT assembly, translated once to each
target position (V translate ops), combined with a balanced union tree
((V-1) unions). Total = 1 + V + (V-1) = 2V ops -- essentially the checker's
own B_hi = 2V-1 naive reference (off by one, which the checker's clamped
scoring formula absorbs), so this scores ~0.10 by design. No structure is
exploited: every voxel is placed independently.
"""
import sys


def main():
    data = sys.stdin.buffer.read().split()
    v = int(data[0])
    if v == 0:
        return
    pts = [(data[1 + 3 * k], data[2 + 3 * k], data[3 + 3 * k]) for k in range(v)]

    out = []

    def emit(line):
        out.append(line)
        return len(out) - 1

    unit = emit("U")
    leaves = []
    for (x, y, z) in pts:
        t = emit("T %d %s %s %s" % (unit, x.decode(), y.decode(), z.decode()))
        leaves.append(t)

    level = leaves
    while len(level) > 1:
        nxt = []
        i, n = 0, len(level)
        while i + 1 < n:
            m = emit("M %d %d" % (level[i], level[i + 1]))
            nxt.append(m)
            i += 2
        if i < n:
            nxt.append(level[i])
        level = nxt

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
