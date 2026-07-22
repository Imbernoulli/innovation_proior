# TIER: strong
"""The insight: a rake stroke of width w costs w regardless of how much sag
it removes, while a point stroke costs 1 per unit of sag removed. So a rake
only pays for itself when the block's SHARED component is large relative to
its width -- exactly what you can only see by measuring the per-op residual
reduction a candidate rake would buy, not by guessing "big blocks first" or
"always rake". This solution walks the block hierarchy coarse-to-fine,
computes each candidate rake's actual point-stroke savings, commits only the
strokes that pay for themselves, and mops up whatever the rakes could not
reach (oscillating / genuinely local disturbance) by hand."""
import sys


def point_cost(vals, T):
    return sum(max(0, abs(x) - T) for x in vals)


def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx]); idx += 1
    T = int(data[idx]); idx += 1
    _MAXOPS = int(data[idx]); idx += 1
    h = [int(data[idx + i]) for i in range(N)]

    ops = []

    w = N
    while w >= 2:
        a = 0
        while a < N:
            block = h[a:a + w]
            s = sum(block)
            avg = s // w  # floor division, matches counter.py exactly
            if avg != 0:
                cost_before = point_cost(block, T)
                after = [x - avg for x in block]
                cost_after = point_cost(after, T)
                # only rake if the point-strokes it saves outweigh its own cost
                if cost_before - cost_after > w:
                    for i in range(a, a + w):
                        h[i] -= avg
                    ops.append("B %d %d" % (a, w))
            a += w
        w //= 2

    for i in range(N):
        while abs(h[i]) > T:
            h[i] += -1 if h[i] > 0 else 1
            ops.append("P %d" % i)

    out = [str(len(ops))]
    out.extend(ops)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
