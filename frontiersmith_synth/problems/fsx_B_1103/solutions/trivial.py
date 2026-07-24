# TIER: trivial
"""Reproduce the checker's baseline exactly: walk the segment list IN INPUT
ORDER; whenever the needle is at neither endpoint, jump to u; then stitch.
No reordering, no color planning -- score is exactly 0.1 by construction."""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0

    def ni():
        nonlocal p
        v = int(data[p]); p += 1
        return v

    V = ni(); E = ni(); C = ni(); K = ni()
    for _ in range(V):
        ni(); ni()
    edges = [(ni(), ni(), ni()) for _ in range(E)]

    ops = []
    pos = 0
    for (u, v, c) in edges:
        if pos != u and pos != v:
            ops.append("J %d" % u)
            pos = u
        ops.append("S %d %d" % (u, v))
        pos = v if pos == u else u
    out = [str(len(ops))]
    out.extend(ops)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
