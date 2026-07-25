# TIER: invalid
"""Plausibly-shaped but infeasible: stitch the first input segment TWICE.
The duplicate-stitch check fails -> Ratio 0.0 on every case."""
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
    u = ni(); v = ni(); ni()
    sys.stdout.write("3\nJ %d\nS %d %d\nS %d %d\n" % (u, u, v, u, v))


if __name__ == "__main__":
    main()
