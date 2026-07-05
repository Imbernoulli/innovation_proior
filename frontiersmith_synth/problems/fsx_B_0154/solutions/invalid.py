# TIER: invalid
"""Emit a garbage program: execute every maneuver in place with NO routing SWAPs.
Almost all interactions are on non-adjacent slots -> equivalence check fails ->
Ratio: 0.0."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    ni = lambda: int(next(it))
    n = ni(); m = ni(); k = ni()
    for _ in range(m):
        ni(); ni()
    for _ in range(n):
        ni()
    out = []
    for g in range(k):
        out.append("GATE %d" % g)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
