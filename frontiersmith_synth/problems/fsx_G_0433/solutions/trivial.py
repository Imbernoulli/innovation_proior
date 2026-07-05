# TIER: trivial
# Baseline: fit a CIRCLE through the training centroid with the mean radius.
# This reproduces the grader's own internal construction, so it scores ~0.1.
# A circle vaguely follows the orbit but cannot bend to the eccentric,
# never-observed arc.
import sys
import math


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0].split()[0])
    xs = []
    ys = []
    for ln in data[1:1 + n]:
        p = ln.split()
        if len(p) >= 2:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    R = sum(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)) / len(xs)
    print("(x - (%r))**2 + (y - (%r))**2 - (%r)" % (cx, cy, R * R))


if __name__ == "__main__":
    main()
