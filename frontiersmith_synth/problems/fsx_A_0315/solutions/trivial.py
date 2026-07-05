# TIER: trivial
# Reproduces the checker's internal baseline: light the first ceil(N/3) segments
# at their ceiling, everything else off. Scores ~0.1 by construction.
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    u = [float(t) for t in toks[1:1 + N]]
    m = (N + 2) // 3
    f = [u[i] if i < m else 0.0 for i in range(N)]
    sys.stdout.write(" ".join("%.6f" % x for x in f) + "\n")


if __name__ == "__main__":
    main()
