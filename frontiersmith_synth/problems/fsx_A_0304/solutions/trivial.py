# TIER: trivial
# Baseline: cluster the N stations on a SMALL central ring (regular N-gon,
# centre (0.5,0.5), radius 0.5/sqrt(5)). This reproduces the checker's internal
# baseline construction, so it scores ~0.1.
import sys, math


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    cx, cy = 0.5, 0.5
    r = 0.5 / math.sqrt(5.0)
    out = []
    for k in range(N):
        t = 2.0 * math.pi * k / N
        out.append("%.12f %.12f" % (cx + r * math.cos(t), cy + r * math.sin(t)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
