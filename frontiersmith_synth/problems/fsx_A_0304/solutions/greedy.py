# TIER: greedy
# Spread the stations on the reef plot's FULL incircle (regular N-gon, centre
# (0.5,0.5), radius 0.5) instead of the small central ring. Using the whole
# plot immediately lifts the minimum triangle area well above the clustered
# baseline. No further optimisation.
import sys, math


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    cx, cy, r = 0.5, 0.5, 0.5
    out = []
    for k in range(N):
        t = 2.0 * math.pi * k / N
        out.append("%.12f %.12f" % (cx + r * math.cos(t), cy + r * math.sin(t)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
