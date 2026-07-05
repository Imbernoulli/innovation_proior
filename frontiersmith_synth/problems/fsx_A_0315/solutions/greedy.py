# TIER: greedy
# Spread the light over the WHOLE boulevard: set every lamp to its ceiling.
# A uniform-ish full-support profile has a far flatter autocorrelation than the
# concentrated baseline, so it beats trivial (c1 ~ 2 vs ~ 6).
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    u = [float(t) for t in toks[1:1 + N]]
    sys.stdout.write(" ".join("%.6f" % x for x in u) + "\n")


if __name__ == "__main__":
    main()
