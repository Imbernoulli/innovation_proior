# TIER: trivial
# Dump every added station at the reef centre (0.5,0.5) -- reproduces the checker baseline.
import sys


def main():
    toks = sys.stdin.read().split()
    d = int(toks[0]); M = int(toks[1]); K = int(toks[2])
    A = M - K
    out = []
    for _ in range(A):
        out.append("0.5 0.5")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
