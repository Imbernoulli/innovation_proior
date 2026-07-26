# TIER: invalid
# Well-formed artifact syntactically, but claims every output equals x[0] -- wrong
# for almost every basis vector -- so the feasibility gate must reject it (Ratio 0).
import sys


def main():
    n, q, w = map(int, sys.stdin.read().split())
    out = ["%d 0" % n]
    out.append("O " + " ".join(["0"] * n))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
