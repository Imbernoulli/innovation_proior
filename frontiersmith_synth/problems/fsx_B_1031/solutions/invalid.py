# TIER: invalid
# Garbage / infeasible artifact: wrong token count, out-of-range and
# non-finite values mixed in. Must score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    H = int(toks[0])
    out = []
    for i in range(H + 3):
        if i % 4 == 0:
            out.append("2")
        elif i % 4 == 1:
            out.append("-1")
        elif i % 4 == 2:
            out.append("nan")
        else:
            out.append("1")
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
