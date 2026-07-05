# TIER: invalid
# Deploys three mutually-interfering routes: 00..0, 00..1, 00..2 (they sum to
# 0 mod 3 coordinate-wise), an explicit collinear triple.  The feasibility gate
# must reject this -> score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    a = "0" * n
    b = "0" * (n - 1) + "1"
    c = "0" * (n - 1) + "2"
    sys.stdout.write("3\n%s\n%s\n%s\n" % (a, b, c))


if __name__ == "__main__":
    main()
