# TIER: invalid
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    C = int(next(it)); P = int(next(it)); H = int(next(it)); Lmax = int(next(it))
    A_full = int(next(it)); t = int(next(it))
    # Deliberately infeasible: negative reserve estimates (money that isn't
    # money) for every cohort.
    out = ["-1000.0"] * C
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
