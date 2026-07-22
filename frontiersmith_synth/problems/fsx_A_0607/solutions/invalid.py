# TIER: invalid
# Schema-valid but WRONG: one dummy instruction, every query answered with the
# literal 0.  Passes token-count parsing but fails the exact-equivalence gate -> 0.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); K = int(next(it))
    toks = ["1", "+", "x0", "x0"] + ["0"] * K
    sys.stdout.write(" ".join(toks) + "\n")


if __name__ == "__main__":
    main()
