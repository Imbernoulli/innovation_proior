# TIER: invalid
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); m = int(next(it)); K = int(next(it)); L = int(next(it)); seed = int(next(it))
    p = int(next(it))
    for _ in range(p):
        next(it); next(it); next(it)
    re_ = int(next(it))
    for _ in range(re_):
        next(it); next(it)
    ce_ = int(next(it))
    for _ in range(ce_):
        next(it); next(it)
    q = int(next(it))
    # deliberately garbage: wrong count AND a non-finite token, to fail feasibility hard
    out = ["nan"] * max(1, q - 1)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
