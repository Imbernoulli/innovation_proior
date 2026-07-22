# TIER: invalid
import sys


def main():
    toks = sys.stdin.read().split()
    W = int(toks[0])
    # emit garbage: wrong count and an out-of-range / non-finite value mixed in
    vals = [str(999)] * (W - 1) + ["nan"]
    print(" ".join(vals))


if __name__ == "__main__":
    main()
