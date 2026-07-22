# TIER: trivial
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    W = int(toks[idx]); idx += 1
    L = int(toks[idx]); idx += 1
    C = int(toks[idx]); idx += 1
    g = int(toks[idx]); idx += 1
    r = int(toks[idx]); idx += 1
    # ignore the rest (base curve + surge profiles)

    # everyone starts at the same, first allowed hour -- maximal clustering,
    # the "do the simplest possible thing" construction.
    r0 = r  # r is already in [0,g), and r%g==r so it's allowed
    print(" ".join(str(r0) for _ in range(W)))


if __name__ == "__main__":
    main()
