# TIER: greedy
import sys


def allowed_hours(g, r):
    return [h for h in range(24) if h % g == r]


def main():
    toks = sys.stdin.read().split()
    idx = 0
    W = int(toks[idx]); idx += 1
    L = int(toks[idx]); idx += 1
    C = int(toks[idx]); idx += 1
    g = int(toks[idx]); idx += 1
    r = int(toks[idx]); idx += 1
    base = [int(toks[idx + i]) for i in range(24)]; idx += 24
    K = int(toks[idx]); idx += 1
    # greedy never looks at the surge profiles at all -- it just spreads the
    # roster evenly across the allowed hours, anchored at the first one.
    # This is the "obvious" answer: cover the clock uniformly.

    A = allowed_hours(g, r)
    n = len(A)
    starts = [A[(i * n) // W] for i in range(W)]
    print(" ".join(str(x) for x in starts))


if __name__ == "__main__":
    main()
