# TIER: trivial
"""Static hedge: buy the initial delta-neutral position once and never rebalance.
This reproduces the checker's own internal baseline B exactly."""
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    N = int(toks[idx]); idx += 1
    idx += N + 1                       # skip S
    D0 = float(toks[idx])              # D[0]
    # (we only need D[0]; skip the rest of D, all of G, and the cost line)
    h0 = D0
    print(" ".join("%.10g" % h0 for _ in range(N)))


if __name__ == "__main__":
    main()
