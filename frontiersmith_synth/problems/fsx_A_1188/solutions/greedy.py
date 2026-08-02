# TIER: greedy
"""
The obvious first attempt: treat this as "spend the budget on the
highest-flux edges" -- sort ALL nodes by their individual flux weight and
probe the top Q, one query per node, with no notion of which nodes belong to
the same confounded cluster or what fully separating a cluster costs.

This is exactly the seed's stated trap: nodes that are already uniquely
determined by baseline alone (singleton clusters) often carry the highest
individual flux and get probed for zero benefit, while a query budget poured
into one oversized cluster (because its nodes also rank high) never reaches
the point of actually separating it -- leaving several smaller,
fully-separable clusters elsewhere completely untouched.
"""
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    testId = int(toks[idx]); idx += 1
    N = int(toks[idx]); idx += 1
    L = int(toks[idx]); idx += 1
    Q = int(toks[idx]); idx += 1
    W = int(toks[idx]); idx += 1
    weights = [int(toks[idx + i]) for i in range(N)]
    idx += N
    # adjacency / baseline not needed by this tier -- skip parsing them.

    order = sorted(range(1, N + 1), key=lambda hid: -weights[hid - 1])
    picked = order[:Q]
    print(len(picked))
    print(" ".join(str(x) for x in picked))


if __name__ == "__main__":
    main()
