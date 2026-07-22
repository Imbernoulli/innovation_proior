# TIER: greedy
"""Myopic-greedy: at this truck, buy it iff it is affordable right now AND
strictly increases the exact lambda_2 of the network built so far. The obvious
"accept anything that helps" online rule -- it has no notion of pacing, so it
happily spends on every small-but-real early gain and can run dry before the
high-leverage late trucks show up. No cross-call memory is needed."""
import sys, json
import numpy as np


def lam2(n, edges):
    L = np.zeros((n, n))
    for u, v, w in edges:
        L[u, u] += w; L[v, v] += w; L[u, v] -= w; L[v, u] -= w
    return float(sorted(np.linalg.eigvalsh(L))[1])


def main():
    inst = json.load(sys.stdin)
    n = inst["n"]
    u, v, cost = inst["u"], inst["v"], inst["cost"]
    remaining = inst["remaining"]
    edges = [(a, b, 1) for a, b in inst["backbone"]] + [(a, b, 1) for a, b in inst["accepted"]]

    accept = False
    if cost <= remaining + 1e-9:
        base = lam2(n, edges)
        trial = edges + [(u, v, 1)]
        val = lam2(n, trial)
        if val > base + 1e-9:
            accept = True

    print(json.dumps({"accept": accept, "state": None}))


if __name__ == "__main__":
    main()
