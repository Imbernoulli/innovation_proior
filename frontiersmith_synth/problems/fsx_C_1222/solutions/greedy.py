# TIER: greedy
"""
The obvious "recipe" move any average strong coder reaches for first:
classic last-writer-wins (LWW) register semantics. For each key, among ALL
ops that ever touched it (no causal-graph analysis at all), keep whichever
has the largest wall-clock timestamp; ties broken by larger op id. Never
merges (always converges trivially, spends zero budget).

This always converges (one deterministic value per key) but has no notion
of "concurrent" vs "causally superseded": on a genuinely concurrent key it
silently drops every writer but the timestamp-winner, and if the
timestamp-winner happens to be the least important one (nothing in a raw
wall clock guarantees otherwise) it throws away the most value possible.
"""
import sys


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    R = int(next(it)); K = int(next(it)); N = int(next(it)); _BUDGET = int(next(it))
    _mtypes = [int(next(it)) for _ in range(K)]
    _mcosts = [int(next(it)) for _ in range(K)]
    ops = []
    for _ in range(N):
        replica = int(next(it)); key = int(next(it)); value = int(next(it))
        weight = int(next(it)); ts = int(next(it))
        vc = [int(next(it)) for _ in range(R)]
        ops.append((replica, key, value, weight, ts, vc))

    by_key = [[] for _ in range(K)]
    for idx, op in enumerate(ops, start=1):
        by_key[op[1]].append(idx)

    out_lines = []
    for k in range(K):
        ids = by_key[k]
        best = max(ids, key=lambda i: (ops[i - 1][4], i))
        out_lines.append(f"{k} P {best} {ops[best - 1][2]}")

    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
