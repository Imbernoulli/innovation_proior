# TIER: trivial
"""
Reproduces the checker's own calibration baseline exactly: compute each
key's causal frontier (unavoidable when the frontier has size 1 -- there is
only one valid choice), then for any genuinely concurrent key (frontier
size >= 2) arbitrarily pick the MINIMUM-weight frontier member and never
attempt a merge. No strategy at all for the actual conflict case.
"""
import sys


def leq(a, b):
    return all(x <= y for x, y in zip(a, b))


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    R = int(next(it)); K = int(next(it)); N = int(next(it)); _BUDGET = int(next(it))
    _mtypes = [int(next(it)) for _ in range(K)]
    _mcosts = [int(next(it)) for _ in range(K)]
    ops = []
    for _ in range(N):
        replica = int(next(it)); key = int(next(it)); value = int(next(it))
        weight = int(next(it)); _ts = int(next(it))
        vc = [int(next(it)) for _ in range(R)]
        ops.append((replica, key, value, weight, vc))

    by_key = [[] for _ in range(K)]
    for idx, op in enumerate(ops, start=1):
        by_key[op[1]].append(idx)

    out_lines = []
    for k in range(K):
        ids = by_key[k]
        frontier = []
        for i in ids:
            vci = ops[i - 1][4]
            dominated = False
            for j in ids:
                if i == j:
                    continue
                vcj = ops[j - 1][4]
                if leq(vci, vcj) and vci != vcj:
                    dominated = True
                    break
            if not dominated:
                frontier.append(i)
        chosen = min(frontier, key=lambda i: ops[i - 1][3])
        out_lines.append(f"{k} P {chosen} {ops[chosen - 1][2]}")

    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
