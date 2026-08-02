# TIER: greedy
# The obvious textbook fix over the trivial dump: when a probed checkpoint's
# cumulative value can only be explained by a run of L unprobed edges,
# distribute its known sum EQUALLY across the L edges (the minimum-variance /
# minimum-norm answer under "no further information" -- exactly what
# straightforward least-squares would return). Edges touched by no probe
# still get a flat, uncalibrated constant guess. This is a completely
# reasonable first attempt -- it just never looks at HOW the tree branches
# within an ambiguous run, so it is blind to the planted subtree-size skew.
import sys


def read_input():
    toks = sys.stdin.read().split()
    pos = 0
    test_id = int(toks[pos]); pos += 1
    N = int(toks[pos]); pos += 1
    M = int(toks[pos]); pos += 1
    parent = [-1] + [int(toks[pos + i]) for i in range(N - 1)]
    pos += N - 1
    measured = {}
    for _ in range(M):
        v = int(toks[pos]); pos += 1
        dv = float(toks[pos]); pos += 1
        measured[v] = dv
    return N, parent, measured


def compute_chains(N, parent, measured_dict):
    chains = []
    support = set()
    for v in sorted(measured_dict.keys()):
        chain = []
        cur = v
        while True:
            chain.append(cur)
            nxt = parent[cur]
            if nxt == 0 or nxt in measured_dict:
                break
            cur = nxt
        chain.reverse()
        anchor = parent[chain[0]]
        anchor_val = 0.0 if anchor == 0 else measured_dict[anchor]
        target = measured_dict[v] - anchor_val
        chains.append((chain, target))
        support.update(chain)
    orphan_edges = [v for v in range(1, N) if v not in support]
    return chains, orphan_edges


def main():
    N, parent, measured = read_input()
    chains, orphan_edges = compute_chains(N, parent, measured)
    cost = [0.0] * N
    for chain, target in chains:
        share = target / len(chain)
        for e in chain:
            cost[e] = share
    for e in orphan_edges:
        cost[e] = 1.0
    print("\n".join(f"{cost[v]:.6f}" for v in range(1, N)))


if __name__ == "__main__":
    main()
