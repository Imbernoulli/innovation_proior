# TIER: strong
# Insight: the probes only ever pin the SUM of an ambiguous run, never the
# split -- so within that run report the canonical representative implied by
# the tree-structure prior instead of an arbitrary/uninformed one. Compute
# leaf counts directly from the given topology (visible to every solver,
# independent of which nodes happen to be probed) and split each run's known
# sum PROPORTIONALLY to those leaf counts, rather than evenly. For edges no
# probe touches at all, calibrate the cost-per-leaf constant empirically from
# the fully-identified (single-edge) runs in THIS instance and extrapolate
# with the same leaf-count law, instead of guessing a fixed number.
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


def leaf_counts(N, parent):
    children = [[] for _ in range(N)]
    for v in range(1, N):
        children[parent[v]].append(v)
    lc = [0] * N
    for v in range(N - 1, -1, -1):
        if not children[v]:
            lc[v] = 1
        if v != 0:
            lc[parent[v]] += lc[v]
    return lc


def main():
    N, parent, measured = read_input()
    chains, orphan_edges = compute_chains(N, parent, measured)
    lc = leaf_counts(N, parent)

    cost = [0.0] * N
    resolved_alpha_samples = []
    for chain, target in chains:
        weight_sum = sum(lc[e] for e in chain)
        if weight_sum <= 0:
            share = target / len(chain)
            for e in chain:
                cost[e] = share
        else:
            for e in chain:
                cost[e] = target * (lc[e] / weight_sum)
        if len(chain) == 1:
            e = chain[0]
            if lc[e] > 0:
                resolved_alpha_samples.append(target / lc[e])

    if resolved_alpha_samples:
        alpha_hat = sum(resolved_alpha_samples) / len(resolved_alpha_samples)
    else:
        alpha_hat = 2.0  # reasonable family-level fallback prior

    for e in orphan_edges:
        cost[e] = max(0.0, alpha_hat * lc[e])

    print("\n".join(f"{cost[v]:.6f}" for v in range(1, N)))


if __name__ == "__main__":
    main()
