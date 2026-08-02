# TIER: trivial
# Feasible but deliberately naive: whenever a probed checkpoint's cumulative
# value can only be explained by a run of L>=2 unprobed edges, dump the
# ENTIRE run's known sum onto the edge nearest the PROBED leaf-side endpoint
# and set every other edge on the run to zero. Edges touched by no probe at
# all get a flat, instance-independent guess. This matches the checker's own
# internal reference baseline exactly.
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
        cost[chain[-1]] = target
        for e in chain[:-1]:
            cost[e] = 0.0
    for e in orphan_edges:
        cost[e] = 1.0
    print("\n".join(f"{cost[v]:.6f}" for v in range(1, N)))


if __name__ == "__main__":
    main()
