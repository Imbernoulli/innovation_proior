# TIER: strong
"""PC-style conditional-independence skeleton + equal-disturbance (entropy) ordering.

Skeleton (constraint-based, PC algorithm up to conditioning order 2):
  * Order 0: link every pair whose marginal mutual information exceeds a threshold.
  * Orders 1 and 2: for each surviving link i-j, if there is a small set S of other
    currently-adjacent subsystems such that the CONDITIONAL mutual information
    MI(i;j | S) drops below the threshold, the pair is conditionally independent
    given S -> delete the link.  This screens off INDIRECT (transitive) associations
    that marginal MI mistakes for real edges, while co-parents of a collider (which
    are marginally independent) were never linked in the first place.

Orientation (equal-disturbance / entropy gradient):
  Under equal-variance disturbances a source subsystem has a small-variance latent
  that discretizes mostly into the central 'nominal' code (low marginal entropy),
  whereas a descendant accumulates variance and spreads across the outer codes
  (higher marginal entropy).  Sorting subsystems by ASCENDING marginal entropy
  estimates a topological order; each skeleton link is oriented from the
  lower-entropy (earlier) subsystem to the higher-entropy (later) one.

Residual longer-than-2 chains, the occasional entropy tie / mis-order, and the
weakest links washed out by finite samples keep it short of perfect, especially
on the larger, denser, data-poorer held-out rigs."""
import sys, json
import itertools
import numpy as np


def mutual_info(x, y, K):
    N = len(x)
    joint = np.zeros((K, K))
    np.add.at(joint, (x, y), 1.0)
    joint /= N
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    m = 0.0
    for a in range(K):
        if px[a] <= 0:
            continue
        for b in range(K):
            if joint[a, b] > 0 and py[b] > 0:
                m += joint[a, b] * np.log(joint[a, b] / (px[a] * py[b]))
    return m


def cond_mutual_info(data, i, j, S, K):
    """MI(i;j | S) by stratifying on the joint value of the conditioning set S."""
    N = data.shape[0]
    if not S:
        return mutual_info(data[:, i], data[:, j], K)
    key = np.zeros(N, dtype=np.int64)
    for s in S:
        key = key * K + data[:, s]
    total = 0.0
    for v in np.unique(key):
        mask = key == v
        nz = int(mask.sum())
        if nz < 8:                         # too few samples in this stratum to trust
            continue
        total += (nz / N) * mutual_info(data[:, i][mask], data[:, j][mask], K)
    return total


def entropy(x, K):
    N = len(x)
    counts = np.bincount(x, minlength=K) / N
    return -sum(p * np.log(p) for p in counts if p > 0)


def main():
    inst = json.load(sys.stdin)
    data = np.asarray(inst["data"], dtype=np.int64)
    d = int(inst["n_nodes"])
    K = int(inst.get("n_categories", int(data.max()) + 1))
    thr = 0.02
    max_ord = 2

    # ---- PC skeleton ----
    adj = {i: set() for i in range(d)}
    for i in range(d):
        for j in range(i + 1, d):
            if mutual_info(data[:, i], data[:, j], K) > thr:
                adj[i].add(j)
                adj[j].add(i)

    for L in range(1, max_ord + 1):
        for i in range(d):
            for j in list(adj[i]):
                if j < i:
                    continue
                neigh = list(adj[i] - {j})
                if len(neigh) < L:
                    continue
                remove = False
                for S in itertools.combinations(neigh, L):
                    if cond_mutual_info(data, i, j, S, K) < thr:
                        remove = True
                        break
                if remove:
                    adj[i].discard(j)
                    adj[j].discard(i)

    skeleton = [(i, j) for i in range(d) for j in adj[i] if i < j]

    # ---- orientation by ascending marginal entropy ----
    ent = np.array([entropy(data[:, c], K) for c in range(d)])
    order = np.argsort(ent, kind="mergesort")          # ascending entropy ~ topo order
    rank = np.empty(d, dtype=np.int64)
    for pos, node in enumerate(order):
        rank[node] = pos

    edges = []
    for (i, j) in skeleton:
        if rank[i] < rank[j]:
            edges.append([i, j])
        else:
            edges.append([j, i])
    print(json.dumps({"edges": edges}))


main()
