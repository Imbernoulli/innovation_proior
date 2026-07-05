#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0467 -- "Ivory Tower: Semi-Supervised Citation Sorting"
(family: ml-node-propagation; format B, quality-metric).

THEME.  A digital library holds a CITATION GRAPH: each node is a paper, an
undirected edge means one paper cites the other, and every paper belongs to
exactly one research subfield (its class label).  A tiny fraction of papers has
been hand-curated with a subfield label (the LABELED seeds); the rest are
UNLABELED.  Each paper also carries a bag-of-topics FEATURE vector that is
subfield-informative but noisy.  The library wants to file every unlabeled paper
into the correct subfield.

This is transductive semi-supervised node classification.  Papers of the same
subfield cite each other more often than across subfields (HOMOPHILY), so a good
solver PROPAGATES / AGGREGATES the seed labels along citation edges and blends in
the noisy feature signal.  The model designs that propagation + aggregation rule.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "n": int,                     # number of papers (node ids 0..n-1)
             "k": int,                     # number of subfields (labels 0..k-1)
             "dim": int,                   # feature dimension
             "features": [[float]*dim]*n,  # features[i] = topic vector of paper i
             "edges": [[u, v], ...],       # undirected citations, 0 <= u < v < n
             "train_ids":    [int, ...],   # labeled seed papers
             "train_labels": [int, ...],   # parallel labels for train_ids (in 0..k-1)
             "query_ids":    [int, ...]}   # papers whose subfield must be predicted
  stdout: ONE JSON object:
            {"labels": [int, ...]}         # parallel to query_ids; each in 0..k-1

  The answer is VALID iff `labels` is a list of exactly len(query_ids) integers,
  each in [0, k).  A crash, timeout, non-JSON, wrong length, or an out-of-range /
  non-integer label -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance:
    acc_cand = fraction of query papers the candidate labels correctly
    acc_base = accuracy of the MAJORITY-CLASS predictor (label every query paper
               with the most frequent seed label; ties -> lowest class index),
               computed by THIS parent process -- the weak reference.
  We normalize with an affine anchor (majority -> 0.1, perfect -> 1.0):
    r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / max(1e-9, 1.0 - acc_base), 0, 1 )
  Matching the majority predictor scores ~0.1; perfect classification scores 1.0;
  doing worse than majority scores < 0.1.  Because the graph is homophilous but
  noisy and the feature signal is imperfect, even a good propagation rule stays
  strictly below 1.0 -> headroom for open-ended improvement.

ISOLATION.  The candidate is untrusted and runs in a FRESH OS-SANDBOXED SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance.  The true query
labels and the majority reference live ONLY in this parent, so a frame-walking /
filesystem-snooping candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math, random
import isorun


# ----------------------------- instance family -----------------------------
def _build_instance(seed, n, k, dim, p_in, p_out, signal, noise, label_frac):
    """Deterministic synthetic citation graph with community/homophily structure.

    Returns a dict with the FULL instance (public fields + hidden query labels).
    """
    rng = random.Random(seed)

    # 1) assign each paper a true subfield, roughly balanced
    truth = [i % k for i in range(n)]
    rng.shuffle(truth)

    # 2) per-class topic centroids (fixed for this instance)
    centroids = [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(k)]

    # 3) node features: class centroid * signal + gaussian noise, rounded for
    #    reproducible JSON serialization
    features = []
    for i in range(n):
        c = truth[i]
        row = [round(signal * centroids[c][d] + rng.gauss(0.0, noise), 4)
               for d in range(dim)]
        features.append(row)

    # 4) homophilous citation edges via a stochastic block model
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            p = p_in if truth[u] == truth[v] else p_out
            if rng.random() < p:
                edges.append([u, v])

    # 5) label split: a small labeled seed set, at least one seed per class
    order = list(range(n))
    rng.shuffle(order)
    n_label = max(k, int(round(label_frac * n)))
    train_ids = order[:n_label]
    # guarantee every class appears among the seeds
    have = set(truth[i] for i in train_ids)
    if len(have) < k:
        pool = order[n_label:]
        for c in range(k):
            if c not in have:
                for j, node in enumerate(pool):
                    if truth[node] == c:
                        train_ids.append(node)
                        pool.pop(j)
                        break
    train_set = set(train_ids)
    query_ids = [i for i in order if i not in train_set]

    train_labels = [truth[i] for i in train_ids]
    query_truth = [truth[i] for i in query_ids]

    inst = {
        "name": f"cite{seed}",
        "n": n, "k": k, "dim": dim,
        "features": features,
        "edges": edges,
        "train_ids": train_ids,
        "train_labels": train_labels,
        "query_ids": query_ids,
        "_query_truth": query_truth,   # HIDDEN: never sent to the candidate
    }
    return inst


def _build_instances():
    """Deterministic instance family.
    (seed, n, k, dim, p_in, p_out, signal, noise, label_frac)."""
    specs = [
        # moderate homophily, moderate noise
        (101, 160, 3, 8, 0.10, 0.015, 1.0, 1.3, 0.15),
        (102, 180, 3, 8, 0.09, 0.015, 1.0, 1.4, 0.15),
        (103, 200, 3, 10, 0.08, 0.012, 1.0, 1.4, 0.12),
        (104, 170, 4, 10, 0.10, 0.012, 1.0, 1.3, 0.15),
        # weaker features (rely more on the graph)
        (205, 190, 3, 8, 0.11, 0.012, 0.8, 1.6, 0.15),
        (206, 210, 4, 10, 0.10, 0.010, 0.8, 1.6, 0.12),
        # weaker graph (rely more on the features)
        (307, 180, 3, 10, 0.06, 0.020, 1.1, 1.2, 0.15),
        (308, 200, 4, 12, 0.05, 0.018, 1.1, 1.2, 0.12),
        # harder / larger held-out instances
        (411, 260, 4, 10, 0.07, 0.014, 0.9, 1.5, 0.10),
        (412, 300, 5, 12, 0.06, 0.010, 0.9, 1.5, 0.10),
        (413, 240, 3, 8, 0.07, 0.016, 0.85, 1.6, 0.10),
        (414, 320, 5, 12, 0.055, 0.010, 0.9, 1.55, 0.08),
    ]
    out = []
    for s in specs:
        out.append(_build_instance(*s))
    return out


# ----------------------------- references ----------------------------------
def _majority_label(train_labels, k):
    counts = [0] * k
    for c in train_labels:
        counts[c] += 1
    best = 0
    for c in range(1, k):
        if counts[c] > counts[best]:
            best = c
    return best


def _accuracy(pred, truth):
    if not truth:
        return 0.0
    correct = sum(1 for a, b in zip(pred, truth) if a == b)
    return correct / len(truth)


# ----------------------------- validation ----------------------------------
def _candidate_accuracy(inst, answer):
    """Validate the answer; return accuracy over query nodes, or None."""
    if not isinstance(answer, dict):
        return None
    labels = answer.get("labels")
    if not isinstance(labels, list):
        return None
    q = inst["query_ids"]
    k = inst["k"]
    if len(labels) != len(q):
        return None
    for c in labels:
        if isinstance(c, bool) or not isinstance(c, int):
            return None
        if c < 0 or c >= k:
            return None
    return _accuracy(labels, inst["_query_truth"])


# ----------------------------- scoring driver ------------------------------
def _public_view(inst):
    return {
        "name": inst["name"], "n": inst["n"], "k": inst["k"], "dim": inst["dim"],
        "features": inst["features"], "edges": inst["edges"],
        "train_ids": inst["train_ids"], "train_labels": inst["train_labels"],
        "query_ids": inst["query_ids"],
    }


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        maj = _majority_label(inst["train_labels"], inst["k"])
        acc_base = _accuracy([maj] * len(inst["_query_truth"]), inst["_query_truth"])
        denom = 1.0 - acc_base
        if denom < 1e-9:
            denom = 1e-9

        public = _public_view(inst)
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            acc_cand = _candidate_accuracy(inst, ans)
        except Exception:
            acc_cand = None
        if acc_cand is None:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (acc_cand - acc_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
