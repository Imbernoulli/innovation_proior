#!/usr/bin/env python3
"""Deterministic checker for 'Phantom Ward Cartography' (fsx_A_1091).

CLI: python3 verify.py <in> <out> <ans>   (ans is an empty placeholder)

Validates the submitted degree-preserving 2-swap list strictly, replays the
frozen census pipeline (normalized Laplacian -> k smallest eigenvectors ->
deterministic k-means -> best-permutation mismatch vs the planted wards),
and prints the normalized score on its own final line:

    sc = min(1000.0, 100.0 * F / max(1e-9, B));  Ratio = sc/1000

where B is the misclassification achieved by the checker's own canonical
seeded-random blurring attack (documented in the statement).
Bit-for-bit deterministic: all randomness is seeded; BLAS is pinned to one
thread before numpy loads.
"""
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import itertools
import random

import numpy as np

MAX_OUT_BYTES = 8 * 1024 * 1024
INT_TOKEN = set("0123456789+-")


# ---------------------------------------------------------------- instance --
def read_instance(path):
    with open(path, "r") as f:
        tok = f.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        if pos >= len(tok):
            raise ValueError("truncated instance")
        t = tok[pos]
        pos += 1
        return int(t)

    n = nxt(); m = nxt(); k = nxt(); budget = nxt()
    labels = [nxt() for _ in range(n)]
    edges = []
    for _ in range(m):
        a = nxt(); b = nxt()
        edges.append((a, b))
    return n, m, k, budget, labels, edges


# ------------------------------------------------- canonical baseline ------
def canonical_attack(n, edges, budget):
    """The checker's internal baseline: a seeded UNIFORM-random blurring
    attack. This exact procedure is documented in the statement so the
    trivial reference can reproduce it. Deterministic across machines
    (random.Random is Mersenne-Twister, stable)."""
    rng = random.Random(1000003 + n * 131 + budget)
    edge_set = set(edges)
    elist = list(edges)
    swaps = []
    attempts = 0
    limit = 60 * budget + 200
    while len(swaps) < budget and attempts < limit:
        attempts += 1
        i = rng.randrange(len(elist))
        j = rng.randrange(len(elist))
        if i == j:
            continue
        a, b = elist[i]
        c, d = elist[j]
        if a == c or a == d or b == c or b == d:
            continue
        e1 = (a, d) if a < d else (d, a)
        e2 = (b, c) if b < c else (c, b)
        if e1 == e2 or e1 in edge_set or e2 in edge_set:
            continue
        edge_set.discard((a, b))
        edge_set.discard((c, d))
        edge_set.add(e1)
        edge_set.add(e2)
        elist[i] = e1
        elist[j] = e2
        swaps.append((a, b, c, d))
    return swaps


def apply_swaps(edges, swaps):
    es = set(edges)
    for (a, b, c, d) in swaps:
        e_ab = (a, b) if a < b else (b, a)
        e_cd = (c, d) if c < d else (d, c)
        es.discard(e_ab)
        es.discard(e_cd)
        e1 = (a, d) if a < d else (d, a)
        e2 = (b, c) if b < c else (c, b)
        es.add(e1)
        es.add(e2)
    return sorted(es)


# ------------------------------------------------- frozen census pipeline --
def spectral_labels(n, edges, k):
    """Frozen deterministic spectral clustering:
    symmetric normalized Laplacian, k smallest eigenpairs, row-normalized
    embedding, deterministic k-means. Sign/column-permutation invariant:
    k-means uses only pairwise distances, and farthest-point + argmin/argmax
    tie-breaking is fully deterministic."""
    A = np.zeros((n, n), dtype=np.float64)
    for (a, b) in edges:
        A[a, b] = 1.0
        A[b, a] = 1.0
    deg = A.sum(axis=1)
    with np.errstate(divide="ignore"):
        dinv = np.where(deg > 0.0, 1.0 / np.sqrt(np.maximum(deg, 1e-300)), 0.0)
    S = A * dinv[:, None] * dinv[None, :]
    L = np.eye(n) - S
    _, V = np.linalg.eigh(L)          # ascending eigenvalues, deterministic
    U = V[:, :k].copy()
    nrm = np.sqrt((U * U).sum(axis=1))
    nz = nrm > 0.0
    U[nz] /= nrm[nz, None]
    return kmeans_det(U, k)


def kmeans_det(X, k):
    n = X.shape[0]
    sq = (X * X).sum(axis=1)
    centers = [X[int(np.argmax(sq))].copy()]     # argmax: first-max tie-break
    while len(centers) < k:
        d2 = np.full(n, np.inf)
        for c in centers:
            d2 = np.minimum(d2, ((X - c) ** 2).sum(axis=1))
        centers.append(X[int(np.argmax(d2))].copy())
    C = np.stack(centers)
    assign = np.full(n, -1, dtype=np.int64)
    for _ in range(300):
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d2, axis=1)              # argmin: lowest-index tie-break
        if np.array_equal(new, assign):
            break
        assign = new
        dmin = d2[np.arange(n), assign]
        for c in range(k):
            mask = assign == c
            if mask.any():
                C[c] = X[mask].mean(axis=0)
            else:
                C[c] = X[int(np.argmax(dmin))].copy()
    return assign


def misclassification(labels, pred, k):
    n = len(labels)
    best = n + 1
    for perm in itertools.permutations(range(k)):   # fixed order, first-min wins
        mism = 0
        for i in range(n):
            if perm[pred[i]] != labels[i]:
                mism += 1
        if mism < best:
            best = mism
    return best / float(n)


# ------------------------------------------------------------- validation --
def parse_output(path, n, budget):
    """Return swap list or raise ValueError. Strict schema: integer tokens
    only (rejects nan/inf/garbage), S in [0, budget], exactly 4S further
    tokens, vertex ids in range."""
    with open(path, "rb") as f:
        blob = f.read(MAX_OUT_BYTES + 1)
    if len(blob) > MAX_OUT_BYTES:
        raise ValueError("output too large")
    text = blob.decode("utf-8", "strict")
    toks = text.split()
    if not toks:
        raise ValueError("empty output")
    vals = []
    for t in toks:
        if not t or any(ch not in INT_TOKEN for ch in t) or t in ("+", "-"):
            raise ValueError("non-integer token")
        v = int(t)
        vals.append(v)
    S = vals[0]
    if S < 0 or S > budget:
        raise ValueError("swap count out of range")
    if len(vals) != 1 + 4 * S:
        raise ValueError("wrong token count")
    swaps = []
    for i in range(S):
        a, b, c, d = vals[1 + 4 * i: 5 + 4 * i]
        for v in (a, b, c, d):
            if v < 0 or v >= n:
                raise ValueError("vertex id out of range")
        swaps.append((a, b, c, d))
    return swaps


def validate_and_apply(edges, swaps):
    """Sequentially validate degree-preserving 2-swaps; return final edge
    list or raise ValueError."""
    es = set(edges)
    for (a, b, c, d) in swaps:
        if len({a, b, c, d}) != 4:
            raise ValueError("swap endpoints not distinct")
        e_ab = (a, b) if a < b else (b, a)
        e_cd = (c, d) if c < d else (d, c)
        if e_ab == e_cd:
            raise ValueError("same edge twice")
        if e_ab not in es or e_cd not in es:
            raise ValueError("removing a non-edge")
        e1 = (a, d) if a < d else (d, a)
        e2 = (b, c) if b < c else (c, b)
        if e1 == e2:
            raise ValueError("degenerate swap")
        if e1 in es or e2 in es:
            raise ValueError("adding an existing edge")
        es.discard(e_ab)
        es.discard(e_cd)
        es.add(e1)
        es.add(e2)
    return sorted(es)


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return
    inp, outp = sys.argv[1], sys.argv[2]
    try:
        n, m, k, budget, labels, edges = read_instance(inp)
        edges = sorted({(min(a, b), max(a, b)) for (a, b) in edges if a != b})
    except Exception:
        print("checker: bad instance")
        print("Ratio: 0.0")
        return

    # internal baseline: canonical seeded-random blurring attack
    base_edges = apply_swaps(edges, canonical_attack(n, edges, budget))
    B = misclassification(labels, spectral_labels(n, base_edges, k), k)

    try:
        swaps = parse_output(outp, n, budget)
        final_edges = validate_and_apply(edges, swaps)
    except Exception as e:
        sys.stderr.write("infeasible: %s\n" % e)
        print("Ratio: 0.0")
        return

    F = misclassification(labels, spectral_labels(n, final_edges, k), k)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
