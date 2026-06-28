#!/usr/bin/env python3
"""Deterministic local scorer for "Capacitated k-Means".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score. HIGHER is better. INFEASIBLE -> 0.

Scoring rule (see context.md "Evaluation settings"):

  * Instance:
        n k cap
        x_i y_i        (n lines)
    n points in the integer plane, to be partitioned into k clusters, each
    cluster holding AT MOST `cap` points.

  * SOLUTION format (read from <solution_file>): the solver writes
        - n integers a_0 .. a_{n-1}: a_i in [0, k) is the cluster index of point i;
        - then 2*k numbers: the k center coordinates cx_j cy_j (any real values).
    Tokens are whitespace-separated; line breaks do not matter. The center
    coordinates are READ but NOT trusted for scoring: the scorer recomputes each
    cluster's centroid from the assignment itself (the centroid is what minimizes
    the within-cluster squared distance, so this only ever helps a correct
    solver). The center block must still be present and parseable.

  * FEASIBILITY (any violation -> score 0):
      - the file parses as exactly n + 2*k numbers (n integer labels, then 2*k
        reals);
      - every label a_i is an integer in [0, k);
      - NO cluster is empty: every cluster index 0..k-1 is used at least once
        (an empty cluster has no centroid and wastes a center -- it is illegal);
      - every cluster's size is <= cap (the hard cardinality cap).
    If any of these fail, the whole solution is INFEASIBLE and scores 0.

  * COST (lower is better) of a feasible solution: recompute each cluster's
    centroid c_j = mean of the points assigned to j (real arithmetic), then
        cost = sum over points i of || p_i - c_{a_i} ||^2
    (squared Euclidean distance to the assigned cluster's recomputed centroid).

  * SCORE (higher better), normalized against a deterministic baseline the
    scorer recomputes itself -- UNCAPPED Lloyd's k-means (k-means++ seeded) run
    to convergence, then a GREEDY cap-repair that evicts the farthest-from-center
    overflow points into the nearest cluster with spare capacity:
        score = round(1_000_000 * baseline_cost / max(1e-9, solver_cost))
    The baseline scores ~1_000_000; a lower-cost (better) clustering scores more.
    INFEASIBLE -> 0.
"""
import sys
import math
import random


# ----------------------------------------------------------------------------- IO
def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    k = int(next(it))
    cap = int(next(it))
    pts = []
    for _ in range(n):
        x = float(next(it))
        y = float(next(it))
        pts.append((x, y))
    return n, k, cap, pts


def read_solution(path, n, k, cap):
    """Parse + fully validate. Return label list (len n) or None if infeasible."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    # exactly n integer labels followed by 2*k real center coordinates.
    if len(toks) != n + 2 * k:
        return None
    labels = []
    for i in range(n):
        t = toks[i]
        # labels must be plain integers in [0, k).
        try:
            a = int(t)
        except ValueError:
            return None
        if a < 0 or a >= k:
            return None
        labels.append(a)
    # the 2*k center tokens must parse as reals (values not otherwise used).
    for j in range(2 * k):
        try:
            float(toks[n + j])
        except ValueError:
            return None
    # feasibility: no empty cluster, every cluster size <= cap.
    counts = [0] * k
    for a in labels:
        counts[a] += 1
    for j in range(k):
        if counts[j] == 0:
            return None
        if counts[j] > cap:
            return None
    return labels


# ------------------------------------------------------------------------- cost
def assignment_cost(n, k, pts, labels):
    """Recompute centroids from the assignment and return total squared distance."""
    sx = [0.0] * k
    sy = [0.0] * k
    cnt = [0] * k
    for i in range(n):
        a = labels[i]
        sx[a] += pts[i][0]
        sy[a] += pts[i][1]
        cnt[a] += 1
    cx = [0.0] * k
    cy = [0.0] * k
    for j in range(k):
        if cnt[j] > 0:
            cx[j] = sx[j] / cnt[j]
            cy[j] = sy[j] / cnt[j]
    cost = 0.0
    for i in range(n):
        a = labels[i]
        dx = pts[i][0] - cx[a]
        dy = pts[i][1] - cy[a]
        cost += dx * dx + dy * dy
    return cost


# ---------------------------------------------------- baseline: Lloyd + cap-repair
def _kmeans_pp_init(pts, k, rng):
    n = len(pts)
    first = rng.randrange(n)
    centers = [pts[first]]
    d2 = [((p[0] - pts[first][0]) ** 2 + (p[1] - pts[first][1]) ** 2) for p in pts]
    for _ in range(1, k):
        tot = sum(d2)
        if tot <= 0:
            centers.append(pts[rng.randrange(n)])
        else:
            r = rng.uniform(0, tot)
            acc = 0.0
            idx = 0
            for i in range(n):
                acc += d2[i]
                if acc >= r:
                    idx = i
                    break
            centers.append(pts[idx])
        cnew = centers[-1]
        for i in range(n):
            nd = (pts[i][0] - cnew[0]) ** 2 + (pts[i][1] - cnew[1]) ** 2
            if nd < d2[i]:
                d2[i] = nd
    return list(centers)


def baseline_cost(n, k, cap, pts):
    """Uncapped Lloyd's k-means (k-means++ seed), then greedy cap-repair.

    1. Run plain k-means (nearest centroid + centroid recompute) to convergence,
       ignoring the cap. 2. Repair the cap: while some cluster exceeds `cap`,
       evict its farthest-from-center members into the nearest cluster that still
       has spare capacity. This is a legitimate, always-feasible heuristic and the
       cost it achieves is the normalizer.
    """
    rng = random.Random(0xBADC0FFEE ^ (n * 1000003 + k * 101 + cap))

    best = None
    for _trial in range(4):
        centers = _kmeans_pp_init(pts, k, rng)
        labels = [0] * n
        for _it in range(60):
            changed = False
            for i in range(n):
                bestj = 0
                bestd = None
                for j in range(k):
                    dx = pts[i][0] - centers[j][0]
                    dy = pts[i][1] - centers[j][1]
                    d = dx * dx + dy * dy
                    if bestd is None or d < bestd:
                        bestd = d
                        bestj = j
                if labels[i] != bestj:
                    labels[i] = bestj
                    changed = True
            sx = [0.0] * k
            sy = [0.0] * k
            cnt = [0] * k
            for i in range(n):
                a = labels[i]
                sx[a] += pts[i][0]
                sy[a] += pts[i][1]
                cnt[a] += 1
            for j in range(k):
                if cnt[j] > 0:
                    centers[j] = (sx[j] / cnt[j], sy[j] / cnt[j])
                else:
                    centers[j] = pts[rng.randrange(n)]
            if not changed:
                break

        # greedy cap-repair against the converged (uncapped) centers.
        labels = _cap_repair(n, k, cap, pts, centers, labels)
        if labels is None:
            continue
        c = assignment_cost(n, k, pts, labels)
        if best is None or c < best:
            best = c
    if best is None:
        # extremely defensive fallback: round-robin assignment (always feasible
        # because k*cap >= n by construction).
        labels = [i % k for i in range(n)]
        best = assignment_cost(n, k, pts, labels)
    return best


def _cap_repair(n, k, cap, pts, centers, labels):
    cnt = [0] * k
    for a in labels:
        cnt[a] += 1
    # repeatedly drain overfull clusters by evicting their farthest member into
    # the nearest cluster that still has room.
    for j in range(k):
        while cnt[j] > cap:
            # find farthest-from-center member of cluster j
            worst_i = -1
            worst_d = -1.0
            for i in range(n):
                if labels[i] != j:
                    continue
                dx = pts[i][0] - centers[j][0]
                dy = pts[i][1] - centers[j][1]
                d = dx * dx + dy * dy
                if d > worst_d:
                    worst_d = d
                    worst_i = i
            if worst_i < 0:
                return None
            # nearest cluster with spare room
            bestj = -1
            bestd = None
            for t in range(k):
                if t == j or cnt[t] >= cap:
                    continue
                dx = pts[worst_i][0] - centers[t][0]
                dy = pts[worst_i][1] - centers[t][1]
                d = dx * dx + dy * dy
                if bestd is None or d < bestd:
                    bestd = d
                    bestj = t
            if bestj < 0:
                return None  # no room anywhere (should not happen, k*cap >= n)
            labels[worst_i] = bestj
            cnt[j] -= 1
            cnt[bestj] += 1
    # make sure no cluster ended up empty (cap-repair can in principle drain one)
    for j in range(k):
        if cnt[j] == 0:
            # steal the nearest point from the most populous cluster
            src = max(range(k), key=lambda t: cnt[t])
            if cnt[src] <= 1:
                return None
            bi = -1
            bd = None
            for i in range(n):
                if labels[i] != src:
                    continue
                dx = pts[i][0] - centers[j][0]
                dy = pts[i][1] - centers[j][1]
                d = dx * dx + dy * dy
                if bd is None or d < bd:
                    bd = d
                    bi = i
            if bi < 0:
                return None
            labels[bi] = j
            cnt[j] += 1
            cnt[src] -= 1
    return labels


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, k, cap, pts = read_instance(sys.argv[1])

    labels = read_solution(sys.argv[2], n, k, cap)
    if labels is None:
        print(0)
        return

    solver = assignment_cost(n, k, pts, labels)
    base = baseline_cost(n, k, cap, pts)
    score = int(round(1_000_000.0 * base / max(1e-9, solver)))
    print(score)


if __name__ == "__main__":
    main()
