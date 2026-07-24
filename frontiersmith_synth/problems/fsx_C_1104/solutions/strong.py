# TIER: strong
# CRT residue-tiling strategy.
#
# Insight: a job with d_i = 1 and 4 | p_i has ALL of its transmission instants
# inside ONE residue class mod 4 (t == o_i (mod 4)).  If one residue class is
# much cheaper than the others, every such "flexible" job can be routed
# wholesale into that class: the family's unavoidable mutual excess (its total
# transmission events minus the class capacity) is then priced at the cheap
# rate instead of the expensive one.  Rigid jobs (runs spanning several
# classes, or periods coprime to 4) cannot do this, so they must claim the
# expensive classes FIRST, while those instants are still empty.
#
# Two periodic jobs i, j can collide at all only at instants
# t == o_i == o_j (mod gcd(p_i, p_j)); aligning/staggering residues is how the
# flexible family is packed into its class.
#
# Algorithm:
#   1. Detect the cheap residue class c* (mod 4) from the posted weights.
#   2. Place rigid jobs first (duty-cycle order, exact marginal cost).
#   3. Route every flexible job to offset c* (maximal mutual alignment inside
#      the cheap class -- excess events there cost w ~ 1 each).
#   4. Polish with exact-marginal local search (2 passes over all jobs).
import sys
import numpy as np


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    M = int(next(it))
    n = int(next(it))
    jobs = [(int(next(it)), int(next(it))) for _ in range(n)]
    w = np.array([float(next(it)) for _ in range(M)], dtype=np.float64)

    # ---- 1. cheap residue class mod 4 ----
    class_sum = [float(w[r::4].sum()) for r in range(4)]
    cstar = min(range(4), key=lambda r: (class_sum[r], r))
    light_ratio = (class_sum[cstar] / (M // 4)) / max(1e-9, w.mean())
    route = light_ratio < 0.6

    flexible = [(d == 1 and p % 4 == 0) for (p, d) in jobs]

    load = np.zeros(M, dtype=np.int32)
    offs = [0] * n
    tau = np.arange(0, max(d for (_, d) in jobs))

    def costvec(p, d):
        occ = (load >= 1).astype(np.float64)
        col = (w * occ).reshape(M // p, p).sum(axis=0)
        ext = np.concatenate([col, col[:d - 1]]) if d > 1 else col
        csum = np.concatenate([[0.0], np.cumsum(ext)])
        return csum[d:d + p] - csum[:p]

    def apply(i, o, delta):
        p, d = jobs[i]
        base = (o + tau[:d]) % p
        idx = (np.arange(M // p, dtype=np.int64)[:, None] * p + base[None, :]).ravel()
        load[idx] += delta

    # ---- 2. rigid jobs first, duty-cycle order, exact marginal ----
    rigid_order = sorted((i for i in range(n) if not flexible[i]),
                         key=lambda i: (-jobs[i][1] / jobs[i][0], jobs[i][0], i))
    for i in rigid_order:
        p, d = jobs[i]
        o = int(np.argmin(costvec(p, d)))
        offs[i] = o
        apply(i, o, +1)

    # ---- 3. route the flexible family into the cheap residue class ----
    flex_ids = [i for i in range(n) if flexible[i]]
    for i in flex_ids:
        p, d = jobs[i]
        if route and cstar < p:
            o = cstar
        else:
            o = int(np.argmin(costvec(p, d)))
        offs[i] = o
        apply(i, o, +1)

    # ---- 4. exact-marginal local search (rigid first, flexible last) ----
    polish_order = rigid_order + flex_ids
    for _ in range(2):
        for i in polish_order:
            p, d = jobs[i]
            apply(i, offs[i], -1)
            cv = costvec(p, d)
            o_best = int(np.argmin(cv))
            if cv[o_best] < cv[offs[i]] - 1e-9:
                offs[i] = o_best
            apply(i, offs[i], +1)

    print(" ".join(map(str, offs)))


if __name__ == "__main__":
    main()
