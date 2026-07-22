#!/usr/bin/env python3
# gen.py <testId>  ->  prints ONE instance of "congruence-sublattice-harvest" to stdout.
#
# THEME: mining dense veins inside a crystal lattice.
# We build a full-rank integer lattice L in Z^n whose UNUSUALLY SHORT vectors are all
# concentrated in a single planted congruence class {v : a.v == 0 (mod p)} (the "vein").
# The public basis B is a scrambled (unimodular) basis whose rows are long, so the vein
# is hidden.  The linear form a (the vein's residue rule) is SECRET; only n, p, k are shown.
#
# Output (stdin the solver receives):
#   line 1:  n p k
#   next n lines: the public basis B (row i = i-th basis vector, n integers)
import sys
import os
from fractions import Fraction as Fr
import random


def _gram_schmidt(B):
    n = len(B); m = len(B[0])
    Bs = []; mu = [[0.0] * n for _ in range(n)]; norm = [0.0] * n
    for i in range(n):
        bi = [float(x) for x in B[i]]
        for j in range(i):
            if norm[j] == 0:
                mu[i][j] = 0.0; continue
            mu[i][j] = sum(B[i][t] * Bs[j][t] for t in range(m)) / norm[j]
            bi = [bi[t] - mu[i][j] * Bs[j][t] for t in range(m)]
        Bs.append(bi); norm[i] = sum(x * x for x in bi)
    return Bs, mu, norm


def _lll(B, delta=0.99):
    B = [r[:] for r in B]; n = len(B); m = len(B[0]) if n else 0
    if n == 0:
        return B
    Bs, mu, norm = _gram_schmidt(B); k = 1; guard = 0
    while k < n and guard < 2000 * n:
        guard += 1
        for j in range(k - 1, -1, -1):
            q = int(round(mu[k][j]))
            if q != 0:
                B[k] = [B[k][t] - q * B[j][t] for t in range(m)]
                Bs, mu, norm = _gram_schmidt(B)
        if norm[k] >= (delta - mu[k][k - 1] ** 2) * norm[k - 1]:
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            Bs, mu, norm = _gram_schmidt(B); k = max(k - 1, 1)
    return B


def _pick(cands, k):
    chosen = []; basis = []; seen = set()
    for v in sorted(cands, key=lambda r: sum(x * x for x in r)):
        key = tuple(v)
        if key in seen or all(x == 0 for x in v):
            continue
        seen.add(key)
        row = [float(x) for x in v]
        for b, pc in basis:
            if row[pc] != 0:
                f = row[pc] / b[pc]
                row = [row[t] - f * b[t] for t in range(len(row))]
        pc = next((c for c in range(len(row)) if abs(row[c]) > 1e-7), None)
        if pc is None:
            continue
        basis.append((row, pc)); chosen.append(v)
        if len(chosen) == k:
            break
    return chosen


def rank(rows):
    # exact rank over Q via fraction Gaussian elimination
    M = [[Fr(x) for x in r] for r in rows]
    if not M:
        return 0
    ncol = len(M[0])
    r = 0
    for c in range(ncol):
        piv = None
        for i in range(r, len(M)):
            if M[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]
        M[r] = [x / pv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [M[i][j] - f * M[r][j] for j in range(ncol)]
        r += 1
        if r == len(M):
            break
    return r


def main():
    t = int(sys.argv[1])
    rng = random.Random(20260710 + 9173 * t)

    # difficulty ladder: keep n small (7 or 8) so a solver can enumerate the ~2^n
    # candidate congruence sublattices within the time limit; difficulty grows via the
    # scramble strength and the number of vein vectors requested.
    n = 7 + (t % 2)                 # alternates 7, 8
    p = 2
    k = n - 2                       # harvest k independent short vectors

    # ---- planted secret linear form a over Z/p, weight >= 2 ----
    while True:
        a = [rng.randint(0, 1) for _ in range(n)]
        if 2 <= sum(a) <= n - 1:
            break

    def residue(v):
        return sum(a[i] * v[i] for i in range(n)) % p

    fix_idx = [i for i in range(n) if a[i] == 1]

    # ---- vein: k short, independent vectors, all with residue 0 ----
    R = []
    while len(R) < k:
        v = [rng.choice([-1, -1, 0, 0, 0, 0, 1, 1]) for _ in range(n)]
        if all(x == 0 for x in v):
            continue
        if residue(v) != 0:
            i = rng.choice(fix_idx)
            v[i] += 1 if v[i] < 1 else -1
            if residue(v) != 0:
                continue
        if rank(R + [v]) == len(R) + 1:
            R.append(v)

    # ---- connectors: longer vectors with residue != 0, completing L to full rank ----
    while len(R) < n:
        v = [rng.randint(-9, 9) for _ in range(n)]
        j = rng.randrange(n)
        v[j] += rng.choice([-1, 1]) * rng.randint(6, 12)
        if residue(v) == 0:
            i = rng.choice(fix_idx)
            v[i] += 1
        if residue(v) == 0:
            continue
        if rank(R + [v]) == len(R) + 1:
            R.append(v)

    # ---- scramble into a public basis B via a unimodular sequence of row operations ----
    # The public basis must (a) HIDE the vein (no short row survives) and (b) present a
    # CONTROLLED baseline: the k shortest row-norms should sit in a fixed band above the vein
    # optimum, so the harvest score is stable and leaves headroom.  We reject-sample scrambles
    # (deterministically, from the seeded stream) until the baseline lands in the band.
    # optimum proxy: harvest the vein from the correct congruence sublattice of L (we know a),
    # exactly what the strong reference achieves.  Band the baseline against THIS.
    def sublat(Brows):
        res = [sum(a[j] * Brows[i][j] for j in range(n)) % p for i in range(n)]
        piv = next((i for i in range(n) if res[i] % p != 0), None)
        if piv is None:
            return [r[:] for r in Brows]
        out = []
        for i in range(n):
            if i == piv:
                continue
            if res[i] % p == 0:
                out.append(Brows[i][:])
            else:
                out.append([Brows[i][t] + Brows[piv][t] for t in range(n)])
        out.append([p * x for x in Brows[piv]])
        return out

    red = _lll(sublat(R))
    fopt_pick = _pick(red, k)
    Fopt = sum(sum(x * x for x in v) for v in fopt_pick) if len(fopt_pick) == k else \
        sum(sum(x * x for x in R[i]) for i in range(k))
    lo, hi = int(6.5 * Fopt), int(7.8 * Fopt)

    def scramble(seed_rng, n_ops):
        Bn = [row[:] for row in R]
        for _ in range(n_ops):
            i = seed_rng.randrange(n)
            j = seed_rng.randrange(n)
            if i == j:
                continue
            c = seed_rng.choice([-2, -1, 1, 2])
            Bn[i] = [Bn[i][x] + c * Bn[j][x] for x in range(n)]
        for _ in range(n):
            i = seed_rng.randrange(n)
            if seed_rng.random() < 0.5:
                Bn[i] = [-x for x in Bn[i]]
        seed_rng.shuffle(Bn)
        return Bn

    B = None
    best = None
    best_gap = None
    for attempt in range(6000):
        n_ops = 8 + (attempt % 9)          # 8..16 mixing operations
        cand = scramble(rng, n_ops)
        rawtopk = sum(sorted(sum(x * x for x in row) for row in cand)[:k])
        if lo <= rawtopk <= hi:
            B = cand
            break
        gap = min(abs(rawtopk - lo), abs(rawtopk - hi))
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best = cand
    if B is None:
        B = best

    out = [f"{n} {p} {k}"]
    for row in B:
        out.append(" ".join(str(x) for x in row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
