# TIER: strong
# INSIGHT (congruence-sublattice harvest): the unusually short vectors are not spread through
# the whole lattice -- they all live in ONE planted residue class {v : a.v == 0 (mod p)}.
# Reducing the full scrambled basis wanders around generic density.  Instead, for EACH candidate
# linear form a in (Z/p)^n we intersect the lattice with its index-p congruence sublattice
# L_a = {v in L : a.v == 0 (mod p)}, LLL-reduce THAT (a much better-conditioned, vein-aligned
# problem), and harvest the k shortest independent vectors found across all candidates.  The one
# correct a exposes the dense vein; all wrong a's are generic, so taking the global best wins.
import sys


def gram_schmidt(B):
    n = len(B)
    m = len(B[0])
    Bs = []
    mu = [[0.0] * n for _ in range(n)]
    norm = [0.0] * n
    for i in range(n):
        bi = [float(x) for x in B[i]]
        for j in range(i):
            if norm[j] == 0:
                mu[i][j] = 0.0
                continue
            mu[i][j] = sum(B[i][t] * Bs[j][t] for t in range(m)) / norm[j]
            bi = [bi[t] - mu[i][j] * Bs[j][t] for t in range(m)]
        Bs.append(bi)
        norm[i] = sum(x * x for x in bi)
    return Bs, mu, norm


def lll(B, delta=0.99):
    B = [r[:] for r in B]
    n = len(B)
    m = len(B[0]) if n else 0
    if n == 0:
        return B
    Bs, mu, norm = gram_schmidt(B)
    k = 1
    guard = 0
    limit = 2000 * n
    while k < n and guard < limit:
        guard += 1
        for j in range(k - 1, -1, -1):
            q = int(round(mu[k][j]))
            if q != 0:
                B[k] = [B[k][t] - q * B[j][t] for t in range(m)]
                Bs, mu, norm = gram_schmidt(B)
        if norm[k] >= (delta - mu[k][k - 1] ** 2) * norm[k - 1]:
            k += 1
        else:
            B[k], B[k - 1] = B[k - 1], B[k]
            Bs, mu, norm = gram_schmidt(B)
            k = max(k - 1, 1)
    return B


def sublattice_basis(B, a, p):
    """Basis of L_a = {v in L : a.v == 0 mod p} given basis B (rows) and form a."""
    n = len(B)
    res = [sum(a[j] * B[i][j] for j in range(n)) % p for i in range(n)]
    piv = next((i for i in range(n) if res[i] % p != 0), None)
    if piv is None:
        return [r[:] for r in B]
    # p == 2 here: combine odd-residue rows with the pivot, double the pivot.
    out = []
    for i in range(n):
        if i == piv:
            continue
        if res[i] % p == 0:
            out.append(B[i][:])
        else:
            out.append([B[i][t] + B[piv][t] for t in range(n)])
    out.append([p * x for x in B[piv]])
    return out


def independent_pick(cands, k):
    chosen = []
    basis = []
    seen = set()
    for v in sorted(cands, key=lambda r: sum(x * x for x in r)):
        key = tuple(v)
        if key in seen:
            continue
        seen.add(key)
        if all(x == 0 for x in v):
            continue
        row = [float(x) for x in v]
        for b, pc in basis:
            if row[pc] != 0:
                f = row[pc] / b[pc]
                row = [row[t] - f * b[t] for t in range(len(row))]
        pc = next((c for c in range(len(row)) if abs(row[c]) > 1e-7), None)
        if pc is None:
            continue
        basis.append((row, pc))
        chosen.append(v)
        if len(chosen) == k:
            break
    return chosen


def obj(vecs):
    return sum(sum(x * x for x in v) for v in vecs)


def main():
    toks = sys.stdin.read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    p = int(toks[idx]); idx += 1
    k = int(toks[idx]); idx += 1
    B = []
    for i in range(n):
        row = [int(toks[idx + j]) for j in range(n)]
        idx += n
        B.append(row)

    pool = []
    # baseline reduction of the full lattice (what a plain LLL would give)
    pool.extend(lll(B))

    # enumerate every nonzero congruence form a in {0,1}^n (p == 2); reduce each sublattice.
    best = None
    best_obj = None
    for mask in range(1, 1 << n):
        a = [(mask >> j) & 1 for j in range(n)]
        Ba = sublattice_basis(B, a, p)
        red = lll(Ba)
        pool.extend(red)
        pick = independent_pick(red, k)
        if len(pick) == k:
            o = obj(pick)
            if best_obj is None or o < best_obj:
                best_obj = o
                best = pick

    glob = independent_pick(pool, k)
    if len(glob) == k and (best_obj is None or obj(glob) < best_obj):
        best = glob
        best_obj = obj(glob)

    out = []
    for v in best:
        out.append(" ".join(str(x) for x in v))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
