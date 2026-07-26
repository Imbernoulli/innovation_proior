# TIER: greedy
# "Obvious" textbook approach: normal bases (Frobenius-conjugate bases {theta,
# theta^p, ..., theta^{p^{k-1}}}) are the standard folklore trick for cheap finite
# field arithmetic, so try the Frobenius orbit of a handful of low-weight elements
# theta (x+c, 2x+c, x^2+c, x+x^3, ...) plus the monomial basis itself, and submit
# whichever of this SMALL FIXED family scores best. No number theory, no check of
# WHICH (p,k) admit a provably-optimal ("type-1/2") Gauss-period basis, no search
# beyond this fixed family.
import sys


def reduce_full(raw, f, p, k):
    a = raw[:]
    for deg in range(len(a) - 1, k - 1, -1):
        c = a[deg]
        if c:
            shift = deg - k
            for i in range(k):
                a[shift + i] = (a[shift + i] - c * f[i]) % p
            a[deg] = 0
    return a[:k]


def pmulmod(a, b, f, p, k):
    raw = [0] * (2 * k - 1)
    for i in range(k):
        ai = a[i]
        if not ai:
            continue
        for j in range(k):
            bj = b[j]
            if bj:
                raw[i + j] = (raw[i + j] + ai * bj) % p
    return reduce_full(raw, f, p, k)


def frob_step(a, f, p, k):
    raw = [0] * ((k - 1) * p + 1)
    for i, c in enumerate(a):
        if c:
            raw[i * p] = (raw[i * p] + c) % p
    return reduce_full(raw, f, p, k)


def frob_orbit(beta, f, p, k):
    rows = []
    cur = beta[:]
    for _ in range(k):
        rows.append(cur[:k] + [0] * (k - len(cur)))
        cur = frob_step(cur, f, p, k)
    return rows


def mat_inv_mod_p(M, p, k):
    A = [row[:] + [1 if i == j else 0 for j in range(k)] for i, row in enumerate(M)]
    for col in range(k):
        piv = None
        for r in range(col, k):
            if A[r][col] % p != 0:
                piv = r
                break
        if piv is None:
            return None
        A[col], A[piv] = A[piv], A[col]
        inv = pow(A[col][col], p - 2, p)
        A[col] = [(x * inv) % p for x in A[col]]
        for r in range(k):
            if r != col and A[r][col] % p != 0:
                c = A[r][col]
                A[r] = [(A[r][j] - c * A[col][j]) % p for j in range(2 * k)]
    return [row[k:] for row in A]


def max_lane(M, f, p, k):
    Minv = mat_inv_mod_p(M, p, k)
    if Minv is None:
        return None
    per_l = [0] * k
    for i in range(k):
        for j in range(i, k):
            prod = pmulmod(M[i], M[j], f, p, k)
            c = [sum(prod[t] * Minv[t][l] for t in range(k)) % p for l in range(k)]
            for l in range(k):
                if c[l] != 0:
                    per_l[l] += 1
    return max(per_l)


def identity(k):
    return [[1 if i == j else 0 for j in range(k)] for i in range(k)]


def greedy_family(f, p, k):
    best = identity(k)
    bv = max_lane(best, f, p, k)
    elems = []
    for c in range(0, min(p, 4)):
        e = [0] * k; e[0] = c; e[1] = 1; elems.append(e)
    for c in range(0, min(p, 3)):
        e = [0] * k; e[0] = c; e[1] = 2 % p if p > 2 else 1; elems.append(e)
    if k > 2:
        for c in range(0, min(p, 3)):
            e = [0] * k; e[0] = c; e[2] = 1; elems.append(e)
    if k > 3:
        e = [0] * k; e[1] = 1; e[3] = 1; elems.append(e)
        e = [0] * k; e[0] = 1; e[1] = 1; e[k - 1] = 1; elems.append(e)
    if k > 4:
        e = [0] * k; e[0] = 1; e[k - 1] = 1; elems.append(e)
        e = [0] * k; e[1] = 1; e[k - 2] = 1; elems.append(e)
    for elem in elems:
        rows = frob_orbit(elem, f, p, k)
        if mat_inv_mod_p(rows, p, k) is None:
            continue
        v = max_lane(rows, f, p, k)
        if v is not None and v < bv:
            bv, best = v, rows
    return best


def main():
    data = sys.stdin.read().split()
    p = int(data[0]); k = int(data[1])
    f = [int(x) for x in data[2:2 + k]]
    M = greedy_family(f, p, k)
    out = [" ".join(map(str, row)) for row in M]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
