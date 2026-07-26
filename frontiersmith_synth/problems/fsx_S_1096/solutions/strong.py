# TIER: strong
# The genuine insight: FIRST decide which arithmetic regime (p,k) is in.
#   - Type-1: n = k+1 is prime and p is a primitive root mod n.
#   - Type-2: n = 2k+1 is prime, n = 3 (mod 4), and p has order k mod n
#             (p generates the quadratic-residue subgroup).
# In either case a Gauss period of a primitive n-th root of unity generates a
# PROVABLY near-optimal normal basis (every Frobenius-shifted output lane carries
# the identical load, by construction). We build it directly: find a generator g0
# of the full multiplicative group of F_p[x]/(f(x)) (order p^k-1), take
# gamma = g0^((p^k-1)/n) (a primitive n-th root of unity), and set
# beta = gamma (type 1) or beta = gamma + gamma^-1 (type 2); the Frobenius orbit
# of beta is the candidate basis.
# When NEITHER condition holds, the optimal complexity is a genuinely open
# question (no known closed form) -- the best we can do is a wider "permuted
# period" search over many Frobenius-orbit generators (a hybrid/best-effort
# construction, not a proof), always compared against the trivial monomial basis
# and against greedy's own small family so we are never worse than either.
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


def xpow_mod(base, e, f, p, k):
    result = [0] * k; result[0] = 1
    b = base[:]
    while e > 0:
        if e & 1:
            result = pmulmod(result, b, f, p, k)
        e >>= 1
        if e:
            b = pmulmod(b, b, f, p, k)
    return result


def is_zero(a):
    return all(c == 0 for c in a)


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


# ---- number theory: which regime is (p,k) in? ----
def is_prime(m):
    if m < 2:
        return False
    i = 2
    while i * i <= m:
        if m % i == 0:
            return False
        i += 1
    return True


def factorize(m):
    fac = {}
    d = 2; x = m
    while d * d <= x:
        while x % d == 0:
            fac[d] = fac.get(d, 0) + 1; x //= d
        d += 1
    if x > 1:
        fac[x] = fac.get(x, 0) + 1
    return fac


def order_mod(p, n):
    if n == 1:
        return 1
    x = p % n; o = 1; cur = x
    while cur != 1:
        cur = (cur * x) % n; o += 1
        if o > n:
            return None
    return o


def detect_regime(p, k):
    n1 = k + 1
    if is_prime(n1) and order_mod(p, n1) == k:
        return 1, n1
    n2 = 2 * k + 1
    if is_prime(n2) and (n2 % 4 == 3) and order_mod(p, n2) == k:
        return 2, n2
    return 0, None


def poly_inv(a, f, p, k):
    def dgv(v):
        d = len(v) - 1
        while d > 0 and v[d] == 0:
            d -= 1
        return d
    r0, r1 = f[:] + [1], (a[:] if any(a) else [0])
    while len(r1) > 1 and r1[-1] == 0:
        r1.pop()
    s0, s1 = [0] * (k + 1), [1] + [0] * k
    while not (len(r1) == 1 and r1[0] == 0):
        dr1 = dgv(r1)
        q = [0] * (k + 1)
        rem = r0[:]
        while True:
            dr = dgv(rem)
            if dr < dr1 or (len(rem) == 1 and rem[0] == 0):
                break
            c = (rem[dr] * pow(r1[dr1], p - 2, p)) % p
            shift = dr - dr1
            q[shift] = (q[shift] + c) % p
            for i in range(dr1 + 1):
                rem[shift + i] = (rem[shift + i] - c * r1[i]) % p
            while len(rem) > 1 and rem[-1] == 0:
                rem.pop()
        qs1 = [0] * (2 * k + 1)
        for i in range(k + 1):
            if q[i]:
                for j in range(len(s1)):
                    if s1[j]:
                        qs1[i + j] = (qs1[i + j] + q[i] * s1[j]) % p
        news = [(s0[i] - (qs1[i] if i < len(qs1) else 0)) % p for i in range(k + 1)]
        r0, r1 = r1, rem
        s0, s1 = s1, news
    dr0 = dgv(r0)
    if dr0 != 0 or r0[0] == 0:
        raise ValueError("not invertible")
    invc = pow(r0[0], p - 2, p)
    return [(c * invc) % p for c in s0[:k]]


def field_gen(p, k, f, seed, max_tries=250):
    N = p ** k - 1
    fac = list(factorize(N).keys())
    import random
    rng = random.Random(seed)
    cand_list = []
    for c in range(1, min(p, 4)):
        cand_list.append([c] + [0] * (k - 1))
    for i in range(1, k):
        cand_list.append([1] + [0] * (i - 1) + [1] + [0] * (k - i - 1))
    for _ in range(max_tries):
        cand_list.append([rng.randrange(p) for _ in range(k)])
    for cand in cand_list:
        if is_zero(cand):
            continue
        ok = True
        for q in fac:
            if xpow_mod(cand, N // q, f, p, k) == ([1] + [0] * (k - 1)):
                ok = False
                break
        if ok:
            return cand
    return None


def gauss_period_basis(p, k, f, seed):
    typ, n = detect_regime(p, k)
    if typ == 0:
        return None
    g0 = field_gen(p, k, f, seed=seed)
    if g0 is None:
        return None
    N = p ** k - 1
    gamma = xpow_mod(g0, N // n, f, p, k)
    if typ == 1:
        beta = gamma
    else:
        try:
            ginv = poly_inv(gamma, f, p, k)
        except ValueError:
            return None
        beta = [(gamma[i] + ginv[i]) % p for i in range(k)]
    return frob_orbit(beta, f, p, k)


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
    return best, bv


def input_seed(p, k, f):
    s = p * 1000003 + k * 97
    for i, c in enumerate(f):
        s = (s * 131 + (c + 1) * (i + 7)) % (2 ** 61 - 1)
    return s


def strong_solve(p, k, f):
    seed = input_seed(p, k, f)
    best, bv = greedy_family(f, p, k)  # never worse than greedy's own small family
    rows = gauss_period_basis(p, k, f, seed=seed ^ 0x1234)
    if rows is not None and mat_inv_mod_p(rows, p, k) is not None:
        v = max_lane(rows, f, p, k)
        if v is not None and v < bv:
            bv, best = v, rows
    # "permuted period" fallback / cross-check: wider pool of Frobenius orbits of
    # low-weight two- and three-term elements (a legitimate, but non-optimal,
    # constructive search -- the frontier is open when no regime condition holds).
    cand = []
    for c in range(1, p):
        e = [0] * k; e[0] = c; cand.append(e)
    for i in range(1, k):
        for c in range(1, p):
            e = [0] * k; e[0] = 1; e[i] = c; cand.append(e)
    cnt = 0
    for i in range(1, k):
        for j in range(i + 1, k):
            e = [0] * k; e[0] = 1; e[i] = 1; e[j] = 1; cand.append(e); cnt += 1
            if cnt > 150:
                break
        if cnt > 150:
            break
    for elem in cand:
        r2 = frob_orbit(elem, f, p, k)
        if mat_inv_mod_p(r2, p, k) is None:
            continue
        v = max_lane(r2, f, p, k)
        if v is not None and v < bv:
            bv, best = v, r2
    return best


def main():
    data = sys.stdin.read().split()
    p = int(data[0]); k = int(data[1])
    f = [int(x) for x in data[2:2 + k]]
    M = strong_solve(p, k, f)
    out = [" ".join(map(str, row)) for row in M]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
