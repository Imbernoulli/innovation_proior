# ============================================================================
# SHARED CORE (byte-identical block pasted into gen.py AND verify.py).
# Do NOT import this file directly from either -- it is a source-of-truth
# staging copy only; the harness must not see an importable ground-truth
# module sitting in the problem directory.
# ============================================================================
import random
from fractions import Fraction

SMALL_PRIMES = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

LADDER = {
    1:  dict(d=3, E=3, m0=2, bpp=5, Q=14, kf=0.50),
    2:  dict(d=3, E=3, m0=2, bpp=5, Q=14, kf=0.45),
    3:  dict(d=3, E=4, m0=3, bpp=5, Q=15, kf=0.45),
    4:  dict(d=4, E=3, m0=3, bpp=6, Q=16, kf=0.40),
    5:  dict(d=4, E=3, m0=4, bpp=6, Q=16, kf=0.40),
    6:  dict(d=4, E=4, m0=4, bpp=6, Q=17, kf=0.40),
    7:  dict(d=5, E=3, m0=4, bpp=7, Q=18, kf=0.35),
    8:  dict(d=5, E=4, m0=5, bpp=7, Q=18, kf=0.35),
    9:  dict(d=5, E=4, m0=5, bpp=8, Q=19, kf=0.30),
    10: dict(d=5, E=5, m0=6, bpp=8, Q=20, kf=0.30),
}


def mat_mult(A, B):
    n = len(A); m = len(B[0]); k = len(B)
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(m)] for i in range(n)]


def mat_add_scalar(A, s):
    n = len(A)
    return [[A[i][j] + (s if i == j else 0) for j in range(n)] for i in range(n)]


def trace(A):
    return sum(A[i][i] for i in range(len(A)))


def charpoly_int(M):
    """Exact integer characteristic-polynomial coefficients via Faddeev-LeVerrier
    (Fraction arithmetic, asserted integral -- always true for an integer matrix).
    Returns c[0..d-1] = c_1..c_d where det(xI-M) = x^d + c_1 x^{d-1} + ... + c_d."""
    d = len(M)
    Mf = [[Fraction(x) for x in row] for row in M]
    c = []
    Mk = None
    for k in range(1, d + 1):
        if k == 1:
            Mk = Mf
        else:
            Mk = mat_mult(Mf, mat_add_scalar(Mk, c[-1]))
        ck = Fraction(-1, k) * trace(Mk)
        c.append(ck)
    ints = []
    for x in c:
        assert x.denominator == 1
        ints.append(int(x))
    return ints


def seq_trace_powers(M, N):
    """a[0]=tr(M^1) ... a[N-1]=tr(M^N), exact Python big ints."""
    cur = [row[:] for row in M]
    a = [trace(cur)]
    for _ in range(2, N + 1):
        cur = mat_mult(cur, M)
        a.append(trace(cur))
    return a


def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        ok = False
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                ok = True
                break
        if not ok:
            return False
    return True


def next_prime_from(rng, lo, hi):
    cand = rng.randrange(lo, hi) | 1
    while not is_prime(cand):
        cand += 2
        if cand >= hi:
            cand = lo | 1
    return cand


def solve_mod(rowsA, rhs, p, d):
    """Exact Gaussian elimination over F_p. Returns x (len d) or None if not full rank."""
    A = [row[:] + [rhs[i] % p] for i, row in enumerate(rowsA)]
    n = len(A)
    piv_row = 0
    pivcols = []
    for col in range(d):
        sel = None
        for r in range(piv_row, n):
            if A[r][col] % p != 0:
                sel = r
                break
        if sel is None:
            continue
        A[piv_row], A[sel] = A[sel], A[piv_row]
        inv = pow(A[piv_row][col], p - 2, p)
        A[piv_row] = [(x * inv) % p for x in A[piv_row]]
        for r in range(n):
            if r != piv_row and A[r][col] % p != 0:
                f = A[r][col]
                A[r] = [(A[r][j] - f * A[piv_row][j]) % p for j in range(d + 1)]
        pivcols.append(col)
        piv_row += 1
        if piv_row == d:
            break
    if piv_row < d:
        return None
    x = [0] * d
    for i, col in enumerate(pivcols):
        x[col] = A[i][d] % p
    return x


def find_windows(rows_by_k, p, d):
    ks = sorted(k for k, pp, r in rows_by_k if pp == p)
    rowmap = {k: r for k, pp, r in rows_by_k if pp == p}
    ks_set = set(ks)
    eqs = []
    for k in ks:
        window = [k + i for i in range(d + 1)]
        if all(w in ks_set for w in window):
            eqs.append([rowmap[w] for w in window])
    return eqs


def solve_recurrence_mod_p(rows, p, d):
    """a_{k+d} = b_1*a_{k+d-1} + ... + b_d*a_k. Returns b_1..b_d mod p, or None."""
    eqs = find_windows(rows, p, d)
    if len(eqs) < d:
        return None
    A = [eq[:d][::-1] for eq in eqs]
    rhs = [eq[d] for eq in eqs]
    return solve_mod(A, rhs, p, d)


def hidden_construct(testId):
    """Deterministically (re)build the hidden matrix M, the observation rows, and the
    query list for a given testId. Identical in gen.py and verify.py -- NEVER import
    this across files; the checker must not depend on gen.py's presence."""
    cfg = LADDER[testId]
    d, E, m0, bpp, Q, kf = cfg['d'], cfg['E'], cfg['m0'], cfg['bpp'], cfg['Q'], cfg['kf']
    for attempt in range(600):
        rng = random.Random(900001 + 97 * testId + 131 * attempt)
        M = [[rng.randint(-E, E) for _ in range(d)] for __ in range(d)]
        c = charpoly_int(M)
        bound = max(abs(x) for x in c)
        # prefer instances where even the single largest pool prime can't pin the law
        # down alone (keeps the single-modulus trap genuinely active)
        if attempt < 400 and max(SMALL_PRIMES) > 2 * bound:
            continue
        pool = SMALL_PRIMES[:]
        rng2 = random.Random(500000 + testId * 13 + attempt * 7919)
        rng2.shuffle(pool)
        need = 2 * bound + 5
        primes = []
        prod = 1
        idx = 0
        while (len(primes) < m0 or prod <= need) and idx < len(pool):
            primes.append(pool[idx])
            prod *= pool[idx]
            idx += 1
        if prod <= need:
            continue
        L = d + 1
        total_blocks = len(primes) * bpp
        order = []
        for _rep in range(bpp):
            blockorder = primes[:]
            rng2.shuffle(blockorder)
            order.extend(blockorder)
        assign = []
        for b in range(total_blocks):
            assign += [order[b]] * L
        N = len(assign)
        a = seq_trace_powers(M, N)
        rows = [(k + 1, assign[k], a[k] % assign[k]) for k in range(N)]
        # every chosen prime's windowed system must be full rank, else retry
        if not all(solve_recurrence_mod_p(rows, p, d) is not None for p in primes):
            continue
        rngq = random.Random(700000 + testId * 31 + attempt * 131)
        n_known = round(Q * kf)
        queries = []
        for i in range(Q):
            if i < n_known:
                P = primes[i % len(primes)]
            else:
                P = next_prime_from(rngq, 10**6, 10**7)
            if i % 3 == 0:
                K = N + rngq.randint(1, 50)
            else:
                K = rngq.randint(10**6, 10**15)
            queries.append((K, P))
        rngq.shuffle(queries)
        return dict(d=d, N=N, rows=rows, Q=Q, queries=queries, M=M, c=c, primes=primes)
    raise RuntimeError("construction failed for test %d" % testId)
# ============================================================================
# END SHARED CORE
# ============================================================================


def mat_mult_mod(A, B, P):
    n = len(A); m = len(B[0]); k = len(B)
    return [[sum(A[i][t] * B[t][j] for t in range(k)) % P for j in range(m)] for i in range(n)]


def mat_pow_mod(A, e, P):
    n = len(A)
    R = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = [row[:] for row in A]
    while e > 0:
        if e & 1:
            R = mat_mult_mod(R, base, P)
        base = mat_mult_mod(base, base, P)
        e >>= 1
    return R


def true_answer(M, K, P):
    Mm = [[x % P for x in row] for row in M]
    Kp = mat_pow_mod(Mm, K, P)
    return trace(Kp) % P


INT_RE = None


def parse_output(path, Q):
    import re
    global INT_RE
    if INT_RE is None:
        INT_RE = re.compile(r"^-?[0-9]+$")
    try:
        txt = open(path, "r").read()
    except Exception:
        return None
    toks = txt.split()
    if len(toks) != Q:
        return None
    vals = []
    for t in toks:
        if not INT_RE.match(t):
            return None
        vals.append(int(t))
    return vals


def circ_norm(pred, actual, P):
    dd = (pred - actual) % P
    dd = min(dd, P - dd)
    return dd / (P / 2.0)


def main():
    import sys
    inf, outf = sys.argv[1], sys.argv[2]
    with open(inf) as f:
        toks = f.read().split()
    it = iter(toks)
    testId = int(next(it)); d = int(next(it)); N = int(next(it)); Q = int(next(it))
    rows = []
    for _ in range(N):
        k = int(next(it)); p = int(next(it)); r = int(next(it))
        rows.append((k, p, r))
    queries = []
    for _ in range(Q):
        K = int(next(it)); P = int(next(it))
        queries.append((K, P))

    # rebuild ground truth (must match gen.py's hidden_construct byte-for-byte)
    inst = hidden_construct(testId)
    if inst['d'] != d or inst['N'] != N or inst['Q'] != Q:
        print("Ratio: 0.0  # instance metadata mismatch")
        return
    M = inst['M']

    preds = parse_output(outf, Q)
    if preds is None:
        print("Ratio: 0.0  # malformed output: need exactly %d finite base-10 integers" % Q)
        return

    losses = []
    b_losses = []
    for (pred, (K, P)) in zip(preds, queries):
        actual = true_answer(M, K, P)
        losses.append(circ_norm(pred, actual, P))
        b_losses.append(circ_norm(0, actual, P))
    F = sum(losses) / len(losses)
    B = sum(b_losses) / len(b_losses)
    sc = min(880.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
