# TIER: strong
# The insight: ALL logged clock-primes are windows onto the SAME global integer
# recurrence (the char poly of the hidden resonance matrix), so per-prime
# Berlekamp-Massey-style solves are not independent fits to reconcile -- they
# are noisy views of ONE law. Solve the order-d linear system mod each logged
# prime separately (exact, via Gaussian elimination over F_p), then CRT-fuse
# the per-prime coefficient vectors into the exact INTEGER recurrence (valid
# because the true coefficients are small enough that the product of the
# logged primes exceeds twice their bound -- so the centered CRT lift is
# unique). From the exact integer characteristic polynomial, Newton's
# identities bootstrap the exact initial state a_0..a_{d-1} with NO further
# data at all. Any future night's brightness, modulo ANY modulus (logged or
# brand new), then follows by one fast companion-matrix exponentiation mod
# that modulus -- exact, regardless of how far in the future it is asked.
import sys


def solve_mod(rowsA, rhs, p, d):
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
    eqs = find_windows(rows, p, d)
    if len(eqs) < d:
        return None
    A = [eq[:d][::-1] for eq in eqs]
    rhs = [eq[d] for eq in eqs]
    return solve_mod(A, rhs, p, d)


def ext_gcd(a, b):
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = ext_gcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def crt_combine(residues):
    """residues: list of (r,m) pairs, m's pairwise coprime -> unique r mod lcm(m)."""
    r_acc, m_acc = 0, 1
    for r2, m2 in residues:
        g, x, y = ext_gcd(m_acc, m2)
        lcm = m_acc * m2
        r_acc = (r_acc + m_acc * ((x * (r2 - r_acc)) % m2)) % lcm
        m_acc = lcm
    return r_acc, m_acc


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


def bootstrap_p(c, d):
    """Newton's identities: exact a_0..a_d from the char-poly coefficients c alone."""
    p = [d]
    for k in range(1, d + 1):
        s = sum(c[i - 1] * p[k - i] for i in range(1, k))
        p.append(-(s) - k * c[k - 1])
    return p


def companion_answer(c, d, K, P):
    b = [(-ci) % P for ci in c]
    p = bootstrap_p(c, d)
    state = [p[i] % P for i in range(d - 1, -1, -1)]
    C = [[b[j] for j in range(d)]] + [[1 if j == i - 1 else 0 for j in range(d)] for i in range(1, d)]
    t = K - (d - 1)
    Ct = mat_pow_mod(C, t, P)
    return sum(Ct[0][j] * state[j] for j in range(d)) % P


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    testId = int(next(it)); d = int(next(it)); N = int(next(it)); Q = int(next(it))
    rows = []
    for _ in range(N):
        k = int(next(it)); p = int(next(it)); r = int(next(it))
        rows.append((k, p, r))
    queries = []
    for _ in range(Q):
        K = int(next(it)); P = int(next(it))
        queries.append((K, P))

    primes_present = sorted(set(p for k, p, r in rows))
    sols = {}
    for p in primes_present:
        s = solve_recurrence_mod_p(rows, p, d)
        if s is not None:
            sols[p] = s

    out = []
    if not sols:
        out = ["0"] * Q
    else:
        mods = list(sols.keys())
        b_int = []
        for i in range(d):
            residues = [(sols[p][i], p) for p in mods]
            r_acc, m_acc = crt_combine(residues)
            if r_acc > m_acc // 2:
                r_acc -= m_acc
            b_int.append(r_acc)
        c_int = [-b for b in b_int]
        for K, P in queries:
            out.append(str(companion_answer(c_int, d, K, P)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
