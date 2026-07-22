# TIER: greedy
# The "obvious" recipe: for each clock prime that shows up in the log, fit a
# single order-d linear recurrence directly on that prime's own residues (find
# windows of d+1 consecutive nights sharing that prime, solve mod that prime).
# This DOES answer a query correctly when the asked modulus P equals a prime it
# already solved for (any representative of the recurrence mod P propagates
# correctly through Newton's identities and the mod-P companion power -- no
# fusion needed there). But for a query against a brand-new modulus never
# logged, it just takes the coefficients solved mod its most-common logged
# prime, centers them into (-p/2, p/2], and reuses that guess AS IF it were
# the true integer law. That shortcut is only valid when every true
# coefficient already fits under that ONE prime's Nyquist bound -- exactly the
# regime the generator avoids, so the guess is aliased and the extrapolation
# for novel moduli comes out essentially arbitrary.
import sys
from collections import Counter


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
    """Newton's identities: a_0..a_d purely from the char-poly coefficients c
    (no data needed). Pure polynomial (no division) -> commutes with mod P."""
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

    cnt = Counter(p for k, p, r in rows)
    majority_p = cnt.most_common(1)[0][0]
    primes_present = sorted(set(p for k, p, r in rows))
    sols = {}
    for p in primes_present:
        s = solve_recurrence_mod_p(rows, p, d)
        if s is not None:
            sols[p] = s
    maj_sol = sols.get(majority_p)

    out = []
    for K, P in queries:
        if P in sols:
            c_p = [-b for b in sols[P]]
            out.append(str(companion_answer(c_p, d, K, P)))
        elif maj_sol is not None:
            b_guess = [(x - majority_p) if x > majority_p // 2 else x for x in maj_sol]
            c_guess = [-b for b in b_guess]
            out.append(str(companion_answer(c_guess, d, K, P)))
        else:
            out.append("0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
