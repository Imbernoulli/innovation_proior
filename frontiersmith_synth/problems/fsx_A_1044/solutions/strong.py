# TIER: strong
import sys

def conv(a, b, n, p):
    res = [0] * n
    for idx in range(n):
        ai = a[idx]
        if ai == 0:
            continue
        for j in range(n):
            k = idx + j
            if k >= n:
                k -= n
            res[k] = (res[k] + ai * b[j]) % p
    return res

def poly_pow(c0, t, n, p):
    result = [0] * n
    result[0] = 1 % p
    base = c0[:]
    while t > 0:
        if t & 1:
            result = conv(result, base, n, p)
        t >>= 1
        if t > 0:
            base = conv(base, base, n, p)
    return result

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); p = int(next(it)); r = int(next(it))
    s = int(next(it)); k = int(next(it))
    coeffs = [int(next(it)) for _ in range(2 * r + 1)]
    targets = []
    for _ in range(k):
        t = int(next(it)); pos = int(next(it)); val = int(next(it)); w = int(next(it))
        targets.append((t, pos, val, w))

    c0 = [0] * n
    for d in range(-r, r + 1):
        c0[d % n] = coeffs[d + r] % p

    distinct_t = sorted(set(t for (t, _, _, _) in targets))
    Ct_cache = {t: poly_pow(c0, t, n, p) for t in distinct_t}

    # Each target cell is a linear functional of x0: row_i[m] = C_{t_i}[(pos_i-m) mod n].
    # Choosing which portraits to satisfy exactly under a hard sparsity budget is therefore
    # a rank / support-geometry question over F_p, not a search over seed placements: run
    # real Gaussian elimination (with pivoting) over the target equations, most important
    # first. Each accepted equation consumes exactly one budget slot (its pivot column) and,
    # crucially, back-substitution re-derives every earlier pivot's value from the FULL
    # final system, so an equation accepted early is never silently invalidated by an
    # equation accepted later -- unlike a forward-only solve. If two portraits ever were
    # linearly dependent, the second would be caught by the reduction below (row becomes
    # all-zero) and resolved for free without spending any budget on it.
    order = sorted(range(k), key=lambda i: (-targets[i][3], i))

    basis = []  # list of (pivot_col, row (length n, mod p, row[pivot_col]==1), rhs)
    for idx in order:
        t, pos, val, w = targets[idx]
        Ct = Ct_cache[t]
        row = [Ct[(pos - m) % n] % p for m in range(n)]
        rhs = val % p

        for (pc, brow, brhs) in basis:
            coeff = row[pc] % p
            if coeff:
                row = [(row[j] - coeff * brow[j]) % p for j in range(n)]
                rhs = (rhs - coeff * brhs) % p

        if all(v == 0 for v in row):
            # Implied by the current basis: either automatically consistent (free match)
            # or a genuine conflict with an already-committed portrait -- either way, no
            # new seed cell is spent here.
            continue

        if len(basis) >= s:
            continue  # budget exhausted; leave this portrait unmatched rather than force it

        pivot_col = next(j for j in range(n) if row[j] != 0)
        inv = pow(row[pivot_col], p - 2, p)
        row = [(v * inv) % p for v in row]
        rhs = (rhs * inv) % p
        basis.append((pivot_col, row, rhs))

    x0 = [0] * n
    for (pc, row, rhs) in reversed(basis):
        total = rhs
        for j in range(n):
            if j != pc and row[j] and x0[j]:
                total = (total - row[j] * x0[j]) % p
        x0[pc] = total % p

    print(" ".join(map(str, x0)))

main()
