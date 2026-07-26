# TIER: trivial
"""Over-engineered baseline: nobody trusts "just leave it alone".

This construction gets the CRT split right (finds m1, q; puts the fixing
requirement on the m1 side; puts a Hull-Dobell-violating affine map with one
fixed point on the q side) but distrusts the idea that the m1-side map can
simply be the identity.  "Permutation" suggests real cycling structure, so it
builds h1 as an *actual* permutation of Z_m1: the k required points are fixed,
and every other residue is folded into one big cycle -- via a from-scratch
Lagrange interpolation over the whole (prime) field Z_m1, evaluated through
Horner's rule.  This is the maximally-general, mechanically-safe way to turn a
hand-built truth table into a straight-line program, but it burns O(m1)
instructions to reproduce something the identity would have given for free,
and it also glues the two CRT halves together with the un-simplified
weighted-idempotent sum (same as greedy).
"""
import sys


def factor_small(n):
    d = 2
    while d * d <= n:
        if n % d == 0:
            return d, n // d
        d += 1
    raise ValueError("n has no small factor")


def find_primitive_root(q):
    if q == 2:
        return 1
    phi = q - 1
    factors = set()
    x = phi
    d = 2
    while d * d <= x:
        if x % d == 0:
            factors.add(d)
            while x % d == 0:
                x //= d
        d += 1
    if x > 1:
        factors.add(x)
    for a in range(2, q):
        if all(pow(a, phi // p, q) != 1 for p in factors):
            return a
    raise ValueError("no primitive root found")


def lagrange_monomial_coeffs(values, mod):
    """values[i] = target h(i) for i in 0..mod-1 (mod prime). Returns monomial
    coefficients c_0..c_{mod-1} (mod `mod`) of the unique interpolating
    polynomial of degree < mod through nodes 0..mod-1."""
    m = mod
    nodes = list(range(m))
    result = [0] * m
    for j in range(m):
        poly = [1]
        denom = 1
        for i in range(m):
            if i == j:
                continue
            newpoly = [0] * (len(poly) + 1)
            for t, coef in enumerate(poly):
                newpoly[t + 1] = (newpoly[t + 1] + coef) % m
                newpoly[t] = (newpoly[t] - coef * nodes[i]) % m
            poly = newpoly
            denom = (denom * ((nodes[j] - nodes[i]) % m)) % m
        inv_denom = pow(denom, -1, m)
        scale = (values[j] * inv_denom) % m
        for t, coef in enumerate(poly):
            result[t] = (result[t] + coef * scale) % m
    return result


def build_h1_table(m1, fixed_y):
    """Permutation of Z_m1: identity on `fixed_y`, one big cycle on the rest."""
    fixed = set(fixed_y)
    rest = sorted(set(range(m1)) - fixed)
    table = [0] * m1
    for y in fixed:
        table[y] = y
    if rest:
        for idx, z in enumerate(rest):
            table[z] = rest[(idx + 1) % len(rest)]
    return table


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    S = [int(data[idx + i]) for i in range(k)]; idx += k
    L = int(data[idx]); idx += 1

    m1, q = factor_small(n)
    c = S[0] % q if k > 0 else 0
    fixed_y = sorted(set(s % m1 for s in S))

    a = find_primitive_root(q)
    b = (c - (a * c) % q) % q

    inv_m1_mod_q = pow(m1, -1, q)
    e2 = (m1 * inv_m1_mod_q) % n
    e1 = (1 - e2) % n

    h1_table = build_h1_table(m1, fixed_y)
    coeffs = lagrange_monomial_coeffs(h1_table, m1)  # coeffs[t] in [0, m1)

    prog = []
    # Horner evaluation of the m1-degree-(m1-1) polynomial into register 3,
    # using register 1 (guaranteed 0 so far) as a zero source for the
    # leading-coefficient load.
    prog.append("ADDC 1 %d 3" % coeffs[m1 - 1])
    for t in range(m1 - 2, -1, -1):
        prog.append("MUL 3 0 3")
        prog.append("ADDC 3 %d 3" % coeffs[t])
    # combine: F(x) = e1*h1(x) + e2*(a*x+b)  (mod n)   [unfolded, as-written]
    prog.append("MULC 3 %d 1" % e1)   # r1 = h1(x)*e1
    prog.append("MULC 0 %d 2" % a)    # r2 = x*a
    prog.append("ADDC 2 %d 2" % b)    # r2 = r2 + b
    prog.append("MULC 2 %d 2" % e2)   # r2 = r2*e2
    prog.append("ADD 1 2 0")          # r0 = r1 + r2

    out = [str(len(prog))] + prog
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
