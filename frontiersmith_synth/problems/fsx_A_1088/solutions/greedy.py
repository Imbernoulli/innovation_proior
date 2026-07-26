# TIER: greedy
"""Textbook CRT reconstruction, applied literally.

The "average strong coder" reaches for the standard Hull-Dobell full-cycle
recipe first (a=1, pure translation) -- but that map is fixed-point free, so it
can never satisfy the "fix S pointwise" requirement whenever S is non-empty.
The next thing anyone reaches for is the *textbook* Chinese-Remainder
reconstruction formula: build the two per-factor maps (identity mod m1, an
affine map with the right fixed point mod q) and glue them with the standard
weighted-idempotent sum

    F(x) = e1*x + e2*(a*x + b)   (mod n)

exactly as it is written in every CRT reference.  This IS a fully correct
construction, but it is emitted TERM BY TERM -- the solver never notices that,
because both per-factor pieces are affine, the whole right-hand side collapses
algebraically into a single affine map.  That missed simplification costs
2.5x the instructions of the folded (strong) version.
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


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    S = [int(data[idx + i]) for i in range(k)]; idx += k
    L = int(data[idx]); idx += 1

    m1, q = factor_small(n)
    c = S[0] % q if k > 0 else 0

    a = find_primitive_root(q)
    b = (c - (a * c) % q) % q

    inv_m1_mod_q = pow(m1, -1, q)
    e2 = (m1 * inv_m1_mod_q) % n
    e1 = (1 - e2) % n

    # h1(x) = x (identity mod m1), applied to register 0 directly.
    # h2(x) = a*x + b (mod q-component), computed then weighted by e2.
    out = []
    prog = [
        "MULC 0 %d 1" % e1,   # r1 = x*e1
        "MULC 0 %d 2" % a,    # r2 = x*a
        "ADDC 2 %d 2" % b,    # r2 = r2 + b
        "MULC 2 %d 2" % e2,   # r2 = r2*e2
        "ADD 1 2 0",          # r0 = r1 + r2
    ]
    out.append(str(len(prog)))
    out.extend(prog)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
