# TIER: strong
"""CRT-surgery insight: fold the whole construction into ONE global affine map.

Factor n = m1 * q (m1 the small factor, q the large prime factor).  On the m1
side we let f act as the IDENTITY (the cheapest possible "fixing gadget": since
every element of S shares one residue c mod q, being the identity mod m1
automatically fixes ALL of S at once -- no per-point machinery is needed at
all).  On the q side we deliberately pick an affine map that VIOLATES the
Hull-Dobell full-cycle condition (coefficient a = a primitive root mod q,
not a=1): that trades exactly one unit of cycle length (q-1 instead of q) for
a single controllable fixed point at c, which is exactly what the fixing
requirement needs.

Because BOTH per-factor maps are affine, their CRT recombination
    F(x) = e1*x + e2*(a*x+b)  (mod n)
is *itself* affine: F(x) = A*x + B (mod n).  Recognizing and folding this
collapses the whole construction to 2 machine instructions, independent of
k = |S| and independent of n's size.
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
    e2 = (m1 * inv_m1_mod_q) % n           # e2 == 0 mod m1, == 1 mod q
    e1 = (1 - e2) % n                      # e1 == 1 mod m1, == 0 mod q

    A = (e1 + (e2 * a) % n) % n
    B = (e2 * b) % n

    out = []
    out.append("2")
    out.append("MULC 0 %d 0" % A)
    out.append("ADDC 0 %d 0" % B)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
