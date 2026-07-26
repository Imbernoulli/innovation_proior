#!/usr/bin/env python3
"""gen.py <testId> -- generates one instance of "Cheapest Clockwork" (fsx_A_1088).

Hidden construction (NOT printed to the solver): n = m1 * q, two distinct primes
with m1 small and q large.  A shared secret residue c in [1,q) is picked; the
required fixed-point set S consists of k values that all reduce to c mod q but
have distinct (nonzero) residues mod m1.  L is set to a generous fraction of
(q-1), which is exactly the cycle length any CRT-surgery affine map of the
"identity mod m1 / near-full multiplicative cycle mod q" shape achieves.

Only n, k, S and L are printed -- m1, q, c are never revealed.
"""
import sys, random


def is_prime(x):
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    d = 3
    while d * d <= x:
        if x % d == 0:
            return False
        d += 2
    return True


def next_prime(x):
    x = max(x, 2)
    if x % 2 == 0:
        x += 1
    while not is_prime(x):
        x += 2
    return x


def crt_combine(m1, r1, q, r2):
    """Unique x in [0, m1*q) with x % m1 == r1 and x % q == r2 (gcd(m1,q)=1)."""
    n = m1 * q
    inv_q = pow(q, -1, m1)
    inv_m1 = pow(m1, -1, q)
    x = (r1 * q * inv_q + r2 * m1 * inv_m1) % n
    return x


def gen(test_id: int):
    rng = random.Random(1000003 * test_id + 17)

    m1 = 11 if test_id % 2 == 1 else 13
    q_target = 60 + 140 * (test_id - 1)
    q = next_prime(q_target)
    while q == m1:
        q = next_prime(q + 1)
    n = m1 * q

    k = 3 + ((test_id - 1) % 3)  # 3, 4, or 5

    c = rng.randint(1, q - 1)
    residues = list(range(1, m1))  # nonzero residues mod m1
    rng.shuffle(residues)
    y_list = sorted(residues[:k])

    S = sorted(crt_combine(m1, y, q, c) for y in y_list)

    L = int(0.85 * (q - 1))
    L = max(L, 1)

    return n, k, S, L


def main():
    test_id = int(sys.argv[1])
    n, k, S, L = gen(test_id)
    out = []
    out.append(str(n))
    out.append(str(k))
    out.append(" ".join(str(s) for s in S))
    out.append(str(L))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
