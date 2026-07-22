# TIER: strong
"""Strong approach: exploits the two composed mechanisms explicitly.

1. modulus-recovery-from-orbit: gcd of the determinant combinations z_k gives a value g
   that is m itself, OR (when a prime factor p is 'frozen': multiplier==1, increment==0
   in that residue class) exactly m*p, because every z_k picks up a bonus factor of p^2
   from that component. Factoring g (bounded prime pool [101,4999], per the statement)
   and taking the *set* of distinct prime factors -- ignoring multiplicity -- recovers
   the true m = product of that set regardless of which case occurred.

2. cycle-structure-fast-forward: for EACH recovered prime p separately, reduce the
   logged draws mod p and diagnose the local regime from data (frozen constant vs a
   genuine affine map), solving the local multiplier/increment independently per prime
   -- a component that is frozen mod one prime never poisons the others. Fast-forward
   each component with affine binary exponentiation, then CRT-recombine the K
   per-prime predictions into the answer mod m.
"""
import sys
import math

PRIME_LO, PRIME_HI = 101, 4999


def _sieve(limit):
    isc = [True] * (limit + 1)
    isc[0] = isc[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if isc[i]:
            for j in range(i * i, limit + 1, i):
                isc[j] = False
    return [i for i in range(2, limit + 1) if isc[i]]


_POOL = [p for p in _sieve(PRIME_HI) if p >= PRIME_LO]


def fast_forward(a, c, mod, x0, k):
    e = k - 1
    res_a, res_c = 1, 0
    cur_a, cur_c = a % mod, c % mod
    while e > 0:
        if e & 1:
            res_a, res_c = (cur_a * res_a) % mod, (cur_a * res_c + cur_c) % mod
        cur_a, cur_c = (cur_a * cur_a) % mod, (cur_a * cur_c + cur_c) % mod
        e >>= 1
    return (res_a * x0 + res_c) % mod


def crt(residues, moduli):
    x, M = 0, 1
    for r, mod in zip(residues, moduli):
        r = r % mod
        t = ((r - x) * pow(M, -1, mod)) % mod
        x = x + M * t
        M *= mod
    return x % M


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0].strip())
    xs = list(map(int, data[1].split()))
    q = int(data[2].strip())
    ks = list(map(int, data[3].split()))

    ys = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    zs = [ys[i + 1] ** 2 - ys[i + 2] * ys[i] for i in range(len(ys) - 2)]
    g = 0
    for z in zs:
        g = math.gcd(g, abs(z))

    # factor g over the disclosed prime pool; take the DISTINCT prime set as the true
    # factorization of m (strips the bonus p^2 a frozen component would otherwise add).
    rem = g
    primes_true = []
    for p in _POOL:
        if rem % p == 0:
            primes_true.append(p)
            while rem % p == 0:
                rem //= p
        if rem == 1:
            break
    primes_true.sort()

    if len(primes_true) < 2:
        # recovery failed outright: fall back to repeating the last draw.
        print(" ".join(str(xs[-1]) for _ in ks))
        return

    m = 1
    for p in primes_true:
        m *= p

    # per-prime local diagnosis + closed form
    per_prime_params = []  # (p, a_i, c_i, x0_i) with a_i,c_i meaningful only if not frozen
    for p in primes_true:
        loc = [v % p for v in xs]
        yy = [(loc[i + 1] - loc[i]) % p for i in range(len(loc) - 1)]
        frozen = all(v == 0 for v in yy)
        if frozen:
            per_prime_params.append((p, 1, 0, loc[0], True))
            continue
        a_i = c_i = None
        for i in range(len(yy) - 1):
            if yy[i] != 0:
                a_i = (yy[i + 1] * pow(yy[i], -1, p)) % p
                c_i = (loc[i + 1] - a_i * loc[i]) % p
                break
        if a_i is None:
            # extremely unlikely fallback: treat as frozen at the last observed residue
            per_prime_params.append((p, 1, 0, loc[-1], True))
        else:
            per_prime_params.append((p, a_i, c_i, loc[0], False))

    preds = []
    for k in ks:
        comp_vals = []
        for (p, a_i, c_i, x0_i, frozen) in per_prime_params:
            if frozen:
                comp_vals.append(x0_i)
            else:
                comp_vals.append(fast_forward(a_i, c_i, p, x0_i, k))
        preds.append(crt(comp_vals, primes_true))

    print(" ".join(map(str, preds)))


if __name__ == "__main__":
    main()
