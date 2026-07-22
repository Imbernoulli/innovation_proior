#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE lottery-machine-forensics instance to stdout.

Hidden law (never printed): x_{k+1} = (a*x_k + c) mod m, m = product of 4 distinct
secret primes in [101,4999]. Reveals the first N_REVEAL draws and asks for Q_QUERIES
future draws at indices up to 1e18. All randomness is seeded ONLY from testId.
"""
import sys
import random


def _sieve(limit):
    isc = [True] * (limit + 1)
    isc[0] = isc[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if isc[i]:
            for j in range(i * i, limit + 1, i):
                isc[j] = False
    return [i for i in range(2, limit + 1) if isc[i]]


_PRIMES = _sieve(4999)
_BANDS = [(101, 300), (300, 800), (800, 1800), (1800, 3200), (3200, 4999)]
N_REVEAL = 120
K_PRIMES = 4
_ANCHORS = [130, 900, 15000, 2_000_000, 900_000_000,
            400_000_000_000, 200_000_000_000_000, 900_000_000_000_000_000]


def _crt(residues, moduli):
    x, M = 0, 1
    for r, mod in zip(residues, moduli):
        r = r % mod
        t = ((r - x) * pow(M, -1, mod)) % mod
        x = x + M * t
        M *= mod
    return x % M


def make_secret(t):
    """Deterministic hidden law for instance t. Even t -> one prime factor gets the
    planted a_i=1,c_i=0 'frozen' regime (trap: breaks any single global modular
    inverse of consecutive differences, since that prime then divides EVERY
    revealed difference)."""
    rnd = random.Random(1_000_003 * t + 7)
    band = _BANDS[(t - 1) // 2]
    cand = [p for p in _PRIMES if band[0] <= p <= band[1]]
    primes = sorted(rnd.sample(cand, K_PRIMES))
    trap = (t % 2 == 0)
    trap_idx = rnd.randrange(K_PRIMES) if trap else -1
    a_list, c_list, x0_list = [], [], []
    for idx, p in enumerate(primes):
        if idx == trap_idx:
            a_list.append(1)
            c_list.append(0)
        else:
            a_list.append(rnd.randrange(2, p - 1))
            c_list.append(rnd.randrange(0, p))
        x0_list.append(rnd.randrange(0, p))
    m = 1
    for p in primes:
        m *= p
    a = _crt(a_list, primes)
    c = _crt(c_list, primes)
    x0 = _crt(x0_list, primes)
    ks = []
    for anc in _ANCHORS:
        j = rnd.randint(0, max(1, anc // 25))
        ks.append(anc + j)
    ks = sorted(set(ks))
    return primes, m, a, c, x0, trap_idx, ks


def gen_sequence(m, a, c, x0, n):
    xs = [x0 % m]
    for _ in range(n - 1):
        xs.append((a * xs[-1] + c) % m)
    return xs


def main():
    t = int(sys.argv[1])
    primes, m, a, c, x0, trap_idx, ks = make_secret(t)
    xs = gen_sequence(m, a, c, x0, N_REVEAL)
    # NOTE: the instance id t is deliberately NOT printed -- t determines the RNG seed
    # via a fixed, guessable formula, so exposing it would let a submission reimplement
    # make_secret(t) directly instead of exploiting the revealed draws. The checker
    # re-identifies the instance from the (high-entropy) revealed sequence itself.
    out = []
    out.append(str(N_REVEAL))
    out.append(" ".join(map(str, xs)))
    out.append(str(len(ks)))
    out.append(" ".join(map(str, ks)))
    print("\n".join(out))


if __name__ == "__main__":
    main()
