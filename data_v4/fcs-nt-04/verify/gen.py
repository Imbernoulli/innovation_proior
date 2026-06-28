import sys
import random

# Random small-case generator for differential testing against the trial-division oracle.
# All generated n are kept <= 10^12 so the O(sqrt(n)) oracle stays fast.
# Cases deliberately include: primes, prime powers, products of two large primes
# (the hard case for Pollard-Rho), products of many small primes, perfect powers,
# and small / boundary values.

CAP = 10**12

def small_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

PRIMES = small_primes(2000000)  # primes up to 2e6; product of two ~ up to 4e12, capped

def is_probable_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def random_prime_below(cap, rnd):
    if cap < 2:
        return 2
    while True:
        c = rnd.randint(2, cap)
        c |= 1
        if c < 2:
            c = 3
        if is_probable_prime(c):
            return c

def make_case(rnd):
    kind = rnd.randint(0, 8)
    if kind == 0:
        return rnd.randint(1, 50)                       # tiny
    if kind == 1:
        return random_prime_below(CAP, rnd)             # a single large prime
    if kind == 2:                                       # product of two large primes (hard)
        p = random_prime_below(10**6, rnd)
        q = random_prime_below(CAP // p, rnd)
        return p * q
    if kind == 3:                                       # prime power
        p = rnd.choice([2, 3, 5, 7, 11, 13])
        n = p
        while n * p <= CAP and rnd.random() < 0.8:
            n *= p
        return n
    if kind == 4:                                       # product of several small primes
        n = 1
        while True:
            p = rnd.choice(PRIMES[:200])
            if n * p > CAP:
                break
            n *= p
            if rnd.random() < 0.3:
                break
        return max(n, 2)
    if kind == 5:                                       # square of a medium prime
        p = random_prime_below(10**6, rnd)
        return p * p
    if kind == 6:                                       # fully random in range
        return rnd.randint(1, CAP)
    if kind == 7:                                       # 1 and 2 boundaries
        return rnd.choice([1, 2, 3, 4])
    # kind == 8: power of two
    e = rnd.randint(1, 39)
    return 1 << e

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rnd = random.Random(seed)
    q = rnd.randint(1, 12)
    cases = [make_case(rnd) for _ in range(q)]
    out = [str(q)]
    out.extend(str(c) for c in cases)
    sys.stdout.write("\n".join(out) + "\n")

main()
