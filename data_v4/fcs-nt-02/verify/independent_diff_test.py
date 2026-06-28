#!/usr/bin/env python3
import math
import random
import subprocess
import sys


SOL = "/tmp/fcs_nt_02_sol"
LIMIT = 10**18


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


def primes_upto(n):
    sieve = bytearray(b"\x01") * (n + 1)
    if n >= 0:
        sieve[0] = 0
    if n >= 1:
        sieve[1] = 0
    p = 2
    while p * p <= n:
        if sieve[p]:
            start = p * p
            sieve[start:n + 1:p] = b"\x00" * (((n - start) // p) + 1)
        p += 1
    return [i for i in range(n + 1) if sieve[i]]


def comb_small_mod_prime(n, r, p):
    if r < 0 or r > n:
        return 0
    return math.comb(n, r) % p


def lucas_mod_prime(n, r, p):
    if r < 0 or r > n:
        return 0
    ans = 1
    while n or r:
        ni = n % p
        ri = r % p
        if ri > ni:
            return 0
        ans = (ans * comb_small_mod_prime(ni, ri, p)) % p
        n //= p
        r //= p
    return ans


def exact_oracle(n, r, m):
    if m == 1:
        return 0
    if r < 0 or r > n:
        return 0
    return math.comb(n, r) % m


def run_sol(n, r, m):
    proc = subprocess.run(
        [SOL],
        input=f"{n} {r} {m}\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"solver exited {proc.returncode}: {proc.stderr}")
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"solver produced no output for n={n} r={r} m={m}")
    return int(out)


def prime_powers(limit):
    out = []
    for p in (2, 3, 5, 7):
        x = p
        while x <= limit:
            out.append(x)
            x *= p
    return out


def squarefree_composites(primes, limit, count, rng):
    out = set()
    small = [p for p in primes if p <= 97]
    while len(out) < count:
        rng.shuffle(small)
        val = 1
        used = 0
        for p in small:
            if val * p > limit:
                continue
            val *= p
            used += 1
            if used >= rng.randint(2, 5) and rng.random() < 0.65:
                break
        if used >= 2:
            out.add(val)
    return sorted(out)


def main():
    rng = random.Random(20260628)
    primes = primes_upto(1_000_000)
    ppows = prime_powers(1_000_000)
    sqfree = squarefree_composites(primes, 1_000_000, 80, rng)
    full_composites = [
        1_000_000, 999_936, 999_900, 999_720, 998_244, 997_920,
        831_600, 720_720, 554_400, 360_360, 277_200, 166_320,
        110_880, 83_160, 75_600, 55_440, 45_360, 25_200,
    ]

    cases = []

    edge_ms = [1, 2, 3, 4, 5, 7, 8, 9, 16, 25, 27, 32, 49, 64,
               81, 125, 343, 625, 1024, 3125, 15625, 1000000,
               999983, 999979, 720720, 831600]
    edge_ns = [0, 1, 2, 3, 10, 97, 1000, LIMIT]
    for m in edge_ms:
        for n in edge_ns:
            cases.append((n, 0, m, "edge r=0", None))
            cases.append((n, n, m, "edge r=n", None))
            if n < LIMIT:
                cases.append((n, n + 1, m, "edge r>n", None))
        cases.append((LIMIT, LIMIT + 1 if False else LIMIT, m, "edge n=r=1e18", None))

    for m in ppows + sqfree + full_composites:
        for n, r in [(0, 0), (1, 2), (10, 3), (64, 31), (127, 63),
                     (256, 128), (999, 500), (1500, 750), (5000, 2500)]:
            cases.append((n, r, m, "adversarial exact", exact_oracle(n, r, m)))

    for _ in range(700):
        bucket = rng.randrange(6)
        if bucket == 0:
            m = rng.choice([1] + ppows)
        elif bucket == 1:
            m = rng.choice(sqfree)
        elif bucket == 2:
            m = rng.choice(full_composites)
        elif bucket == 3:
            m = rng.choice(primes)
        else:
            m = rng.randint(1, 1_000_000)
        n = rng.randint(0, 5000)
        choice = rng.random()
        if choice < 0.12:
            r = 0
        elif choice < 0.24:
            r = n
        elif choice < 0.34:
            r = n + rng.randint(1, 50)
        else:
            r = rng.randint(0, n)
        cases.append((n, r, m, "random exact", exact_oracle(n, r, m)))

    large_primes = [2, 3, 5, 97, 65537, 999983]
    for p in large_primes:
        for n, r in [
            (LIMIT, 1),
            (LIMIT, 2),
            (LIMIT, 12345),
            (LIMIT, LIMIT // 2),
            (LIMIT - 12345, 987654321),
            (10**17 + 1234567, 10**12 + 345),
            (min(LIMIT, p**3 + p**2 + 17), min(p**2 + 11, min(LIMIT, p**3 + p**2 + 17))),
        ]:
            cases.append((n, r, p, "large prime Lucas", lucas_mod_prime(n, r, p)))

    for _ in range(120):
        p = rng.choice(primes)
        n = rng.randint(0, LIMIT)
        if rng.random() < 0.2:
            r = n + rng.randint(1, 1000)
        else:
            shape = rng.random()
            if shape < 0.45:
                r = rng.randint(0, min(n, 10_000))
            elif shape < 0.9:
                r = max(0, n - rng.randint(0, min(n, 10_000)))
            else:
                r = rng.randint(0, n)
        cases.append((n, r, p, "random large prime Lucas", lucas_mod_prime(n, r, p)))

    seen = set()
    unique_cases = []
    for item in cases:
        key = item[:3]
        if key not in seen:
            seen.add(key)
            unique_cases.append(item)

    for idx, (n, r, m, label, expected) in enumerate(unique_cases, 1):
        if expected is None:
            expected = exact_oracle(n, r, m) if n <= 5000 else (1 % m if r == 0 or r == n else 0)
        got = run_sol(n, r, m)
        if got != expected:
            print(f"mismatch #{idx} [{label}]: n={n} r={r} m={m}", file=sys.stderr)
            print(f"expected {expected}, got {got}", file=sys.stderr)
            return 1

    print(f"PASS {len(unique_cases)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
