#!/usr/bin/env python3
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Choose a regime so that brute force stays fast (n small enough) but
    # the modulus m exercises prime-power + composite structure.
    regime = rng.randint(0, 6)

    if regime == 0:
        # tiny n, tiny m
        n = rng.randint(0, 30)
        m = rng.randint(1, 50)
    elif regime == 1:
        # n moderate, m a prime power (forces Granville factorial path)
        primes = [2, 3, 5, 7, 11, 13]
        p = rng.choice(primes)
        e = rng.randint(1, 4)
        m = p ** e
        while m > 10**6:
            e -= 1; m = p ** e
        n = rng.randint(0, 400)
    elif regime == 2:
        # composite m with several distinct prime factors (CRT path)
        cand = [6, 10, 12, 15, 30, 36, 60, 72, 100, 144, 210, 720, 1000, 999983]
        m = rng.choice(cand)
        n = rng.randint(0, 500)
    elif regime == 3:
        # m squarefree product of small primes
        m = 1
        for p in [2, 3, 5, 7, 11]:
            if rng.random() < 0.5:
                m *= p
        if m == 1:
            m = 2
        n = rng.randint(0, 600)
    elif regime == 4:
        # high prime power of 2 / 3 to stress Wilson sign of blocks
        choice = rng.choice([2**10, 2**16, 3**10, 5**8, 7**6])
        m = choice
        while m > 10**6:
            m //= 2 if m % 2 == 0 else 3
        n = rng.randint(0, 800)
    elif regime == 5:
        # r near the boundary (0, n, n+1, negative-ish via r>n)
        n = rng.randint(0, 200)
        m = rng.randint(1, 1000)
        # bias r to boundaries
        choices = [0, n, max(0, n - 1), 1, rng.randint(0, n + 2)]
        r = rng.choice(choices)
        print(n, r, m)
        return
    else:
        # m up to 1e6 fully random, n moderate
        m = rng.randint(1, 10**6)
        n = rng.randint(0, 500)

    r = rng.randint(0, n + 1)  # may equal n+1 to test r>n -> 0
    print(n, r, m)

if __name__ == "__main__":
    main()
