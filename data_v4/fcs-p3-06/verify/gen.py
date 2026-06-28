#!/usr/bin/env python3
"""
Random test generator for the derangement-mod-p problem.

Usage: gen.py SEED [MODE]

Emits a valid stdin:
  line 1: T P     (T queries, prime modulus P)
  next T lines or tokens: n_i

Modes bias toward different regimes; the differential test driver picks modes
so that we cover small-n (where a hardcoded table would 'work') AND larger n
(where it would not). The brute oracle is exact-integer, so 'larger' here is
capped to keep the Python oracle fast, but the structure still exercises the
recurrence well beyond any plausible hardcoded prefix.
"""
import sys
import random

# A pool of primes (small and large) to use as the modulus.
PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 97, 101, 9973,
    99991, 999983, 1000000007, 998244353, 1000000009, 2147483647,
]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "mixed"
    rng = random.Random(seed)

    p = rng.choice(PRIMES)

    if mode == "tiny":
        # only very small n -- the 'tempting to hardcode' regime
        t = rng.randint(1, 8)
        ns = [rng.randint(0, 7) for _ in range(t)]
    elif mode == "small":
        t = rng.randint(1, 8)
        ns = [rng.randint(0, 40) for _ in range(t)]
    elif mode == "mid":
        t = rng.randint(1, 6)
        ns = [rng.randint(0, 2000) for _ in range(t)]
    elif mode == "edge":
        # boundary values around the small/large divide
        choices = [0, 1, 2, 3, 7, 8, 12, 20, 100, 1000, 5000]
        t = rng.randint(1, 8)
        ns = [rng.choice(choices) for _ in range(t)]
    else:  # mixed
        t = rng.randint(1, 8)
        ns = []
        for _ in range(t):
            r = rng.random()
            if r < 0.3:
                ns.append(rng.randint(0, 7))
            elif r < 0.6:
                ns.append(rng.randint(0, 60))
            else:
                ns.append(rng.randint(0, 3000))

    lines = ["%d %d" % (t, p)]
    lines.append(" ".join(str(x) for x in ns))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
