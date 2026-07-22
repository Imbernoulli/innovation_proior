#!/usr/bin/env python3
"""gen.py <testId> -- prints one Mirror Pond Rake Budget instance to stdout.
Deterministic: fully determined by testId (fixed table + a seeded PRNG for the
'general/random' cases). No wall-clock, no external state.
"""
import sys
import random

MAXOPS = 20000


def uniform(N, A):
    return [A] * N


def checkerboard(N, A):
    return [A if i % 2 == 0 else -A for i in range(N)]


def halfstep(N, A):
    return [A] * (N // 2) + [-A] * (N // 2)


def quarterblock(N, A, q):
    h = [0] * N
    w = N // 4
    for i in range(q * w, (q + 1) * w):
        h[i] = A
    return h


def mixed_uc(N, base, noise):
    return [base + (noise if i % 2 == 0 else -noise) for i in range(N)]


def randomcase(N, A, seed):
    rng = random.Random(seed)
    return [rng.randint(-A, A) for _ in range(N)]


# (N, T, disturbance) table -- testId 1..10, a difficulty/trap ladder.
CASES = {
    1: (8, 2, lambda: randomcase(8, 6, 2001)),
    2: (16, 2, lambda: uniform(16, 8)),
    3: (16, 2, lambda: checkerboard(16, 50)),
    4: (32, 3, lambda: uniform(32, 10)),
    5: (32, 3, lambda: halfstep(32, 11)),
    6: (32, 3, lambda: checkerboard(32, 150)),
    7: (64, 3, lambda: quarterblock(64, 40, 1)),
    8: (64, 3, lambda: mixed_uc(64, 25, 18)),
    9: (64, 3, lambda: randomcase(64, 80, 2009)),
    10: (64, 4, lambda: uniform(64, 10)),
}


def main():
    tid = int(sys.argv[1])
    key = ((tid - 1) % 10) + 1
    N, T, f = CASES[key]
    h0 = f()
    assert len(h0) == N
    out = []
    out.append(f"{N} {T} {MAXOPS}")
    out.append(" ".join(str(x) for x in h0))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
