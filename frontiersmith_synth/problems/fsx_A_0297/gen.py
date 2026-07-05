#!/usr/bin/env python3
"""gen.py <testId> -> one instance on stdout.

testId 1..10 is a difficulty ladder over the manifest size n. Everything is seeded by
testId only, so generation is bit-for-bit reproducible.

Instance format:
    n M k
    f_1 ... f_k
where the max weight M is sized (a tight ~1.25x) to the Mian-Chowla Sidon packing of size n,
so a perfect difference-dominant packing is feasible but the interval stays tight enough that
naive spreading collides. k reserved weights are drawn deterministically.
"""
import random
import sys

N_LADDER = [12, 20, 32, 48, 64, 90, 120, 150, 175, 200]


def sidon_max(n):
    """Max element of the greedy (Mian-Chowla) Sidon set of size n."""
    A = []
    sums = set()
    c = 0
    while len(A) < n:
        ok = True
        for a in A:
            if a + c in sums:
                ok = False
                break
        if ok:
            for a in A:
                sums.add(a + c)
            sums.add(2 * c)
            A.append(c)
        c += 1
    return A[-1]


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    if t < 1:
        t = 1
    if t > len(N_LADDER):
        t = len(N_LADDER)
    n = N_LADDER[t - 1]

    E = sidon_max(n)
    M = int(E * 1.25) + 30

    rnd = random.Random(700000 + 97 * t + n)
    k = max(1, n // 25)
    forb = set()
    while len(forb) < k:
        forb.add(rnd.randint(0, M))
    forb = sorted(forb)

    out = ["%d %d %d" % (n, M, k), " ".join(str(x) for x in forb)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
