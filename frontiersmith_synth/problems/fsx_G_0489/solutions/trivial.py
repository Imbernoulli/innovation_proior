# TIER: trivial
# Reproduces the grader's coarse double-residue-sieve reference construction -> ~0.1.
import sys


def primes_upto(n):
    if n < 2:
        return []
    s = bytearray([1]) * (n + 1)
    s[0] = s[1] = 0
    i = 2
    while i * i <= n:
        if s[i]:
            s[i * i::i] = bytearray(len(s[i * i::i]))
        i += 1
    return [i for i in range(2, n + 1) if s[i]]


def double_sieve(W):
    alive = bytearray([1]) * (W + 1)
    alive[0] = 0
    count = W
    for p in primes_upto(W):
        rem = (0,) if p == 2 else (0, 1)
        for r in rem:
            start = r if r != 0 else p
            for m in range(start, W + 1, p):
                if alive[m]:
                    alive[m] = 0
                    count -= 1
        if p >= count:
            break
    return [i for i in range(W + 1) if alive[i]]


def main():
    W = int(sys.stdin.read().split()[0])
    S = double_sieve(W)
    print(" ".join(map(str, S)))


main()
