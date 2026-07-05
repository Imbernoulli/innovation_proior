# TIER: greedy
# Naive greedy residue elimination on the window [0, W]: whenever a prime's residues are
# fully covered, delete the MOST-occupied class (a poor choice that removes many elements).
# Valid and admissible, but noticeably smaller than a good sieve.
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


def greedy_max(W):
    S = set(range(W + 1))
    while True:
        k = len(S)
        changed = False
        for p in primes_upto(k):
            occ = [0] * p
            for x in S:
                occ[x % p] += 1
            if all(occ):
                r = max(range(p), key=lambda r: (occ[r], -r))
                S = {x for x in S if x % p != r}
                changed = True
        if not changed:
            break
    return sorted(S)


def main():
    W = int(sys.stdin.read().split()[0])
    S = greedy_max(W)
    print(" ".join(map(str, S)))


main()
