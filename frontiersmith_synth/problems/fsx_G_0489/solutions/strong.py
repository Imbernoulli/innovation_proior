# TIER: strong
# Greedy MIN-occupancy residue elimination (delete the least-occupied class -> remove the
# fewest elements) run over several deterministic window offsets, keeping the densest set.
# Offsetting the window changes which small residues are pre-avoided, so the best offset
# yields a denser admissible tuple than the fixed [0,W] greedy. Still far from the (unknown)
# optimum, leaving headroom.
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


def greedy_min(vals, W):
    S = set(vals)
    while True:
        k = len(S)
        changed = False
        for p in primes_upto(k):
            occ = [0] * p
            for x in S:
                occ[x % p] += 1
            if all(occ):
                r = min(range(p), key=lambda r: (occ[r], r))
                S = {x for x in S if x % p != r}
                changed = True
        if not changed:
            break
    return sorted(S)


def strong(W):
    offs = {0, 1, W // 2, W}
    P = 1
    for p in primes_upto(40):
        P *= p
        offs.add(P % (W + 1))
    best = []
    for off in offs:
        cand = greedy_min(range(off, off + W + 1), W)
        if len(cand) > len(best):
            best = cand
    return best


def main():
    W = int(sys.stdin.read().split()[0])
    S = strong(W)
    print(" ".join(map(str, S)))


main()
