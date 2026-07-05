# TIER: strong
# Seeded random-restart hill climbing that directly maximizes |A-A|
# (the number of distinct signed spacings). Each restart samples n random
# distinct positions in [0,M], then repeatedly relocates a single stage to an
# unused slot, accepting the move iff it increases |A-A|. Best over all restarts.
import sys, random

def diff_set_size(A):
    s = set()
    for a in A:
        for b in A:
            s.add(a - b)
    return len(s)

def main():
    d = sys.stdin.read().split()
    n = int(d[0]); M = int(d[1])
    rng = random.Random(20260701 + 97 * n + M)

    restarts = 8
    moves_per = 400

    best = None
    bestv = -1
    for r in range(restarts):
        if M + 1 <= n:
            A = list(range(n))
        else:
            A = rng.sample(range(M + 1), n)
        cur = diff_set_size(A)
        used = set(A)
        for _ in range(moves_per):
            idx = rng.randrange(n)
            q = rng.randrange(M + 1)
            if q in used:
                continue
            old = A[idx]
            A[idx] = q
            v = diff_set_size(A)
            if v > cur:
                cur = v
                used.discard(old)
                used.add(q)
            else:
                A[idx] = old
        if cur > bestv:
            bestv = cur
            best = A[:]

    print(n)
    print("\n".join(map(str, best)))

main()
