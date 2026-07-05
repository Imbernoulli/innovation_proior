# TIER: strong
# Full-budget maximizer for |A+A| inside [0, M]:
#   (1) seed with a greedy Sidon set (all sums distinct),
#   (2) AUGMENT to the full budget n by repeatedly adding the offset that
#       creates the most NEW distinct signatures (uses every element of the
#       budget, unlike the pure-Sidon greedy which stops early),
#   (3) seeded local search: try replacing a member with an unused offset,
#       keep it only if |A+A| strictly increases.
# Deterministic (seed derived from the instance).
import sys, random

def sumset(A):
    s = set()
    for i in range(len(A)):
        ai = A[i]
        for j in range(i, len(A)):
            s.add(ai + A[j])
    return s

def main():
    toks = sys.stdin.read().split()
    n, M = int(toks[0]), int(toks[1])
    rng = random.Random(9173 * n + 31 * M + 7)

    # (1) greedy Sidon seed
    A = []
    sums = set()
    x = 0
    while x <= M and len(A) < n:
        new = set()
        ok = True
        for a in A:
            s = a + x
            if s in sums or s in new:
                ok = False
                break
            new.add(s)
        if ok:
            s2 = 2 * x
            if s2 in sums or s2 in new:
                ok = False
            else:
                new.add(s2)
        if ok:
            A.append(x)
            sums |= new
        x += 1

    present = set(A)
    cur = sumset(A)

    # (2) augment to budget n, each step maximizing new distinct sums
    while len(A) < n:
        best_x, best_gain = None, -1
        for x in range(0, M + 1):
            if x in present:
                continue
            cand = set(a + x for a in A)
            cand.add(2 * x)
            gain = len(cand - cur)
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x is None:
            break
        A.append(best_x)
        present.add(best_x)
        cur = sumset(A)

    # (3) seeded local search: swap a member for an unused value if it helps
    best_size = len(cur)
    iters = 1500
    for _ in range(iters):
        if len(A) == 0:
            break
        i = rng.randrange(len(A))
        x = rng.randrange(0, M + 1)
        if x in present:
            continue
        old = A[i]
        A[i] = x
        ns = sumset(A)
        if len(ns) > best_size:
            best_size = len(ns)
            present.discard(old)
            present.add(x)
        else:
            A[i] = old  # revert

    if not A:
        A = [0]
    sys.stdout.write(" ".join(map(str, A)) + "\n")

if __name__ == "__main__":
    main()
