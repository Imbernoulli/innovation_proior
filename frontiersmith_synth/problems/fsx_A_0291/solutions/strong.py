# TIER: strong
# Best-of-many restarts of the greedy cap-set scaffold under different priority
# orderings (lexicographic + several deterministically-seeded random shuffles).
# Deterministic given the fixed RNG seed. Always >= the single lexicographic greedy.
import sys, random

def build(order, blocked, n):
    S = []
    Sset = set()
    F = set()
    for p in order:
        if p in blocked or p in Sset or p in F:
            continue
        for x in S:
            z = tuple((3 - ((p[i] + x[i]) % 3)) % 3 for i in range(n))
            F.add(z)
        S.append(p)
        Sset.add(p)
    return S

def main():
    tok = sys.stdin.read().split()
    idx = 0
    n = int(tok[idx]); idx += 1
    m = int(tok[idx]); idx += 1
    blocked = set()
    for _ in range(m):
        v = tuple(int(tok[idx + i]) for i in range(n)); idx += n
        blocked.add(v)
    space = 3 ** n

    def dec(e):
        d = []
        for _ in range(n):
            d.append(e % 3); e //= 3
        return tuple(d)

    allp = [dec(e) for e in range(space)]

    # restart budget shrinks as the space grows to stay well within time limits
    if n <= 5:
        R = 40
    elif n == 6:
        R = 24
    elif n == 7:
        R = 10
    else:
        R = 5

    best = build(allp, blocked, n)  # lexicographic scaffold (== greedy tier)
    rng = random.Random(987654321 + n)
    for _ in range(R):
        order = allp[:]
        rng.shuffle(order)
        cand = build(order, blocked, n)
        if len(cand) > len(best):
            best = cand

    out = [str(len(best))]
    for v in best:
        out.append(' '.join(map(str, v)))
    sys.stdout.write('\n'.join(out) + '\n')

main()
