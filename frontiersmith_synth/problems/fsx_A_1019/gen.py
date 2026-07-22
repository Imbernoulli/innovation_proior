#!/usr/bin/env python3
"""gen.py <testId> -- prints one instance of the Relief Bazaar endowment problem.

Instance = N households in K trade-communities (comm_i), G relief goods with a
fixed per-good satiation cap, a fixed shipment S_g of each good, and a per-
household, per-good FLAT per-unit weight w_{i,g} (utility from good g is
w_{i,g} * min(x_{i,g}, cap_g) -- linear up to the satiation cap, then worthless).

Every test plants a "basin" trap: community c collectively values good c the
most (many moderate-weight members), but a handful of "champion" households
living in the NEXT community (c+1 mod K) have an individually enormous weight
for good c. A myopic argmax-by-marginal-utility allocator (the obvious first
approach) fully saturates those few champions before anyone else -- and since
the champions can never trade good c back across the community boundary (the
fixed trading protocol only links households inside the same community), and
because a saturate-then-move-on allocator also concentrates community c's own
leftover share onto only its top few members, a good fraction of community c's
own households end up with ZERO of the good their community was supposed to
receive -- households holding nothing cannot trade at all, so the friction-
limited replay can never repair it either.
"""
import sys
import random

PARAMS = {
    1:  dict(N=6,  K=2, cap=4, eps=1, R=12, champ=1),
    2:  dict(N=8,  K=2, cap=5, eps=1, R=15, champ=1),
    3:  dict(N=10, K=2, cap=5, eps=1, R=15, champ=1),
    4:  dict(N=12, K=3, cap=5, eps=1, R=18, champ=1),
    5:  dict(N=15, K=3, cap=6, eps=1, R=18, champ=2),
    6:  dict(N=18, K=3, cap=6, eps=1, R=21, champ=2),
    7:  dict(N=20, K=4, cap=6, eps=1, R=22, champ=2),
    8:  dict(N=24, K=4, cap=7, eps=2, R=24, champ=3),
    9:  dict(N=30, K=5, cap=7, eps=2, R=27, champ=3),
    10: dict(N=36, K=6, cap=8, eps=2, R=30, champ=3),
}

MOD_LO, MOD_HI = 8, 14      # moderate ("own community") weight range
HIGH_LO, HIGH_HI = 45, 61   # champion weight range (strictly above moderate)
LOW_LO, LOW_HI = 0, 4       # indifferent weight range (strictly below moderate)
SUPPLY_FRAC = 0.85          # shipment size relative to the owning community's capacity


def build(testId):
    p = PARAMS[testId]
    N, K, cap, eps, R, champ = p['N'], p['K'], p['cap'], p['eps'], p['R'], p['champ']
    G = K
    rng = random.Random(1000003 * testId + 17)

    base, rem = N // K, N % K
    sizes = [base + (1 if c < rem else 0) for c in range(K)]
    comm_of, local_rank = [], []
    for c, sz in enumerate(sizes):
        comm_of += [c] * sz
        local_rank += list(range(sz))

    caps = [cap] * G

    W = [[0] * G for _ in range(N)]
    for i in range(N):
        c = comm_of[i]
        lr = local_rank[i]
        for g in range(G):
            if g == c:
                w = rng.randrange(MOD_LO, MOD_HI)
            elif g == (c - 1) % K:
                w = rng.randrange(HIGH_LO, HIGH_HI) if lr < champ else rng.randrange(LOW_LO, LOW_HI)
            else:
                w = rng.randrange(LOW_LO, LOW_HI)
            W[i][g] = w

    S = []
    for g in range(G):
        mg = sizes[g]
        Sg = max(mg, int(SUPPLY_FRAC * mg * cap))
        S.append(Sg)

    lines = [f"{N} {G} {K} {R} {eps}",
             " ".join(map(str, caps)),
             " ".join(map(str, S))]
    for i in range(N):
        lines.append(f"{comm_of[i]} " + " ".join(map(str, W[i])))
    return "\n".join(lines)


def main():
    testId = int(sys.argv[1])
    sys.stdout.write(build(testId) + "\n")


if __name__ == "__main__":
    main()
