import sys, random

# ---------------------------------------------------------------------------
# dfa-transition-tour-cover  (format C, MINIMIZE total test-suite length)
#   `python3 gen.py <testId>` prints ONE instance to stdout.
#   Deterministic in testId only.
#
# Instance:
#   line 1:  n k s0
#   line 2:  k alphabet symbols (single lowercase letters)
#   next n lines: k integers -- row i = target state for each symbol from state i
#
# Construction: a canonical backbone directed n-cycle on symbol #0 guarantees
# strong connectivity (every state reachable from s0, s0 reachable from every
# state). Every other symbol routes to a uniformly random state, which creates
# realistic in/out-degree imbalance (some states become "hubs") -- this is what
# forces genuine Chinese-postman rebalancing rather than a plain Euler walk.
#
# The canonical states are then RELABELLED by a permutation before being
# printed. For testId==1 the permutation is the identity (a small, easy sanity
# case). For testId>=2 the permutation is a full REVERSAL of the state ids.
# Reversal turns "walk to the next transition in state-id order" (the natural
# first-instinct greedy) into a near-worst-case traversal of the one-way
# backbone cycle, while a graph-distance-aware (Chinese-postman) solver is
# completely unaffected by how the states happen to be labelled -- this is the
# planted trap.
# ---------------------------------------------------------------------------

# (n, k) ladder: small sanity case first, then growing/adversarial instances.
SIZE_LADDER = {
    1: (5, 2), 2: (6, 2), 3: (7, 2), 4: (8, 2), 5: (9, 2),
    6: (10, 3), 7: (11, 2), 8: (12, 3), 9: (13, 2), 10: (15, 3),
}


def main():
    t = int(sys.argv[1])
    t = max(1, min(10, t))
    rng = random.Random(20000 + 97 * t)

    n, k = SIZE_LADDER[t]
    symbols = "abc"[:k]

    # canonical transition table (state ids 0..n-1, backbone cycle on symbol 0)
    canon = [[0] * k for _ in range(n)]
    for i in range(n):
        canon[i][0] = (i + 1) % n
    for i in range(n):
        for s in range(1, k):
            canon[i][s] = rng.randrange(n)

    if t == 1:
        perm = list(range(n))
    else:
        perm = [n - 1 - i for i in range(n)]

    ext = [[0] * k for _ in range(n)]
    for i in range(n):
        for s in range(k):
            ext[perm[i]][s] = perm[canon[i][s]]
    s0 = perm[0]

    out = []
    out.append("%d %d %d" % (n, k, s0))
    out.append(" ".join(symbols))
    for i in range(n):
        out.append(" ".join(str(x) for x in ext[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
