import sys, random

# ---------------------------------------------------------------------------
# sandpile-profile-sculptor  (format C, minimize)
# Emit ONE instance for testId 1..10 (deterministic; seeded by testId only).
#
#   line 1:  N K S L G
#   line 2:  t[0] ... t[N-1]        (target height profile, non-negative ints)
#
# The target is a COMB: tall teeth (height ~H) alternating with deep notches
# (height 1).  Because a tooth of height H forces its notch neighbour up to at
# least H-S (angle of repose), and H-1 > S, EVERY notch is unavoidably flooded
# once its teeth are built -- a large, permanent overshoot volume the score can
# never erase.  That overshoot is charged (averaged) over every stage it exists,
# so raising the teeth early (the obvious max-deficit recipe) floods the notches
# for the whole schedule, while the insight is to hold the pile under target and
# raise the teeth only in the final stages, minimising the time the notches sit
# above target.
# ---------------------------------------------------------------------------

LADDER = [
    # N   S  L   H   K   rbud
    ( 9,  1, 1,  6,   9, 220),
    (11,  1, 1,  6,  10, 220),
    (13,  2, 1,  7,  11, 220),
    (15,  1, 1,  6,  12, 218),
    (18,  2, 1,  8,  13, 222),
    (22,  1, 1,  7,  15, 220),
    (26,  2, 1,  8,  17, 222),
    (30,  1, 1,  7,  19, 220),
    (34,  2, 1,  9,  21, 222),
    (40,  1, 1,  8,  25, 220),
]


def build_target(N, S, H, rng):
    t = [1] * N
    # teeth at odd interior indices; notches (height 1) between them
    for x in range(1, N - 1):
        if x % 2 == 1:
            t[x] = H + rng.randint(0, 2)     # asymmetric tooth heights
    # guarantee the wall exceeds the slope limit everywhere
    for x in range(1, N - 1):
        if x % 2 == 1 and t[x] - 1 <= S:
            t[x] = S + 2
    return t


def main():
    i = int(sys.argv[1])
    N, S, L, H, K, rbud = LADDER[i - 1]
    rng = random.Random(45080 + 17 * i)

    t = build_target(N, S, H, rng)

    tot = sum(t)
    # Grains are ample enough to raise every tooth, but the per-stage cap G means
    # building spans a large fraction of the schedule, so a solver cannot defer
    # ALL the flooding to the very last stage -- ordering is what pays off.
    G = (rbud * tot) // (100 * K)
    G = max(G, S + 2)

    out = ["%d %d %d %d %d" % (N, K, S, L, G),
           " ".join(str(x) for x in t)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
