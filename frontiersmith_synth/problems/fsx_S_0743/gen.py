import sys, random

# gen.py <testId> -- prints ONE "symmetric emblem" instance to stdout.
#
# Builds A_TRUE generic representative cells (never on the diagonal / anti-diagonal,
# so each has a full 8-cell D4 orbit about the NxN grid center) with pairwise-DISJOINT
# orbits, then emits an ANCHOR LIST that, for each representative, includes a random
# SUBSET of its own 8-cell orbit (size in [redund_lo, redund_hi]) instead of a single
# point. Low-testId cases use subset size 1 (no redundancy: every anchor is a genuinely
# distinct orbit, so naive "one macro over all given anchors" already IS the dedup-minimal
# choice). Higher-testId cases plant heavy redundancy (subset size up to 8): many given
# anchors are just OTHER IMAGES of an orbit already represented elsewhere in the list,
# shuffled in so proximity gives no hint. A solver that treats every given anchor as an
# independent seed point (instead of testing anchors against each other under the 8
# symmetries) pays for the same orbit many times over -- this is the planted trap.

# (N, A_true, redund_lo, redund_hi)
SPECS = {
    1:  (24, 3,  1, 1),
    2:  (24, 5,  1, 1),
    3:  (28, 8,  1, 2),
    4:  (28, 10, 1, 3),
    5:  (32, 14, 3, 6),
    6:  (32, 16, 3, 7),
    7:  (36, 20, 4, 8),
    8:  (40, 25, 4, 8),
    9:  (44, 30, 5, 8),
    10: (48, 40, 5, 8),
}


def apply_t(t, x, y, N):
    if t == 0: return (x, y)
    if t == 1: return (N - 1 - y, x)
    if t == 2: return (N - 1 - x, N - 1 - y)
    if t == 3: return (y, N - 1 - x)
    if t == 4: return (N - 1 - x, y)
    if t == 5: return (x, N - 1 - y)
    if t == 6: return (y, x)
    if t == 7: return (N - 1 - y, N - 1 - x)
    raise ValueError("bad transform id")


def main():
    tid = int(sys.argv[1])
    N, A_true, r_lo, r_hi = SPECS[tid]
    rng = random.Random(700000 + 37 * tid)

    used = set()
    reps = []
    guard = 0
    while len(reps) < A_true:
        guard += 1
        if guard > 2_000_000:
            raise RuntimeError("sampler stuck")
        x = rng.randrange(N)
        y = rng.randrange(N)
        if x == y or x + y == N - 1:
            continue  # would lie on a symmetry axis -> orbit smaller than 8
        orbit = [apply_t(t, x, y, N) for t in range(8)]
        if len(set(orbit)) != 8:
            continue
        if used.intersection(orbit):
            continue
        used.update(orbit)
        reps.append((x, y))

    anchors = []
    for (x, y) in reps:
        orbit = [apply_t(t, x, y, N) for t in range(8)]
        k = rng.randint(r_lo, min(r_hi, 8))
        idxs = rng.sample(range(8), k)
        for i in idxs:
            anchors.append(orbit[i])
    rng.shuffle(anchors)

    out = ["%d" % N, "%d" % len(anchors)]
    for (x, y) in anchors:
        out.append("%d %d" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
