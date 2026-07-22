import sys, random

# ---------------------------------------------------------------------------
# Bookbinder's pocket fold -- instance generator.
#
# An instance is a 1-D map: N unit cells in a row with per-cell thickness h[.].
# Between cell c and c+1 sits crease c (0..N-2). Some creases are "reinforced"
# (a stitched spine) and cost W stress each time a fold bends them. The map must
# be folded down to footprint width <= T. See statement.md for the fold rules.
#
# Difficulty ladder (testId 1..10): small/uniform -> large/adversarial. Several
# cases plant a THICK CENTRAL BLOCK and a reinforced central crease so that the
# obvious "halve it each time" schedule folds the heavy middle early (and bends
# the spine), while a thin-margins-first schedule keeps carried thickness low
# and routes around the spine.
# ---------------------------------------------------------------------------


def emit(N, T, reinforced, h):
    W = 0
    # W scales with a typical carried thickness so the spine matters but does
    # not dwarf the layer-stress term.
    if reinforced:
        W = max(4, sum(h) // 3)
    reinforced = sorted(set(reinforced))
    out = []
    out.append("%d %d %d %d" % (N, T, len(reinforced), W))
    out.append(" ".join(str(x) for x in h))
    out.append(" ".join(str(x) for x in reinforced))
    sys.stdout.write("\n".join(out) + "\n")


def build(i, rng):
    if i == 1:
        # tiny uniform warm-up
        N = 6
        h = [1] * N
        return N, 1, [], h
    if i == 2:
        N = 8
        h = [rng.randint(1, 2) for _ in range(N)]
        return N, 1, [], h
    if i == 3:
        # TRAP: thick central block, no spine -- halving folds the heavy middle first
        N = 10
        h = [1] * N
        for c in (4, 5):
            h[c] = 7
        return N, 2, [], h
    if i == 4:
        # TRAP: thick center + reinforced central crease
        N = 12
        h = [1] * N
        for c in (5, 6):
            h[c] = 6
        return N, 2, [5], h
    if i == 5:
        # two thick zones + two reinforced creases; T leaves room to route around
        N = 12
        h = [1] * N
        for c in (2, 3, 8, 9):
            h[c] = 5
        return N, 3, [3, 8], h
    if i == 6:
        # TRAP: big central slab, reinforced middle
        N = 14
        h = [1] * N
        for c in (6, 7):
            h[c] = 9
        return N, 2, [6], h
    if i == 7:
        # heavy margins, reinforced creases near the ends; halving ignores this
        N = 14
        h = [rng.randint(1, 2) for _ in range(N)]
        for c in (0, 1, N - 2, N - 1):
            h[c] = 6
        return N, 4, [2, 10], h
    if i == 8:
        # TRAP: large central block, larger map
        N = 16
        h = [1] * N
        for c in (6, 7, 8, 9):
            h[c] = 6
        return N, 2, [7], h
    if i == 9:
        # adversarial: alternating thick pockets + several reinforced creases
        N = 16
        h = [rng.randint(1, 2) for _ in range(N)]
        for c in (3, 4, 11, 12):
            h[c] = 7
        return N, 4, [4, 8, 11], h
    # i == 10: largest adversarial mix
    N = 18
    h = [rng.randint(1, 2) for _ in range(N)]
    for c in (7, 8, 9, 10):
        h[c] = 8
    for c in (0, 17):
        h[c] = 5
    return N, 3, [5, 12], h


def main():
    i = int(sys.argv[1])
    rng = random.Random(90531 + 17 * i)
    N, T, reinforced, h = build(i, rng)
    emit(N, T, reinforced, h)


if __name__ == "__main__":
    main()
