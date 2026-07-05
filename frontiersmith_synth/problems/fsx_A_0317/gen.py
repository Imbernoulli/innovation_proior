import sys

# ----------------------------------------------------------------------------
# Difficulty ladder for the "Waggle-Resonance Apiary" sumset-maximization task.
#   n = number of hive boxes to place at DISTINCT integer posts in [0, M].
#   M = length of the orchard rail (posts 0..M).
#
# M is deliberately kept in the "Sidon-packing" regime  M ~ 0.58 * C(n,2):
#   * A *perfect* B_2 (Sidon) set -- all C(n,2)+n pairwise sums distinct --
#     needs a rail of length ~ n^2, so it does NOT fit here. The theoretical
#     maximum |A+A| = n(n+1)/2 is therefore an UNREACHABLE normalizer.
#   * The window itself can hold 2M+1 distinct sums, which is strictly larger
#     than any construction reaches, so window-saturation is not the binding
#     constraint either.
#   => The true optimum is genuinely unknown -> open-ended search.
#
# seed is a deterministic tag the solver MAY use to seed local search. It does
# NOT influence scoring.
# ----------------------------------------------------------------------------
LADDER = {
    1:  (6,   10),
    2:  (8,   18),
    3:  (10,  28),
    4:  (12,  40),
    5:  (14,  55),
    6:  (16,  72),
    7:  (20, 110),
    8:  (24, 160),
    9:  (28, 220),
    10: (32, 290),
}


def main():
    i = int(sys.argv[1])
    n, M = LADDER.get(i, LADDER[1])
    # deterministic seed derived from the test id only (no wall-clock / rng)
    seed = 1000003 * i + 7
    sys.stdout.write("%d %d %d\n" % (n, M, seed))


if __name__ == "__main__":
    main()
