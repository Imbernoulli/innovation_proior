import sys

# Difficulty ladder for testId 1..10.
# k grows 16 -> 48; the corridor multiplier c cycles through TIGHT (<1, a full
# Sidon layout of size k cannot fit -> hard packing) and LOOSE (>=1, a full
# Sidon layout fits but must still be constructed) regimes, so the strategies
# diverge across the ladder.
K_LADDER = [16, 16, 16, 24, 24, 24, 32, 32, 48, 48]
C_LADDER = [0.35, 0.60, 1.10, 0.35, 0.60, 1.10, 0.50, 1.30, 0.50, 1.30]


def main():
    i = int(sys.argv[1])
    idx = (i - 1) % len(K_LADDER)
    k = K_LADDER[idx]
    c = C_LADDER[idx]
    V = int(c * k * k)
    if V < k:
        V = k
    if V > 2 * k * k:
        V = 2 * k * k
    sys.stdout.write("%d %d\n" % (k, V))


if __name__ == "__main__":
    main()
