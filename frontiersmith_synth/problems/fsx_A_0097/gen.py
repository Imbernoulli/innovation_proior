import sys

# Difficulty ladder for testId 1..10.
# k grows 14 -> 40; the range multiplier c cycles through TIGHT (<1, Sidon
# impossible at full size -> hard packing) and LOOSE (>=1, a full Sidon set fits
# but must still be constructed) regimes so the strategies diverge.
K_LADDER = [14, 14, 14, 22, 22, 22, 30, 30, 40, 40]
C_LADDER = [0.35, 0.60, 1.00, 0.35, 0.60, 1.00, 0.50, 1.20, 0.50, 1.20]


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
