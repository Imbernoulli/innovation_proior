import sys

# Difficulty ladder for the "carnival ride circuit" low-discrepancy point-set problem.
# Each test fixes (d, M): place M ride-anchors in the [0,1]^d fairground.
# Purely a deterministic function of testId (no randomness in the score anywhere).
LADDER = {
    1:  (2, 8),
    2:  (2, 12),
    3:  (2, 16),
    4:  (2, 20),
    5:  (2, 24),
    6:  (2, 28),
    7:  (2, 32),
    8:  (2, 40),
    9:  (3, 16),
    10: (3, 20),
}

def main():
    i = int(sys.argv[1])
    if i not in LADDER:
        # clamp out-of-range ids into the ladder so the harness never crashes
        i = ((i - 1) % len(LADDER)) + 1
    d, M = LADDER[i]
    # Instance: first line "d M". Nothing else is needed to score star discrepancy.
    sys.stdout.write("%d %d\n" % (d, M))

if __name__ == "__main__":
    main()
