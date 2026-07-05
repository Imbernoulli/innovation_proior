import sys

# Difficulty ladder for the orbital debris-cleanup additive-basis problem.
#   n = sweeper-depot budget (max distinct phase slots you may occupy)
#   M = highest phase slot index (depots live in [0, M])
# M is kept at ~4n: loose enough that a spread-out additive basis genuinely
# helps, tight enough that even the best basis reaches only ~2-3.5x the
# arithmetic-progression baseline -> the scored range stays open-ended (no
# construction saturates the cap; the exact extremal reach is unknown).
LADDER = {
    1:  (6,    24),
    2:  (10,   40),
    3:  (16,   64),
    4:  (25,  100),
    5:  (40,  160),
    6:  (60,  240),
    7:  (90,  360),
    8:  (130, 520),
    9:  (170, 680),
    10: (200, 800),
}

def main():
    i = int(sys.argv[1])
    n, M = LADDER.get(i, LADDER[1])
    sys.stdout.write("%d %d\n" % (n, M))

if __name__ == "__main__":
    main()
