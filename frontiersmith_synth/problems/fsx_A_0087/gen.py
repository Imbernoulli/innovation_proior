import sys

# Difficulty ladder. n = number of stages, M = promenade length (positions in [0,M]).
# M is chosen tight: M ~ 0.6 * C(n,2), so a full Sidon/Golomb ruler (all pairwise
# distances distinct) does NOT fit -> distinct-distance packing is genuinely hard.
LADDER = {
    1:  (6,   9),
    2:  (7,  13),
    3:  (8,  17),
    4:  (10, 27),
    5:  (12, 40),
    6:  (14, 55),
    7:  (16, 72),
    8:  (20, 114),
    9:  (25, 180),
    10: (32, 298),
}

def main():
    i = int(sys.argv[1])
    n, M = LADDER.get(i, LADDER[1])
    sys.stdout.write("%d %d\n" % (n, M))

if __name__ == "__main__":
    main()
