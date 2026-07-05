import sys

# Difficulty ladder. n = number of antennas (slots), M = largest phase slot.
# M is chosen tight (~0.55-0.7 of the minimal collision-free span for n points),
# so a fully Sidon layout (all sums AND all differences distinct) does NOT fit ->
# packing distinct sums + distinct differences is a genuine trade-off.
LADDER = {
    1:  (6,  15),
    2:  (7,  20),
    3:  (8,  25),
    4:  (10, 40),
    5:  (12, 55),
    6:  (14, 80),
    7:  (16, 110),
    8:  (18, 145),
    9:  (20, 180),
    10: (24, 280),
}

def main():
    i = int(sys.argv[1])
    n, M = LADDER.get(i, LADDER[1])
    sys.stdout.write("%d %d\n" % (n, M))

if __name__ == "__main__":
    main()
