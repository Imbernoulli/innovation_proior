import sys

# Deterministic difficulty ladder, keyed only by testId (1..10). No RNG needed: every
# instance is a fixed (alphabet size a, window length k, cyclic-string length L) triple,
# chosen so L is an exact multiple of k and L is strictly less than k * (total number of
# necklaces of length k over an a-letter alphabet) -- so a full necklace cover is never
# reachable and the packing choice is genuinely open-ended.
#
# (a, k, L) table, increasing scale test1 -> test10:
TABLE = {
    1:  (2, 3, 9),
    2:  (3, 3, 12),
    3:  (2, 4, 16),
    4:  (4, 3, 18),
    5:  (2, 5, 20),
    6:  (3, 4, 24),
    7:  (4, 4, 28),
    8:  (3, 5, 30),
    9:  (4, 5, 30),
    10: (2, 7, 42),
}


def main():
    i = int(sys.argv[1])
    a, k, L = TABLE[i]
    sys.stdout.write("%d %d %d\n" % (a, k, L))


if __name__ == "__main__":
    main()
