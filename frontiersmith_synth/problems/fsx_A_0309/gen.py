import sys

# Difficulty ladder for the watchtower-placement instances.
# Each test fixes (d, M): d = dimension of the reserve, M = number of towers.
# testId 1..6 are the flat 2-D reserve with growing tower counts;
# testId 7..10 add the elevation band (3-D) reserve.
TESTS = [
    (2, 8),
    (2, 12),
    (2, 16),
    (2, 20),
    (2, 24),
    (2, 30),
    (3, 10),
    (3, 14),
    (3, 18),
    (3, 22),
]

def main():
    i = int(sys.argv[1])
    if i < 1:
        i = 1
    if i > len(TESTS):
        i = len(TESTS)
    d, M = TESTS[i - 1]
    sys.stdout.write("%d %d\n" % (d, M))

if __name__ == "__main__":
    main()
