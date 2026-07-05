import sys

# Difficulty ladder: increasing channel size n. Small -> large.
LADDER = [60, 150, 300, 600, 1000, 1800, 3000, 4500, 6000, 8000]


def main():
    t = int(sys.argv[1])
    n = LADDER[(t - 1) % len(LADDER)]
    sys.stdout.write("%d\n" % n)


if __name__ == "__main__":
    main()
