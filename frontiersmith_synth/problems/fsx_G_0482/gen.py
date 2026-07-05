import sys

# testId 1..10 -> increasing number of antennas (difficulty ladder).
# Seeded solely by testId; no randomness in the instance.
LADDER = [7, 8, 9, 10, 11, 12, 13, 14, 16, 18]

def main():
    tid = int(sys.argv[1])
    idx = (tid - 1) % len(LADDER)
    n = LADDER[idx]
    sys.stdout.write("%d\n" % n)

if __name__ == "__main__":
    main()
