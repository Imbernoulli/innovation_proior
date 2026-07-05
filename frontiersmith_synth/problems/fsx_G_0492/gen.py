import sys

# Difficulty ladder: the S-box width n grows small -> large, then a second
# sweep so each width appears twice (10 cases). The instance is just n; the
# challenge is to *construct* a high-nonlinearity n-bit permutation.
LADDER = [4, 5, 6, 7, 8, 4, 5, 6, 7, 8]

def main():
    i = int(sys.argv[1])
    n = LADDER[(i - 1) % len(LADDER)]
    sys.stdout.write("%d\n" % n)

if __name__ == "__main__":
    main()
