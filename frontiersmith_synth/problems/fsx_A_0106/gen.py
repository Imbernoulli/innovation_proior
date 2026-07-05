import sys

# integer-matrix-construction family, LARGE scale, variant #2.
# Difficulty ladder: side length N grows. All chosen sizes admit a Hadamard
# matrix (either a power of two -> Sylvester, or N-1 an odd prime == 3 mod 4
# -> Paley type I), so a structured optimum exists and the score never trivially
# caps. Sizes are deliberately LARGE (24..64) versus the small-scale sibling.
SIZES = [24, 32, 44, 48, 60, 64]

def main():
    i = int(sys.argv[1])
    n = SIZES[(i - 1) % len(SIZES)]
    sys.stdout.write("%d\n" % n)

if __name__ == "__main__":
    main()
