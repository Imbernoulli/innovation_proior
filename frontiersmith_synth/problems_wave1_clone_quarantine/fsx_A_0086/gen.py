import sys

# Difficulty ladder: side length N grows, all even and >= 8.
# Small scale, variant #0 of the integer-matrix-construction family.
SIZES = [8, 12, 16, 20, 24, 32]

def main():
    i = int(sys.argv[1])
    idx = (i - 1) % len(SIZES)
    n = SIZES[idx]
    sys.stdout.write("%d\n" % n)

if __name__ == "__main__":
    main()
