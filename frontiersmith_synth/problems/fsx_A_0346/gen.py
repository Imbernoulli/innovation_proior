import sys

# Difficulty ladder for the traffic-signal-grid polarity problem.
# All sizes are ODD, so no Hadamard matrix exists at any order: the
# theoretical N^(N/2) bound is only an UNREACHABLE normalizer, never an
# attainable polynomial-time optimum. Medium scale, variant #13 of the
# integer-matrix-construction family.
SIZES = [13, 17, 21, 25, 27, 29]

def main():
    i = int(sys.argv[1])
    idx = (i - 1) % len(SIZES)
    n = SIZES[idx]
    sys.stdout.write("%d\n" % n)

if __name__ == "__main__":
    main()
