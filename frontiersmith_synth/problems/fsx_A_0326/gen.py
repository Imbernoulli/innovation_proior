import sys

def main():
    # Difficulty ladder: testId 1..10 -> odd N = 15,17,...,33 (all odd => no
    # Hadamard order exists, no closed-form optimum). Instance is fully
    # determined by testId (seeded via testId only), no wall-time/randomness.
    t = int(sys.argv[1])
    N = 13 + 2 * t
    if N % 2 == 0:
        N += 1
    sys.stdout.write("%d\n" % N)

if __name__ == "__main__":
    main()
