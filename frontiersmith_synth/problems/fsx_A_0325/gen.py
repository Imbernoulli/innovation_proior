import sys

# Difficulty ladder for the "peak-resonance" autocorrelation problem.
# testId 1..10 : arena length n grows 40 -> 220 (large search space),
# with a fixed integer intensity cap M. No randomness is needed -- the
# instance is fully determined by the difficulty index (seeded via testId).

def main():
    i = int(sys.argv[1])
    if i < 1:
        i = 1
    n = 20 * (i + 1)      # i=1 -> 40, ..., i=10 -> 220 (all even)
    M = 1000              # each time-slot intensity is an integer in [0, M]
    sys.stdout.write("%d %d\n" % (n, M))

if __name__ == "__main__":
    main()
