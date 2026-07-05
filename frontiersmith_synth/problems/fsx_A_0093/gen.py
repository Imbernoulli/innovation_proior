# gen.py -- prints ONE instance to stdout.
# Difficulty ladder via testId only (deterministic; no randomness).
#   larger testId  ->  more zones (harder packing) and a thinner annulus.
import sys

def main():
    t = int(sys.argv[1])
    if t < 1:
        t = 1
    N = 8 + 3 * t              # t=1 -> 11 ... t=10 -> 38   (within 10..40)
    R = 1.0
    r_in = round(0.15 + 0.02 * t, 4)   # t=1 -> 0.17 ... t=10 -> 0.35
    print("%d %s %s" % (N, repr(R), repr(r_in)))

if __name__ == "__main__":
    main()
