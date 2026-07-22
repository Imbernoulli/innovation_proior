# TIER: invalid
# Deliberately infeasible: pours a wildly-oversized volume of feedstock 1 into tank 1
# (blows past availability) and also feeds tank 1 with 3 distinct feedstocks directly
# (violates the <=2-distinct-feedstock manifold rule). Any single one of these is enough
# to force Ratio: 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    F = int(toks[0]) if toks else 1
    M = int(toks[1]) if len(toks) > 1 else 1
    F = max(F, 1)
    M = max(M, 1)
    print("POUR 1 1 1000000000")
    print("POUR 1 %d 5" % (min(2, F)))
    print("POUR 1 %d 5" % (min(3, F)))


if __name__ == "__main__":
    main()
