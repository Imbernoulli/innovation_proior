# TIER: greedy
# The full {0,1}^n hypercube (never use phase 2). No three distinct {0,1}-vectors
# can be all-distinct at any intersection, so this is a valid (maximal) cap of
# size 2^n -- a real improvement over the trivial baseline but far from maximum.
import sys, itertools


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    out = ["".join(bits) for bits in itertools.product("01", repeat=n)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
