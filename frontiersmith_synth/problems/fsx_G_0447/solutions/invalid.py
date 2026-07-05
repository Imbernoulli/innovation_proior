# TIER: invalid
# Emits a well-formed but WRONG circuit: claims every target equals 0.
# The targets are non-zero polynomials, so exact-equivalence fails -> score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); m = int(next(it))
    # no instructions; output constant 0 for every target
    sys.stdout.write("0\n")
    sys.stdout.write("OUT " + " ".join(["#0"] * m) + "\n")


if __name__ == "__main__":
    main()
