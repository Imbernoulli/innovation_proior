# TIER: invalid
# Emits n copies of the same slot -> not pairwise distinct -> checker rejects -> Ratio 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0]); M = int(toks[1])
    A = [0] * n                 # all identical: violates distinctness
    sys.stdout.write(" ".join(map(str, A)) + "\n")


if __name__ == "__main__":
    main()
