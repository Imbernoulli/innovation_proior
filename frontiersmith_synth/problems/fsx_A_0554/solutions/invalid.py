# TIER: invalid
# Emits vectors that are NOT in the lattice (huge generic integers) -> feasibility fails -> 0.
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0]); k = int(toks[2])
    out = []
    for i in range(k):
        out.append(" ".join(str(1000000007 + i + j) for j in range(n)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
