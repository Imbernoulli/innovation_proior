# TIER: trivial
# Assign the k drones consecutive offsets 0,1,...,k-1 (an arithmetic progression).
# |A+A| = 2k-1 -- exactly reproduces the checker's internal baseline -> score ~0.1.
import sys


def main():
    tok = sys.stdin.read().split()
    k, M = int(tok[0]), int(tok[1])
    A = list(range(k))
    out = [str(len(A))] + [str(x) for x in A]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
