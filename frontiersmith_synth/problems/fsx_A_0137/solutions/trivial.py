# TIER: trivial
# Arithmetic-progression cordon 0,1,...,n-1. Its sum set is the contiguous block
# [0, 2n-2], so it reproduces the checker's internal baseline reach = 2n-2 -> ~0.1.
import sys

def main():
    data = sys.stdin.read().split()
    n, M = int(data[0]), int(data[1])
    A = list(range(n))
    out = [str(len(A))] + [str(x) for x in A]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
