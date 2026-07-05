# TIER: trivial
# Densest trivial deployment: consecutive offsets {0,1,...,n-1} (an arithmetic
# progression).  This exactly reproduces the checker's internal baseline, so it
# scores ~0.1.  An AP has the FEWEST distinct pairwise sums (2n-1) of any set.
import sys

def main():
    toks = sys.stdin.read().split()
    n, M = int(toks[0]), int(toks[1])
    A = list(range(n))            # 0..n-1, all <= M since M >= 4n
    sys.stdout.write(" ".join(map(str, A)) + "\n")

if __name__ == "__main__":
    main()
