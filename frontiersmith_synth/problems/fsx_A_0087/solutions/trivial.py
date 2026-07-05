# TIER: trivial
# Arithmetic progression 0,1,...,n-1. This reproduces the checker's internal
# baseline (|A-A| = 2n-1) and therefore scores ~0.1.
import sys

def main():
    d = sys.stdin.read().split()
    n = int(d[0]); M = int(d[1])
    pos = list(range(n))  # fits: M >= n-1 always
    print(n)
    print("\n".join(map(str, pos)))

main()
