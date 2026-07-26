# TIER: invalid
# Emits N copies of slot 0 -> not a permutation -> infeasible -> must score 0.
import sys

def main():
    d = sys.stdin.read().split()
    N = int(d[0])
    print(N)
    print("\n".join(["0"] * N))

main()
