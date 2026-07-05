# TIER: invalid
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    n = int(next(it)); k = int(next(it))
    # emit an all-zero matrix: entries are out of the {-1,+1} range -> infeasible
    print("\n".join(" ".join("0" for _ in range(n)) for _ in range(n)))

main()
