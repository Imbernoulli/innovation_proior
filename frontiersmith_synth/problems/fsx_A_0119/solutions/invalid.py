# TIER: invalid
# Emits points far outside [0,1]^2 -> feasibility gate must score 0.
import sys

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    dim = int(next(it)); M = int(next(it)); K = int(next(it))
    print("\n".join("5.000000 5.000000" for _ in range(M)))

if __name__ == "__main__":
    main()
