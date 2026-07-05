# TIER: invalid
# Emits a decomposition that does NOT reconstruct M (a single all-ones term) ->
# feasibility gate fails -> Ratio 0.0.
import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it))
    print(1)
    print(" ".join(["1"] * n))
    print(" ".join(["1"] * m))

if __name__ == "__main__":
    main()
