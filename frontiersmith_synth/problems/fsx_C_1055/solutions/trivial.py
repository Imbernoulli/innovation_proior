# TIER: trivial
# Reproduces the checker's own reference construction: round-robin flux
# across all columns, oblivious to the target relief and to the shadowing
# dynamics entirely.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    L = int(next(it)); R = int(next(it)); M = int(next(it)); T = int(next(it))
    # target values not needed for round-robin
    schedule = [t % L for t in range(T)]
    print(" ".join(map(str, schedule)))


if __name__ == "__main__":
    main()
