# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    # Emit an infeasible artifact on purpose: claim every song plus a few indices that
    # don't exist. Fails both the "distinct/in-range" and "duration budget" checks.
    m = N + 5
    order = list(range(m))
    print(m)
    print(" ".join(map(str, order)))


if __name__ == "__main__":
    main()
