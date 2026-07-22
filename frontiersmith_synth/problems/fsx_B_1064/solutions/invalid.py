# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    # deliberately infeasible: deposits far outside [0, D_MAX], will be rejected
    print(" ".join(str(999999) for _ in range(n)))


if __name__ == "__main__":
    main()
