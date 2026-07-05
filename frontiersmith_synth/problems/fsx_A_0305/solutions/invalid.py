# TIER: invalid
# Emits an infeasible profile: all-zero total mass (sum f = 0), which the checker
# must reject with Ratio 0.0.
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    print(" ".join(["0.0"] * n))


if __name__ == "__main__":
    main()
