# TIER: invalid
# Emits an out-of-range action token on every step -- must be rejected by the
# checker's feasibility gate regardless of the instance's numeric parameters.
import sys


def main():
    data = sys.stdin.read().split()
    T = int(data[0])
    print(" ".join(["5"] * T))


if __name__ == "__main__":
    main()
