# TIER: invalid
# Emits negative densities -> violates the feasibility gate -> must score 0.
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    print(" ".join("-1.0" for _ in range(n)))


if __name__ == "__main__":
    main()
