# TIER: trivial
"""Full commitment: build every module in one giant stage, never checking in.
This is exactly the checker's internal baseline ('always fully commit')."""
import sys


def main():
    data = sys.stdin.read().split()
    M = int(data[0])
    print(1)
    print(M)


if __name__ == "__main__":
    main()
