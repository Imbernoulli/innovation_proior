# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0]) if data else 0
    # every token is wildly out of any package's version range -> rejected
    # by the checker's strict bounds check.
    print(" ".join("1000000" for _ in range(max(n, 1))))


if __name__ == "__main__":
    main()
