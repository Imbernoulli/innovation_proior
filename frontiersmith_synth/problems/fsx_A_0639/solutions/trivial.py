# TIER: trivial
import sys


def main():
    d = sys.stdin.read().split()
    n = int(d[0])
    print(" ".join(str(i) for i in range(n)))


if __name__ == "__main__":
    main()
