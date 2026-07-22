# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        return
    N = int(data[0])
    print(" ".join(str(i) for i in range(1, N + 1)))


if __name__ == "__main__":
    main()
