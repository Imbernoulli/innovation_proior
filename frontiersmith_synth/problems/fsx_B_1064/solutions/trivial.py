# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    V = int(next(it))
    D_MAX = int(next(it))
    Fpm = int(next(it))
    for _ in range(n):
        for _ in range(7):
            next(it)
    # (ignore edges entirely)

    U = 90
    print(" ".join(str(U) for _ in range(n)))


if __name__ == "__main__":
    main()
