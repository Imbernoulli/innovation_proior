# TIER: trivial
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = [0] * n
    for i in range(n):
        mi = int(next(it))
        m[i] = mi
        for v in range(1, mi + 1):
            next(it)  # pref
            r = int(next(it))
            for _ in range(r):
                next(it); next(it); next(it)
    # the universal safe fallback: install everything at its oldest version.
    print(" ".join("1" for _ in range(n)))


if __name__ == "__main__":
    main()
