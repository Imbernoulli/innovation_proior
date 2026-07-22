# TIER: invalid
# Deliberately infeasible: claims to cover T with a single chain that in fact
# only lights its own anchor village (r=1, L=1), leaving the rest of T dark.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    p = int(next(it)); alpha = int(next(it))
    m = int(next(it))
    T = [int(next(it)) for _ in range(m)]

    print(1)
    print("%d 1 1" % T[0])


if __name__ == "__main__":
    main()
