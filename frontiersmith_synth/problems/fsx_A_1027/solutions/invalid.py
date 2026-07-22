# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    p = int(next(it))
    T = int(next(it))
    # Blatantly infeasible: claim far more terms than the budget T allows.
    k = T + 5
    print(k)
    for i in range(k):
        print(i, 1)


if __name__ == "__main__":
    main()
