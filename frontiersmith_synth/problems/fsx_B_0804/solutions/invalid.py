# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    P = int(next(it))
    next(it)
    next(it)  # cn, cd
    n = 0
    for _p in range(P):
        k = int(next(it))
        for _j in range(k):
            next(it)
        n += k
    # negative application times: always infeasible regardless of instance shape
    sys.stdout.write("\n".join(["-1"] * n) + "\n")


if __name__ == "__main__":
    main()
