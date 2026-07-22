# TIER: invalid
# Declares more test points than the budget T allows -> feasibility gate
# rejects it (count out of [0, T]) -> Ratio 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    D = int(toks[0])
    T = int(toks[1])
    c = T + 5
    xs = [0] * c
    print(c)
    print(" ".join(map(str, xs)))


if __name__ == "__main__":
    main()
