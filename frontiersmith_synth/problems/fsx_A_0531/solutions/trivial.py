# TIER: trivial
# Accordion from the left: fold one slot at a time (k=1) until width <= T.
# This is exactly the checker's internal baseline, so it scores ~0.1.
import sys


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); T = int(next(it)); R = int(next(it)); W = int(next(it))
    folds = [1] * (N - T)
    print(len(folds))
    if folds:
        print(" ".join(str(x) for x in folds))


if __name__ == "__main__":
    main()
