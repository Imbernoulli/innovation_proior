# TIER: invalid
# Pretends the target molecule can simply be BOUGHT off the shelf (it never is
# purchasable in this family), so the checker must reject it with Ratio: 0.0.
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); T = int(next(it)); P = int(next(it))
    print("BUY %d 1" % T)
    print("ROOT 1")


if __name__ == "__main__":
    main()
