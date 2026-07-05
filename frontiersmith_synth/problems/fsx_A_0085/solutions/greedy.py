# TIER: greedy
# Spread habitat uniformly across the whole corridor -- the obvious anti-crowding
# move. Cuts the peak-encounter index from ~4 (half-block) to ~2.
import sys


def main():
    n = int(sys.stdin.read().split()[0])
    f = [1.0] * n
    print(" ".join("%.6f" % x for x in f))


if __name__ == "__main__":
    main()
