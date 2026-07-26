# TIER: invalid
# Emits a single move that overflows a final track's hard capacity (moves L+1
# cars straight from the working track onto train 0's final track, whose cap
# is exactly L) -> capacity violation -> must score 0.
import sys


def main():
    data = sys.stdin.read().split()
    N, T, L, Y = (int(x) for x in data[:4])
    print(1)
    print("1 0.0 0 %d %d" % (Y + 1, L + 1))


if __name__ == "__main__":
    main()
