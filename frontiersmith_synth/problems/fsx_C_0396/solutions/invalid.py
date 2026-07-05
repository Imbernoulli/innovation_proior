# TIER: invalid
# Emits an out-of-range grid (every cell = N, which exceeds the max phase N-1),
# which also clobbers the prefilled cells.  Must score 0.
import sys


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    w = sys.stdout.write
    for i in range(N):
        w(" ".join(str(N) for _ in range(N)) + "\n")


if __name__ == "__main__":
    main()
