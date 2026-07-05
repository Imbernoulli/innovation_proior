# TIER: invalid
# Emits a dense contiguous block {0,1,2,...,60} which is riddled with 3-term
# arithmetic progressions (e.g. 0,1,2). Must score exactly 0.
import sys


def main():
    p = int(sys.stdin.read().split()[0])
    k = min(60, p - 1)
    print(" ".join(str(i) for i in range(k)))


if __name__ == "__main__":
    main()
