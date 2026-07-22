# TIER: trivial
# Presents the samples in the exact order they arrived in the input, using
# no palate cleansers at all -- the checker's own reference construction.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    # remaining fields (mechanism constants, V, W) are irrelevant to this tier
    print(" ".join(str(i) for i in range(1, N + 1)))


if __name__ == "__main__":
    main()
