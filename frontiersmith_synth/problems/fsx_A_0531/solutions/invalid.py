# TIER: invalid
# Emits zero folds: the footprint stays at width N > T, so the schedule never
# reaches the target and the checker must score it 0.
import sys


def main():
    sys.stdin.read()
    print(0)


if __name__ == "__main__":
    main()
