# TIER: invalid
# Emits a full contiguous block [1..n], which is riddled with 3-term arithmetic
# progressions (1,2,3 ; 2,3,4 ; ...). The checker must reject it -> Ratio 0.0.
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    print(" ".join(str(i) for i in range(1, n + 1)))


if __name__ == "__main__":
    main()
