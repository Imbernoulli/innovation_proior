# TIER: invalid
# Emits a single anchor (all-zeros).  For any n > r this leaves the far half of the cube
# uncovered, so it is NOT a covering code -> the checker must score it 0.
import sys


def main():
    data = sys.stdin.read().split()
    n, r = int(data[0]), int(data[1])
    sys.stdout.write("0" * n + "\n")


if __name__ == "__main__":
    main()
