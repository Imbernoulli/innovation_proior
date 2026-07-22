# TIER: invalid
# Presses every cell onto groove 0 -- always violates the groove-capacity
# limit (N is always far larger than cap), so the checker must score 0.
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    sys.stdout.write(" ".join("0" for _ in range(N)) + "\n")


if __name__ == "__main__":
    main()
