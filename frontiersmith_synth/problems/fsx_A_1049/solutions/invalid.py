# TIER: invalid
# Emits a well-shaped but garbage grid (non-binary characters) -- must
# score exactly 0.
import sys


def main():
    data = sys.stdin.read().split("\n")
    head = data[0].split()
    k = int(head[1])
    W = 5 * k
    for _ in range(7):
        print("9" * W)


if __name__ == "__main__":
    main()
