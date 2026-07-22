# TIER: invalid
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[0]) if data else 1
    # Not a permutation: repeats vertex 1 for every slot.
    print(" ".join("1" for _ in range(N)))


if __name__ == "__main__":
    main()
