# TIER: trivial
# Baseline recipe: minor-collect every single step. Always safe (never lets
# the young generation build up), but pays a fixed per-step overhead whether
# or not there is anything worth collecting.
import sys


def main():
    data = sys.stdin.read().split()
    T = int(data[0])
    print(" ".join(["1"] * T))


if __name__ == "__main__":
    main()
