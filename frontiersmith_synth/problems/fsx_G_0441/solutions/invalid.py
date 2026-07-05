# TIER: invalid
"""Emits an incomplete network (a single comparator) that cannot sort n>=3
inputs. The checker's zero-one verification fails -> Ratio 0.0."""
import sys


def main():
    int(sys.stdin.read().split()[0])
    sys.stdout.write("0 1\n")


if __name__ == "__main__":
    main()
