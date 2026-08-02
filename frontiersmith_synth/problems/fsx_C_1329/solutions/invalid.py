# TIER: invalid
"""Deliberately infeasible: names a template index one past the end of the
library. Must score 0.0 on every case."""
import sys


def main():
    data = sys.stdin.read().split()
    K = int(data[0])
    print("%d 150.0 10.0" % K)


if __name__ == "__main__":
    main()
