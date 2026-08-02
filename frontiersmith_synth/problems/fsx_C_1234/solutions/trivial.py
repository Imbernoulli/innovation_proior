# TIER: trivial
"""Reproduces the checker's own baseline construction exactly: no protocol at
all on any lock (plain strict-priority scheduling, uncorrected priority
inversion). This is the textbook-naive "do nothing about locks" answer."""
import sys


def main():
    data = sys.stdin.read().split()
    L = int(data[0])
    print(" ".join(["0"] * L))  # 0 = none, on every lock


if __name__ == "__main__":
    main()
