# TIER: trivial
"""Trivial baseline: makes no attempt to crack the recurrence at all -- just repeats the
last logged draw for every future query."""
import sys


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0].strip())
    xs = list(map(int, data[1].split()))
    q = int(data[2].strip())
    ks = list(map(int, data[3].split()))
    last = xs[-1]
    print(" ".join(str(last) for _ in ks))


if __name__ == "__main__":
    main()
