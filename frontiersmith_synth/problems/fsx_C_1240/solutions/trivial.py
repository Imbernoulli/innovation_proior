# TIER: trivial
"""Baseline construction: round-robin key placement assign[i] = i % K. This
exactly reproduces the checker's own internal reference baseline, so it
should score ~0.1 on every case."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    K = int(data[1])
    print(" ".join(str(i % K) for i in range(n)))


if __name__ == "__main__":
    main()
