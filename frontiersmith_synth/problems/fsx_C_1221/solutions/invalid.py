# TIER: invalid
"""Garbage artifact: claims max_attempts=3 but only supplies one backoff
token (should supply two) on every line -- fails the strict per-line token
count check, must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    N = int(data[1])
    for _ in range(N):
        print("3 5")


if __name__ == "__main__":
    main()
