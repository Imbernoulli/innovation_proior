# TIER: invalid
"""Emits out-of-range diameter indices -> must be rejected (Ratio: 0.0)."""
import sys


def main():
    data = sys.stdin.read().split('\n')
    data = [t for t in data if t.strip() != ""]
    n = int(data[0].split()[0])
    print(" ".join(str(999999) for _ in range(1, n)))


if __name__ == "__main__":
    main()
