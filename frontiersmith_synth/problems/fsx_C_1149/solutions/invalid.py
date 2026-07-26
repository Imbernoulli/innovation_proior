# TIER: invalid
"""Emits cuts at the two ends (0 and L), which are never valid interior
cut positions (0 < b_j < L is required) -- must score 0 on every instance."""
import sys


def main():
    data = sys.stdin.buffer.read().split()
    L = int(data[0])
    mid = L // 2
    print(3)
    print(f"0 {mid} {L}")


if __name__ == "__main__":
    main()
