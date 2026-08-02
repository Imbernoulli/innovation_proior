# TIER: invalid
"""Shifts every key index up by one (key 0 never appears, key K is out of
range) -- fails the checker's "every key 0..K-1 exactly once" schema check
on every instance, regardless of K. Must score 0 on every test case."""
import sys


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    _R = int(next(it)); K = int(next(it))
    lines = []
    for k in range(K):
        lines.append(f"{k + 1} P 1 0")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
