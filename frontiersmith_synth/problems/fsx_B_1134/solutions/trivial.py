# TIER: trivial
"""Do-nothing baseline: leave the sheet completely flat (every fold angle = 0).
This reproduces the checker's own internal baseline construction exactly."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    K = int(next(it))
    n = 4 * K + (K - 1)
    print(" ".join("0.0" for _ in range(n)))


if __name__ == "__main__":
    main()
