# TIER: invalid
"""Deliberately infeasible: negative thicknesses (violates h_i > 0)."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    int(next(it))  # test_id
    N = int(next(it))
    out = []
    for _ in range(N - 1):
        out.append("-50.0 1200.0")
    out.append("1200.0")
    print("\n".join(out))


if __name__ == "__main__":
    main()
