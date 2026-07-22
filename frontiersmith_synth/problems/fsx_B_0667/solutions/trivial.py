# TIER: trivial
"""Do nothing: leave every cable at its already-installed base weight 1,
spending none of the upgrade budget."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    _w_budget = int(next(it))
    for _ in range(m):
        next(it)
        next(it)
    for _ in range(n):
        next(it)
    sys.stdout.write("\n".join(["1"] * m) + "\n")


if __name__ == "__main__":
    main()
