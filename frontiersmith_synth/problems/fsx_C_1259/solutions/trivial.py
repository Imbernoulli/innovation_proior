# TIER: trivial
"""Do-nothing baseline: keep every group on the Host. Reproduces the checker's
internal baseline B exactly (all-host makespan), so this always scores ~0.1."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); m = int(next(it)); g = int(next(it))
    print(" ".join(["0"] * g))


if __name__ == "__main__":
    main()
