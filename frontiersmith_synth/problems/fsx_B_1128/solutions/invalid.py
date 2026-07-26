# TIER: invalid
"""References a tank index that doesn't exist -- always infeasible regardless
of how generous the flour budget happens to be. Must score 0."""
import sys


def main():
    tokens = sys.stdin.read().split()
    T, H, M, BUDGET = (int(x) for x in tokens[:4])
    lines = ["1", f"{T} 0"]  # tank index T is out of range [0,T)
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
