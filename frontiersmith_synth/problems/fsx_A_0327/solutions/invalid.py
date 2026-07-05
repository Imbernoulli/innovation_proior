# TIER: invalid
"""Invalid: emit n copies of the same altitude (0). Not distinct -> infeasible -> scores 0."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    sys.stdout.write(" ".join(["0"] * n) + "\n")


if __name__ == "__main__":
    main()
