# TIER: trivial
"""Draw the mask exactly as the target pattern. This is the checker's own
baseline construction: it reproduces B exactly, so it always scores 0.1."""
import sys


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    rows = data[1:1 + n]
    sys.stdout.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
