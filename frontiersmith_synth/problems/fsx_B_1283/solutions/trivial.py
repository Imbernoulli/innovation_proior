# TIER: trivial
"""Do-nothing roster: staff zero workers in every shift block, everywhere.
Every hour's demand is met entirely by (unlimited, expensive) agency staff.
This exactly reproduces the checker's own internal baseline construction."""
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    T = int(data[p]); p += 1
    n_starts = int(data[p]); p += 1
    p += n_starts  # skip starts[]
    # (rest of the input is irrelevant to this solution)
    for _ in range(n_starts):
        print(0, 0)


if __name__ == "__main__":
    main()
