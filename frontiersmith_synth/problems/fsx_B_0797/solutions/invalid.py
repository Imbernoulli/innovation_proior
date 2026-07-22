# TIER: invalid
import sys


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    D = int(next(it)); K = int(next(it)); M = int(next(it))
    minlen = int(next(it)); maxlen = int(next(it))
    docs = [next(it) for _ in range(D)]

    # deliberately infeasible: dump a huge pile of duplicate/over-length/
    # over-budget entries so every feasibility gate is violated at once.
    junk = "a" * (maxlen + 5)          # too long
    entries = [junk] * (M + 10)        # too many, and all duplicates
    print(len(entries))
    for e in entries:
        print(e)


if __name__ == "__main__":
    main()
