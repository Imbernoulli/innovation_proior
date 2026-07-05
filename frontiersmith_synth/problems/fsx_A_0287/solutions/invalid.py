# TIER: invalid
# Emits an infeasible artifact: too many stations, some out of range and duplicated.
# Must score 0 (checker rejects on count / range / distinctness).
import sys


def main():
    toks = sys.stdin.read().split()
    n, M = int(toks[0]), int(toks[1])
    bad = [0, 0, M + 999] * (n)  # duplicates + out-of-range + wrong count
    sys.stdout.write(" ".join(map(str, bad)) + "\n")


if __name__ == "__main__":
    main()
