# TIER: invalid
"""Deliberately infeasible: emits a word one character too long AND uses a
letter outside the given alphabet budget K, so the checker's feasibility gate
must reject it."""
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    K = int(toks[1])
    # wrong token count (n+1) and a letter index guaranteed out of range for K
    bad = K + 5
    W = [bad] * (n + 1)
    sys.stdout.write(" ".join(map(str, W)) + "\n")


if __name__ == "__main__":
    main()
