# TIER: invalid
"""Deliberately infeasible: reverses the ascending-id order (breaks nearly
every dependence edge, since ids are randomly relabelled and dependence
direction is not correlated with descending id) and drops the last token
(wrong length too)."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it))

    ids = list(range(N))
    out = list(reversed(ids))[:-1]
    sys.stdout.write(" ".join(map(str, out)))


if __name__ == "__main__":
    main()
