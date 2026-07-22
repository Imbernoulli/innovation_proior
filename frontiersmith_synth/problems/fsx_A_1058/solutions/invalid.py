# TIER: invalid
"""Deliberately infeasible: floods every edge with its full capacity, ignoring
flow conservation and the global backbone budget entirely. Must score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = int(toks[p]); p += 1; return v

    K = nxt()
    out = []
    for _ in range(K):
        n = nxt(); m = nxt(); s = nxt(); t = nxt(); d = nxt()
        for _ in range(m):
            u = nxt(); v = nxt(); cap = nxt(); cost = nxt(); shared = nxt(); weight = nxt()
            out.append(str(cap))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
