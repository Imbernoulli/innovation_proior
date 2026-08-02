# TIER: trivial
"""Reproduces the checker's own internal baseline exactly: use ONLY the
first provided ingredient, at its own IFRA cap concentration. No search,
no trajectory reasoning, no masking awareness at all -> lands right on the
checker's baseline B (Ratio ~ 0.1)."""
import sys


def main():
    toks = sys.stdin.read().split()
    p = 0
    K = int(toks[p]); p += 1
    D = int(toks[p]); p += 1
    T = int(toks[p]); p += 1
    caps = []
    for _ in range(K):
        p += D  # desc
        p += 1  # k
        cap = float(toks[p]); p += 1
        caps.append(cap)
    # mask / times / targets unused by the trivial baseline construction

    print(1)
    print("%d %.6f" % (1, caps[0]))


if __name__ == "__main__":
    main()
