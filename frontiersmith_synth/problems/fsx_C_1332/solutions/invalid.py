# TIER: invalid
"""Emits a blend that uses ingredient 1 at MORE than its own IFRA
regulatory concentration cap (cap_1 + 1.0). The checker must reject this
(exceeds the per-ingredient cap) -> 0, regardless of the instance."""
import sys


def main():
    toks = sys.stdin.read().split()
    p = 0
    K = int(toks[p]); p += 1
    D = int(toks[p]); p += 1
    p += 1  # T
    p += D  # ingredient 1's desc
    p += 1  # k
    cap0 = float(toks[p]); p += 1

    print(1)
    print("%d %.6f" % (1, cap0 + 1.0))


if __name__ == "__main__":
    main()
