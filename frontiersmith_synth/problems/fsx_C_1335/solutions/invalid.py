# TIER: invalid
"""Deliberately infeasible: holds at the hottest level and injects the
per-step maximum EVERY step, blowing straight through the total precursor
budget C0 (sum of injections far exceeds C0). Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]; pos += 1
        return v

    T = int(nxt()); L = int(nxt()); S = int(nxt())
    nxt(); nxt()
    nxt(); nxt()
    for _ in range(L):
        nxt(); nxt(); nxt()
    for _ in range(S):
        nxt(); nxt()
    nxt()                       # C0 (deliberately ignored)
    max_inject = float(nxt())
    nxt(); nxt()

    temp = [L - 1] * T
    inject = [max_inject] * T   # sum = T * max_inject >> C0
    surf = 0

    out = [" ".join(map(str, temp)), " ".join(f"{x:.6f}" for x in inject), str(surf)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
