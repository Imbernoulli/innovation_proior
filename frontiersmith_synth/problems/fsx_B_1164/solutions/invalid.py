# TIER: invalid
"""Deliberately infeasible: dumps every job onto robot 1, all starting at
time 0, with no decon cycles at all. This violates chronological ordering
(the second job on robot 1 starts before the first one finished) and, once
grades differ, also violates the contamination-grade rule. Must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    L = int(next(it)); R = int(next(it)); J = int(next(it))
    T = int(next(it)); C = int(next(it)); KCOST = int(next(it))
    # jobs are irrelevant to this deliberately-broken construction
    for _ in range(J):
        for _ in range(5):
            next(it)

    out = ["0"]  # NC = 0, no decon cycles at all
    out.append(f"ROBOT 1 {J}")
    for jid in range(1, J + 1):
        out.append(f"J {jid} 0")
    for r in range(2, R + 1):
        out.append(f"ROBOT {r} 0")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
