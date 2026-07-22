# TIER: invalid
"""Deliberately broken: sweeps row 1 twice (also skips the free-row floor
entirely) so the checker must reject it."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); T = int(next(it))
    plan = [(1, 1), (2, 1)]  # row 1 swept twice -> infeasible
    out = [str(len(plan))]
    for (t, g) in plan:
        out.append(f"{t} {g}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
