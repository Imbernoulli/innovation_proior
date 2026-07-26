# TIER: invalid
"""Deliberately broken: pumps only the very first foundation cell and ignores
every other one, leaving the rest of the crypt floor wet -- must score 0."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    R = int(next(it)); C = int(next(it))
    n_rech = int(next(it))
    for _ in range(n_rech):
        next(it); next(it)
    n_wall = int(next(it))
    for _ in range(n_wall):
        next(it); next(it)
    n_found = int(next(it))
    first = None
    for i in range(n_found):
        r = int(next(it)); c = int(next(it)); next(it)
        if first is None:
            first = (r, c)
    # remaining tokens (reach_l, screen_l, fixed_cost, qmax, budget) ignored

    if first is None:
        print("0")
        return
    r, c = first
    print("1")
    print(f"{r} {c} 1")


if __name__ == "__main__":
    main()
