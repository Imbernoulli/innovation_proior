#!/usr/bin/env python3
"""
Independent brute-force oracle for "Divisor Nim".

Game: there are n piles. A move picks a pile of size x > 1 and replaces it by a
proper divisor y of x (y | x, 1 <= y < x). A pile of size 1 has no move. The
player who cannot move loses (normal play). Determine whether the first player wins.

This oracle does NOT assume Sprague-Grundy. It runs a full minimax over the
*combined* multiset of piles with memoization: a position is losing (for the player
to move) iff every move leads to a winning position; winning iff some move leads to
a losing position. Output "First" if the first player wins, else "Second".

Read stdin (same format as the C++ solution), write "First"/"Second".
"""
import sys
from functools import lru_cache

sys.setrecursionlimit(1000000)


def proper_divisors(x):
    ds = []
    i = 1
    while i < x:
        if x % i == 0:
            ds.append(i)
        i += 1
    return ds


# Minimax over the full multiset state. State = sorted tuple of pile sizes,
# with piles of size 1 dropped (they are dead -- no move). Returns True if the
# player to move wins.
@lru_cache(maxsize=None)
def first_wins(state):
    # state: sorted tuple of pile sizes, each >= 2 (size-1 piles already removed)
    if not state:
        return False  # no move available -> player to move loses
    state_list = list(state)
    for idx, x in enumerate(state):
        for y in proper_divisors(x):
            rest = state_list[:idx] + state_list[idx + 1:]
            if y >= 2:
                rest = rest + [y]
            nxt = tuple(sorted(rest))
            if not first_wins(nxt):
                return True  # moving to a losing-for-opponent position wins
    return False


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    piles = []
    for _ in range(n):
        x = int(next(it))
        if x >= 2:
            piles.append(x)
    state = tuple(sorted(piles))
    print("First" if first_wins(state) else "Second")


if __name__ == "__main__":
    main()
