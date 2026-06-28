#!/usr/bin/env python3
"""
Independent brute-force oracle for the Box Stacking problem.

Problem recap:
- n box types, each with 3 positive integer dimensions.
- Unlimited copies of each type, any of 3 orientations (choose which dim is height).
- A box may be placed on top of another iff BOTH base dimensions are strictly smaller.
- Maximize total stacked height. Empty stack (height 0) is always allowed.

Strategy (independent of sol.cpp):
- Expand every box type into its 3 oriented boxes; store base as a frozenset-ish
  pair but here we keep base sorted as (w, d) with w <= d, plus height h.
- Build a DAG: edge u -> v if base(u) strictly contains base(v) in both dims.
- Longest weighted path where node weight = height. Since the relation is a strict
  partial order (acyclic), do a memoized DFS / topological longest path.
- This is deliberately written differently from sol.cpp (graph + recursion w/ memo)
  to be a genuine independent check.
"""
import sys
from functools import lru_cache


def solve(data):
    it = iter(data)
    try:
        n = next(it)
    except StopIteration:
        return 0
    boxes = []  # (w, d, h) with w <= d
    for _ in range(n):
        x = next(it); y = next(it); z = next(it)
        dims = [x, y, z]
        for k in range(3):
            h = dims[k]
            a = dims[(k + 1) % 3]
            b = dims[(k + 2) % 3]
            if a > b:
                a, b = b, a
            boxes.append((a, b, h))

    m = len(boxes)
    if m == 0:
        return 0

    # adjacency: i can be ABOVE j  (i sits on j)  iff j's base strictly contains i's base
    # We'll compute longest path where weight is the box height of the chosen node,
    # and an edge goes from a box to a box that can sit on top of it.
    # below_to_above[i] = list of boxes that can sit directly on top of box i
    above = [[] for _ in range(m)]
    for i in range(m):
        wi, di, _ = boxes[i]
        for j in range(m):
            if i == j:
                continue
            wj, dj, _ = boxes[j]
            # box j can sit on top of box i if i's base strictly larger in both dims
            if wi > wj and di > dj:
                above[i].append(j)

    sys.setrecursionlimit(10000)

    memo = [None] * m

    def best_from(i):
        # max total height of a stack whose bottom box is i
        if memo[i] is not None:
            return memo[i]
        res = boxes[i][2]
        for j in above[i]:
            res = max(res, boxes[i][2] + best_from(j))
        memo[i] = res
        return res

    ans = 0
    for i in range(m):
        ans = max(ans, best_from(i))
    return ans


def main():
    data = list(map(int, sys.stdin.read().split()))
    print(solve(data))


if __name__ == "__main__":
    main()
