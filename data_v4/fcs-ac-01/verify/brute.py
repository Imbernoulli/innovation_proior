#!/usr/bin/env python3
"""
Independent brute-force oracle for "Parity-Invariant Reachability" (generalized
sliding-tile puzzle).

Problem (see context.md): an R x C board holds tiles 1..R*C-1 plus one blank
(encoded as 0). A move slides a tile orthogonally adjacent to the blank into the
blank cell (equivalently, the blank swaps with one of its 4-neighbours). Given two
boards A and B that are permutations of the same multiset {0,1,...,R*C-1}, decide
whether A can be transformed into B by a sequence of moves. Print YES / NO.

This oracle ignores the clever invariant entirely and just does a BFS over the
(R*C)! reachable states starting from A, checking whether B is encountered. Only
usable for tiny boards (R*C <= ~9).

Input format (stdin):
    R C
    R lines of C integers  -> board A
    R lines of C integers  -> board B
Output: YES or NO
"""
import sys
from collections import deque


def solve(data):
    it = iter(data)
    R = next(it); C = next(it)
    n = R * C
    A = [next(it) for _ in range(n)]
    B = [next(it) for _ in range(n)]
    a = tuple(A)
    b = tuple(B)
    if sorted(a) != sorted(b):
        return "NO"
    if a == b:
        return "YES"
    # BFS over states reachable from A.
    start = a
    target = b
    seen = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        z = cur.index(0)
        r, c = divmod(z, C)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                nz = nr * C + nc
                lst = list(cur)
                lst[z], lst[nz] = lst[nz], lst[z]
                nxt = tuple(lst)
                if nxt == target:
                    return "YES"
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    return "NO"


def main():
    data = list(map(int, sys.stdin.read().split()))
    print(solve(data))


if __name__ == "__main__":
    main()
