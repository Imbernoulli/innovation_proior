#!/usr/bin/env python3
# Independent oracle: minimum number of perfect squares summing to n.
# Uses BFS over the "remaining amount" graph (different algorithmic shape
# from the solution's bottom-up min-DP), so a coincident bug is unlikely.
import sys
from collections import deque


def min_squares_bfs(n):
    if n == 0:
        return 0
    squares = []
    i = 1
    while i * i <= n:
        squares.append(i * i)
        i += 1
    # BFS levels = number of squares used; first time we hit 0 is optimal.
    visited = [False] * (n + 1)
    visited[n] = True
    frontier = [n]
    level = 0
    while frontier:
        level += 1
        nxt = []
        for rem in frontier:
            for sq in squares:
                if sq > rem:
                    break
                nr = rem - sq
                if nr == 0:
                    return level
                if not visited[nr]:
                    visited[nr] = True
                    nxt.append(nr)
        frontier = nxt
    return -1  # unreachable: every n>=1 has the all-ones decomposition


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    print(min_squares_bfs(n))


if __name__ == "__main__":
    main()
