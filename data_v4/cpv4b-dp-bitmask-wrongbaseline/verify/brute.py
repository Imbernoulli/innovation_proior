import sys
from itertools import permutations

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    idx = 0
    n = int(data[idx]); idx += 1
    c = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            c[i][j] = int(data[idx]); idx += 1
    e = [[0]*n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            e[i][k] = int(data[idx]); idx += 1

    if n <= 1:
        print(0)
        return

    best = None
    for perm in permutations(range(n)):
        total = 0
        # adjacency cleaning cost between consecutive runs
        for t in range(1, n):
            total += c[perm[t-1]][perm[t]]
        # two-step carry-over penalty: batch perm[t] gets extra e[perm[t-2]][perm[t]]
        for t in range(2, n):
            total += e[perm[t-2]][perm[t]]
        if best is None or total < best:
            best = total
    print(best)

main()
