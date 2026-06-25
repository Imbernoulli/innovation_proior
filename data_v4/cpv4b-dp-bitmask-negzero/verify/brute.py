import sys
from itertools import permutations

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    p = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            p[i][j] = int(data[idx]); idx += 1

    # Independent brute force: enumerate EVERY partial matching of artists -> stages.
    # A partial matching = pick a subset S of artists and an injective map S -> stages.
    # Equivalently, for each ordered assignment we let each stage hold at most one artist
    # and each artist appear at most once. We just try all functions stage->(-1 or artist)
    # with distinct artists, by brute recursion over stages. Empty roster gives 0.
    best = 0  # empty roster always allowed

    used = [False]*n
    def rec(stage, total):
        nonlocal best
        if total > best:
            best = total
        if stage == n:
            return
        # leave this stage empty
        rec(stage+1, total)
        # assign any free artist to this stage
        for a in range(n):
            if not used[a]:
                used[a] = True
                rec(stage+1, total + p[a][stage])
                used[a] = False

    rec(0, 0)
    print(best)

main()
