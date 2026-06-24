import sys
from itertools import permutations

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    s = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s[i][j] = int(data[idx]); idx += 1
    # Brute force: try every assignment of runners to legs.
    # perm[leg] = runner assigned to that leg. Sum s[runner][leg].
    best = None
    for perm in permutations(range(n)):
        total = 0
        for leg in range(n):
            runner = perm[leg]
            total += s[runner][leg]
        if best is None or total > best:
            best = total
    # n >= 1 always (generator), so best is defined.
    print(best)

main()
