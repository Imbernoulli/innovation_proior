import sys
from itertools import permutations

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    p = [int(next(it)) for _ in range(n)]
    w = [int(next(it)) for _ in range(n)]

    if n == 0:
        print(0)
        return

    # Independent method: exhaustively try EVERY ordering of the jobs and take
    # the minimum total weighted completion time. No greedy reasoning at all.
    best = None
    for perm in permutations(range(n)):
        clock = 0
        total = 0
        for i in perm:
            clock += p[i]
            total += w[i] * clock
        if best is None or total < best:
            best = total
    print(best)

if __name__ == "__main__":
    main()
