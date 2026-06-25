import sys
from itertools import permutations

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    p = []
    w = []
    for _ in range(n):
        p.append(int(next(it)))
        w.append(int(next(it)))

    # Brute force: try every ordering of the n scenes, compute total weighted
    # completion time with Python big integers (exact), take the minimum.
    best = None
    for perm in permutations(range(n)):
        clock = 0
        total = 0
        for i in perm:
            clock += p[i]
            total += w[i] * clock
        if best is None or total < best:
            best = total
    if best is None:
        best = 0
    print(best)

if __name__ == "__main__":
    main()
