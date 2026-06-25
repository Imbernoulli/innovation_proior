import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    m = int(next(it)); n = int(next(it))
    strips = []
    for _ in range(n):
        a = int(next(it)); b = int(next(it)); c = int(next(it))
        # Half-open span [a, b): vents a, a+1, ..., b-1. Independent construction
        # of the covered set as an explicit set of vent indices.
        covered = set(range(a, b))   # range(a,b) is exactly a..b-1
        strips.append((covered, c))

    target = set(range(m))           # all vents 0..m-1 must be sealed
    best = None
    # Try every subset of strips; check if the union covers all vents.
    for sub in range(1 << n):
        union = set()
        tot = 0
        for i in range(n):
            if sub & (1 << i):
                union |= strips[i][0]
                tot += strips[i][1]
        if target <= union:          # union covers every required vent
            if best is None or tot < best:
                best = tot
    print(-1 if best is None else best)

main()
