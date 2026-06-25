import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    k = int(data[idx]); idx += 1
    w = []
    for _ in range(n):
        w.append(int(data[idx])); idx += 1

    maxw = max(w) if w else 1

    # Independent definition of blows: for power p, a boulder of weight x is broken
    # into the fewest pieces each of size <= p. Number of pieces = how many parts of
    # size at most p are needed to cover x units; that is the smallest m with m*p >= x.
    # Blows = pieces - 1. Compute pieces by a literal loop (no ceil trickery).
    def pieces(x, p):
        m = 0
        covered = 0
        while covered < x:
            covered += p
            m += 1
        return m

    def blows(p):
        return sum(pieces(x, p) - 1 for x in w)

    # Linear scan over every candidate power from 1..maxw; pick the smallest feasible.
    ans = maxw
    for p in range(1, maxw + 1):
        if blows(p) <= k:
            ans = p
            break
    print(ans)

main()
