import sys

MOD = 1000000007

def rotate(c, s):
    n = len(c)
    return tuple(c[(i - s) % n] for i in range(n))

def reflect(c, axis):
    # apply reflection then we will also try all rotations of the reflection,
    # but easier: generate the full dihedral group as functions of index.
    n = len(c)
    return tuple(c[(axis - i) % n] for i in range(n))

def canonical(c):
    n = len(c)
    best = None
    # all rotations
    for s in range(n):
        r = rotate(c, s)
        if best is None or r < best:
            best = r
    # all reflections (reverse then all rotations) -- dihedral group
    rev = tuple(reversed(c))
    for s in range(n):
        r = rotate(rev, s)
        if r < best:
            best = r
    return best

def solve(n, k):
    if n == 0:
        # No standard necklace; define count as 0 (we never test n=0).
        return 0
    seen = set()
    count = 0
    # enumerate all colorings as base-k numbers of length n
    for code in range(k ** n):
        c = []
        x = code
        for _ in range(n):
            c.append(x % k)
            x //= k
        c = tuple(c)
        can = canonical(c)
        if can not in seen:
            seen.add(can)
            count += 1
    return count % MOD

data = sys.stdin.read().split()
n = int(data[0]); k = int(data[1])
print(solve(n, k))
