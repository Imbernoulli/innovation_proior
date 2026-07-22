# TIER: greedy
"""Textbook offline dynamic-storage-allocation sweep: process births/deaths in
time order, always reuse the smallest free hole that fits (best-fit), tie
broken by lowest address, else bump-allocate fresh space at the frontier.
This correctly minimizes footprint fragmentation but is completely blind to
which crates get checked often -- it never looks at the check trace at all.
"""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def next_int():
        nonlocal pos
        v = int(data[pos])
        pos += 1
        return v

    N = next_int()
    M = next_int()
    next_int()  # PAGE
    next_int()  # LAMBDA

    sizes = [0] * N
    births = [0] * N
    deaths = [0] * N
    for i in range(N):
        sizes[i] = next_int()
        births[i] = next_int()
        deaths[i] = next_int()
    for _ in range(M):
        next_int()
        next_int()

    events = []
    for i in range(N):
        events.append((births[i], 1, i))
        events.append((deaths[i], 0, i))
    events.sort()

    free = []  # list of [start, length], kept sorted by start
    addr = [0] * N
    peak = 0

    def alloc(size):
        nonlocal peak
        best = -1
        for k in range(len(free)):
            s, l = free[k]
            if l < size:
                continue
            if best == -1 or l < free[best][1] or (l == free[best][1] and s < free[best][0]):
                best = k
        if best == -1:
            start = peak
            peak += size
            return start
        s, l = free[best]
        start = s
        if l == size:
            free.pop(best)
        else:
            free[best][0] = s + size
            free[best][1] = l - size
        return start

    def dealloc(start, size):
        free.append([start, size])
        free.sort()
        merged = []
        for iv in free:
            if merged and merged[-1][0] + merged[-1][1] == iv[0]:
                merged[-1][1] += iv[1]
            else:
                merged.append(iv)
        free[:] = merged

    for _, typ, i in events:
        if typ == 0:
            dealloc(addr[i], sizes[i])
        else:
            addr[i] = alloc(sizes[i])

    sys.stdout.write("\n".join(str(a) for a in addr) + "\n")


if __name__ == "__main__":
    main()
