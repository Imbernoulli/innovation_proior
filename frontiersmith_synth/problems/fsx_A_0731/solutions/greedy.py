# TIER: greedy
#!/usr/bin/env python3
"""Obvious recipe: for each storm, estimate each cell's exposure with the
textbook infinite-rain 'trapping rain water' depth (ceiling = the lower of
the two ridges bounding it *within that storm's window*, ignoring that the
storm only carries a finite volume and ignoring what raising a ridge does to
anyone else). Take each cell's worst exposure across the sweep, rank cells
by exposure*value, and fully patch the worst cells first until the budget
runs out. This reacts only to the raw depth map -- it never notices that a
cheap fix at a low-value chokepoint (a saddle) can keep an entire finite
overflow from ever reaching a valuable cluster."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    Budget = int(next(it))
    K = int(next(it))
    e = [int(next(it)) for _ in range(N)]
    v = [int(next(it)) for _ in range(N)]
    storms = []
    for _ in range(K):
        a = int(next(it))
        b = int(next(it))
        V = int(next(it))
        storms.append((a, b, V))

    fd = [0] * N
    LOCAL_CAP = 20
    for (a, b, V) in storms:
        m = b - a + 1
        # naive per-cell rain share, pretending only a nearby handful of
        # cells (not the whole window) compete for the volume
        avg_rain = V / max(1, min(m, LOCAL_CAP))
        pref = [0] * m
        cur = -10 ** 9
        for idx in range(a, b + 1):
            if e[idx] > cur:
                cur = e[idx]
            pref[idx - a] = cur
        suf = [0] * m
        cur = -10 ** 9
        for idx in range(b, a - 1, -1):
            if e[idx] > cur:
                cur = e[idx]
            suf[idx - a] = cur
        for idx in range(a, b + 1):
            ceiling = min(pref[idx - a], suf[idx - a])
            depth = ceiling - e[idx]
            if avg_rain < depth:
                depth = avg_rain
            if depth > fd[idx]:
                fd[idx] = int(depth)

    order = sorted(range(N), key=lambda i: fd[i] * v[i], reverse=True)

    h = [0] * N
    remaining = Budget
    for i in order:
        if remaining <= 0:
            break
        need = fd[i]
        if need <= 0:
            continue
        give = min(need, remaining)
        h[i] += give
        remaining -= give

    print(" ".join(map(str, h)))


if __name__ == "__main__":
    main()
