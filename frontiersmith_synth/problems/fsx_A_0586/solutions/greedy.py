# TIER: greedy
# The obvious approach: minimise total travel by solving the assignment for
# minimum squared distance (optimal Hungarian match), then fly everyone at
# once (single wave 0).  It is blind to the downwash cones -- so in the
# tall-stack instances the cheapest match is the radial swap, and drones on
# the same strut cross at the same (x,y) and pile into each other's cones.
import sys

def hungarian(cost):
    # square assignment, minimise; O(n^3).  cost[i][j] int.
    n = len(cost)
    INF = float("inf")
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    match = [0] * n
    for j in range(1, n + 1):
        match[p[j] - 1] = j - 1
    return match

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it)); L = int(next(it)); S = int(next(it))
    W = int(next(it)); K = int(next(it))
    for _ in range(6):
        next(it)
    F0 = [(int(next(it)), int(next(it)), int(next(it))) for _ in range(N)]
    F1 = [(int(next(it)), int(next(it)), int(next(it))) for _ in range(N)]

    cost = [[0] * N for _ in range(N)]
    for i in range(N):
        xi, yi, zi = F0[i]
        row = cost[i]
        for j in range(N):
            xj, yj, zj = F1[j]
            dx = xi - xj; dy = yi - yj; dz = zi - zj
            row[j] = dx * dx + dy * dy + dz * dz
    assign = hungarian(cost)

    out = ["%d 0" % assign[i] for i in range(N)]
    sys.stdout.write("\n".join(out) + "\n")

main()
