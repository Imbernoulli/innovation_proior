# TIER: trivial
"""Reproduces the checker's baseline routing construction -> Ratio ~= 0.1.

Route the spring to the zone and give the zone a mild constant-slope stall. No
slope engineering: the do-the-obvious reference the checker normalizes against."""
import sys

NB = ((-1, 0), (0, 1), (1, 0), (0, -1))


def carve(g, path, prof, N, HMAX):
    pf = list(prof)
    for i in range(1, len(pf)):
        if pf[i] >= pf[i - 1]:
            pf[i] = pf[i - 1] - 1
        if pf[i] < 0:
            pf[i] = 0
    pset = set(path)
    for i, (y, x) in enumerate(path):
        g[y][x] = pf[i]
    for (y, x) in path:
        for dy, dx in NB:
            ny, nx = y + dy, x + dx
            if 0 <= ny < N and 0 <= nx < N and (ny, nx) not in pset:
                if g[ny][nx] <= g[y][x]:
                    g[ny][nx] = min(HMAX, g[y][x] + 1)


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    def nx():
        return int(next(it))
    N = nx(); V = nx(); T = nx(); CAP_K = nx(); EROSION = nx()
    DEPOSIT = nx(); HMAX = nx(); SEA = nx()
    sy = nx(); sx = nx()
    zy0 = nx(); zx0 = nx(); zy1 = nx(); zx1 = nx()
    h = [[nx() for _ in range(N)] for _ in range(N)]

    g = [row[:] for row in h]
    zcx = (zx0 + zx1) // 2
    path = []
    x = sx
    while x != zcx:
        path.append((sy, x)); x += 1 if zcx > sx else -1
    for y in range(sy, N):
        path.append((y, zcx))
    start = h[sy][sx]
    lin = [(start * (N - 1 - y)) // (N - 1) for y in range(N)]
    tgt = {}
    for (yy, xx) in path:
        tgt[(yy, xx)] = lin[yy] if yy >= sy else start
    ztop = max(1, (lin[zy0 - 1] if zy0 - 1 >= 0 else start) - 7)
    for k, y in enumerate(range(zy0, zy1 + 1)):
        tgt[(y, zcx)] = max(0, ztop - k)
    bot = tgt[(zy1, zcx)]
    for k, y in enumerate(range(zy1 + 1, N)):
        span = max(1, N - 1 - zy1)
        tgt[(y, zcx)] = max(0, bot - (bot * (k + 1)) // span)
    prof = [tgt[c] for c in path]
    carve(g, path, prof, N, HMAX)

    edits = []
    for y in range(N):
        for x in range(N):
            d = g[y][x] - h[y][x]
            if d != 0:
                edits.append((y, x, d))
    out = [str(len(edits))]
    for y, x, d in edits:
        out.append("%d %d %d" % (y, x, d))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
