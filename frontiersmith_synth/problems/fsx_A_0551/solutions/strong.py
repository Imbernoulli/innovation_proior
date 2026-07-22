# TIER: strong
"""The insight: the score depends on the CHANGE in transport capacity, not the
flow.  Capacity = CAP_K * slope, so sediment is entrained where the slope
STEEPENS and dropped where it FLATTENS.  So engineer the derivative of the land:

  * route gently from the spring,
  * a short STEEP plunge just above the zone -> capacity spikes, the water
    accelerates and entrains a large sediment load L,
  * a GENTLE slope-1 traverse ACROSS the zone -> capacity collapses below L, so
    the load is shed cell by cell (DEPOSIT rate) right on target,
  * a gentle run-out to the sea so the channel never pools and backs up.

Same routing and budget as greedy, but the arriving sediment is multiples larger
because acceleration was engineered upstream.  That is the whole game."""
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
    ZTOP = 4
    STEEP = 4                                # rows of steep plunge above the zone
    start = h[sy][sx]
    lin = [(start * (N - 1 - y)) // (N - 1) for y in range(N)]
    # keep the water HIGH along a walled causeway (fill above the natural land)
    # so the plunge into the zone is far steeper than the terrain itself allows
    HI = max(lin[zy0] if zy0 < N else 1, start - 20)
    st_top = max(sy + 1, zy0 - STEEP)

    path = []
    x = sx
    while x != zcx:
        path.append((sy, x)); x += 1 if zcx > sx else -1
    for y in range(sy, N):
        path.append((y, zcx))

    tgt = {}
    # gentle high causeway: spring -> plunge top at height HI
    for (yy, xx) in path:
        if yy < st_top:
            span = max(1, st_top - sy)
            tgt[(yy, xx)] = start - (start - HI) * (yy - sy) // span
    # steep plunge HI -> ZTOP over STEEP rows -> capacity spikes, entrains a big load
    for y in range(st_top, zy0):
        span = max(1, zy0 - st_top)
        tgt[(y, zcx)] = ZTOP + (HI - ZTOP) * (zy0 - y) // span
    # gentle slope-1 traverse across the zone -> shed the load on target
    for k, y in enumerate(range(zy0, zy1 + 1)):
        tgt[(y, zcx)] = max(1, ZTOP - k)
    # gentle run-out to the sea (never pool)
    zbot = max(1, ZTOP - (zy1 - zy0))
    for k, y in enumerate(range(zy1 + 1, N)):
        span = max(1, N - 1 - zy1)
        tgt[(y, zcx)] = max(0, zbot - zbot * (k + 1) // span)

    prof = [tgt[c] for c in path]
    carve(g, path, prof, N, HMAX)

    edits = []
    used = 0
    for y in range(N):
        for x in range(N):
            d = g[y][x] - h[y][x]
            if d == 0 or g[y][x] < 0 or g[y][x] > HMAX:
                continue
            if used + abs(d) > V:
                continue
            used += abs(d)
            edits.append((y, x, d))
    out = [str(len(edits))]
    for y, x, d in edits:
        out.append("%d %d %d" % (y, x, d))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
