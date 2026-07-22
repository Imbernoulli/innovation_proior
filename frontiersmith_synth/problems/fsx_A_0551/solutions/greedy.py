# TIER: greedy
"""The obvious approach an average coder writes first: ROUTE the spring to the
target zone and DIG A CATCH POND (a flat low basin) so the water pools and drops
its sediment.

It puts water in the right place and pools it -- but it ignores transport
CAPACITY: the water arrives carrying only the modest load the ordinary approach
entrained, so the pond fills slowly and then backs up.  The recipe, not the
insight."""
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
    BASIN = 6
    path = []
    x = sx
    while x != zcx:
        path.append((sy, x)); x += 1 if zcx > sx else -1
    for y in range(sy, N):
        path.append((y, zcx))
    start = h[sy][sx]
    lin = [(start * (N - 1 - y)) // (N - 1) for y in range(N)]
    GS = 6                                            # rows of steep final dig
    dig_top = max(sy, zy0 - GS)
    tgt = {}
    for (yy, xx) in path:
        if yy < dig_top:                              # follow the land gently
            tgt[(yy, xx)] = lin[yy] if yy >= sy else start
        elif yy < zy0:                                # steep dig down to the pond
            span = max(1, zy0 - dig_top)
            tgt[(yy, xx)] = BASIN + (lin[dig_top] - BASIN) * (zy0 - yy) // span
        else:
            tgt[(yy, xx)] = BASIN
    for y in range(zy0, zy1 + 1):
        tgt[(y, zcx)] = BASIN                         # flat catch pond
    for k, y in enumerate(range(zy1 + 1, N)):         # outlet to the sea
        span = max(1, N - 1 - zy1)
        tgt[(y, zcx)] = max(0, BASIN - (BASIN * (k + 1)) // span)
    prof = [tgt[c] for c in path]
    carve(g, path, prof, N, HMAX)

    # widen the pond over the full zone rectangle (obvious "make the pond big")
    for y in range(zy0, zy1 + 1):
        for x in range(zx0, zx1 + 1):
            if g[y][x] > BASIN:
                g[y][x] = BASIN

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
