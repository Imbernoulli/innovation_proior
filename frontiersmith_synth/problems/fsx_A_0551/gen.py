#!/usr/bin/env python3
"""gen.py <testId> -> one instance on stdout.

Family: terrain-sculpt-erosion-delta ("river god sculpting a farm delta").
Deterministic: all randomness seeded from testId only.

Instance:
  A seeded HxH integer heightfield descending from a high spring (top) toward a
  sea coast (bottom row, height 0).  A rectangular target deposition zone sits
  inland.  The solver emits up to V units of dig/fill edits reshaping the field.
  A fixed integer erosion CA then routes T water pulses from the spring; the
  score is sediment mass dropped inside the zone.

Trap tests (>=3 of 10): the zone is laterally OFFSET from the spring column, so
water flowing straight down the natural tilt bypasses the zone entirely -- a
naive straight canal deposits nothing there.
"""
import sys

# ---- fixed CA / terrain constants (echoed into the instance) ----
H0      = 150     # spring / top height of the tilt
HMAX    = 250     # per-cell height ceiling (validation bound on edited field)
SEA     = 0       # sea level; cells <= SEA are coast: water + load exit there
CAP_K   = 1       # transport capacity = CAP_K * local downhill slope
EROSION = 8       # max sediment entrained per cell per step
DEPOSIT = 6       # max sediment dropped per cell per step (spreads a stall out)
T       = 40      # number of water pulses


def rng(seed):
    # tiny deterministic LCG -> ints; no library RNG (portable + reproducible)
    s = (seed * 2654435761 + 1013904223) & 0xFFFFFFFF
    def nxt(mod):
        nonlocal s
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        return s % mod
    return nxt


def build(testId):
    N = 22 + 2 * testId                      # 24 .. 42
    r = rng(1000 + 7 * testId)

    sy, sx = 1, N // 2                        # spring near the top-middle
    # trap tests: zone offset laterally off the spring column
    trap = testId in (4, 5, 6, 8, 9, 10)
    zcx = sx + (N // 4 if trap else 0)
    zcx = max(2, min(N - 3, zcx))
    zx0, zx1 = zcx - 1, zcx + 1              # 3 wide
    zy0 = int(0.50 * N)
    zy1 = zy0 + 4                            # 5 tall
    zy1 = min(zy1, N - 3)

    # base terrain: linear tilt H0 (top) -> 0 (coast) + small seeded noise
    h = [[0] * N for _ in range(N)]
    for y in range(N):
        base = (H0 * (N - 1 - y)) // (N - 1)
        for x in range(N):
            v = base + (r(5) - 2)            # noise in [-2, +2]
            if v < 1:
                v = 1
            h[y][x] = v
    for x in range(N):
        h[N - 1][x] = 0                      # coast row = sea
    h[sy][sx] = H0                           # spring is the fixed high source

    # trap terrain: a pre-carved bypass gully straight down the spring column,
    # diverting the natural flow to the sea AWAY from the offset zone.
    if trap:
        for y in range(sy, N - 1):
            g = h[sy][sx] - (h[sy][sx] * (y - sy)) // (N - 1 - sy) - 4
            if g < 1:
                g = 1
            if g < h[y][sx]:
                h[y][sx] = g

    V = 12 * N * N                           # edit budget (total |delta|)

    out = []
    out.append("%d %d %d %d %d %d %d %d" % (N, V, T, CAP_K, EROSION, DEPOSIT, HMAX, SEA))
    out.append("%d %d" % (sy, sx))
    out.append("%d %d %d %d" % (zy0, zx0, zy1, zx1))
    for y in range(N):
        out.append(" ".join(str(h[y][x]) for x in range(N)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    build(tid)
