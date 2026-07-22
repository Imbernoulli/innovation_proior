# TIER: strong
# Insight: the schedule is a PROGRAM controlling a self-modifying router.
# Small early pulses are spent on purpose -- they aggrade a near-shore platform
# (paying the subsidence tax while they are still small) that flattens the local
# slope and steers later flow sideways instead of down the incised channel.  The
# big late releases then land on that engineered terrain and are retained almost
# in full.  So a CRESCENDO (geometric ramp-up) beats the flat recipe; we search
# the ramp ratio per instance and locally refine it against the exact simulator.
import sys

def simulate(zbed, H, W, sr, schedule, d0, kslope, beta):
    N = H * W
    z = [zbed[r][c] for r in range(H) for c in range(W)]
    bed = z[:]
    T = len(schedule)
    src = sr * W + 0
    neigh = [None] * N
    for r in range(H):
        for c in range(W):
            i = r * W + c
            lst = []
            if c + 1 < W: lst.append(i + 1)
            if c - 1 >= 0: lst.append(i - 1)
            if r + 1 < H: lst.append(i + W)
            if r - 1 >= 0: lst.append(i - W)
            neigh[i] = lst
    for t in range(T):
        if t > 0 and beta > 0.0:
            keep = 1.0 - beta
            for i in range(N):
                d = z[i] - bed[i]
                if d != 0.0:
                    z[i] = bed[i] + d * keep
        M = schedule[t]
        if M <= 0.0:
            continue
        order = sorted(range(N), key=lambda i: (-z[i], i))
        flux = [0.0] * N
        flux[src] += M
        dz = [0.0] * N
        for i in order:
            q = flux[i]
            if q <= 1e-15:
                continue
            zi = z[i]
            drops = []
            tot = 0.0
            for nb in neigh[i]:
                d = zi - z[nb]
                if d > 0.0:
                    drops.append((nb, d)); tot += d
            smax = 0.0
            for _, d in drops:
                if d > smax:
                    smax = d
            dep_frac = d0 + (1.0 - d0) / (1.0 + kslope * smax)
            if (i % W) == W - 1:
                dz[i] += q * dep_frac; continue
            if not drops:
                dz[i] += q; continue
            dep = q * dep_frac
            dz[i] += dep
            rem = q - dep
            for nb, d in drops:
                flux[nb] += rem * (d / tot)
        for i in range(N):
            if dz[i] > 0.0:
                z[i] += dz[i]
    return sum(1 for i in range(N) if z[i] >= 0.0)


def geo(T, M, r):
    w = [r ** i for i in range(T)]
    s = sum(w)
    return [M * x / s for x in w]


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    H = int(next(it)); W = int(next(it)); T = int(next(it))
    Mtot = float(next(it)); d0 = float(next(it)); kslope = float(next(it))
    beta = float(next(it)); sr = int(next(it))
    z = [[float(next(it)) for _ in range(W)] for _ in range(H)]

    def score(sched):
        return simulate(z, H, W, sr, sched, d0, kslope, beta)

    # 1) coarse search over crescendo ratios
    best = None; bl = -1
    for r in (1.2, 1.35, 1.5, 1.7, 1.9, 2.2, 2.6):
        sc = geo(T, Mtot, r)
        l = score(sc)
        if l > bl:
            bl = l; best = sc

    # 2) light coordinate refinement: shift mass between adjacent late releases
    #    (re-tune where the "program" hands off to the retained final pulses)
    cur = best[:]; cl = bl
    step = Mtot * 0.06
    for _ in range(3):
        improved = False
        for j in range(T - 1):
            for a, b in ((j, j + 1), (j + 1, j)):
                if cur[a] < step:
                    continue
                cand = cur[:]
                cand[a] -= step; cand[b] += step
                l = score(cand)
                if l > cl:
                    cl = l; cur = cand; improved = True
        if not improved:
            step *= 0.5

    out = cur if cl >= bl else best
    print(" ".join("%.6f" % x for x in out))

main()
