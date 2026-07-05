# TIER: strong
# Hammersley set (i/nf, radical-inverse base 2) -- an asymptotically
# low-discrepancy sequence -- combined with an ANCHOR-AWARE Cranley-Patterson
# shift search: try a fixed grid of toroidal shifts and keep the one whose
# UNION with the pre-deployed anchor rigs has the smallest star discrepancy.
import sys, math, bisect

SHIFT_G = 5  # 5x5 = 25 candidate shifts (deterministic)


def star_discrepancy(pts):
    n = len(pts)
    xs = sorted(set(p[0] for p in pts))
    ys = sorted(set(p[1] for p in pts))
    xr = {v: i for i, v in enumerate(xs)}
    yr = {v: i for i, v in enumerate(ys)}
    nx, ny = len(xs), len(ys)
    H = [[0] * ny for _ in range(nx)]
    for (px, py) in pts:
        H[xr[px]][yr[py]] += 1
    P = [[0] * (ny + 1) for _ in range(nx + 1)]
    for a in range(nx):
        row = P[a + 1]; prow = P[a]; Ha = H[a]
        run = 0
        for b in range(ny):
            run += Ha[b]
            row[b + 1] = prow[b + 1] + run
    Xc = xs + ([1.0] if xs[-1] != 1.0 else [])
    Yc = ys + ([1.0] if ys[-1] != 1.0 else [])
    n_inv = 1.0 / n
    best = 0.0
    for vx in Xc:
        a_le = bisect.bisect_right(xs, vx) - 1
        a_lt = bisect.bisect_left(xs, vx) - 1
        rle = P[a_le + 1]; rlt = P[a_lt + 1]
        for vy in Yc:
            b_le = bisect.bisect_right(ys, vy) - 1
            b_lt = bisect.bisect_left(ys, vy) - 1
            vol = vx * vy
            pos = rle[b_le + 1] * n_inv - vol
            neg = vol - rlt[b_lt + 1] * n_inv
            if pos > best:
                best = pos
            if neg > best:
                best = neg
    return best


def radinv2(i):
    r = 0.0; f = 0.5
    while i > 0:
        r += f * (i & 1)
        i >>= 1
        f *= 0.5
    return r


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    d = int(next(it)); M = int(next(it)); K = int(next(it))
    anchors = []
    for _ in range(K):
        anchors.append((float(next(it)), float(next(it))))
    nf = M - K

    base = [((i + 0.5) / nf, radinv2(i)) for i in range(nf)]
    best_pts = None
    best_F = None
    for a in range(SHIFT_G):
        for b in range(SHIFT_G):
            sx = a / SHIFT_G; sy = b / SHIFT_G
            pts = [(((x + sx) % 1.0), ((y + sy) % 1.0)) for (x, y) in base]
            F = star_discrepancy(anchors + pts)
            if best_F is None or F < best_F:
                best_F = F
                best_pts = pts

    out = ["%.17g %.17g" % (x, y) for (x, y) in best_pts]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
