# TIER: strong
"""Constrained optimization instead of transform inversion. Alternate: (1) an ART-style
additive projection update that pulls the (continuous-relaxed) reconstruction toward
consistency with every given ray sum, and (2) a local median-smoothing step that acts as
a total-variation-style prior, pulling each pixel toward the flat block it belongs to.
This exploits the object's known piecewise-constant / few-material structure to resolve
the ambiguity a missing angular wedge leaves behind, instead of just inverting the
visible data. Finally, quantize to the nearest palette value."""
import sys, math


def make_geom(N):
    R = 2 * math.ceil(N * math.sqrt(2) / 2) + 1
    off = R // 2
    return R, off


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); _test_id = int(next(it)); R = int(next(it))
    P = int(next(it))
    palette = [int(next(it)) for _ in range(P)]
    K = int(next(it))
    angles = [int(next(it)) for _ in range(K)]
    sino = []
    for _ in range(K):
        sino.append([int(next(it)) for _ in range(R)])

    off = R // 2

    bins_pix_by_angle = []
    for deg in angles:
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        bins_pix = [[] for _ in range(R)]
        for i in range(N):
            x = i - (N - 1) / 2.0
            for j in range(N):
                y = j - (N - 1) / 2.0
                t = x * c + y * s
                b = int(round(t)) + off
                if b < 0: b = 0
                if b >= R: b = R - 1
                bins_pix[b].append((i, j))
        bins_pix_by_angle.append(bins_pix)

    lo, hi = palette[0], palette[-1]
    fimg = [[(lo + hi) / 2.0] * N for _ in range(N)]

    ITERS = 18
    STEP = 0.7
    for _ in range(ITERS):
        # data-consistency (ART) pass: pull toward matching every given ray sum
        for row, bins_pix in zip(sino, bins_pix_by_angle):
            for b in range(R):
                pix = bins_pix[b]
                if not pix:
                    continue
                cur = sum(fimg[i][j] for i, j in pix)
                diff = (row[b] - cur) / len(pix)
                for i, j in pix:
                    fimg[i][j] += diff * STEP
        # total-variation-style prior pass: pull each pixel toward its local median
        new = [[fimg[i][j] for j in range(N)] for i in range(N)]
        for i in range(N):
            for j in range(N):
                vals = [fimg[i][j]]
                if i > 0: vals.append(fimg[i - 1][j])
                if i < N - 1: vals.append(fimg[i + 1][j])
                if j > 0: vals.append(fimg[i][j - 1])
                if j < N - 1: vals.append(fimg[i][j + 1])
                med = sorted(vals)[len(vals) // 2]
                new[i][j] = fimg[i][j] * 0.5 + med * 0.5
        fimg = new

    out = []
    for i in range(N):
        row_out = []
        for j in range(N):
            best = min(palette, key=lambda p: abs(p - fimg[i][j]))
            row_out.append(str(best))
        out.append(" ".join(row_out))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
