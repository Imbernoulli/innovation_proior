# TIER: greedy
"""Classic filtered backprojection (FBP): ramp-filter each given projection, backproject
over all given angles, then auto-quantize the reconstructed intensities onto the palette
via 1-D k-means (a fair, self-calibrating threshold pick, not tuned against any hidden
answer). This is direct transform inversion -- the textbook first approach -- and it
ignores the fact that the object is piecewise-constant. On a full angular sweep it
reconstructs well; on a narrow missing wedge, whole directions are unobserved and the
result is smeared / streaky."""
import sys, math


def make_geom(N):
    R = 2 * math.ceil(N * math.sqrt(2) / 2) + 1
    off = R // 2
    return R, off


def ramp_kernel(hw):
    k = [0.0] * (2 * hw + 1)
    k[hw] = 0.25
    for n in range(1, hw + 1):
        if n % 2 == 1:
            v = -1.0 / (math.pi * math.pi * n * n)
            k[hw + n] = v
            k[hw - n] = v
    return k


def filt_row(row, kernel):
    R = len(row)
    hw = len(kernel) // 2
    out = [0.0] * R
    for b in range(R):
        s = 0.0
        for d in range(-hw, hw + 1):
            bb = b + d
            if 0 <= bb < R:
                s += row[bb] * kernel[hw + d]
        out[b] = s
    return out


def kmeans1d(vals, k, iters=25):
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [lo] * k
    centers = [lo + (hi - lo) * (i + 0.5) / k for i in range(k)]
    for _ in range(iters):
        sums = [0.0] * k; cnts = [0] * k
        for v in vals:
            bi = min(range(k), key=lambda ci: abs(v - centers[ci]))
            sums[bi] += v; cnts[bi] += 1
        newc = [(sums[ci] / cnts[ci] if cnts[ci] > 0 else centers[ci]) for ci in range(k)]
        if newc == centers:
            break
        centers = newc
    return sorted(centers)


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
    kernel = ramp_kernel(min(R // 2, 12))
    acc = [[0.0] * N for _ in range(N)]
    for row, deg in zip(sino, angles):
        frow = filt_row(row, kernel)
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        for i in range(N):
            x = i - (N - 1) / 2.0
            for j in range(N):
                y = j - (N - 1) / 2.0
                t = x * c + y * s
                b = int(round(t)) + off
                if b < 0: b = 0
                if b >= R: b = R - 1
                acc[i][j] += frow[b]

    vals = [acc[i][j] for i in range(N) for j in range(N)]
    centers = kmeans1d(vals, len(palette))

    out = []
    for i in range(N):
        row_out = []
        for j in range(N):
            v = acc[i][j]
            bi = min(range(len(centers)), key=lambda ci: abs(v - centers[ci]))
            row_out.append(str(palette[bi]))
        out.append(" ".join(row_out))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
