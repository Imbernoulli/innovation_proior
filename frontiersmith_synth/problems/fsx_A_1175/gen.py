#!/usr/bin/env python3
"""
gen.py <testId> -> prints ONE sparse-view tomography instance to stdout.

Builds a hidden N x N piecewise-constant object (a few random axis-aligned material
blocks over a small palette), computes its discrete Radon transform (parallel-beam,
pixel-center projection, integer bins) at K given angles, and prints ONLY the sinogram
(never the object itself). Angles are, for most test ids, confined to a narrow angular
WEDGE rather than spread over the full 0..180 range -- the missing-wedge trap: filtered
backprojection over a wedge produces streaky, directionally-biased artifacts that a
constrained-optimization / total-variation-prior reconstruction can avoid.

The hidden object's construction (RNG seed keyed ONLY on testId) is mirrored
byte-for-byte in verify.py so the checker can regenerate the same ground truth and the
same held-out (never-given) verification angles without either being printed here.
"""
import sys, math, random

PALETTE = [0, 1, 2, 3]
HELD_DEG = [5, 35, 65, 95, 125, 155]   # fixed held-out angles, never given to the solver


def make_geom(N):
    R = 2 * math.ceil(N * math.sqrt(2) / 2) + 1
    off = R // 2
    return R, off


def project(img, N, angles_deg, R, off):
    out = []
    for deg in angles_deg:
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        row = [0] * R
        for i in range(N):
            x = i - (N - 1) / 2.0
            for j in range(N):
                y = j - (N - 1) / 2.0
                t = x * c + y * s
                b = int(round(t)) + off
                if b < 0: b = 0
                if b >= R: b = R - 1
                row[b] += img[i][j]
        out.append(row)
    return out


def gen_phantom(N, palette, rng, nrect):
    img = [[palette[0]] * N for _ in range(N)]
    for _ in range(nrect):
        w = rng.randint(max(2, int(N * 0.35)), max(3, int(N * 0.75)))
        h = rng.randint(max(2, int(N * 0.35)), max(3, int(N * 0.75)))
        w = min(w, N); h = min(h, N)
        x0 = rng.randint(0, N - w)
        y0 = rng.randint(0, N - h)
        v = palette[rng.randint(0, len(palette) - 1)]
        for i in range(x0, x0 + w):
            for j in range(y0, y0 + h):
                img[i][j] = v
    return img


# ---- per-testId instance parameters (identical in gen.py and verify.py) ----
def case_params(test_id):
    table = {
        1:  (12, list(range(0, 180, 15)), 5),          # wide coverage, easy
        2:  (14, list(range(0, 180, 18)), 5),          # wide coverage, easy
        3:  (16, list(range(0, 180, 20)), 3),          # wide-ish, moderate
        4:  (16, [0, 10, 20, 30, 40, 50, 60], 6),       # missing-wedge trap
        5:  (16, [5, 15, 25, 35, 45, 55], 6),           # missing-wedge trap, offset
        6:  (18, [0, 10, 20, 30, 40, 50], 5),           # missing-wedge trap, larger
        7:  (18, [0, 20, 40, 50], 3),                    # severe missing-wedge trap
        8:  (18, [10, 25, 40, 55, 70], 7),               # missing-wedge trap, offset+range
        9:  (20, [90, 105, 120, 135], 8),                # missing-wedge trap, elsewhere
        10: (20, [0, 12, 24, 36], 9),                    # severe missing-wedge trap, largest
    }
    if test_id not in table:
        raise ValueError("testId out of range")
    return table[test_id]


def true_phantom(test_id):
    N, angles_deg, nrect = case_params(test_id)
    seed = 900000 + test_id * 104729
    rng = random.Random(seed)
    ph = gen_phantom(N, PALETTE, rng, nrect)
    return N, angles_deg, ph


def main():
    test_id = int(sys.argv[1])
    N, angles_deg, ph = true_phantom(test_id)
    R, off = make_geom(N)
    sino = project(ph, N, angles_deg, R, off)

    out = []
    out.append("%d %d %d" % (N, test_id, R))
    out.append(str(len(PALETTE)))
    out.append(" ".join(str(p) for p in PALETTE))
    out.append(str(len(angles_deg)))
    out.append(" ".join(str(d) for d in angles_deg))
    for row in sino:
        out.append(" ".join(str(v) for v in row))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
