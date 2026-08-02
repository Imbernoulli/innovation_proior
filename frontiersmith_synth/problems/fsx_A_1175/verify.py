#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans ignored)

Deterministic grader for the sparse-view / missing-wedge tomography reconstruction task.

Regenerates the hidden object and the fixed held-out verification angles entirely from
the testId printed in <in> (byte-for-byte mirror of gen.py's construction -- the ground
truth is never printed to the solver and lives only here).

Feasibility: exactly N lines of N tokens each, every token finite and equal (within
1e-6) to a declared palette value.

Objective F = W1 * (fraction of pixels within 1 palette step of truth)
            + W2 * (held-out-angle projection consistency, 1 - normalized L1 residual)
Baseline B  = F of the checker's own single-flat-fill construction (palette value
closest to the overall mean density implied by the given sinogram), floored so a
degenerate instance cannot blow up the ratio.
Ratio = min(1000, 100*F/max(1e-9,B)) / 1000.
"""
import sys, math, random

PALETTE = [0, 1, 2, 3]
HELD_DEG = [5, 35, 65, 95, 125, 155]

W1, W2 = 0.7, 0.3
TOL = 1          # "within 1 palette step" tolerance for structural accuracy
B_FLOOR = 0.05
MAX_TOKENS = 2000  # generous cap: N<=20 -> N*N<=400 tokens expected


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


# ---- geometry + phantom construction: IDENTICAL to gen.py ----
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


def case_params(test_id):
    table = {
        1:  (12, list(range(0, 180, 15)), 5),
        2:  (14, list(range(0, 180, 18)), 5),
        3:  (16, list(range(0, 180, 20)), 3),
        4:  (16, [0, 10, 20, 30, 40, 50, 60], 6),
        5:  (16, [5, 15, 25, 35, 45, 55], 6),
        6:  (18, [0, 10, 20, 30, 40, 50], 5),
        7:  (18, [0, 20, 40, 50], 3),
        8:  (18, [10, 25, 40, 55, 70], 7),
        9:  (20, [90, 105, 120, 135], 8),
        10: (20, [0, 12, 24, 36], 9),
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
# ---- end mirrored section ----


def pixel_sim_tol(img, ph, N, tol):
    c = 0
    for i in range(N):
        for j in range(N):
            if abs(img[i][j] - ph[i][j]) <= tol:
                c += 1
    return c / (N * N)


def heldout_consistency(img, ph, N, R, off):
    sp = project(img, N, HELD_DEG, R, off)
    st = project(ph, N, HELD_DEG, R, off)
    resid = 0.0
    mass = 0.0
    for r1, r2 in zip(sp, st):
        for a, b in zip(r1, r2):
            resid += abs(a - b)
            mass += abs(b)
    return 1 - min(1.0, resid / max(1e-9, mass))


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    in_tokens = open(in_path).read().split()
    if not in_tokens:
        fail("empty input")

    it = iter(in_tokens)
    try:
        N = int(next(it)); test_id = int(next(it)); R_in = int(next(it))
        P = int(next(it))
        pal_in = [int(next(it)) for _ in range(P)]
        K = int(next(it))
        angles_in = [int(next(it)) for _ in range(K)]
        sino = []
        for _ in range(K):
            sino.append([int(next(it)) for _ in range(R_in)])
    except (StopIteration, ValueError):
        fail("malformed input")

    # sanity: the input itself should match what gen.py would produce for this testId
    try:
        N_true, angles_true, ph = true_phantom(test_id)
    except ValueError:
        fail("bad test_id in input")
    R, off = make_geom(N_true)
    if N != N_true or R_in != R or pal_in != PALETTE or angles_in != angles_true:
        fail("input does not match expected instance for its test_id")

    # ---- parse participant output ----
    raw = open(out_path).read()
    out_tokens = raw.split()
    if len(out_tokens) > MAX_TOKENS:
        fail("too many output tokens")
    if len(out_tokens) != N * N:
        fail("expected %d tokens, got %d" % (N * N, len(out_tokens)))

    pal_set = set(PALETTE)
    img = [[0] * N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(N):
            tok = out_tokens[idx]; idx += 1
            try:
                v = float(tok)
            except ValueError:
                fail("unparsable token %r" % tok)
            if not math.isfinite(v):
                fail("non-finite value %r" % tok)
            best = min(PALETTE, key=lambda p: abs(p - v))
            if abs(best - v) > 1e-6:
                fail("value %r not in declared palette" % tok)
            img[i][j] = best

    # ---- objective ----
    ps = pixel_sim_tol(img, ph, N, TOL)
    hc = heldout_consistency(img, ph, N, R, off)
    F = W1 * ps + W2 * hc

    # ---- checker's own baseline: single flat fill at nearest palette to mean density ----
    tot = sum(sum(r) for r in sino)
    avg = tot / (len(angles_true) * N)
    flat_v = min(PALETTE, key=lambda p: abs(p - avg))
    triv_img = [[flat_v] * N for _ in range(N)]
    ps_t = pixel_sim_tol(triv_img, ph, N, TOL)
    hc_t = heldout_consistency(triv_img, ph, N, R, off)
    B = max(W1 * ps_t + W2 * hc_t, B_FLOOR)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.4f B=%.4f ps=%.4f hc=%.4f Ratio: %.6f" % (F, B, ps, hc, sc / 1000.0))


if __name__ == "__main__":
    main()
