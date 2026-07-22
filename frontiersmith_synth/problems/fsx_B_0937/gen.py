import sys, random

# ---------------------------------------------------------------------------
# fsx_B_0937 -- "Hearing the Shape of Pans" (Weyl-echo extrapolation, family
# weyl-echo-extrapolation). gen.py prints ONLY the TRAIN sample: rows of
# (a, b, lam, N_obs) for a family of rectangular pans with aspect ratio in
# [1,3] (compact / near-square pans). It never prints the hidden law, its
# coefficients, or the seed -- only data rows. The held-out extrapolation
# split (extreme aspect ratio 10-30 baking sheets) lives ONLY inside verify.py
# and is regenerated there, deterministically, from nothing but the fixed
# formula baked into that file.
# ---------------------------------------------------------------------------

AREA_GRID = [30, 50, 80, 130, 200, 320, 500, 800, 1200]
R_GRID    = [1.0, 1.2, 1.4, 1.7, 2.0, 2.3, 2.6, 3.0]
LAM_GRID  = [40, 80, 150, 300, 600, 1200, 2400, 4800]
NOISE     = 0.04

C1 = 1.0 / (4.0 * 3.141592653589793)
C2 = 1.0 / (4.0 * 3.141592653589793)
C3 = 1.5


def n_true(a, b, lam):
    A = a * b
    P = 2.0 * (a + b)
    return C1 * A * lam - C2 * P * (lam ** 0.5) + C3


def main():
    t = int(sys.argv[1])
    t = max(1, min(10, t))

    n_areas = max(4, round(len(AREA_GRID) * t / 10.0))
    n_rs    = max(4, round(len(R_GRID) * t / 10.0))
    n_lams  = max(5, round(len(LAM_GRID) * t / 10.0))
    n_areas = min(n_areas, len(AREA_GRID))
    n_rs    = min(n_rs, len(R_GRID))
    n_lams  = min(n_lams, len(LAM_GRID))

    areas = AREA_GRID[:n_areas]
    rs = R_GRID[:n_rs]
    lams = LAM_GRID[:n_lams]

    rng = random.Random(20000 + t)

    rows = []
    for Area in areas:
        for r in rs:
            a = (Area * r) ** 0.5
            b = (Area / r) ** 0.5
            for lam in lams:
                nt = n_true(a, b, lam)
                eps = rng.uniform(-NOISE, NOISE)
                nobs = nt * (1.0 + eps)
                rows.append((a, b, lam, nobs))

    out = []
    out.append(str(len(rows)))
    for a, b, lam, nobs in rows:
        out.append("%.6f %.6f %.6f %.6f" % (a, b, lam, nobs))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
