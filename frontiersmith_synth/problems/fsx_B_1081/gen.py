#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of fsx_B_1081 to stdout.

Format:
  line 1: R C
  line 2: Bbudget
  next R+1 lines: C+1 floats each -- height h(i,j), row-major i=0..R, j=0..C

The "drum skin" is a Monge-patch heightfield: vertex (i,j) sits at 3D point
(i, j, h(i,j)). h is a deterministic sum of NB well-separated Gaussian
bumps with a gently decaying HARMONIC amplitude ladder A_k = A0/(1+SLOPE*k)
(not geometric -- a geometric decay lets a generous budget mop up so much
of the ladder that the residual peak collapses to a sliver; the harmonic
tail stays comparably tall for a long time, so a budget that only reaches
the top several bumps still leaves a meaningful, not-tiny, residual peak:
no reachable "release everything" optimum). Bump centers are placed via a
low-discrepancy (golden-ratio) sequence with a minimum pairwise separation,
so peaks don't interfere with each other's curvature and don't cluster.
All 10 cases are fully determined by testId (fixed tables + a closed-form
placement formula -- no external randomness).

TRAP cases (testId not in FRIENDLY): the seam budget B = R-1, strictly
below R, the length of even the CHEAPEST possible single full-height
vertical cut (every mesh edge has 3D length >= 1, and a full cut crosses R
edges) -- a uniform-stripe cutter can afford literally ZERO cuts, no matter
how the terrain is shaped or where its boundaries would fall. A
curvature-aware solver can still afford several cheap LOCAL peels (cutting
just enough to un-surround one high-curvature vertex costs the perimeter
of a couple of triangles, a small near-constant, not R).

FRIENDLY cases (testId in FRIENDLY = {2,4,6,8,10}): the single DOMINANT
bump is placed exactly at column C//2 (any row) and boosted 3x above the
ladder's base amplitude, so the natural K=2 equal-gore cut (the first thing
a uniform stripe cutter tries) slices right through it and gets real,
calibrated credit -- this shows uniform gores are not USELESS, just not
curvature-aware. Bump count is raised here (budget is looser, growing with
R) so even this generous a budget cannot clear the whole peak ladder.
"""
import math
import sys

SIZES = {1: 12, 2: 13, 3: 15, 4: 16, 5: 18, 6: 19, 7: 21, 8: 23, 9: 25, 10: 28}
FRIENDLY = {2, 4, 6, 8, 10}
NB_TRAP = {1: 7, 3: 7, 5: 8, 7: 9, 9: 9}
NB_FRIENDLY = {2: 12, 4: 14, 6: 16, 8: 20, 10: 24}
SLOPE = 0.22   # harmonic decay A_k = A0/(1+SLOPE*k) -- gentle long tail so the
               # residual peak can't collapse below ~1/10 of the top peak no
               # matter how many cheap releases a generous budget affords
SIGMA = 1.0
MIN_SEP = 3.6
PHI = 0.6180339887498949   # golden-ratio conjugate
SQ2 = 0.41421356237309515  # sqrt(2)-1


def height_field(R, C, bumps):
    h = [[0.0] * (C + 1) for _ in range(R + 1)]
    for i in range(R + 1):
        for j in range(C + 1):
            s = 0.0
            for (A, ci, cj, sig) in bumps:
                s += A * math.exp(-((i - ci) ** 2 + (j - cj) ** 2) / (2.0 * sig * sig))
            h[i][j] = s
    return h


def vlen(h, i, j):
    dz = h[i + 1][j] - h[i][j]
    return math.sqrt(1.0 + dz * dz)


def cut_cost(h, R, j0):
    return sum(vlen(h, i, j0) for i in range(R))


def place_centers(R, C, NB, t, forced_first=None):
    """Low-discrepancy (golden-ratio) candidate stream, each candidate kept
    only if it's >= MIN_SEP from every already-placed center (deterministic
    rejection, no interference between bump curvatures)."""
    centers = []
    if forced_first is not None:
        centers.append(forced_first)
    lo_i, hi_i = 2, R - 2
    lo_j, hi_j = 2, C - 2
    k = 0
    tries = 0
    while len(centers) < NB and tries < 2000:
        fi = (k * PHI + t * 0.371 + 0.13) % 1.0
        fj = (k * SQ2 + t * 0.617 + 0.29) % 1.0
        ci = lo_i + int(fi * max(1, hi_i - lo_i))
        cj = lo_j + int(fj * max(1, hi_j - lo_j))
        k += 1
        tries += 1
        ok = all((ci - pc[0]) ** 2 + (cj - pc[1]) ** 2 >= MIN_SEP * MIN_SEP for pc in centers)
        if ok:
            centers.append((ci, cj))
    return centers


def build_case(t):
    R = C = SIZES[t]
    friendly = t in FRIENDLY
    NB = NB_FRIENDLY[t] if friendly else NB_TRAP[t]
    forced = (R // 2, round(C / 2)) if friendly else None
    centers = place_centers(R, C, NB, t, forced_first=forced)

    bumps = []
    for k, (ci, cj) in enumerate(centers):
        if friendly and k == 0:
            # a pronounced outlier peak sitting exactly on the gore boundary,
            # so releasing it gives a uniform-stripe cutter real, calibrated
            # credit (not just a marginal nudge over the rest of the ladder)
            A = round(3.4 * 3.0, 4)
        else:
            expo = (k - 1) if friendly else k
            A = round(3.4 / (1.0 + SLOPE * expo), 4)
        bumps.append((A, ci, cj, SIGMA))

    if friendly:
        h_probe = height_field(R, C, bumps)
        cost2 = cut_cost(h_probe, R, round(C / 2))
        bs3 = sorted(set(round(k * C / 3) for k in range(1, 3)))
        bs3 = [b for b in bs3 if 1 <= b <= C - 1]
        cost3 = sum(cut_cost(h_probe, R, b) for b in bs3) if bs3 else 1e18
        B = round(min(cost2 + 1.5, cost3 - 0.5), 4)
    else:
        B = float(R - 1)

    return R, C, B, bumps


def main():
    t = int(sys.argv[1])
    R, C, B, bumps = build_case(t)
    h = height_field(R, C, bumps)
    out = [f"{R} {C}", f"{B:.4f}"]
    for i in range(R + 1):
        out.append(" ".join(f"{h[i][j]:.6f}" for j in range(C + 1)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
