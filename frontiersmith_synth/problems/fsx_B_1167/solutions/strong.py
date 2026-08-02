# TIER: strong
"""The insight: don't fit density per-cell (that free-form fit is what makes the
shallowest layer look "cheapest" and is exactly greedy's trap). Instead exploit the
layer-continuity prior explicitly -- search over the SHAPE the true body actually has, a
single depth layer z holding ONE contiguous span of columns [c_lo,c_hi] -- and for each
candidate (z, c_lo, c_hi) solve the one remaining free parameter (a uniform magnitude) by
least squares against every given reading. Because the point-mass kernel's LATERAL SHAPE
(not just its on-axis amplitude) genuinely differs with depth, this multi-station residual
discriminates the true depth even though a single amplitude reading could not.

On top of that, apply depth-weighted regularization to break residual near-ties in favor
of NOT explaining the data with an implausibly small magnitude at shallow depth: since the
kernel's on-axis sensitivity falls as roughly 1/depth^2, we inflate each candidate's
comparison cost by (1 + LAMBDA/depth(z)^2) -- a penalty that is large at z=1 and fades to
~1 for deep z. This is the direct fix for the shallow-plastering bias: a shallow candidate
must fit MUCH better (not just marginally better) than a deep one to be preferred, which
is what stops the search from defaulting to the "cheap" shallow explanation the way an
unweighted fit would."""
import sys

CELL_W = 2.0
LAYER_H = 2.0
KGAIN = 100.0
LAMBDA = 2.0
MAX_WIDTH = 6


def depth(z):
    return (z - 0.5) * LAYER_H


def kernel(dc, z):
    dx = dc * CELL_W
    dpt = depth(z)
    return KGAIN * dpt / (dx * dx + dpt * dpt) ** 1.5


def main():
    data = sys.stdin.read().split()
    idx = 0
    test_id = int(data[idx]); idx += 1
    nx = int(data[idx]); idx += 1
    nz = int(data[idx]); idx += 1
    rho_max = float(data[idx]); idx += 1
    mass_max = float(data[idx]); idx += 1
    ns_given = int(data[idx]); idx += 1
    given = []
    for _ in range(ns_given):
        c = int(data[idx]); idx += 1
        r = float(data[idx]); idx += 1
        given.append((c, r))

    best = None  # (adjusted_cost, z, c_lo, c_hi, mag)
    for z in range(1, nz + 1):
        dpt = depth(z)
        penalty = 1.0 + LAMBDA / (dpt * dpt)
        max_w = min(MAX_WIDTH, nx)
        for c_lo in range(nx):
            for w in range(1, max_w + 1):
                c_hi = c_lo + w - 1
                if c_hi >= nx:
                    break
                num = 0.0
                den = 0.0
                basis = []
                for (cs, r) in given:
                    b = 0.0
                    for c in range(c_lo, c_hi + 1):
                        b += kernel(cs - c, z)
                    basis.append(b)
                    num += b * r
                    den += b * b
                if den < 1e-12:
                    continue
                mag = num / den
                if mag < 0.0:
                    mag = 0.0
                if mag > rho_max:
                    mag = rho_max
                resid = 0.0
                for b, (cs, r) in zip(basis, given):
                    d = mag * b - r
                    resid += d * d
                resid_norm = resid / len(given)
                adjusted = resid_norm * penalty
                if best is None or adjusted < best[0]:
                    best = (adjusted, z, c_lo, c_hi, mag)

    _, z, c_lo, c_hi, mag = best
    grid = [[0.0] * nx for _ in range(nz + 1)]
    for c in range(c_lo, c_hi + 1):
        grid[z][c] = mag

    total = mag * (c_hi - c_lo + 1)
    if total > mass_max and total > 0:
        f = mass_max / total
        for c in range(c_lo, c_hi + 1):
            grid[z][c] *= f

    out = []
    for zz in range(1, nz + 1):
        out.append(" ".join("%.6f" % v for v in grid[zz]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
