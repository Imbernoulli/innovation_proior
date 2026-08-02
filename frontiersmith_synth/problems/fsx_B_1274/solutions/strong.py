# TIER: strong
import sys


def solve_lstsq(A, y, w, P):
    """Weighted normal-equations solve for a P-vector via Gaussian elimination
    with partial pivoting (P is tiny -- 2 or 3 -- so this is exact enough)."""
    M = [[0.0] * P for _ in range(P)]
    b = [0.0] * P
    for row, yi, wi in zip(A, y, w):
        for i in range(P):
            b[i] += wi * row[i] * yi
            for j in range(P):
                M[i][j] += wi * row[i] * row[j]
    for i in range(P):
        M[i][i] += 1e-6   # tiny ridge for numerical safety
    aug = [M[i][:] + [b[i]] for i in range(P)]
    for col in range(P):
        piv = max(range(col, P), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        if abs(pv) < 1e-12:
            continue
        for r in range(P):
            if r != col:
                factor = aug[r][col] / pv
                for cc in range(col, P + 1):
                    aug[r][cc] -= factor * aug[col][cc]
    return [aug[i][P] / aug[i][i] if abs(aug[i][i]) > 1e-12 else 0.0 for i in range(P)]


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    C = int(next(it)); P = int(next(it)); H = int(next(it)); Lmax = int(next(it))
    A_full = int(next(it)); t = int(next(it))

    exposures = []
    ages = []
    mixes = []
    triangles = []
    for _ in range(C):
        exposure = float(next(it)); age = int(next(it))
        mix = [float(next(it)) for _ in range(P)]
        K = int(next(it))
        rvals = [float(next(it)) for _ in range(K)]
        exposures.append(exposure); ages.append(age); mixes.append(mix)
        triangles.append(rvals)

    m = [min(ages[c], A_full) for c in range(C)]

    # THE INSIGHT: separate the COHORT/MIX effect from the DEVELOPMENT effect
    # before projecting.  A cohort's per-unit-exposure ULTIMATE value is a
    # mix-weighted blend of per-type "ultimate value rates" that do NOT
    # depend on how far along development is -- so instead of extrapolating
    # each cohort's own (possibly barely-started) development curve, or
    # trusting an aggregate development-factor pattern that is silently
    # dominated by whichever mix happens to populate each age band (the
    # chain-ladder trap), read the per-type ultimate rates DIRECTLY off the
    # FULLY DEVELOPED cohorts (their reported total no longer needs any
    # development projection at all) via a mix-weighted regression, then
    # apply those rates to EVERY cohort through its OWN known mix.
    mature = [c for c in range(C) if m[c] >= A_full]

    if len(mature) >= P:
        A = [mixes[c] for c in mature]
        y = [triangles[c][-1] / exposures[c] for c in mature]
        w = [exposures[c] for c in mature]
        v_hat = solve_lstsq(A, y, w, P)
        v_hat = [max(0.0, x) for x in v_hat]
    else:
        # fallback if too few mature cohorts to identify all P types:
        # a single pooled ultimate-value-rate applied uniformly
        tot_val = sum(triangles[c][-1] for c in mature) if mature else sum(tr[-1] for tr in triangles)
        tot_exp = sum(exposures[c] for c in mature) if mature else sum(exposures)
        rate = tot_val / max(1e-9, tot_exp)
        v_hat = [rate] * P

    out = []
    for c in range(C):
        ultimate = exposures[c] * sum(mixes[c][k] * v_hat[k] for k in range(P))
        reserve = max(0.0, ultimate - triangles[c][-1])
        out.append("%.6f" % reserve)
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
