# TIER: strong
"""
The bunching-estimator insight: don't regress hours against wage at all -- read the
mechanism out of the ATOM in the pretax-income histogram z = wage*hours.

1. Every worker's z = w*h is computed. A histogram of z has one sharp spike: the
   pile-up of workers who all located at the same notch threshold z0 (a genuine
   point mass, not a curve). Region-A ("smooth") workers spread z continuously
   across a wide range instead. The spike location IS z0 -- no curve-fitting needed.
2. tau_lo is read off the smooth region (through-origin fit of h on w, since the
   true relation there is exactly h = w*(1-tau_lo)).
3. Every bunched worker's wage lies in [w0_lo, w_star] (w0_lo = boundary of the
   smooth region; w_star = the wage of the marginal buncher, who is exactly
   indifferent between the notch and pushing into the upper bracket). Wages are
   uniform on the KNOWN archive window [W_LO, W_HI], so the spike's SIZE (fraction
   of the archived population that bunched) directly locates w_star -- even though
   not a single worker with wage above w_star was ever archived.
4. The given surcharge dT then pins the ONE remaining unknown -- the upper-bracket
   marginal rate tau_hi -- via the exact indifference condition of the marginal
   buncher, in closed form. No held-out data, and no curve through it, is ever used.
"""
import sys


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    N = int(data[idx]); idx += 1
    W_LO = float(data[idx]); idx += 1
    W_HI = float(data[idx]); idx += 1
    dT = float(data[idx]); idx += 1
    ws, hs = [], []
    for _ in range(N):
        w = float(data[idx]); idx += 1
        h = float(data[idx]); idx += 1
        ws.append(w); hs.append(h)

    zs = [w * h for w, h in zip(ws, hs)]

    # --- 1. locate the atom via a histogram of z ---
    zmin, zmax = min(zs), max(zs)
    span = max(zmax - zmin, 1e-9)
    nbins = 300
    bins = [0] * nbins
    bin_idx = []
    for z in zs:
        bi = int((z - zmin) / span * nbins)
        bi = min(max(bi, 0), nbins - 1)
        bins[bi] += 1
        bin_idx.append(bi)
    peak_bin = max(range(nbins), key=lambda i: bins[i])
    lo_bin = max(peak_bin - 1, 0); hi_bin = min(peak_bin + 1, nbins - 1)
    cluster0 = [z for z, bi in zip(zs, bin_idx) if lo_bin <= bi <= hi_bin]
    if len(cluster0) < 5:
        cluster0 = [z for z, bi in zip(zs, bin_idx) if bi == peak_bin] or [zmax]
    z0_hat = sum(cluster0) / len(cluster0)
    if len(cluster0) > 1:
        var = sum((z - z0_hat) ** 2 for z in cluster0) / len(cluster0)
        noise_std = var ** 0.5
    else:
        noise_std = 0.02 * z0_hat
    tol = max(6.0 * noise_std, 0.01 * z0_hat)

    region_B_z = [z for z in zs if abs(z - z0_hat) <= tol]
    z0_hat = sum(region_B_z) / len(region_B_z) if region_B_z else z0_hat
    region_B_mask = [abs(z - z0_hat) <= tol for z in zs]
    B_frac = sum(region_B_mask) / N

    # --- 2. tau_lo from the smooth (non-bunched) region, through-origin fit ---
    sw2 = sum(w * w for w, m in zip(ws, region_B_mask) if not m)
    swh = sum(w * h for w, h, m in zip(ws, hs, region_B_mask) if not m)
    if sw2 > 1e-9:
        slope_lo = swh / sw2
    else:
        slope_lo = 1.0 - 0.18
    tau_lo = min(max(1.0 - slope_lo, 0.0), 0.90)

    # --- 3. bunching-mass fraction locates the marginal buncher's wage w_star ---
    w0_lo = (z0_hat / max(1.0 - tau_lo, 1e-6)) ** 0.5
    B_frac = min(max(B_frac, 1e-6), 0.95)
    w_star = w0_lo + B_frac * (W_HI - W_LO)
    w_star = max(w_star, w0_lo + 1e-6)

    # --- 4. closed-form marginal-buncher indifference: solve for tau_hi ---
    # dT = (1-tau_hi)*(w_star^2*(1-tau_hi)-z0)^2 / (2*w_star^2*(1-tau_hi))
    #    => (1-tau_hi) = (z0 + w_star*sqrt(2*dT)) / w_star^2
    tau_hi = 1.0 - (z0_hat + w_star * (2.0 * max(dT, 0.0)) ** 0.5) / (w_star * w_star)
    tau_hi = min(max(tau_hi, tau_lo), 0.97)

    print("%.6f * min(z, %.6f) + %.6f * max(z - %.6f, 0) + %.6f * (z >= %.6f)"
          % (tau_lo, z0_hat, tau_hi, z0_hat, dT, z0_hat))


if __name__ == "__main__":
    main()
