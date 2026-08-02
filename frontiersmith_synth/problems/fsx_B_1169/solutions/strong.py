# TIER: strong
# The insight: deconvolution amplifies noise exactly where the kernel's
# response is small, so a full-resolution unregularised inverse fit is only
# as good as the calibration burst can support -- past that it just injects
# noise. Instead of maximising in-sample sharpness:
#   1. ESTIMATE the kernel's usable support from the calibration
#      cross-correlation corr[j] = sum_n x_cal[n]*y_cal[n+j]. For
#      well-separated spikes this recovers (a noisy copy of) h[j]*energy;
#      it decays to the noise floor once |j| exceeds the true kernel
#      half-width. Find where it drops into that floor and STOP there,
#      instead of assuming the full radius-7 window is signal.
#   2. Fit a RIDGE-regularised filter restricted to that estimated support,
#      choosing the ridge strength itself by a train/validation split of the
#      calibration burst (pick the strength that generalises within the
#      burst, not the one that fits it best).
# This trades a little in-sample sharpness for a filter that survives being
# rolled onto fresh noise.
import sys
import numpy as np

R = 7
TAPS = list(range(-R, R + 1))


def tap_name(j):
    if j < 0:
        return "ym%d" % (-j)
    if j == 0:
        return "y0"
    return "yp%d" % j


def build_rows(y_cal, x_cal, lo, hi, taps):
    rows, targets = [], []
    for i in range(lo, hi):
        rows.append([y_cal[i + j] for j in taps])
        targets.append(x_cal[i])
    return np.array(rows, dtype=float), np.array(targets, dtype=float)


def ridge_fit(A, b, lam):
    p = A.shape[1]
    ATA = A.T @ A + lam * np.eye(p)
    ATb = A.T @ b
    try:
        return np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def main():
    data = sys.stdin.read().split()
    if len(data) < 3:
        print("0"); return
    ncal = int(data[0])
    vals = data[3:]
    x_cal = [0.0] * ncal
    y_cal = [0.0] * ncal
    for i in range(ncal):
        x_cal[i] = float(vals[2 * i])
        y_cal[i] = float(vals[2 * i + 1])

    lo, hi = R, ncal - R
    if hi <= lo:
        print("0"); return

    # ---- 1. estimate the kernel's usable support via cross-correlation ----
    energy = sum(v * v for v in x_cal)
    corr = {}
    for j in TAPS:
        s = 0.0
        for n in range(ncal):
            m = n + j
            if 0 <= m < ncal:
                s += x_cal[n] * y_cal[m]
        corr[j] = s
    h_hat = {j: (corr[j] / energy if energy > 1e-9 else 0.0) for j in TAPS}

    tail = [abs(h_hat[j]) for j in (R, -R, R - 1, -(R - 1)) if j in h_hat]
    tail.sort()
    floor = tail[len(tail) // 2] if tail else 0.0
    center = abs(h_hat[0])
    threshold = max(4.0 * floor, 0.03 * center, 1e-3)

    w_hat = 0
    for j in range(1, R + 1):
        if max(abs(h_hat[j]), abs(h_hat[-j])) > threshold:
            w_hat = j
        else:
            break
    w_hat = max(w_hat, 1)
    support = list(range(-w_hat, w_hat + 1))

    # ---- 2. ridge-regularised fit restricted to that support; pick lambda
    #          by a validation split of the calibration burst ----
    A, b = build_rows(y_cal, x_cal, lo, hi, support)
    n_rows = A.shape[0]
    if n_rows < 6:
        # too little calibration data to split -- fall back to a firmly
        # regularised fit on everything (still far safer than raw OLS).
        coef = ridge_fit(A, b, lam=max(1.0, 0.5 * n_rows))
    else:
        split = max(3, n_rows // 2)
        idx = list(range(n_rows))
        fit_idx, val_idx = idx[:split], idx[split:]
        Af, bf = A[fit_idx], b[fit_idx]
        Av, bv = A[val_idx], b[val_idx]
        var_b = float(np.mean(bf ** 2)) if len(bf) else 1.0
        candidates = [c * max(var_b, 1e-6) for c in (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)]
        best_lam, best_err = candidates[0], None
        for lam in candidates:
            c = ridge_fit(Af, bf, lam)
            if len(bv):
                err = float(np.mean((Av @ c - bv) ** 2))
            else:
                err = float(np.mean((Af @ c - bf) ** 2))
            if best_err is None or err < best_err:
                best_err, best_lam = err, lam
        coef = ridge_fit(A, b, best_lam)

    terms = []
    for j, c in zip(support, coef):
        if abs(float(c)) < 1e-5:
            continue
        terms.append("%.6f * %s" % (float(c), tap_name(j)))
    if not terms:
        terms = ["0"]
    print(" + ".join(terms))


if __name__ == "__main__":
    main()
