# TIER: strong
# The insight: training residuals of a memoryless rain fit are small but not
# structureless -- around the few brief cold snaps, the noisy snow_proxy
# column visibly RISES while temp is low and FALLS afterward at a rate that
# scales with how far above freezing it gets. That is the signature of a
# persisting storage state, not a curve of today's weather. Recover the
# freeze threshold and melt rate by finding the (Tf, k) pair whose rolled-out
# bucket reconstruction BEST FITS the whole noisy training proxy trace (not
# just a one-tick direction vote), then use the recovered rule's ACTUAL
# per-tick melt (not the storage level) as a regressor for flow. Emit a
# stateful DSL program: a STORE rule that accumulates precip-as-snow below the
# recovered threshold and melts it above, plus an OUT expression that reads
# the realized melt straight from the register via the exact bucket identity
# melt[t] = snow_in[t] - (SW[t] - SW[t-1]). Unlike any finite lag window, the
# register persists across the whole held-out accumulation phase, so it keeps
# carrying the stored water forward to where it actually drives the
# melt-season flow -- including the long dry stretch AFTER the snow finally
# runs out, where a temperature-only fit keeps predicting melt that isn't
# there anymore.
import sys


def solve_ls(X, y, ridge=1e-6):
    m = len(X[0])
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for row, yv in zip(X, y):
        for i in range(m):
            b[i] += row[i] * yv
            for j in range(m):
                A[i][j] += row[i] * row[j]
    for i in range(m):
        A[i][i] += ridge
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-12:
            continue
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        pv = A[col][col]
        for j in range(col, m):
            A[col][j] /= pv
        b[col] /= pv
        for r in range(m):
            if r == col:
                continue
            f = A[r][col]
            if f == 0.0:
                continue
            for j in range(col, m):
                A[r][j] -= f * A[col][j]
            b[r] -= f * b[col]
    return b


def estimate_kmelt_given(temp, proxy, n, Tf):
    num, den = 0.0, 0.0
    for t in range(1, n):
        if temp[t] < Tf:
            continue
        if proxy[t - 1] < 0.05:
            continue
        dp = proxy[t] - proxy[t - 1]
        if dp >= 0:
            continue
        heat = temp[t] - Tf
        num += (-dp) * heat
        den += heat * heat
    if den < 1e-9:
        return 0.4
    return min(0.9, max(0.05, num / den))


def reconstruct_sw_unclamped(precip, temp, Tf, k_melt, n):
    """Un-clamped-at-0 bucket trace used only to COMPARE against the noisy
    proxy sensor (which can dip slightly negative from noise); the real
    grading rollout clamps at 0/CAP, this is purely for calibration."""
    sw = 0.0
    trace = []
    for t in range(n):
        snow_in = precip[t] if temp[t] < Tf else 0.0
        melt_pot = k_melt * (temp[t] - Tf) if temp[t] > Tf else 0.0
        melt = min(sw, melt_pot)
        sw = sw + snow_in - melt
        trace.append(sw)
    return trace


def estimate_Tf_kmelt(precip, temp, proxy, n):
    """Joint recovery: for each candidate freeze threshold, fit the melt rate
    by least squares against the proxy's observed falls, then roll a
    candidate storage trace forward and score it by how well it reconstructs
    the WHOLE noisy proxy trace (not just a one-tick direction vote) -- pick
    the (Tf, k) pair whose reconstruction fits the sensor best."""
    best = None
    best_tk = (0.0, 0.4)
    for i in range(-30, 31):
        Tf = i / 100.0
        k = estimate_kmelt_given(temp, proxy, n, Tf)
        trace = reconstruct_sw_unclamped(precip, temp, Tf, k, n)
        err = sum((trace[t] - proxy[t]) ** 2 for t in range(n)) / n
        if best is None or err < best:
            best = err
            best_tk = (Tf, k)
    return best_tk


CAP = 8.0


def roll_store(precip, temp, Tf, k_melt, n):
    """Grading-faithful rollout (clamped at 0/CAP): returns (sw_trace,
    melt_trace) where melt_trace[t] is the ACTUAL melt consumed this tick."""
    sw = 0.0
    sw_trace = []
    melt_trace = []
    for t in range(n):
        snow_in = precip[t] if temp[t] < Tf else 0.0
        melt_pot = k_melt * (temp[t] - Tf) if temp[t] > Tf else 0.0
        melt = min(sw, melt_pot)
        sw = min(CAP, max(0.0, sw + snow_in - melt))
        sw_trace.append(sw)
        melt_trace.append(melt)
    return sw_trace, melt_trace


def main():
    data = sys.stdin.read().split()
    if not data:
        print("OUT 0.3"); return
    n = int(data[0])
    vals = data[2:]
    p = [0.0] * n
    tm = [0.0] * n
    y = [0.0] * n
    proxy = [0.0] * n
    for i in range(n):
        p[i] = float(vals[4 * i])
        tm[i] = float(vals[4 * i + 1])
        y[i] = float(vals[4 * i + 2])
        proxy[i] = float(vals[4 * i + 3])

    Tf, k_melt = estimate_Tf_kmelt(p, tm, proxy, n)
    sw_hat, melt_hat = roll_store(p, tm, Tf, k_melt, n)

    W = 5
    X = []
    for t in range(n):
        lag_avg = sum(p[t - j] if t - j >= 0 else 0.0 for j in range(1, W + 1)) / W
        rain_in = p[t] if tm[t] >= Tf else 0.0
        X.append([1.0, rain_in, rain_in * lag_avg, melt_hat[t]])
    w = solve_ls(X, y)

    lag_terms = "( pk1 + pk2 + pk3 + pk4 + pk5 ) / 5.0"
    rain_expr = "( p * step ( tm - %.6f ) )" % Tf
    snow_expr = "p * step ( %.6f - tm )" % Tf
    # melt[t] = snow_in[t] - (SW[t] - SWk1[t])  -- exact bucket identity that
    # reads the realized melt straight from the register (no min() needed:
    # the grader's own clip at 0/CAP already enforces "can't melt more than
    # is stored").
    melt_expr = "( %s - ( SW - SWk1 ) )" % snow_expr

    print("STORE %s - %.6f * relu ( tm - %.6f )" % (snow_expr, k_melt, Tf))
    print("OUT   %.6f + %.6f * %s + %.6f * ( %s * ( %s ) ) + %.6f * %s"
          % (w[0], w[1], rain_expr, w[2], rain_expr, lag_terms, w[3], melt_expr))


if __name__ == "__main__":
    main()
