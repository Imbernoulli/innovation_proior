# TIER: strong
# INSIGHT (not "greedy + more probes"): split the three unknowns by TIMESCALE and spend
# budget INVERSELY to stability.
#   (a) w (the reception shape) is perfectly stable all episode -- it falls out for free
#       from a least-squares parabola fit to ANY clean local sweep, no dedicated budget needed.
#   (b) f0(t)'s drift is LINEAR -- the more stable of the two moving quantities.  THREE widely
#       time-separated anchor sweeps (t = 0, ~500, ~995), each locating the peak once via a
#       coarse-then-fine local search, are enough to regress a slope with confidence and
#       extrapolate it deep into the future window.
#   (c) A(t)'s drift is a SINUSOID of unknown period/phase (only a padded period hint is
#       public) and it can turn around inside the very window being predicted -- the LESS
#       stable quantity.  It gets the BULK of the remaining budget: cheap single-probe
#       readings, one per tick, dialled straight onto the just-fitted moving peak across many
#       ticks spread over the whole observed window, then a harmonic (cos/sin) least-squares
#       fit against a small grid of candidate periods drawn from the public hint picks out the
#       true cycle -- instead of ever assuming the trend seen so far continues.
# Prediction evaluates BOTH fitted models (line for f0, harmonic curve for A) at each held-out
# future tick and applies the closed-form reception shape -- never freezing either quantity at
# its last observed value.
import sys, json, math

inst = json.load(sys.stdin)
phase = inst.get("phase")
dial = inst["dial"]
X_LO, X_HI = dial["x_lo"], dial["x_hi"]
T_OBS_MAX = inst["t_obs_max"]
hints = inst["hints"]

T_ANCHOR0 = 0.0
T_ANCHOR1 = 500.0
T_ANCHOR2 = min(995.0, T_OBS_MAX)


def clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def kernel(u, w):
    z = u / w
    v = 1.0 - z * z
    return v if v > 0.0 else 0.0


def solve3(A, b):
    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    D = det3(A)
    if abs(D) < 1e-10:
        return None
    out = []
    for col in range(3):
        M = [row[:] for row in A]
        for r in range(3):
            M[r][col] = b[r]
        out.append(det3(M) / D)
    return out


def quad_fit(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    S0 = float(n); S1 = S2 = S3 = S4 = 0.0
    T0 = T1 = T2 = 0.0
    for x, y in zip(xs, ys):
        x2 = x * x
        S1 += x; S2 += x2; S3 += x2 * x; S4 += x2 * x2
        T0 += y; T1 += x * y; T2 += x2 * y
    sol = solve3([[S0, S1, S2], [S1, S2, S3], [S2, S3, S4]], [T0, T1, T2])
    return sol


def peak_fit_at(history, t0, tol=1e-6):
    pts = [(h["x"], h["y"]) for h in history if abs(h["t"] - t0) < tol]
    if len(pts) < 3:
        return None
    # Localize to the bump BEFORE fitting: a plain OLS parabola over points that span far
    # outside the bump (mostly true signal 0) is badly biased by the flat tails.  Keep only
    # points within one hinted half-width of the best raw reading, so the quadratic model
    # (which is only locally valid) sees points where it actually applies.
    x_best = max(pts, key=lambda p: p[1])[0]
    radius = 1.3 * hints["w_hi"]
    filt = [p for p in pts if abs(p[0] - x_best) <= radius]
    if len(filt) < 3:
        filt = pts
    xs = [p[0] for p in filt]
    ys = [p[1] for p in filt]
    fit = quad_fit(xs, ys)
    if fit is None:
        return None
    c0, c1, c2 = fit
    if c2 >= -1e-9:
        return None
    f0 = -c1 / (2.0 * c2)
    A = c0 + c1 * f0 + c2 * f0 * f0
    if A <= 0.0:
        return None
    w2 = A / (-c2)
    if w2 <= 0.25:
        return None
    return f0, A, math.sqrt(w2)


def default_w():
    return 0.5 * (hints["w_lo"] + hints["w_hi"])


def default_f0():
    return 0.5 * (hints["f0_0_lo"] + hints["f0_0_hi"])


def default_amid():
    return 0.5 * (hints["A_mid_lo"] + hints["A_mid_hi"])


def fit_line_and_w(history):
    """Fit (f0_0, drift) from every anchor whose local quadratic fit succeeded, plus a
    pooled w estimate.  Falls back to hint midpoints / zero drift piece-by-piece."""
    anchors = []
    for t0 in (T_ANCHOR0, T_ANCHOR1, T_ANCHOR2):
        pf = peak_fit_at(history, t0)
        if pf is not None:
            f0, A, w = pf
            anchors.append((t0, f0, A, w))
    w_hat = (sum(a[3] for a in anchors) / len(anchors)) if anchors else default_w()
    if not (w_hat > 0.5):
        w_hat = default_w()
    if len(anchors) >= 2:
        n = len(anchors)
        St = sum(a[0] for a in anchors); Sf = sum(a[1] for a in anchors)
        Stt = sum(a[0] * a[0] for a in anchors); Stf = sum(a[0] * a[1] for a in anchors)
        denom = n * Stt - St * St
        if abs(denom) > 1e-9:
            drift_hat = (n * Stf - St * Sf) / denom
            f0_0_hat = (Sf - drift_hat * St) / n
        else:
            f0_0_hat, drift_hat = anchors[0][1], 0.0
    elif len(anchors) == 1:
        f0_0_hat, drift_hat = anchors[0][1] - anchors[0][0] * 0.0, 0.0
        f0_0_hat = anchors[0][1]
    else:
        f0_0_hat, drift_hat = default_f0(), 0.0
    return f0_0_hat, drift_hat, w_hat, anchors


def fit_amplitude(history, f0_0_hat, drift_hat, w_hat):
    samples = []
    for h in history:
        f0p = f0_0_hat + drift_hat * h["t"]
        if abs(h["x"] - f0p) <= 0.35 * w_hat:
            samples.append((h["t"], h["y"]))
    if len(samples) < 4:
        return None
    plo, phi = hints["period_lo"], hints["period_hi"]
    if plo <= 1.0:
        plo = 1.0
    if phi <= plo:
        phi = plo + 1.0
    n_grid = 25
    best = None
    for i in range(n_grid):
        frac = i / (n_grid - 1) if n_grid > 1 else 0.5
        P = plo + (phi - plo) * frac
        w = 2.0 * math.pi / P
        Sc = Ss = Scc = Sss = Scs = 0.0
        Sy = Syc = Sys = 0.0
        n = float(len(samples))
        for t, y in samples:
            c = math.cos(w * t); s = math.sin(w * t)
            Sc += c; Ss += s; Scc += c * c; Sss += s * s; Scs += c * s
            Sy += y; Syc += y * c; Sys += y * s
        sol = solve3([[n, Sc, Ss], [Sc, Scc, Scs], [Ss, Scs, Sss]], [Sy, Syc, Sys])
        if sol is None:
            continue
        c0, a, b = sol
        sse = 0.0
        for t, y in samples:
            pred = c0 + a * math.cos(w * t) + b * math.sin(w * t)
            sse += (pred - y) ** 2
        if best is None or sse < best[0]:
            best = (sse, P, c0, a, b)
    return best


def predict_amplitude(amp_fit, t):
    if amp_fit is None:
        return default_amid()
    _, P, c0, a, b = amp_fit
    w = 2.0 * math.pi / P
    val = c0 + a * math.cos(w * t) + b * math.sin(w * t)
    return val if val > 0.0 else 0.0


if phase == "query":
    rnd = int(inst.get("round", 0))
    budget = int(inst["budget_left"])
    w_hi = hints["w_hi"]
    max_drift = max(abs(hints["drift_lo"]), abs(hints["drift_hi"]))

    if rnd == 0:
        lo0 = clip(hints["f0_0_lo"] - w_hi, X_LO, X_HI)
        hi0 = clip(hints["f0_0_hi"] + w_hi, X_LO, X_HI)
        lo1 = clip(hints["f0_0_lo"] - w_hi - max_drift * T_ANCHOR1, X_LO, X_HI)
        hi1 = clip(hints["f0_0_hi"] + w_hi + max_drift * T_ANCHOR1, X_LO, X_HI)
        probes = []
        n0 = 8
        for i in range(n0):
            t = i / (n0 - 1) if n0 > 1 else 0.5
            probes.append({"x": lo0 + (hi0 - lo0) * t, "t": T_ANCHOR0})
        n1 = 8
        for i in range(n1):
            t = i / (n1 - 1) if n1 > 1 else 0.5
            probes.append({"x": lo1 + (hi1 - lo1) * t, "t": T_ANCHOR1})
        print(json.dumps({"probes": probes[:budget]}))

    elif rnd == 1:
        history = inst.get("history", [])
        pts0 = [(h["x"], h["y"]) for h in history if abs(h["t"] - T_ANCHOR0) < 1e-6]
        pts1 = [(h["x"], h["y"]) for h in history if abs(h["t"] - T_ANCHOR1) < 1e-6]
        c0 = max(pts0, key=lambda p: p[1])[0] if pts0 else default_f0()
        c1 = max(pts1, key=lambda p: p[1])[0] if pts1 else default_f0()
        probes = []
        n_fine = 4
        for i in range(n_fine):
            t = i / (n_fine - 1) if n_fine > 1 else 0.5
            probes.append({"x": clip(c0 + w_hi * (2.0 * t - 1.0), X_LO, X_HI), "t": T_ANCHOR0})
        for i in range(n_fine):
            t = i / (n_fine - 1) if n_fine > 1 else 0.5
            probes.append({"x": clip(c1 + w_hi * (2.0 * t - 1.0), X_LO, X_HI), "t": T_ANCHOR1})
        drift_rough = (c1 - c0) / T_ANCHOR1 if T_ANCHOR1 > 0 else 0.0
        pred2 = c0 + drift_rough * T_ANCHOR2
        radius2 = 3.0 * w_hi
        lo2 = clip(pred2 - radius2, X_LO, X_HI)
        hi2 = clip(pred2 + radius2, X_LO, X_HI)
        n2 = 6
        for i in range(n2):
            t = i / (n2 - 1) if n2 > 1 else 0.5
            probes.append({"x": lo2 + (hi2 - lo2) * t, "t": T_ANCHOR2})
        print(json.dumps({"probes": probes[:budget]}))

    else:
        history = inst.get("history", [])
        f0_0_hat, drift_hat, w_hat, _ = fit_line_and_w(history)
        n_track = budget
        probes = []
        if n_track > 0:
            span_hi = T_ANCHOR2
            for i in range(n_track):
                t = i / (n_track - 1) if n_track > 1 else 0.5
                tk = span_hi * t
                xk = clip(f0_0_hat + drift_hat * tk, X_LO, X_HI)
                probes.append({"x": xk, "t": tk})
        print(json.dumps({"probes": probes[:budget]}))

else:
    history = inst.get("history", [])
    f0_0_hat, drift_hat, w_hat, _ = fit_line_and_w(history)
    amp_fit = fit_amplitude(history, f0_0_hat, drift_hat, w_hat)

    preds = []
    for q in inst["test_queries"]:
        t = q["t"]; x = q["x"]
        f0p = f0_0_hat + drift_hat * t
        Ap = predict_amplitude(amp_fit, t)
        preds.append(Ap * kernel(x - f0p, w_hat))
    print(json.dumps({"predictions": preds}))
