# TIER: greedy
# The obvious recipe an average strong coder writes first: "scan for the station, zoom in, fit
# its shape precisely" -- and then treat that ONE fit as the whole answer.  It spends the
# ENTIRE probe budget near the very start of the observed window (t == 0): a coarse sweep
# across the public frequency-hint window to locate the station, a fine sweep around the peak
# to nail the shape via a proper least-squares parabola fit (this part is genuinely
# competent -- it reads w, f0(0), A(0) accurately), and a repeat pass at the same fine
# locations to average down noise. It never revisits a later tick, so it has no way to notice
# either drift: it just holds the t=0 snapshot fixed and answers every future query with it.
# That is exactly right when drift is tiny, and badly wrong once the frequency has crept far
# from its t=0 spot and/or the loudness cycle has turned around -- which is exactly where the
# trap instances concentrate their held-out queries.
import sys, json

inst = json.load(sys.stdin)
phase = inst.get("phase")


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
    if abs(D) < 1e-12:
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
    S0 = n; S1 = S2 = S3 = S4 = 0.0
    T0 = T1 = T2 = 0.0
    for x, y in zip(xs, ys):
        x2 = x * x
        S1 += x; S2 += x2; S3 += x2 * x; S4 += x2 * x2
        T0 += y; T1 += x * y; T2 += x2 * y
    A = [[S0, S1, S2], [S1, S2, S3], [S2, S3, S4]]
    b = [T0, T1, T2]
    return solve3(A, b)


if phase == "query":
    rnd = int(inst.get("round", 0))
    budget = int(inst["budget_left"])
    if rnd == 0:
        hints = inst["hints"]
        w_hi = hints["w_hi"]
        lo = hints["f0_0_lo"] - w_hi
        hi = hints["f0_0_hi"] + w_hi
        lo = max(lo, inst["dial"]["x_lo"]); hi = min(hi, inst["dial"]["x_hi"])
        n_coarse = min(15, budget)
        exps = []
        for i in range(n_coarse):
            t = i / (n_coarse - 1) if n_coarse > 1 else 0.5
            exps.append({"x": lo + (hi - lo) * t, "t": 0.0})
        remaining = budget - len(exps)
        # fine + repeat sweep centred on the middle of the coarse window (best-effort guess of
        # where the peak will land -- refined for real once the coarse readings come back, but
        # this candidate never looks at round-0 history again, it only ever probes at t=0)
        centre = 0.5 * (lo + hi)
        radius = w_hi
        n_fine = remaining // 2
        for i in range(n_fine):
            t = i / (n_fine - 1) if n_fine > 1 else 0.5
            x = centre + radius * (2.0 * t - 1.0)
            exps.append({"x": x, "t": 0.0})
        for i in range(remaining - n_fine):
            t = i / (n_fine - 1) if n_fine > 1 else 0.5
            x = centre + radius * (2.0 * t - 1.0)
            exps.append({"x": x, "t": 0.0})
        print(json.dumps({"probes": exps[:budget]}))
    else:
        print(json.dumps({"probes": []}))
else:
    hist = inst.get("history", [])
    # average duplicate x-locations to reduce noise, then fit ONE parabola at t=0
    buckets = {}
    for h in hist:
        key = round(h["x"], 2)
        buckets.setdefault(key, []).append(h["y"])
    xs = []
    ys = []
    for k, vals in buckets.items():
        xs.append(k)
        ys.append(sum(vals) / len(vals))
    fit = quad_fit(xs, ys)
    hints = inst["hints"]
    if fit is None or fit[2] >= -1e-9:
        # fall back to public hint midpoints if the fit is degenerate
        f0_hat = 0.5 * (hints["f0_0_lo"] + hints["f0_0_hi"])
        A_hat = 0.5 * (hints["A_mid_lo"] + hints["A_mid_hi"])
        w_hat = 0.5 * (hints["w_lo"] + hints["w_hi"])
    else:
        c0, c1, c2 = fit
        f0_hat = -c1 / (2.0 * c2)
        A_hat = c0 + c1 * f0_hat + c2 * f0_hat * f0_hat
        if A_hat <= 0.0:
            A_hat = 0.5 * (hints["A_mid_lo"] + hints["A_mid_hi"])
        w_hat = (A_hat / (-c2)) ** 0.5 if A_hat > 0.0 else 0.5 * (hints["w_lo"] + hints["w_hi"])
        if not (w_hat > 0.5):
            w_hat = 0.5 * (hints["w_lo"] + hints["w_hi"])

    preds = []
    for q in inst["test_queries"]:
        preds.append(A_hat * kernel(q["x"] - f0_hat, w_hat))     # STATIC snapshot, no drift
    print(json.dumps({"predictions": preds}))
