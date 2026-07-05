import sys, json, math, random, isorun

# ==========================================================================
# fsx_G_0466 -- ml-timeseries-forecaster (Format B, isolated candidate)
# Theme: "energy demand" -- forecast the hourly electricity load of a
# synthetic grid over a held-out horizon. Each instance is a deterministic
# synthetic load series with a trend, a double-peak daily profile, a
# weekday/weekend calendar effect, a slow "temperature" swing and noise.
#
# The candidate sees ONLY the public history (values + season period +
# horizon length). It must emit a point forecast for the next `horizon`
# hours. The evaluator computes the MASE (mean absolute scaled error) of the
# forecast against the HIDDEN actuals, scaled by the in-sample seasonal-naive
# MAE. Objective: MINIMIZE MASE. A malformed / non-finite forecast scores 0.
#
# Normalization uses the seasonal-naive forecaster as the baseline: a
# candidate that merely repeats the last observed day scores ~0.1; beating
# seasonal-naive (by modelling trend + calendar + slow drift) scores higher.
# ==========================================================================


def _gen(seed, n, H, p):
    """Deterministic synthetic hourly energy-demand series of length n+H."""
    rng = random.Random(seed)
    m = 24                                   # daily season
    total = n + H
    y = []
    for t in range(total):
        # double-peak daily profile (morning + evening) via two harmonics
        daily = (p["a1"] * math.cos(2 * math.pi * t / m)
                 + p["b1"] * math.sin(2 * math.pi * t / m)
                 + p["a2"] * math.cos(2 * math.pi * 2 * t / m)
                 + p["b2"] * math.sin(2 * math.pi * 2 * t / m))
        dow = (t // m) % 7                    # 0..6, 5&6 = weekend
        weekend = p["weekend"] if dow >= 5 else 0.0
        temp = p["temp"] * math.sin(2 * math.pi * t / p["temp_period"] + p["temp_phase"])
        trend = p["slope"] * t
        val = p["level"] + trend + daily + weekend + temp + rng.gauss(0.0, p["noise"])
        y.append(val)
    return y[:n], y[n:]


def make_instances():
    m = 24
    # (seed, days_of_history, horizon_hours, params). A spread of difficulty:
    # some low-noise & strongly structured (learnable), some noisy/held-out.
    specs = [
        (7001, 18, 48, dict(level=520, slope=0.010, noise=6.0,  weekend=-40, temp=25, temp_period=300, temp_phase=0.4, a1=-70, b1=-30, a2=25, b2=15)),
        (7002, 20, 48, dict(level=610, slope=0.020, noise=9.0,  weekend=-55, temp=35, temp_period=400, temp_phase=1.1, a1=-90, b1=-20, a2=30, b2=-10)),
        (7003, 21, 48, dict(level=480, slope=-0.008, noise=5.0, weekend=-30, temp=20, temp_period=260, temp_phase=2.0, a1=-60, b1=-40, a2=20, b2=20)),
        (7004, 16, 24, dict(level=700, slope=0.015, noise=12.0, weekend=-70, temp=45, temp_period=350, temp_phase=0.0, a1=-100, b1=-25, a2=35, b2=-20)),
        (7005, 22, 48, dict(level=560, slope=0.005, noise=7.0,  weekend=-45, temp=30, temp_period=320, temp_phase=1.6, a1=-80, b1=-35, a2=28, b2=12)),
        (7006, 19, 48, dict(level=640, slope=0.025, noise=14.0, weekend=-60, temp=40, temp_period=280, temp_phase=0.8, a1=-95, b1=-15, a2=22, b2=-18)),
        (7007, 21, 48, dict(level=500, slope=-0.012, noise=8.0, weekend=-35, temp=28, temp_period=360, temp_phase=2.4, a1=-65, b1=-45, a2=26, b2=8)),
        (7008, 17, 24, dict(level=680, slope=0.018, noise=16.0, weekend=-80, temp=50, temp_period=300, temp_phase=1.3, a1=-110, b1=-30, a2=40, b2=-25)),
        (7009, 23, 48, dict(level=590, slope=0.008, noise=6.5,  weekend=-50, temp=32, temp_period=340, temp_phase=0.6, a1=-85, b1=-28, a2=24, b2=16)),
        (7010, 20, 48, dict(level=630, slope=0.022, noise=18.0, weekend=-65, temp=42, temp_period=290, temp_phase=1.9, a1=-100, b1=-22, a2=33, b2=-14)),
    ]
    out = []
    for seed, days, H, p in specs:
        n = days * m
        hist, actual = _gen(seed, n, H, p)
        pub = {"y": [round(v, 6) for v in hist], "period": m, "horizon": H}
        out.append({"public": pub, "hidden": {"actual": actual}})
    return out


def _scale(hist, m):
    """In-sample seasonal-naive MAE (the MASE denominator)."""
    diffs = [abs(hist[t] - hist[t - m]) for t in range(m, len(hist))]
    d = sum(diffs) / len(diffs)
    return d if d > 1e-9 else 1e-9


def _seasonal_naive(hist, m, H):
    n = len(hist)
    return [hist[n - m + (i % m)] for i in range(H)]


def _mase(forecast, actual, denom):
    mae = sum(abs(f - a) for f, a in zip(forecast, actual)) / len(actual)
    return mae / denom


def baseline(inst):
    """MASE of the seasonal-naive forecaster (repeat the last observed day)."""
    pub, hid = inst["public"], inst["hidden"]
    hist, m, H = pub["y"], pub["period"], pub["horizon"]
    denom = _scale(hist, m)
    fc = _seasonal_naive(hist, m, H)
    return _mase(fc, hid["actual"], denom)


def score(inst, ans):
    pub, hid = inst["public"], inst["hidden"]
    hist, m, H = pub["y"], pub["period"], pub["horizon"]
    actual = hid["actual"]
    if not isinstance(ans, dict) or "forecast" not in ans:
        return False, 0.0
    fc = ans["forecast"]
    if not isinstance(fc, list) or len(fc) != H:
        return False, 0.0
    clean = []
    for v in fc:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False, 0.0
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            return False, 0.0
        clean.append(v)
    denom = _scale(hist, m)
    obj = _mase(clean, actual, denom)
    if obj != obj or obj < 0.0 or obj in (float("inf"),):
        return False, 0.0
    return True, obj


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, stt = isorun.run_candidate(cand, inst["public"], timeout=20)
        if stt != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
