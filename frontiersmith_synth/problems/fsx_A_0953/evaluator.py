#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0953 -- "Tuning a Drifting Pirate Radio"
(family: drift-tracking-probes; format B, quality-metric; theme: tuning a drifting pirate
radio).

THEME.  A pirate radio station broadcasts a signal whose strength, as a function of your dial
setting x (a real-valued tuning offset) and the current tick t, is

    y(x, t) = A(t) * K( x - f0(t) ; w )                (+ small observation noise when probed)

K(u; w) = max(0, 1 - (u/w)^2)   -- a parabolic "reception bump" of half-width w, centred at 0,
peak value 1.  This SHAPE (the half-width w) is fixed for the whole episode -- the receiver's
own selectivity never changes.  What DOES change, deterministically and on two different
timescales, is where the station sits on the dial and how loud it is:
    f0(t) = f0_0 + drift * t                                    -- LINEAR frequency creep
            (a slow, steady thermal drift of the transmitter's oscillator)
    A(t)  = max(0, A_mid + A_amp * sin(2*pi*t/period + phase))  -- SLOW SINUSOIDAL loudness
            (a diurnal ionospheric fade, period long relative to the observation window)

You may spend a fixed budget of PROBES: pick a dial setting x and a tick t (t must lie in the
past/observed window [0, T_OBS_MAX]) and read back a noisy y.  Afterwards you must PREDICT y
for a frozen suite of (x, t) queries in the FUTURE window [T_FUT_LO, T_FUT_HI] -- strictly
beyond every tick you were allowed to probe, i.e. genuine extrapolation through the drift.

WHY BOTH MECHANISMS ARE FORCED (both shape the score).
  * probe-scheduling -- each probe is a (dial, tick) pair spent from one shared budget; a
    probe's worth is which unknown it resolves, not just "more data".  Probing the wrong dial
    setting (off the moving peak) or the wrong tick (redundant with what you already know)
    wastes budget outright.
  * slow-parameter-drift -- f0(t) and A(t) are BOTH moving, on different footings: f0 is a
    simple straight line (cheap to pin down from a couple of well-separated anchors), while
    A(t) is a bounded sinusoid whose direction can REVERSE inside the very window you must
    predict -- a straight-line reading of "loudness has been rising" during the observed window
    can be exactly backwards by the time the future window arrives.

INNOVATION HOOK (what `strong` exploits): split the unknowns by timescale and spend budget
INVERSELY to stability.
  (a) w (the FORM) is perfectly stable across the whole run -- it falls out for free the moment
      you fit a local parabola to any one clean sweep, so it costs nothing extra.
  (b) f0(t)'s drift is the MORE stable of the two moving quantities (a straight line): a
      handful of widely time-separated anchor sweeps, each locating the peak once, is enough
      to regress a slope and extrapolate it with confidence.
  (c) A(t)'s drift is the LESS stable one (unknown period/phase inside a public hint window,
      and it can turn around): it is given the BULK of the remaining budget, spread thinly
      across many ticks (one probe per tick suffices once f0(t) is known well enough to dial
      straight onto the peak), then fit with a harmonic (cos/sin) regression over a small grid
      of candidate periods drawn from the public hint.
The TRAP instances place the sinusoid's turning point INSIDE the future query window with a
large drift rate, so a strategy that fits everything ONCE near t=0 and never revisits time --
the natural first attempt -- reads the shape and the instantaneous frequency/loudness
correctly, but then extrapolates a STATIC snapshot forward and is blindsided by both the drift
and the reversal, exactly where the held-out queries concentrate.

CANDIDATE CONTRACT (isolated stdin -> stdout program, called ONCE PER ROUND).
  stdin : ONE JSON object.  Common fields:
      {"phase","dial":{"x_lo","x_hi"},"t_obs_max",
       "hints":{"w_lo","w_hi","f0_0_lo","f0_0_hi","drift_lo","drift_hi",
                "A_mid_lo","A_mid_hi","A_amp_lo","A_amp_hi","period_lo","period_hi"},
       "noise_std","budget":{"max_probes"},"budget_left","round","R","max_this_round",
       "history":[{"x","t","y"}, ...]}
    phase == "query"   -> return {"probes": [{"x":x,"t":t}, ...]}  (finite floats; x must lie
                          in [x_lo,x_hi] and t in [0,t_obs_max] or the entry is SILENTLY
                          SKIPPED -- charged against budget, no reading; a malformed entry
                          (wrong type / non-finite) -> 0.0 on the WHOLE instance; at most
                          min(budget_left, max_this_round) entries are honored per round)
    phase == "predict" -> stdin additionally has "test_queries":[{"x","t"}, ...] (frozen,
                          strictly in the future window).  Return
                          {"predictions":[p_0, ..., p_{K-1}]}, one finite real per query, same
                          order.  Wrong length/type/non-finite -> 0.0.
  Any crash / timeout / non-JSON / wrong shape on ANY call -> 0.0 on that instance.

SCORING (deterministic; no wall-time).  Per instance, with true signal y_true(x,t):
    err_ref = mean_j |naive(test_queries[j]) - y_true(test_queries[j])|
              naive(x,t) = A_mid_guess * K(x - f0_guess; w_guess), CONSTANT in t, using only
              the midpoints of the public hint windows -- exactly what predicting with ZERO
              probes reproduces.
    err     = mean_j |pred_j - y_true(test_queries[j])|
    quality = clip(1 - err/err_ref, 0, 1)
    r       = OFFSET + SPAN * quality      # OFFSET=0.10, SPAN=0.82 (cap 0.92, headroom open)
  Final score = mean of r over 10 fixed seeded instances (4 traps + 3 moderate + 3 gentle).

ISOLATION.  The candidate runs in a FRESH sandboxed subprocess via isorun.run_candidate; it
sees only the public fields above.  w, f0_0, drift, A_mid, A_amp, period, phase and the true
y_true live only in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
"""
import sys, json, math
import isorun

# --------------------------- scoring / protocol constants ------------------------
OFFSET = 0.10
SPAN = 0.82                  # r = OFFSET + SPAN*quality ; max attainable r = 0.92 (headroom)
X_LO, X_HI = -95.0, 95.0     # dial range
T_OBS_MAX = 999.0            # ticks [0, T_OBS_MAX] are probeable ("the past")
T_FUT_LO, T_FUT_HI = 1000.0, 1500.0   # frozen query window ("the future") -- strictly later
N_MAX = 45                   # probe budget
R = 3                        # adaptive query rounds
MAX_PER_ROUND = N_MAX        # a round may spend up to the whole remaining budget
K_TEST = 48                  # frozen held-out query suite size
NOISE_STD = 0.35             # public, small observation noise on probe readings


# ------------------------------- deterministic RNG --------------------------------
MASK64 = (1 << 64) - 1


def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & MASK64

    def u01():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & MASK64
        return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    return u01


def _uni(u01, lo, hi):
    return lo + (hi - lo) * u01()


def _clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _noise(seed, x, t, std):
    key = seed & MASK64
    key = (key * 1000003 + (int(round(x * 1000)) & MASK64)) & MASK64
    key = (key * 1000003 + (int(round(t * 1000)) & MASK64)) & MASK64
    u = _rng(key)()
    return std * (2.0 * u - 1.0)


# ------------------------------- physics law ---------------------------------------
def _kernel(u, w):
    z = u / w
    v = 1.0 - z * z
    return v if v > 0.0 else 0.0


def _f0(params, t):
    return params["f0_0"] + params["drift"] * t


def _amp(params, t):
    val = params["A_mid"] + params["A_amp"] * math.sin(
        2.0 * math.pi * t / params["period"] + params["phase"])
    return val if val > 0.0 else 0.0


def _true_y(params, x, t):
    f0 = _f0(params, t)
    A = _amp(params, t)
    return A * _kernel(x - f0, params["w"])


# ------------------------------- padding helpers ------------------------------------
def _pad_abs(u01, true_val, lo_min, lo_max, hi_min, hi_max):
    lo_pad = _uni(u01, lo_min, lo_max)
    hi_pad = _uni(u01, hi_min, hi_max)
    return true_val - lo_pad, true_val + hi_pad


def _pad_prop(u01, true_val, lo_min, lo_max, hi_min, hi_max):
    mag = abs(true_val)
    lo_pad = mag * _uni(u01, lo_min, lo_max)
    hi_pad = mag * _uni(u01, hi_min, hi_max)
    return true_val - lo_pad, true_val + hi_pad


# ------------------------------- test-query generation -------------------------------
def _gen_test_queries(seed, params, K, on_peak_frac):
    u = _rng(seed)
    n_peak = int(round(K * on_peak_frac))
    n_gen = K - n_peak
    queries = []
    for _ in range(n_peak):
        t = _uni(u, T_FUT_LO, T_FUT_HI)
        f0t = _f0(params, t)
        jitter = _uni(u, -0.5, 0.5) * params["w"]
        x = _clip(f0t + jitter, X_LO, X_HI)
        queries.append({"x": x, "t": t})
    for _ in range(n_gen):
        t = _uni(u, T_FUT_LO, T_FUT_HI)
        x = _uni(u, X_LO, X_HI)
        queries.append({"x": x, "t": t})
    return queries


# ------------------------------- instance family -------------------------------------
def _build_one(seed, drift_lo, drift_hi, period_lo, period_hi, t_turn_lo, t_turn_hi,
                on_peak_frac):
    u = _rng(seed)
    w_true = _uni(u, 4.5, 8.5)
    f0_0_true = _uni(u, -20.0, 20.0)
    sign = 1.0 if u() < 0.5 else -1.0
    drift_true = sign * _uni(u, drift_lo, drift_hi)
    A_mid_true = _uni(u, 9.0, 13.0)
    A_amp_true = _uni(u, 3.5, 6.5)
    period_true = _uni(u, period_lo, period_hi)
    if t_turn_lo is not None:
        t_turn = _uni(u, t_turn_lo, t_turn_hi)
        branch = 1.0 if u() < 0.5 else -1.0     # plant a max (+) or a min (-) at t_turn
        phase_true = (branch * math.pi / 2.0) - (2.0 * math.pi * t_turn / period_true)
        phase_true = phase_true % (2.0 * math.pi)
    else:
        phase_true = _uni(u, 0.0, 2.0 * math.pi)

    params = {"w": w_true, "f0_0": f0_0_true, "drift": drift_true, "A_mid": A_mid_true,
              "A_amp": A_amp_true, "period": period_true, "phase": phase_true}

    w_lo, w_hi = _pad_abs(u, w_true, 1.0, 2.5, 1.0, 2.5)
    f0_0_lo, f0_0_hi = _pad_abs(u, f0_0_true, 6.0, 14.0, 6.0, 14.0)
    drift_lo_h, drift_hi_h = _pad_prop(u, drift_true, 0.25, 0.6, 0.25, 0.6)
    A_mid_lo, A_mid_hi = _pad_abs(u, A_mid_true, 2.0, 4.0, 2.0, 4.0)
    A_amp_lo, A_amp_hi = _pad_abs(u, A_amp_true, 1.5, 3.0, 1.5, 3.0)
    period_lo_h, period_hi_h = _pad_prop(u, period_true, 0.10, 0.22, 0.10, 0.22)

    hints = {"w_lo": round(w_lo, 4), "w_hi": round(w_hi, 4),
             "f0_0_lo": round(f0_0_lo, 4), "f0_0_hi": round(f0_0_hi, 4),
             "drift_lo": round(drift_lo_h, 6), "drift_hi": round(drift_hi_h, 6),
             "A_mid_lo": round(A_mid_lo, 4), "A_mid_hi": round(A_mid_hi, 4),
             "A_amp_lo": round(A_amp_lo, 4), "A_amp_hi": round(A_amp_hi, 4),
             "period_lo": round(period_lo_h, 3), "period_hi": round(period_hi_h, 3)}

    test_queries = _gen_test_queries(seed + 777, params, K_TEST, on_peak_frac)
    return {"params": params, "hints": hints, "test_queries": test_queries}


def _build_instances():
    specs = [
        # name,      seed,   drift_lo, drift_hi, period_lo, period_hi, t_turn_lo, t_turn_hi, on_peak_frac
        ("trap1",   95301, 0.024, 0.036, 1900.0, 2500.0, 1000.0, 1350.0, 0.65),
        ("trap2",   95302, 0.024, 0.036, 1900.0, 2500.0, 1000.0, 1350.0, 0.65),
        ("trap3",   95303, 0.024, 0.036, 1900.0, 2500.0, 1000.0, 1350.0, 0.68),
        ("trap4",   95304, 0.024, 0.036, 1900.0, 2500.0, 1000.0, 1350.0, 0.68),
        ("mod1",    95311, 0.012, 0.020, 3000.0, 4000.0, 700.0, 1100.0, 0.45),
        ("mod2",    95312, 0.012, 0.020, 3000.0, 4000.0, 700.0, 1100.0, 0.45),
        ("mod3",    95313, 0.012, 0.020, 3000.0, 4000.0, 700.0, 1100.0, 0.48),
        ("gentle1", 95321, 0.004, 0.010, 5000.0, 7000.0, None, None, 0.30),
        ("gentle2", 95322, 0.004, 0.010, 5000.0, 7000.0, None, None, 0.30),
        ("gentle3", 95323, 0.004, 0.010, 5000.0, 7000.0, None, None, 0.32),
    ]
    out = []
    for name, seed, dlo, dhi, plo, phi, ttlo, tthi, opf in specs:
        inst = _build_one(seed, dlo, dhi, plo, phi, ttlo, tthi, opf)
        inst["name"] = name
        inst["seed"] = seed
        out.append(inst)
    return out


def _err_ref(inst):
    hints = inst["hints"]; params = inst["params"]
    w_g = 0.5 * (hints["w_lo"] + hints["w_hi"])
    f0_g = 0.5 * (hints["f0_0_lo"] + hints["f0_0_hi"])
    A_g = 0.5 * (hints["A_mid_lo"] + hints["A_mid_hi"])
    if w_g <= 1e-6:
        w_g = 1e-6
    s = 0.0
    for q in inst["test_queries"]:
        naive = A_g * _kernel(q["x"] - f0_g, w_g)
        tru = _true_y(params, q["x"], q["t"])
        s += abs(naive - tru)
    return s / len(inst["test_queries"])


# ------------------------------- interactive run ------------------------------------
def _public_query(inst, phase, history, budget_left, rnd):
    # NOTE: the instance "name"/seed is deliberately NOT included in the public payload --
    # only fields a genuine strategy needs (dial range, hints, budget, history) are sent, so
    # there is no per-instance label an RL policy could memorize across repeated training
    # rollouts on this fixed evaluator instead of exploiting the actual probe-allocation
    # insight.
    pub = {"phase": phase,
           "dial": {"x_lo": X_LO, "x_hi": X_HI}, "t_obs_max": T_OBS_MAX,
           "hints": dict(inst["hints"]), "noise_std": NOISE_STD,
           "budget": {"max_probes": N_MAX},
           "budget_left": budget_left, "round": rnd, "R": R,
           "max_this_round": MAX_PER_ROUND, "history": [dict(h) for h in history]}
    if phase == "predict":
        pub["test_queries"] = [dict(q) for q in inst["test_queries"]]
    return pub


def _run_instance(cand, inst):
    params = inst["params"]; seed = inst["seed"]
    history = []
    budget = N_MAX
    for rnd in range(R):
        if budget <= 0:
            break
        pub = _public_query(inst, "query", history, budget, rnd)
        ans, st = isorun.run_candidate(cand, pub, timeout=20)
        if st != "OK" or not isinstance(ans, dict):
            return 0.0
        probes = ans.get("probes", [])
        if not isinstance(probes, list):
            return 0.0
        # Validate the SHAPE of every submitted entry up front -- a malformed entry anywhere
        # in the answer scores the whole instance 0.0, regardless of whether it would end up
        # inside the honored per-round prefix.
        parsed = []
        for e in probes:
            if not isinstance(e, dict):
                return 0.0
            vals = []
            for key in ("x", "t"):
                v = e.get(key)
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    return 0.0
                fv = float(v)
                if fv != fv or fv in (float("inf"), float("-inf")):
                    return 0.0
                vals.append(fv)
            parsed.append(vals)
        allow = min(budget, MAX_PER_ROUND)
        n_taken = 0
        for x, t in parsed:
            if n_taken >= allow or budget <= 0:
                break
            budget -= 1
            n_taken += 1
            if x < X_LO - 1e-9 or x > X_HI + 1e-9 or t < -1e-9 or t > T_OBS_MAX + 1e-9:
                continue          # out of range: skip -- charged against budget, no reading
            y_true = _true_y(params, x, t)
            y_obs = y_true + _noise(seed, x, t, NOISE_STD)
            history.append({"x": x, "t": t, "y": y_obs})
    # ---- prediction phase ----
    pub = _public_query(inst, "predict", history, budget, R)
    ans, st = isorun.run_candidate(cand, pub, timeout=20)
    if st != "OK" or not isinstance(ans, dict):
        return 0.0
    pred = ans.get("predictions")
    K = len(inst["test_queries"])
    if not isinstance(pred, list) or len(pred) != K:
        return 0.0
    err = 0.0
    for j in range(K):
        p = pred[j]
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            return 0.0
        pf = float(p)
        if pf != pf or pf in (float("inf"), float("-inf")):
            return 0.0
        q = inst["test_queries"][j]
        tru = _true_y(params, q["x"], q["t"])
        err += abs(pf - tru)
    err /= K
    eref = _err_ref(inst)
    if eref <= 1e-9:
        eref = 1e-9
    quality = 1.0 - err / eref
    if quality < 0.0:
        quality = 0.0
    elif quality > 1.0:
        quality = 1.0
    r = OFFSET + SPAN * quality
    if r < 0.0:
        r = 0.0
    elif r > 1.0:
        r = 1.0
    return r


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = _build_instances()
    vec = []
    for inst in insts:
        try:
            r = _run_instance(cand, inst)
        except Exception:
            r = 0.0
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        vec.append(r)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
