#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0940 -- "Budgeted Tastings: Mapping a Hidden Phase Plateau"
(family: calorimeter-mixing-probe; format B, quality-metric; theme: budgeted tastings to
map a phase change).

THEME.  A calorimetry bench holds three reagents: hot solvent A (initial temperature TA),
a cold phase-change reagent B (initial temperature TB0, always below its own hidden melting
point), and a neutral diluent C (initial temperature TC).  You may run MIXING EXPERIMENTS of
your own design: choose nonnegative masses (mA, mB, mC) -- summing under a per-experiment
material budget -- and observe the final equilibrium temperature.  You get a fixed budget of
such experiments, spent across a few adaptive rounds (you see every reading so far and choose
the next batch).  Afterwards you must PREDICT the final temperature for a frozen suite of
unseen test mixes.

WHY THE TWO MECHANISMS ARE FORCED (both shape the score).
  * budgeted-mixing-experiments -- the experiment count is tight relative to the composition
    space.  An experiment's worth is the STRUCTURAL information it buys (which thermal regime
    a composition falls in), not the raw reading.
  * latent-phase-plateau -- reagent B absorbs a hidden per-unit-mass latent heat while
    crossing its (also hidden) melting point T*.  The energy-conservation-only prediction
    T_lin(m) (a weighted average of the three initial temperatures, computable from PUBLIC
    data with zero experiments) is EXACT whenever the mixture never gets warm enough to
    trouble B's melting point -- but inside a narrow band where the naive T_lin would cross
    T*, the true outcome locks flat at T* (partial melt absorbs the excess) until enough
    excess energy is present to fully melt B, after which it resumes rising, now offset below
    T_lin.  That flat band is a genuine measure-zero-ish manifold in composition space: its
    width scales with the melted reagent's mass fraction, so it is easy to miss with any
    experiment design that does not deliberately hunt for it, and easy to resolve once a
    design commits real budget to it.

INNOVATION HOOK (what `strong` exploits).  Nearly the whole experiment budget should hunt the
transition manifold -- allocation, not modeling, is the game.  `strong` (a) spends a handful of
anchor experiments broadly to sanity-check the linear trend away from the transition, (b)
commits the bulk of its budget to a fine composition sweep INSIDE the public [Tstar_lo,
Tstar_hi] hint window (using a moderately large B-fraction so the plateau is easy to land
on), (c) recognizes repeated FLAT readings as the plateau signature (that value IS T*) and
bisects toward its two edges to pin the transition and the latent-heat scale, then (d)
predicts every test mix by classifying its regime (cold / plateau / hot) from the fitted
(T*, latent-heat) pair and applying the matching closed form.  The TRAP instances concentrate
most of their held-out test mixes near the true transition with a narrow public window, so the
obvious recipe -- a composition-space-filling sweep that ignores the window hint and then fits
one smooth 1-D curve of reading-vs-T_lin -- lands far from strong there.

CANDIDATE CONTRACT (isolated stdin -> stdout program, called ONCE PER ROUND).
  The evaluator drives an interactive loop by RE-INVOKING the candidate each round with the
  full public state.  The candidate is stateless across calls; it re-derives its plan from the
  history it is handed.
    stdin : ONE JSON object.  Common fields:
        {"name","phase","materials":{"TA","TB0","TC"},
         "bounds":{"Tstar_lo","Tstar_hi","ell_lo","ell_hi"},
         "budget":{"max_experiments","mass_cap_per_experiment"},
         "budget_left","round","R","max_this_round",
         "history":[{"mA","mB","mC","T_f"}, ...]}
      phase == "query"   -> return {"experiments": [{"mA":x,"mB":y,"mC":z}, ...]}
                            (nonnegative floats; each experiment's mA+mB+mC must be <=
                            mass_cap_per_experiment and > 0; at most min(budget_left,
                            max_this_round) experiments are honored per round; a malformed
                            entry (wrong type / negative / non-finite) -> 0.0 on the whole
                            instance; an entry over the per-experiment mass cap or with zero
                            total mass is SKIPPED (charged against budget, no reading))
      phase == "predict" -> return {"predictions": [p_0, ..., p_{K-1}]} matching
                            inst["test_mixes"] in order (each a finite real number)
  Any crash / timeout / non-JSON / wrong shape on ANY call -> 0.0 on that instance.

SCORING (deterministic; no wall-time).  Per instance, with true final temperature T_f(m):
    err_ref = mean_j |T_lin(test_mixes[j]) - T_f(test_mixes[j])|   # zero-experiment, energy-
                                                                    # conservation-only guess
    err     = mean_j |pred_j - T_f(test_mixes[j])|
    quality = clip(1 - err/err_ref, 0, 1)
    r       = OFFSET + SPAN * quality              # OFFSET=0.10, SPAN=0.82 (cap 0.92)
  Predicting T_lin everywhere (no experiments) scores exactly OFFSET=0.10; finite-resolution
  bisection slack keeps even a strong strategy below the 0.92 cap, so headroom stays open above
  the reference.  Final score = mean of r over 10 fixed seeded instances (4 traps + 3 moderate
  + 3 gentle/held-out).

ISOLATION.  The candidate is untrusted and runs in a FRESH bwrap-SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it sees only the public state.  T*, the latent-heat scale, and the
scoring machinery live only in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
"""
import sys, json
import isorun

# --------------------------- scoring / protocol constants ------------------------
OFFSET = 0.10
SPAN = 0.82                 # r = OFFSET + SPAN*quality ; max attainable r = 0.92 (headroom)
N_MAX = 18                  # experiment-count budget
MASS_CAP = 120.0            # per-experiment mass cap (mA+mB+mC <= MASS_CAP)
R = 4                       # adaptive query rounds
MAX_PER_ROUND = N_MAX       # a round may spend up to the whole remaining budget
K_TEST = 44                 # frozen held-out test-mix suite size


# ------------------------------- deterministic RNG -------------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def u01():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    return u01


def _uni(u01, lo, hi):
    return lo + (hi - lo) * u01()


# ------------------------------- physics law --------------------------------------
def _t_lin(materials, mA, mB, mC):
    M = mA + mB + mC
    if M <= 1e-12:
        return None
    return (mA * materials["TA"] + mB * materials["TB0"] + mC * materials["TC"]) / M


def _true_tf(materials, Tstar, ell, mA, mB, mC):
    M = mA + mB + mC
    Tl = _t_lin(materials, mA, mB, mC)
    xB = mB / M
    if Tl <= Tstar:
        return Tl
    hi = Tstar + ell * xB
    if Tl < hi:
        return Tstar
    return Tl - ell * xB


def _feasible_xb_range(materials, target):
    """xB values at which composing A/B/C can achieve T_lin == target (B-fraction fixed at
    xB, remaining mass split between A and C).  Both reach-range endpoints
    xB*TB0+(1-xB)*TC (lo) and xB*TB0+(1-xB)*TA (hi) are monotone in xB (since TB0<TC<TA by
    construction), so this is a closed-form interval, not a search."""
    TA, TB0, TC = materials["TA"], materials["TB0"], materials["TC"]
    xb_hi_max = (TA - target) / (TA - TB0) if TA != TB0 else 1.0
    xb_lo_min = (TC - target) / (TC - TB0) if TC != TB0 else 0.0
    lo = max(0.0, xb_lo_min)
    hi = min(1.0, xb_hi_max)
    return lo, hi


def _compose(materials, xB, Tlin_target, M):
    """Pick (mA, mB, mC) with mA+mB+mC=M, mB=xB*M, and T_lin as close to Tlin_target as
    achievable by splitting the remaining mass between A and C."""
    TA, TB0, TC = materials["TA"], materials["TB0"], materials["TC"]
    mB = xB * M
    rem = (1.0 - xB) * M
    if rem <= 1e-9:
        return {"mA": 0.0, "mB": round(mB, 4), "mC": 0.0}
    if abs(TA - TC) < 1e-9:
        t = 0.5
    else:
        # T_lin = xB*TB0 + (1-xB)*[t*TA + (1-t)*TC]  =>  solve for t
        t = (Tlin_target - xB * TB0 - (1.0 - xB) * TC) / ((1.0 - xB) * (TA - TC))
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    mA = rem * t
    mC = rem * (1.0 - t)
    return {"mA": round(mA, 4), "mB": round(mB, 4), "mC": round(mC, 4)}


def _gen_test_mixes(seed, materials, Tstar, ell, K, plateau_frac):
    u = _rng(seed)
    n_plateau = int(round(K * plateau_frac))
    n_general = K - n_plateau
    lo_reach = min(materials["TA"], materials["TB0"], materials["TC"])
    hi_reach = max(materials["TA"], materials["TB0"], materials["TC"])
    mixes = []
    for _ in range(n_plateau):
        xB = _uni(u, 0.12, 0.78)
        frac = _uni(u, -0.35, 1.35)          # position relative to [Tstar, Tstar+ell*xB]
        # fixed-point: xB and the target both depend on each other (the plateau's own edge
        # shifts with xB); a few clip-and-recompute passes converge since the correction is
        # small and contracting
        for _ in range(3):
            Tlin_target = Tstar + frac * ell * xB
            lo_f, hi_f = _feasible_xb_range(materials, Tlin_target)
            xB = min(max(xB, lo_f), hi_f)
        Tlin_target = Tstar + frac * ell * xB
        Tlin_target = max(lo_reach, min(hi_reach, Tlin_target))
        mixes.append(_compose(materials, xB, Tlin_target, 100.0))
    for _ in range(n_general):
        xB = _uni(u, 0.0, 0.95)
        Tlin_target = _uni(u, lo_reach, hi_reach)
        mixes.append(_compose(materials, xB, Tlin_target, 100.0))
    return mixes


# ------------------------------- instance family -----------------------------------
def _build_one(seed, plateau_frac, pad_lo, pad_hi, width_lo, width_hi):
    u = _rng(seed)
    TA = _uni(u, 68.0, 88.0)
    TB0 = _uni(u, 0.0, 9.0)
    TC = _uni(u, 16.0, 40.0)
    Tstar = _uni(u, TB0 + 10.0, TA - 12.0)
    width_at_ref = _uni(u, width_lo, width_hi)     # plateau width target at xB=0.4
    ell = width_at_ref / 0.4
    ell_lo = ell * _uni(u, 0.45, 0.7)
    ell_hi = ell * _uni(u, 1.3, 1.9)
    materials = {"TA": round(TA, 3), "TB0": round(TB0, 3), "TC": round(TC, 3)}

    # NOTE: the window is deliberately SKEWED (not a symmetric half-width) by a large,
    # guaranteed fraction of its own scale, so the window's midpoint does NOT approximate
    # Tstar -- a candidate that skips experiments and just reports (Tstar_lo+Tstar_hi)/2 sees a
    # value displaced well past the plateau's own width, not a usable estimate of the
    # transition.  A genuine strategy is unaffected: it still knows Tstar in [Tstar_lo,
    # Tstar_hi] and sweeps the whole window to find the flat-reading signature directly.
    # The pad magnitude is then SHRUNK (still skewed) until the resulting window is provably
    # reachable at some single B-fraction with a workable (not vanishingly narrow) plateau --
    # otherwise the instance would be un-huntable by ANY strategy, not a genuine skill test.
    half_width = _uni(u, pad_lo, pad_hi)
    skew_frac = _uni(u, 0.55, 0.9)
    sign = 1.0 if u() < 0.5 else -1.0
    for _shrink in range(12):
        if sign > 0:
            pad_left = half_width * (1.0 - skew_frac)
            pad_right = half_width * (1.0 + skew_frac)
        else:
            pad_left = half_width * (1.0 + skew_frac)
            pad_right = half_width * (1.0 - skew_frac)
        t_lo = Tstar - pad_left
        t_hi = Tstar + pad_right
        lo1, hi1 = _feasible_xb_range(materials, t_lo)
        lo2, hi2 = _feasible_xb_range(materials, t_hi)
        xb_lo, xb_hi = max(lo1, lo2), min(hi1, hi2)
        if xb_hi > xb_lo + 0.02:
            xb_pick = xb_lo + 0.7 * (xb_hi - xb_lo)
            if ell * xb_pick >= 0.5:
                break
        half_width *= 0.8
    bounds = {"Tstar_lo": round(t_lo, 3), "Tstar_hi": round(t_hi, 3),
              "ell_lo": round(ell_lo, 3), "ell_hi": round(ell_hi, 3)}
    test_mixes = _gen_test_mixes(seed + 777, materials, Tstar, ell, K_TEST, plateau_frac)
    return {"materials": materials, "bounds": bounds, "Tstar": Tstar, "ell": ell,
            "test_mixes": test_mixes}


def _build_instances():
    specs = [
        # name,      seed,  plateau_frac, pad_lo, pad_hi, width_lo, width_hi
        ("trap1",    94001, 0.65, 4.0, 6.0, 1.0, 1.8),
        ("trap2",    94002, 0.68, 4.0, 5.5, 1.0, 1.7),
        ("trap3",    94003, 0.62, 4.5, 6.0, 1.1, 1.9),
        ("trap4",    94004, 0.70, 4.0, 5.5, 0.9, 1.6),
        ("mod1",     94011, 0.40, 6.0, 9.0, 1.5, 2.5),
        ("mod2",     94012, 0.42, 6.5, 9.0, 1.6, 2.4),
        ("mod3",     94013, 0.38, 6.0, 8.5, 1.5, 2.6),
        ("gentle1",  94021, 0.25, 8.0, 12.0, 2.0, 3.5),
        ("gentle2",  94022, 0.22, 8.5, 12.0, 2.2, 3.4),
        ("gentle3",  94023, 0.27, 8.0, 11.5, 2.0, 3.6),
    ]
    out = []
    for name, seed, pf, plo, phi, wlo, whi in specs:
        inst = _build_one(seed, pf, plo, phi, wlo, whi)
        inst["name"] = name
        out.append(inst)
    return out


def _err_ref(inst):
    materials = inst["materials"]; Tstar = inst["Tstar"]; ell = inst["ell"]
    s = 0.0
    for m in inst["test_mixes"]:
        tl = _t_lin(materials, m["mA"], m["mB"], m["mC"])
        tf = _true_tf(materials, Tstar, ell, m["mA"], m["mB"], m["mC"])
        s += abs(tl - tf)
    return s / len(inst["test_mixes"])


# ------------------------------- interactive run ---------------------------------
def _public_query(inst, phase, history, budget_left, rnd):
    pub = {"name": inst["name"], "phase": phase, "materials": inst["materials"],
           "bounds": inst["bounds"],
           "budget": {"max_experiments": N_MAX, "mass_cap_per_experiment": MASS_CAP},
           "budget_left": budget_left, "round": rnd, "R": R,
           "max_this_round": MAX_PER_ROUND, "history": [dict(h) for h in history]}
    if phase == "predict":
        pub["test_mixes"] = [dict(m) for m in inst["test_mixes"]]
    return pub


def _run_instance(cand, inst):
    materials = inst["materials"]; Tstar = inst["Tstar"]; ell = inst["ell"]
    history = []
    budget = N_MAX
    for rnd in range(R):
        if budget <= 0:
            break
        pub = _public_query(inst, "query", history, budget, rnd)
        ans, st = isorun.run_candidate(cand, pub, timeout=20)
        if st != "OK" or not isinstance(ans, dict):
            return 0.0
        exps = ans.get("experiments", [])
        if not isinstance(exps, list):
            return 0.0
        # Validate the SHAPE of every submitted entry up front (regardless of whether it will
        # end up inside the honored per-round count) -- a malformed entry anywhere in the
        # answer scores the whole instance 0.0, exactly as stated, not just the honored prefix.
        parsed = []
        for e in exps:
            if not isinstance(e, dict):
                return 0.0
            vals = []
            for key in ("mA", "mB", "mC"):
                v = e.get(key)
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    return 0.0
                fv = float(v)
                if fv != fv or fv in (float("inf"), float("-inf")) or fv < 0.0:
                    return 0.0
                vals.append(fv)
            parsed.append(vals)
        allow = min(budget, MAX_PER_ROUND)
        n_taken = 0
        for vals in parsed:
            if n_taken >= allow or budget <= 0:
                break
            mA, mB, mC = vals
            budget -= 1
            n_taken += 1
            M = mA + mB + mC
            if M <= 1e-9 or M > MASS_CAP + 1e-6:
                continue          # skip: charged against budget, no reading (like out-of-window)
            tf = _true_tf(materials, Tstar, ell, mA, mB, mC)
            history.append({"mA": mA, "mB": mB, "mC": mC, "T_f": tf})
    # ---- prediction phase ----
    pub = _public_query(inst, "predict", history, budget, R)
    ans, st = isorun.run_candidate(cand, pub, timeout=20)
    if st != "OK" or not isinstance(ans, dict):
        return 0.0
    pred = ans.get("predictions")
    K = len(inst["test_mixes"])
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
        m = inst["test_mixes"][j]
        tf = _true_tf(materials, Tstar, ell, m["mA"], m["mB"], m["mC"])
        err += abs(pf - tf)
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
