#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1300 -- "Locking Down Without Locking Down"
(family: epidemic-response-policy; format B, quality-metric).

THEME.  A deterministic outbreak simulator.  For 21 days ("history") the
outbreak has grown UNCONTROLLED (no intervention).  You inherit the response
from day 21 onward for a fixed future horizon of `future_days` days.  Every
day you pick an intervention LEVEL from a fixed menu (0..3): each level has a
transmission-reduction multiplier `m` and a flat daily economic COST.  Two
mechanisms make this harder than "pick the strictest level and hold it":

  1. DELAY BETWEEN SIGNAL AND EFFECT.  You never see the true new-infection
     count.  You see two derived, delayed proxies computed over the history
     window: `reported_cases` (incubation + testing + reporting delay
     `d_rep`, e.g. 7-10 days -- the standard "case count" signal) and
     `leading_indicator` (a much shorter delay `d_lead`, e.g. 2 days -- an
     early-warning proxy such as syndromic/wastewater surveillance), both with
     bounded multiplicative noise.  Because `reported_cases` is blind to the
     most recent `d_rep` days, extrapolating from it alone systematically
     UNDER-estimates how far the outbreak has already progressed by "now"
     (the end of history) whenever growth is fast -- the freshest, most
     decision-relevant days are exactly the ones it cannot show you yet.

  2. BEHAVIORAL FATIGUE.  Compliance with an active (level > 0) restriction
     decays the longer it is held continuously, eroding its real-world
     effect toward doing nothing; a level-0 "rest" day lets compliance
     recover.  Holding the strictest level for weeks straight both costs the
     most AND, by the end, barely reduces transmission -- so sustaining
     maximum restriction is doubly wasteful.  Pulsing (short bursts of
     restriction separated by rest days) buys back compliance for less
     cumulative economic cost.

  3. INTERVENTION COST.  Every active day costs money regardless of how
     effective it currently is (mechanism composes with #2: a fatigued,
     ineffective lockdown still bills you full price).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance) -- see statement.md for the
          exact schema (future_days, d_rep, d_lead, levels menu, fatigue
          params, hospital_capacity/overflow_penalty/health_weight,
          reported_cases[0..H-1], leading_indicator[0..H-1]).
  stdout: ONE JSON object: {"levels": [g_0, ..., g_{future_days-1}]}, each
          g_i an integer intervention level in [0, len(levels)-1] for future
          day i (day H+i of the simulation).

  Any malformed answer (wrong type, wrong length, an out-of-range or
  non-integer level, a crash, a timeout, non-JSON output) scores 0 on that
  instance.

SCORING (deterministic).  The evaluator replays the TRUE future trajectory
(known only to this parent process: the exact growth rate and the exact true
infection count/susceptible pool at the end of history) under the candidate's
level schedule, applying the fatigue/compliance and cost/health accounting
described in statement.md, to get an objective `obj = -(health_cost +
econ_cost)` (higher is better -- MAXIMIZE).  Two references, computed by THIS
process using the TRUE trajectory (never sent to the candidate):
    obj_base    = objective of the constant "always level 1" policy (a naive,
                  data-blind response -- the weak reference / trivial tier).
    obj_perfect = objective of a coordinate-descent local search that ALSO
                  gets the true trajectory (a "perfect information" oracle --
                  strictly more information than any real candidate has, so
                  it opens headroom above what data-driven inference can
                  reach).  We deliberately stretch the denominator by 1.5x so
                  even the oracle only defines the top of a *compressed*
                  scale, leaving room above realistic strong solutions.
    r = clamp(0.1 + 0.9 * (obj_cand - obj_base) / (1.5 * (obj_perfect - obj_base)), 0, 1)
A candidate matching the naive baseline scores ~0.1; nothing can reach 1.0 by
construction (the 1.5x stretch keeps even the oracle itself below ~0.7 on this
scale, so no achievable candidate saturates).

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The true
trajectory and both references are computed by THIS parent process, so a
frame-walking / introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys
import json
import math
import isorun


# ----------------------------- deterministic RNG ----------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    return nxt


# ----------------------------- instance family ------------------------------
_DEFAULTS = dict(
    N=300000, H=21, FUT=45, D_REP=7, D_LEAD=2,
    LEVELS=[(1.00, 0), (0.80, 6), (0.55, 18), (0.30, 45)],
    FATIGUE_DECAY=0.06, FATIGUE_RECOVER=0.10, FATIGUE_FLOOR=0.35,
    H_CAP=6000, OVERFLOW_PEN=4.0, BASE_WEIGHT=1.0, X0=25,
)

# 7 fast-growth ("trap") seeds + 3 slower-growth control seeds; 3 of the 10
# additionally perturb the structural parameters (worse reporting delay,
# bigger population, longer history window) as held-out generalization cases.
_INSTANCE_SPECS = [
    dict(name="wave-a1", seed=4101, R0=1.42),
    dict(name="wave-a2", seed=4102, R0=1.38),
    dict(name="wave-a3", seed=4103, R0=1.33),
    dict(name="wave-a4", seed=4104, R0=1.30),
    dict(name="wave-a5", seed=4105, R0=1.36),
    dict(name="wave-a6", seed=4106, R0=1.40, D_REP=10),
    dict(name="wave-a7", seed=4107, R0=1.31, N=600000, H_CAP=12000),
    dict(name="wave-b1", seed=4201, R0=1.10),
    dict(name="wave-b2", seed=4202, R0=1.06),
    dict(name="wave-b3", seed=4203, R0=1.15, H=28),
]

_STRETCH = 1.5   # opens headroom above the perfect-information oracle (see module docstring)


def _params(spec):
    p = dict(_DEFAULTS)
    p.update(spec)
    return p


def _simulate_history(p):
    """Days 0..H-1, level=0 throughout (uncontrolled outbreak). Returns the
    TRUE new-infection trace only implicitly; what's returned is the two
    OBSERVED (delayed, noisy) proxy series plus the true (x_last, cum_last)
    state at the end of history -- the only "hidden" state needed to
    continue the simulation forward."""
    N, H, X0, R0 = p["N"], p["H"], p["X0"], p["R0"]
    D_REP, D_LEAD = p["D_REP"], p["D_LEAD"]
    seed = p["seed"]
    x = [0.0] * H
    cum = 0.0
    xt = float(X0)
    for t in range(H):
        S = max(0.0, N - cum)
        x[t] = min(S, xt)
        cum += x[t]
        S2 = max(0.0, N - cum)
        xt = min(S2, x[t] * R0 * (S2 / N))
    reported = [0] * H
    leading = [0] * H
    nr = _rng(seed * 11 + 3)
    nl = _rng(seed * 17 + 5)
    for t in range(H):
        if t - D_REP >= 0:
            noise = 0.85 + 0.30 * nr()
            reported[t] = max(0, round(x[t - D_REP] * noise))
        if t - D_LEAD >= 0:
            noise = 0.75 + 0.50 * nl()
            leading[t] = max(0, round(x[t - D_LEAD] * noise))
    return reported, leading, x[H - 1], cum


def _simulate_future(p, x_last, cum_last, levels_seq):
    """Replay days H..T-1 under `levels_seq` (length FUT). Returns
    (health_cost, econ_cost); both >= 0. Uses the TRUE R0."""
    N = p["N"]
    LEVELS = p["LEVELS"]
    DECAY, RECOVER, FLOOR = p["FATIGUE_DECAY"], p["FATIGUE_RECOVER"], p["FATIGUE_FLOOR"]
    H_CAP, OVERFLOW_PEN, BASE_WEIGHT = p["H_CAP"], p["OVERFLOW_PEN"], p["BASE_WEIGHT"]
    R0 = p["R0"]
    health = 0.0
    econ = 0.0
    xt = x_last
    cum = cum_last
    streak = 0
    compliance = 1.0
    for lv in levels_seq:
        m, c = LEVELS[lv]
        if lv > 0:
            streak += 1
            compliance = max(FLOOR, 1.0 - DECAY * streak)
        else:
            streak = 0
            compliance = min(1.0, compliance + RECOVER)
        eff_m = 1.0 - compliance * (1.0 - m)          # fatigue erodes the level's benefit
        S = max(0.0, N - cum)
        xt = min(S, xt * R0 * eff_m * (S / N))
        cum += xt
        health += BASE_WEIGHT * xt + OVERFLOW_PEN * max(0.0, xt - H_CAP)
        econ += c
    return health, econ


def _obj(h, e):
    return -(h + e)


def _oracle_policy(p, x_last, cum_last):
    """Perfect-information reference: coordinate-descent local search over the
    future schedule, using the TRUE R0/state (something no real candidate has
    -- it only ever sees noisy delayed proxies). Multiple deterministic seeded
    starting schedules avoid a bad single local optimum. Fully deterministic
    (no RNG; fixed iteration order)."""
    FUT = p["FUT"]
    K = len(p["LEVELS"])

    def cost_of(seq):
        h, e = _simulate_future(p, x_last, cum_last, seq)
        return h + e

    starts = [[0] * FUT, [3] * FUT, [2] * FUT,
              [min(3, t // 10) for t in range(FUT)]]
    for on_level in (2, 3):
        for on_len in (5, 7):
            seq = []
            while len(seq) < FUT:
                seq += [on_level] * on_len + [0] * 3
            starts.append(seq[:FUT])

    best_cost = None
    for s0 in starts:
        seq = list(s0)
        improved = True
        while improved:
            improved = False
            for day in range(FUT):
                cur = seq[day]
                best_lv, best_c = cur, cost_of(seq)
                for lv in range(K):
                    if lv == cur:
                        continue
                    seq[day] = lv
                    c = cost_of(seq)
                    if c < best_c - 1e-9:
                        best_c, best_lv, improved = c, lv, True
                seq[day] = best_lv
        c = cost_of(seq)
        if best_cost is None or c < best_cost:
            best_cost = c
    return best_cost


def make_instances():
    out = []
    for spec in _INSTANCE_SPECS:
        p = _params(spec)
        reported, leading, x_last, cum_last = _simulate_history(p)
        public = {
            "name": p["name"],
            "future_days": p["FUT"],
            "d_rep": p["D_REP"],
            "d_lead": p["D_LEAD"],
            "levels": [{"m": m, "cost": c} for m, c in p["LEVELS"]],
            "fatigue": {"decay": p["FATIGUE_DECAY"], "recover": p["FATIGUE_RECOVER"],
                        "floor": p["FATIGUE_FLOOR"]},
            "hospital_capacity": p["H_CAP"],
            "overflow_penalty": p["OVERFLOW_PEN"],
            "health_weight": p["BASE_WEIGHT"],
            "reported_cases": reported,
            "leading_indicator": leading,
        }
        hidden = {"params": p, "x_last": x_last, "cum_last": cum_last}
        out.append({"public": public, "hidden": hidden})
    return out


def baseline(inst):
    """Weak reference: constant 'always level 1', ignoring all data."""
    p = inst["hidden"]["params"]
    x_last = inst["hidden"]["x_last"]
    cum_last = inst["hidden"]["cum_last"]
    h, e = _simulate_future(p, x_last, cum_last, [1] * p["FUT"])
    return _obj(h, e)


def score(inst, answer):
    """Validate + score `answer` against the TRUE hidden trajectory. Returns
    (ok, obj)."""
    p = inst["hidden"]["params"]
    x_last = inst["hidden"]["x_last"]
    cum_last = inst["hidden"]["cum_last"]
    FUT = p["FUT"]
    K = len(p["LEVELS"])

    if not isinstance(answer, dict):
        return False, None
    seq = answer.get("levels")
    if not isinstance(seq, list) or len(seq) != FUT:
        return False, None
    for v in seq:
        if isinstance(v, bool) or not isinstance(v, int):
            return False, None
        if v < 0 or v >= K:
            return False, None

    h, e = _simulate_future(p, x_last, cum_last, seq)
    return True, _obj(h, e)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = make_instances()

    vec = []
    for inst in instances:
        p = inst["hidden"]["params"]
        x_last = inst["hidden"]["x_last"]
        cum_last = inst["hidden"]["cum_last"]

        obj_base = baseline(inst)
        oracle_cost = _oracle_policy(p, x_last, cum_last)
        obj_perfect = -oracle_cost
        denom = max(1e-6, (obj_perfect - obj_base) * _STRETCH)

        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (obj - obj_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        vec.append(max(0.0, min(1.0, r)))

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
