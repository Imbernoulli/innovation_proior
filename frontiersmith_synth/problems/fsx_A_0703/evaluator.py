#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0703 -- "Two-Speed Reservoir: Dam Release Scheduling Against
Seeded Rainfall" (family: multiscale-reservoir-release; format B, quality-metric).

THEME.  A dam operator sees a full T-day rainfall record in advance and must choose a
release for every day. Each day's inflow mixes two speeds baked into ONE plain sequence
of numbers: a slow, smooth seasonal drift plus occasional fast, large storm bursts (1-2
days each) riding on top -- nothing in the input tags which day is which. The outlet can
only pass at most `Rmax` units per day (FAST-SLOW-TIMESCALE COUPLING: the control's own
reaction speed is capped while the forcing signal has both a slow and a fast component).
If the level ends a day above `cap`, the excess spills at a `flood_coef` penalty per unit
(RESERVOIR-OVERFLOW-SPILL); if it ends below `min_level`, the deficit costs a
`shortage_coef` penalty per unit. Objective: minimize total penalty over the record.

INNOVATION HOOK.  Because `Rmax` bounds same-day reaction, a storm burst whose excess (over
what the recent local baseline would suggest) exceeds `Rmax` CANNOT be absorbed reactively
on the day it lands, no matter how aggressively you release that day -- the only way to
avoid (or shrink) the spill is to have already lowered the level in the days BEFORE the
storm, using the fact that the whole record -- including the slow seasonal trend that will
determine how much headroom is safe to spend -- is known in advance. A solver that reacts
to a single fixed "keep the reservoir full for water security" target (the obvious first
approach) gets overtopped by any storm bigger than `Rmax` above its target's slack. A
solver that scans the record, detects where the inflow departs from its own local rolling
baseline by more than `Rmax`, and preemptively lowers its target in the days before that
excess arrives (while respecting `min_level` so it doesn't trade flood risk for shortage
risk) recovers most of that lost score.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "T": int, "cap": float, "Rmax": float, "L0": float,
             "min_level": float, "flood_coef": float, "shortage_coef": float,
             "inflow": [T floats, >= 0]}
  stdout: ONE JSON object:
            {"releases": [r_0, ..., r_{T-1}]}   # EXACTLY T finite numbers, each in [0, Rmax]

  Wrong length, a non-numeric/out-of-range/non-finite entry, a crash, a timeout, or
  non-JSON output -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Simulate, starting L = L0, for t = 0..T-1:
    avail          = L + inflow[t]
    actual_release = clip(releases[t], 0, Rmax, avail)
    raw            = avail - actual_release
    if raw > cap:  penalty += flood_coef * (raw - cap);  L = cap
    else:          L = raw; if L < min_level: penalty += shortage_coef * (min_level - L)
  y_cand = total penalty (candidate). y_base = penalty of releasing 0 every day (the
  evaluator's own weak "do nothing" reference, computed on the SAME instance).
    r = clamp( 0.1 + 0.9 * (y_base - y_cand) / (1.2 * y_base), 0, 1 )
  Matching do-nothing scores ~0.1; doing worse scores 0 (clamped); beating it scores
  higher. The denominator anchors against a reference BETTER than the physically-optimal
  zero-penalty outcome (unreachable), so even a flawless zero-penalty run scores ~0.85,
  never 1.0 -- headroom stays open above any reference solution. Final score is the mean
  of r over 10 instances (varied capacity/Rmax/storm timing&size, some tight-budget /
  held-out cases).

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance above (every number needed
to plan well -- inflow, Rmax, coefficients, floor -- is already public). All references
(y_base) and all validation happen in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    box = [state]

    def nxt_float():
        box[0] = (box[0] * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((box[0] >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    return nxt_float


# ----------------------------- instance family -----------------------------
def _build_one(seed, T, cap, Rmax, L0, min_level, flood_coef, shortage_coef,
               base_flow, slow_amp, slow_period, storm_days, storm_mag, tag):
    """Deterministic instance: `inflow` = slow seasonal sinusoid (+ small seeded noise)
    with `storm_days` fast bursts of size `storm_mag` (plus a decaying tail day) added on
    top. Only the combined series is exposed publicly."""
    rng = _rng(seed)
    phase = rng() * slow_period
    inflow = []
    for t in range(T):
        s = base_flow + slow_amp * math.sin(2 * math.pi * (t - phase) / slow_period)
        s = s + (rng() - 0.5) * base_flow * 0.08
        inflow.append(max(0.0, s))
    mags = storm_mag if isinstance(storm_mag, list) else [storm_mag] * len(storm_days)
    for d, mag in zip(storm_days, mags):
        if d < T:
            inflow[d] += mag
        if d + 1 < T:
            inflow[d + 1] += mag * 0.35
    inflow = [round(x, 6) for x in inflow]
    return {"name": tag, "T": T, "cap": float(cap), "Rmax": float(Rmax), "L0": float(L0),
            "min_level": float(min_level), "flood_coef": float(flood_coef),
            "shortage_coef": float(shortage_coef), "inflow": inflow}


def _build_instances():
    specs = [
        dict(seed=301, T=60, cap=1000, Rmax=40, L0=500, min_level=150, flood_coef=6, shortage_coef=3,
             base_flow=14, slow_amp=8, slow_period=60, storm_days=[40], storm_mag=700, tag="s301_singlestorm"),
        dict(seed=302, T=70, cap=1200, Rmax=50, L0=600, min_level=200, flood_coef=5, shortage_coef=4,
             base_flow=16, slow_amp=9, slow_period=70, storm_days=[50], storm_mag=600, tag="s302_cluster"),
        dict(seed=303, T=80, cap=900, Rmax=30, L0=400, min_level=120, flood_coef=8, shortage_coef=2,
             base_flow=11, slow_amp=6, slow_period=80, storm_days=[20, 55], storm_mag=500, tag="s303_twostorm"),
        dict(seed=304, T=50, cap=800, Rmax=45, L0=300, min_level=100, flood_coef=4, shortage_coef=5,
             base_flow=13, slow_amp=7, slow_period=50, storm_days=[], storm_mag=0, tag="s304_nostorm"),
        dict(seed=305, T=90, cap=1500, Rmax=60, L0=700, min_level=250, flood_coef=6, shortage_coef=3,
             base_flow=17, slow_amp=10, slow_period=90, storm_days=[10, 45, 80], storm_mag=550, tag="s305_threestorm"),
        dict(seed=306, T=65, cap=1000, Rmax=35, L0=550, min_level=180, flood_coef=7, shortage_coef=6,
             base_flow=10, slow_amp=8, slow_period=65, storm_days=[30], storm_mag=650, tag="s306_storm_then_dry"),
        dict(seed=307, T=75, cap=1100, Rmax=40, L0=500, min_level=200, flood_coef=5, shortage_coef=5,
             base_flow=15, slow_amp=8, slow_period=75, storm_days=[15, 60], storm_mag=500, tag="s307_mixed"),
        dict(seed=308, T=100, cap=1300, Rmax=45, L0=650, min_level=220, flood_coef=6, shortage_coef=4,
             base_flow=13, slow_amp=8, slow_period=100, storm_days=[25, 55, 85], storm_mag=600, tag="s308_holdout_tight"),
        dict(seed=309, T=55, cap=700, Rmax=25, L0=350, min_level=100, flood_coef=9, shortage_coef=2,
             base_flow=12, slow_amp=6, slow_period=55, storm_days=[35], storm_mag=450, tag="s309_holdout_verytight"),
        dict(seed=310, T=85, cap=1200, Rmax=50, L0=600, min_level=210, flood_coef=5, shortage_coef=7,
             base_flow=14, slow_amp=9, slow_period=85, storm_days=[20, 65], storm_mag=520, tag="s310_holdout_shortagecoef"),
    ]
    return [_build_one(**s) for s in specs]


# ----------------------------- simulation / references ---------------------
def _simulate(inst, releases):
    cap = inst["cap"]; min_level = inst["min_level"]; Rmax = inst["Rmax"]
    flood_coef = inst["flood_coef"]; shortage_coef = inst["shortage_coef"]
    L = inst["L0"]; total = 0.0
    for t in range(inst["T"]):
        inflow_t = inst["inflow"][t]
        req = releases[t] if t < len(releases) else 0.0
        avail = L + inflow_t
        actual = min(max(req, 0.0), Rmax, avail)
        raw = avail - actual
        if raw > cap:
            total += flood_coef * (raw - cap)
            L = cap
        else:
            L = raw
            if L < min_level:
                total += shortage_coef * (min_level - L)
    return total


def _baseline(inst):
    return _simulate(inst, [0.0] * inst["T"])


def _validate(inst, answer):
    if not isinstance(answer, dict):
        return None
    rel = answer.get("releases")
    if not isinstance(rel, list):
        return None
    T = inst["T"]; Rmax = inst["Rmax"]
    if len(rel) != T:
        return None
    out = []
    for x in rel:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        xf = float(x)
        if not math.isfinite(xf):
            return None
        if xf < -1e-9 or xf > Rmax + 1e-6:
            return None
        out.append(max(0.0, min(Rmax, xf)))
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        y_base = _baseline(inst)
        denom = 1.2 * y_base
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "T": inst["T"], "cap": inst["cap"], "Rmax": inst["Rmax"],
                  "L0": inst["L0"], "min_level": inst["min_level"], "flood_coef": inst["flood_coef"],
                  "shortage_coef": inst["shortage_coef"], "inflow": list(inst["inflow"])}
        ans, st = isorun.run_candidate(cand, public, timeout=8)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            rel = _validate(inst, ans)
        except Exception:
            rel = None
        if rel is None:
            vec.append(0.0)
            continue
        y_cand = _simulate(inst, rel)
        r = 0.1 + 0.9 * (y_base - y_cand) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
