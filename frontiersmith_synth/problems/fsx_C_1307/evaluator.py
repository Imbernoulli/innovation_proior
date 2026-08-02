#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1307 -- "Cull Schedule for a Delay-Coupled
Vegetation-Prey-Predator Cascade" (family: predator-prey-manage; format B,
quality-metric).

THEME. A reserve holds three coupled populations, each expressed as a
fraction of its own carrying capacity: vegetation V, a herbivore/prey H
(the species management wants to save), and a predator Pr that is currently
overabundant and suppressing H. Managers may cull the predator every step of
a fixed horizon. The candidate submits the WHOLE cull schedule up front (the
entire instance -- initial state, rate constants, delays, floors -- is
public, exactly as this corpus's Format-B contract expects); the schedule is
then replayed CAUSALLY, step by step, against one fixed recurrence.

This composes three mechanisms into one objective:
  - trophic-cascade: grazing removes vegetation as a function of the CURRENT
    herbivore population; predation removes herbivores as a function of the
    CURRENT predator population (a Holling type-II saturating response).
    Effects propagate V -> H -> Pr and Pr -> H -> V every step.
  - time-delayed-response: H's OWN growth and starvation-mortality this step
    depend on vegetation from `tauV` steps ago (a maturation/gestation lag:
    this generation's births and starvation reflect the food conditions it
    was conceived under, not today's); Pr's numeric response depends on H
    from `tauH` steps ago. Grazing/predation, by contrast, are immediate.
  - population-floor-constraint: any of the three that drops below its
    (instance-specific) floor is clamped to EXACTLY 0 and never recovers --
    the recurrence is multiplicative in each species, so 0 is an absorbing,
    permanent local extinction.

THE TRAP. A reactive proportional controller -- cull hard while H is below
target, ease off once it recovers -- looks like the right "control-theory"
answer. But predator relief is felt by H immediately (predation drops NOW),
so H overshoots past what current vegetation regrowth (rV) can support,
while the resulting vegetation crash only reaches H's OWN growth/mortality
term `tauV` steps later -- by which time the controller has already stopped
culling (it saw H "doing fine") and cannot prevent H's now-inevitable
delayed starvation crash through the floor. Longer tauV widens the gap
between the correction and its consequence, and cripples the reactive
policy worse. THE INSIGHT: commit to a small, constant, SUSTAINED cull rate
-- calibrated DOWN as tauV grows -- that relieves predation gently enough
for vegetation regrowth to keep pace, never triggering the cascade.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance -- everything is public, no
          hidden oracle state; the difficulty is dynamical, not informational):
            {"T":int,"V0":float,"H0":float,"Pr0":float,
             "rV":float,"gH":float,"rH":float,"aPred":float,"hHalf":float,
             "mH":float,"rPr":float,"tauV":int,"tauH":int,
             "floor":{"V":float,"H":float,"Pr":float},"cull_max":float}
  stdout: ONE JSON object:
            {"cull":[c_0, c_1, ..., c_{T-1}]}     # length T, each in [0,cull_max]

  VALIDITY: `cull` must be a list of exactly T finite numbers (bool rejected),
  each in [0, cull_max] (small epsilon tolerance). Any violation, a malformed
  JSON shape, a crash, a timeout, or non-JSON output makes the WHOLE instance
  score 0.0.

SCORING (deterministic; no wall-time). The evaluator replays `cull` causally
from (V0,H0,Pr0) through the recurrence documented above/in statement.md,
clamping each updated value to [0,2] and then applying the population floor.
Per instance:
    avgH        = mean over t=1..T of min(1, H_t)
    persistence = (#{V,H,Pr} alive at t=T) / 3
    score       = 0.7*avgH + 0.3*persistence          (already in [0,1])
averaged over 10 fixed seeded instances.

ISOLATION. The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance. All replay and
grading are computed by THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- fixed instance bank ---------------------------
# 7 "short-delay" instances (tauV=2) where a reactive controller is not
# catastrophic (it recovers prey reasonably well, but still leaves real
# headroom for the calibrated small-sustained-cull insight), plus 3
# "long-delay" TRAP instances (tauV in {4,5,6}) where the reactive
# controller's overshoot-then-delayed-starvation cascade drives prey (and
# often the predator, gutted by its own initial over-cull) through the
# floor into permanent extinction.
INSTANCE_SPECS = [
    # name,  V0,   H0,   Pr0,  rV,   gH,   rH,   aPred, hHalf, mH,   rPr,  tauV, tauH, cull_max, T,  floorV, floorH, floorPr
    ("r01", 0.85, 0.20, 0.55, 0.28, 0.80, 0.55, 0.25, 0.45, 0.35, 0.35, 2, 2, 0.90, 44, 0.05, 0.040, 0.030),
    ("r02", 0.82, 0.20, 0.58, 0.28, 0.80, 0.55, 0.25, 0.45, 0.35, 0.35, 2, 3, 0.85, 46, 0.05, 0.045, 0.030),
    ("r03", 0.88, 0.20, 0.52, 0.28, 0.80, 0.55, 0.25, 0.45, 0.35, 0.35, 2, 4, 0.90, 42, 0.055, 0.040, 0.025),
    ("r04", 0.80, 0.20, 0.60, 0.28, 0.80, 0.55, 0.25, 0.45, 0.35, 0.35, 2, 5, 0.88, 48, 0.05, 0.040, 0.030),
    ("r05", 0.86, 0.20, 0.50, 0.28, 0.80, 0.55, 0.25, 0.45, 0.35, 0.35, 2, 2, 0.90, 45, 0.045, 0.035, 0.030),
    ("r06", 0.83, 0.20, 0.56, 0.28, 0.80, 0.55, 0.25, 0.45, 0.35, 0.35, 2, 3, 0.85, 43, 0.05, 0.040, 0.035),
    ("r07", 0.87, 0.20, 0.54, 0.28, 0.80, 0.55, 0.25, 0.45, 0.35, 0.35, 2, 6, 0.90, 47, 0.05, 0.040, 0.030),
    ("r08", 0.84, 0.20, 0.57, 0.28, 0.80, 0.55, 0.25, 0.45, 0.35, 0.35, 4, 4, 0.90, 48, 0.05, 0.040, 0.030),
    ("r09", 0.86, 0.20, 0.53, 0.28, 0.80, 0.55, 0.25, 0.45, 0.35, 0.35, 5, 5, 0.88, 50, 0.05, 0.040, 0.030),
    ("r10", 0.82, 0.20, 0.59, 0.28, 0.80, 0.55, 0.25, 0.45, 0.35, 0.35, 6, 5, 0.90, 50, 0.05, 0.040, 0.030),
]


def build_instances():
    out = []
    for (name, V0, H0, Pr0, rV, gH, rH, aPred, hHalf, mH, rPr,
         tauV, tauH, cull_max, T, fV, fH, fPr) in INSTANCE_SPECS:
        out.append({
            "name": name, "T": T, "V0": V0, "H0": H0, "Pr0": Pr0,
            "rV": rV, "gH": gH, "rH": rH, "aPred": aPred, "hHalf": hHalf,
            "mH": mH, "rPr": rPr, "tauV": tauV, "tauH": tauH,
            "floor": {"V": fV, "H": fH, "Pr": fPr}, "cull_max": cull_max,
        })
    return out


# ----------------------------- recurrence / replay ---------------------------
def replay(inst, cull):
    """Deterministically replay `cull` (length-T list of floats in [0,cull_max])
    against the fixed recurrence. Returns (Vh, Hh, Prh), each length T+1."""
    T = inst["T"]
    rV, gH = inst["rV"], inst["gH"]
    rH, aPred, hHalf, mH = inst["rH"], inst["aPred"], inst["hHalf"], inst["mH"]
    rPr = inst["rPr"]
    tauV, tauH = inst["tauV"], inst["tauH"]
    fV, fH, fPr = inst["floor"]["V"], inst["floor"]["H"], inst["floor"]["Pr"]

    Vh = [inst["V0"]]; Hh = [inst["H0"]]; Prh = [inst["Pr0"]]
    for t in range(T):
        c = cull[t]
        Vd = Vh[t - tauV] if t - tauV >= 0 else Vh[0]
        Hd = Hh[t - tauH] if t - tauH >= 0 else Hh[0]
        Vt, Ht, Prt = Vh[t], Hh[t], Prh[t]

        graze = gH * Ht * Vt
        Vn = Vt + rV * Vt * (1 - Vt) - graze

        denom = Ht + hHalf
        predation = aPred * Prt * Ht / denom if denom > 1e-12 else 0.0
        Hn = Ht + rH * Ht * Vd * (1 - Ht) - mH * Ht * (1 - Vd) - predation

        Prn = Prt + rPr * Prt * Hd * (1 - Prt) - c * Prt

        Vn = max(0.0, min(2.0, Vn))
        Hn = max(0.0, min(2.0, Hn))
        Prn = max(0.0, min(2.0, Prn))
        if Vn < fV: Vn = 0.0
        if Hn < fH: Hn = 0.0
        if Prn < fPr: Prn = 0.0

        Vh.append(Vn); Hh.append(Hn); Prh.append(Prn)
    return Vh, Hh, Prh


# ----------------------------- validation + scoring ---------------------------
def score_answer(inst, answer):
    """Validate + replay `answer`. Returns (ok, score in [0,1])."""
    if not isinstance(answer, dict):
        return False, 0.0
    cull = answer.get("cull")
    if not isinstance(cull, list):
        return False, 0.0
    T = inst["T"]
    if len(cull) != T:
        return False, 0.0
    cull_max = inst["cull_max"]
    vals = []
    for c in cull:
        if isinstance(c, bool) or not isinstance(c, (int, float)):
            return False, 0.0
        if not math.isfinite(c):
            return False, 0.0
        if c < -1e-9 or c > cull_max + 1e-9:
            return False, 0.0
        vals.append(max(0.0, min(cull_max, float(c))))

    Vh, Hh, Prh = replay(inst, vals)
    avgH = sum(min(1.0, x) for x in Hh[1:]) / T
    persistence = ((1.0 if Vh[-1] > 0 else 0.0) +
                    (1.0 if Hh[-1] > 0 else 0.0) +
                    (1.0 if Prh[-1] > 0 else 0.0)) / 3.0
    score = 0.7 * avgH + 0.3 * persistence
    if not (score == score) or score in (float("inf"), float("-inf")):
        return False, 0.0
    return True, max(0.0, min(1.0, score))


# ----------------------------- scoring driver --------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = build_instances()

    vec = []
    for inst in instances:
        # NOTE: no held-out/oracle data anywhere -- the whole instance IS the
        # public view (this problem's difficulty is dynamical, not
        # informational), except the internal "name" tag, which we strip so a
        # candidate cannot special-case these 10 fixed instances by id.
        public = {k: v for k, v in inst.items() if k != "name"}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, s = score_answer(inst, ans)
        except Exception:
            ok, s = False, 0.0
        vec.append(s if ok else 0.0)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
