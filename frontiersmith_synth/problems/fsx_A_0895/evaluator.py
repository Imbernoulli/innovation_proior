#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0895 -- "Steady-State Titration Ward"
(family: accumulation-aware-titration; format B, quality-metric).

THEME.  A ward runs a fixed, uniform PROBE protocol on every incoming patient: a
short, publicly-known sequence of test doses is administered and the (noisy)
resulting drug concentration is read after each one.  The drug behaves as a
one-compartment pharmacokinetic accumulator: each step a known RETENTION
fraction `rho` of the current concentration carries over, and the new dose
contributes `S_i * dose`, where `S_i` is the patient's HIDDEN sensitivity
(unknown to the candidate, constant per patient, drawn from a population
range).  After the probe, the candidate must commit -- in ONE shot, with no
further feedback -- to the full remaining dosing plan for every patient, aimed
at keeping concentration inside a therapeutic window without drifting into
toxicity.

MECHANISM 1 (hidden-sensitivity-narrowing): S_i must be inferred from the
noisy multi-step probe curve, not read off directly.

MECHANISM 2 (pharmacokinetic-accumulation): concentration is a decaying sum of
ALL past doses (geometric memory via `rho`), not a fresh one-step response, so
naive "assume no carry-over" arithmetic is systematically wrong -- and the
error compounds every subsequent step of the treatment phase.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance) -- see statement.md.
  stdout: ONE JSON object {"doses": [[d_0..d_{T-1}] per patient, in order]}.

SCORING (deterministic; no wall-time).  For each patient the evaluator
replays the TRUE deterministic dynamics (true rho, true hidden S_i, starting
from the patient's TRUE end-of-probe concentration) forward through the
candidate's proposed doses, and averages a per-step reward that gives full
credit inside the therapeutic window, partial credit for under-dosing,
decaying credit above the window ceiling, and a growing penalty beyond the
toxic ceiling.  Per-instance raw scores (averaged over patients) are mapped
through a fixed affine window (calibrated so a population-average constant
dose clusters low and headroom remains above the reference "strong" solver)
and clamped to [0,1].

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance. Hidden
sensitivities and true concentrations never leave this parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun

# ----------------------------- fixed constants ------------------------------
SEED = 8950895
N_INSTANCES = 10
N_PATIENTS = 5
T_TREAT = 8                       # candidate-controlled treatment steps
PROBE_DOSES = [1.5, 2.5, 2.0, 3.0]
M_PROBE = len(PROBE_DOSES)
DOSE_MAX = 80.0
S_LO, S_HI = 0.7, 1.6
NOISE_REL = 0.22
RAW_MIN, RAW_MAX = -2.4, 1.3      # fixed affine normalization anchors

# rho (retention fraction) schedule across the 10 instances: a deliberate mix
# of strong-accumulation ("slow clearance") regimes and mild ones, so that a
# controller which ignores accumulation is punished hard on some instances and
# only mildly on others (a purely random schedule could accidentally under- or
# over-represent the trap; this one guarantees >=3 trap instances by design).
RHO_SCHEDULE = [0.85, 0.80, 0.30, 0.88, 0.20, 0.75, 0.40, 0.90, 0.60, 0.15]


def _lcg_placeholder():
    pass


def _basis(rho, doses):
    """basis[t-1] = sum_{j=0}^{t-1} rho^{t-1-j} * doses[j], i.e. the true
    concentration at probe step t for a patient with S=1 and c0=0."""
    b = []
    c = 0.0
    for d in doses:
        c = rho * c + d
        b.append(c)
    return b


def _step_score(c, lo, hi, tox):
    """Per-step reward: full credit in [lo,hi]; partial credit approaching
    from below; decaying credit above hi up to tox; growing penalty beyond
    tox (capped so a single bad step cannot ruin the whole trajectory)."""
    if c <= 0:
        return 0.0
    if c < lo:
        return 0.5 * (c / lo)
    if c <= hi:
        return 1.0
    if c <= tox:
        return 1.0 - 0.6 * (c - hi) / (tox - hi)
    over = (c - tox) / (tox - hi)
    return -min(2.0, over)


def make_instances():
    import random
    insts = []
    for i in range(N_INSTANCES):
        rho = RHO_SCHEDULE[i]
        rng = random.Random(SEED * 1000003 + i * 97 + 13)
        target = round(rng.uniform(8.0, 15.0), 2)
        lo = round(0.92 * target, 3)
        hi = round(1.08 * target, 3)
        tox = round(1.35 * target, 3)
        tb = _basis(rho, PROBE_DOSES)          # per-unit-S basis at each probe step
        patients_pub, S_list, c_end_list = [], [], []
        for p in range(N_PATIENTS):
            S = round(rng.uniform(S_LO, S_HI), 4)
            true_c = [S * b for b in tb]
            noisy = []
            for c in true_c:
                sigma = NOISE_REL * max(1.0, c)
                noisy.append(max(0.0, round(c + rng.gauss(0.0, sigma), 4)))
            patients_pub.append({"patient_id": p, "probe_readings": noisy})
            S_list.append(S)
            c_end_list.append(true_c[-1])
        public = {
            "instance_id": i,
            "rho": rho,
            "target": target,
            "window_low": lo,
            "window_high": hi,
            "toxic_ceiling": tox,
            "probe_doses": list(PROBE_DOSES),
            "treatment_steps": T_TREAT,
            "dose_max": DOSE_MAX,
            "patients": patients_pub,
        }
        hidden = {"S_list": S_list, "c_end_probe": c_end_list,
                  "lo": lo, "hi": hi, "tox": tox, "basis": tb}
        insts.append({"public": public, "hidden": hidden})
    return insts


def baseline(inst):
    """Internal reference: a population-average constant dose, computed
    entirely by the evaluator (never exposed to the candidate) -- assumes
    S=1 and treats target as the per-step dose (no accumulation modeling).
    Used only as a sanity anchor, not part of the scoring formula itself."""
    pub, hid = inst["public"], inst["hidden"]
    rho = pub["rho"]
    doses = [pub["target"]] * pub["treatment_steps"]
    raws = []
    for S, c0 in zip(hid["S_list"], hid["c_end_probe"]):
        c = c0
        steps = []
        for d in doses:
            c = rho * c + S * d
            steps.append(_step_score(c, hid["lo"], hid["hi"], hid["tox"]))
        raws.append(sum(steps) / len(steps))
    return sum(raws) / len(raws)


def score(inst, answer):
    """Validate `answer` strictly against inst['public']+inst['hidden'];
    return (ok, raw_objective). raw_objective is the mean per-step reward
    across all patients, BEFORE the final affine normalization (done in
    main() so the normalization anchors are visible in one place)."""
    pub, hid = inst["public"], inst["hidden"]
    P = len(pub["patients"])
    Tt = pub["treatment_steps"]
    dose_max = pub["dose_max"]
    rho = pub["rho"]
    lo, hi, tox = hid["lo"], hid["hi"], hid["tox"]

    if not isinstance(answer, dict):
        return False, 0.0
    doses = answer.get("doses")
    if not isinstance(doses, list) or len(doses) != P:
        return False, 0.0

    raws = []
    for p in range(P):
        seq = doses[p]
        if not isinstance(seq, list) or len(seq) != Tt:
            return False, 0.0
        clean = []
        for d in seq:
            if isinstance(d, bool) or not isinstance(d, (int, float)):
                return False, 0.0
            fd = float(d)
            if fd != fd or fd in (float("inf"), float("-inf")):
                return False, 0.0
            if fd < 0.0 or fd > dose_max:
                return False, 0.0
            clean.append(fd)
        S = hid["S_list"][p]
        c = hid["c_end_probe"][p]
        steps = []
        for d in clean:
            c = rho * c + S * d
            steps.append(_step_score(c, lo, hi, tox))
        raws.append(sum(steps) / len(steps))

    return True, sum(raws) / len(raws)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()

    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, raw = score(inst, ans)
        except Exception:
            ok, raw = False, 0.0
        if not ok:
            vec.append(0.0)
            continue
        r = (raw - RAW_MIN) / (RAW_MAX - RAW_MIN)
        if not (r == r):        # NaN guard
            r = 0.0
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
