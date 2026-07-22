#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0863 -- "Titrate the Unknown: Closed-Loop Equivalence-Point
Control" (family: adaptive-titration-controller; format B, quality-metric; theme:
titrate unknown samples to a target pH curve).

THEME.  A sealed sample (an unknown mixture of 1-3 weak-acid "species") sits in a vessel
at V=0.  You may add a strong-base titrant, but ONLY forward: cumulative added volume V is
monotone non-decreasing and additions can NEVER be undone (irreversible, exactly like real
titrant in a beaker).  After each addition the vessel's pH can be read.  You must drive the
solution to a given target pH using as little titrant and as few readings as possible,
without overshooting past a narrow high-slope "equivalence jump" that the sample may hide
anywhere along the way.

FORWARD MODEL (hidden; the evaluator alone computes it).  The sample's pH as a function of
cumulative titrant volume V is a sum of independent logistic "neutralization steps", one per
species:  pH(V) = pH0 + sum_i A_i * sigmoid(k_i * (V - Veq_i)).  Each species contributes a
smooth rise of height A_i centered at its own equivalence volume Veq_i.  Species with a SMALL
k_i produce a wide, gently-buffered plateau (high buffer capacity: pH moves little per mL);
species with a LARGE k_i produce a narrow, steep jump (low buffer capacity: pH moves a lot
per mL).  Composing species with sharply different k_i in one sample is what makes the curve
"buffer-nonlinear": local slope (dpH/dV) can change by an order of magnitude over a couple of
mL, and where that happens is not disclosed -- only the final target pH value is.

WHY THE MECHANISMS ARE FORCED (all three shape the score).
  * buffer-nonlinearity -- the local slope dpH/dV varies hugely along the same curve (flat
    plateau vs. steep jump), so no FIXED step size or FIXED gain is right everywhere.
  * closed-loop-inversion -- since titrant can only be ADDED, "bracket and bisect" (which
    requires deliberately overshooting on purpose to bracket a root) is not a safe option:
    once you pass the target you cannot go back.  The only sound approach is to invert the
    LOCAL forward model online -- estimate the current slope from recent (dV, dpH) pairs and
    take a Newton-like step toward the target -- while staying provably cautious about
    unexplored territory just ahead.
  * sample-efficiency -- every reading costs one of a small number of rounds, and every mL
    used beyond what was actually needed is charged; spending the whole budget crawling a
    flat plateau in tiny steps is also a failure mode, not just overshoot.

INNOVATION HOOK (what `strong` exploits).  `strong` maintains an online ESTIMATE of local
buffer capacity from the two most recent (V, pH) readings (a secant slope) and takes a damped
Newton step (target-pH)/slope toward the target.  Crucially it does NOT trust that estimate
blindly: it runs a genuine TRUST-REGION policy -- the allowed step is capped (a) as a small,
curvature-checked fraction of the remaining volume budget (comparing the two most recent
secant slopes; a big change signals an unseen regime boundary might be near), (b) to a bounded
multiple of the step that produced the current reading (never leap far past ground you have
not explored), and (c) tightened further as the (possibly stale) predicted pH gap shrinks,
since the target sits, by construction, wherever the curve's steepest region may be.  This
composition -- online slope inversion PLUS trust-region caution, purely reconstructed each
call from the given history (no hidden persistent state) -- is what lets it close in on a
hidden equivalence jump instead of vaulting across it.

TRAP.  6 of the 10 instances (a "plateau-then-jump" pair, a "jump-then-plateau" pair, and two
3-species "triple" instances) place the target pH at or beyond an equivalence point whose
local buffer capacity is 5-15x lower than the region the search has to cross first.  A single
FIXED titrant increment (the `trivial` recipe) or a FIXED-GAIN proportional controller (the
`greedy` recipe -- an average coder's first idea: step proportional to the pH error with one
gain picked from the full pH range) both calibrate their step size to the AVERAGE slope of the
whole domain: correct for neither regime, so on these instances they either crawl the gentle
region for most of the round budget or blow straight through the steep one once they reach it.

CANDIDATE CONTRACT (isolated stdin -> stdout program, called ONCE PER ROUND, stateless -- it
must re-derive everything from the `history` it is handed each call).
  stdin (JSON): {"phase":"step","round":r,"max_rounds":R,"rounds_left":R-r,
                 "V":<float>,"pH":<float>,"history":[[V0,pH0],...,[V,pH]],
                 "V_max":<float>,"target_pH":<float>}
  stdout (JSON): {"add": <float, mL of titrant to add next; <=0 means STOP now>}
The evaluator -- not the candidate -- performs 100% of the forward simulation from each
`add`, so there is no way to report a small-looking addition while secretly achieving a
different one: the sequence of `add` values IS the titration.

FEASIBILITY (any violation -> 0.0 on that instance): `add` must be a finite real number (not
NaN/Inf/bool); if the running total V+add would exceed V_max, the answer is infeasible; and no
single `add` may exceed `max_add = 0.35*V_max` (a stated burette single-dispense safety limit,
disclosed in the public view every round) -- every instance needs more than 0.35*V_max to reach
its target, so this makes "jump straight to a known equivalence volume in one shot" structurally
impossible: genuine multi-round, feedback-driven convergence is always required.

SCORING (deterministic; no wall-time).  Let V_final, pH_final be the state when the candidate
stops (add<=0) or the round budget (R) is exhausted, and steps the number of additions made.
    pH_err  = |pH_final - target_pH|
    excess  = max(0, V_final - V*) / V*         (V* = the unique true root pH(V*)=target_pH)
    cost    = pH_err + 0.12*excess + 0.15*(steps/R)
  A reference `trivial` construction (uniform fixed-size increments, stopping the first time
  pH crosses the target) gives cost_ref.  quality = clip(1 - cost/cost_ref, 0, 1);
  r = 0.10 + 0.82*quality  (so a trivial-quality run maps to exactly 0.10, and the 0.82 span
  caps every run at 0.92 -- headroom is enforced by construction, not by an unbounded ratio).
  Infeasible / malformed / crashed / timed-out candidates score 0.0 on that instance.
  Final Ratio = mean r over 10 fixed seeded instances.

ISOLATION.  The candidate is untrusted and runs in a FRESH bwrap-SANDBOXED SUBPROCESS via
`isorun.run_candidate`, once per round; it sees only the public per-round state above.  The
species parameters, Veq's, steepnesses, and V* live only in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
"""
import sys, json, math
import isorun

# --------------------------- scoring / protocol constants ------------------------
OFFSET = 0.10
SPAN = 0.82           # r = OFFSET + SPAN*quality ; max attainable r = 0.92 (headroom)
MAX_ROUNDS = 18        # per-instance round budget (closed-loop calls to the candidate)
ALPHA = 1.0            # pH-error weight
BETA = 0.12            # excess-titrant weight
GAMMA = 0.15           # step-count (sample-efficiency) weight
MAX_ADD_FRAC = 0.35    # burette single-dispense safety limit: any one `add` <= this
                       # fraction of V_max (a STATED, public rule -- see statement.md).
                       # Every instance needs V* > MAX_ADD_FRAC*V_max, so this makes a
                       # single "jump straight to the equivalence point" answer literally
                       # impossible: genuine multi-round, feedback-driven convergence is
                       # structurally required regardless of what a candidate might already
                       # believe it knows about the target.


# ------------------------------- deterministic RNG --------------------------------
def _rng(seed):
    state = seed & ((1 << 64) - 1)

    def u01():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    return u01


def _uni(u, lo, hi):
    return lo + (hi - lo) * u()


# ------------------------------- forward model -------------------------------------
def _sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _pH_of(V, spec):
    pH = spec["pH0"]
    for (A, k, Veq) in spec["species"]:
        pH += A * _sigmoid(k * (V - Veq))
    return pH


def _solve_Vstar(spec, target, V_max):
    """The forward model is strictly increasing in V, so the root is unique; bisect it
    (this is a GENERATOR-side numerical convenience, not something the candidate can do --
    it never sees `spec`)."""
    lo, hi = 0.0, V_max
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _pH_of(mid, spec) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------- instance family ------------------------------------
def _make_one(seed, kind):
    u = _rng(seed)
    pH0 = round(_uni(u, 1.8, 2.6), 4)
    if kind == "gentle":
        A = round(_uni(u, 7.0, 9.0), 4)
        k = round(_uni(u, 0.12, 0.22), 4)
        Veq = round(_uni(u, 18.0, 26.0), 4)
        species = [(A, k, Veq)]
        V_max = round(Veq * 1.6, 4)
        target_idx = 0
    elif kind == "trap_plateau_then_jump":
        A1 = round(_uni(u, 3.0, 4.0), 4); k1 = round(_uni(u, 0.10, 0.18), 4)
        Veq1 = round(_uni(u, 8.0, 12.0), 4)
        A2 = round(_uni(u, 3.5, 5.0), 4); k2 = round(_uni(u, 1.0, 2.0), 4)
        Veq2 = round(Veq1 + _uni(u, 4.0, 7.0), 4)
        species = [(A1, k1, Veq1), (A2, k2, Veq2)]
        V_max = round(Veq2 * 1.4, 4)
        target_idx = 1
    elif kind == "trap_jump_then_plateau":
        A1 = round(_uni(u, 3.0, 4.5), 4); k1 = round(_uni(u, 1.2, 2.2), 4)
        Veq1 = round(_uni(u, 3.0, 6.0), 4)
        A2 = round(_uni(u, 3.5, 5.0), 4); k2 = round(_uni(u, 0.10, 0.18), 4)
        Veq2 = round(Veq1 + _uni(u, 10.0, 16.0), 4)
        species = [(A1, k1, Veq1), (A2, k2, Veq2)]
        V_max = round(Veq2 * 1.35, 4)
        target_idx = 1
    else:  # "triple"
        A1 = round(_uni(u, 2.2, 3.0), 4); k1 = round(_uni(u, 1.0, 2.0), 4)
        Veq1 = round(_uni(u, 3.0, 5.0), 4)
        A2 = round(_uni(u, 2.2, 3.0), 4); k2 = round(_uni(u, 0.12, 0.20), 4)
        Veq2 = round(Veq1 + _uni(u, 6.0, 9.0), 4)
        A3 = round(_uni(u, 2.5, 3.5), 4); k3 = round(_uni(u, 1.0, 2.0), 4)
        Veq3 = round(Veq2 + _uni(u, 4.0, 7.0), 4)
        species = [(A1, k1, Veq1), (A2, k2, Veq2), (A3, k3, Veq3)]
        V_max = round(Veq3 * 1.3, 4)
        target_idx = 2

    spec = {"pH0": pH0, "species": species}
    Veq_t = species[target_idx][2]
    target = round(_pH_of(Veq_t, spec), 4)
    Vstar = _solve_Vstar(spec, target, V_max)
    return {"spec": spec, "V_max": V_max, "target": target, "Vstar": Vstar, "kind": kind}


def _build_instances():
    specs = [
        (86301, "gentle"), (86302, "gentle"), (86303, "gentle"),
        (86304, "trap_plateau_then_jump"), (86305, "trap_plateau_then_jump"),
        (86306, "trap_plateau_then_jump"),
        (86307, "trap_jump_then_plateau"), (86308, "trap_jump_then_plateau"),
        (86309, "triple"), (86310, "triple"),
    ]
    return [_make_one(seed, kind) for seed, kind in specs]


# ------------------------------- protocol / cost -------------------------------------
def _public_view(inst, rnd, V, pH, history):
    return {
        "phase": "step", "round": rnd, "max_rounds": MAX_ROUNDS,
        "rounds_left": MAX_ROUNDS - rnd,
        "V": V, "pH": pH, "history": [[h[0], h[1]] for h in history],
        "V_max": inst["V_max"], "target_pH": inst["target"],
        "max_add": round(MAX_ADD_FRAC * inst["V_max"], 6),
    }


def _cost(inst, V_final, pH_final, steps):
    pH_err = abs(pH_final - inst["target"])
    Vstar = max(inst["Vstar"], 1e-6)
    excess = max(0.0, V_final - inst["Vstar"]) / Vstar
    return ALPHA * pH_err + BETA * excess + GAMMA * (steps / float(MAX_ROUNDS))


def _reference_cost(inst):
    """Uniform fixed-size increments (no feedback beyond a crossing check) -- the same
    construction the `trivial` solution implements; the evaluator computes it itself so a
    trivial candidate normalizes to exactly OFFSET."""
    spec = inst["spec"]; V_max = inst["V_max"]; target = inst["target"]
    step = V_max / MAX_ROUNDS
    V = 0.0
    pH = _pH_of(0.0, spec)
    steps = 0
    for _ in range(MAX_ROUNDS):
        if pH >= target:
            break
        remaining = V_max - V
        if remaining <= 1e-12:
            break
        V += min(step, remaining)
        pH = _pH_of(V, spec)
        steps += 1
    return _cost(inst, V, pH, steps)


def _run_instance(cand, inst):
    spec = inst["spec"]; V_max = inst["V_max"]
    V = 0.0
    pH = _pH_of(0.0, spec)
    history = [[V, pH]]
    steps = 0
    for rnd in range(MAX_ROUNDS):
        pub = _public_view(inst, rnd, V, pH, history)
        ans, st = isorun.run_candidate(cand, pub, timeout=20)
        if st != "OK" or not isinstance(ans, dict):
            return 0.0
        add = ans.get("add")
        if isinstance(add, bool) or not isinstance(add, (int, float)):
            return 0.0
        add = float(add)
        if add != add or add in (float("inf"), float("-inf")):
            return 0.0
        if add <= 0.0:
            break
        if add > MAX_ADD_FRAC * V_max + 1e-6:
            return 0.0
        new_V = V + add
        if new_V > V_max + 1e-6:
            return 0.0
        V = min(new_V, V_max)
        pH = _pH_of(V, spec)
        history.append([V, pH])
        steps += 1

    cost = _cost(inst, V, pH, steps)
    ref = _reference_cost(inst)
    if ref <= 1e-9:
        ref = 1e-9
    quality = 1.0 - cost / ref
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
