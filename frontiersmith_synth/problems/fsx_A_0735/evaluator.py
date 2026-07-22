#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0735 -- "Roots on the Gamble: Carbon Allocation over
Hidden Nutrient Patches" (family: root-carbon-foraging-policy; format B,
quality-metric).

THEME.  A young plant has K root TIPS starting at the soil surface (depth 0).
Each tip follows its OWN fixed vertical shaft down to a maximum depth L; the
soil along each shaft has a hidden, patchy NUTRIENT CONCENTRATION profile
(mostly low-value background, with a few localized rich PATCHES at various
depths -- resource heterogeneity). Over T developmental steps the plant has a
fixed CARBON BUDGET B to spend that step, split however it likes across its
currently active tips (carbon-allocation-policy). Spending c carbon on a tip
extends it by c depth units and harvests the nutrient integrated over the
newly grown stretch, at a fixed construction COST per unit length. A tip can
only ever SENSE the concentration AT ITS OWN CURRENT POSITION -- it has no
knowledge of what lies deeper on itself or on any other tip until carbon is
actually spent to grow there (foraging-bet-hedging under partial
observability). Because a patch that currently reads best is only known
LOCALLY, committing the whole budget to today's best-looking tip (greedy) can
permanently strand the plant on a small/shallow patch while a far larger
patch sits undiscovered deeper on a tip that currently looks unpromising. The
optimal policy is a PORTFOLIO over tips: keep probing every live tip a little
every step so new information keeps arriving, while still weighting
investment toward the most promising evidence so far.

CANDIDATE CONTRACT (isolated stdin -> stdout program, called ONCE PER STEP).
The evaluator interacts with the candidate program T times per instance (a
fresh sandboxed subprocess each call -- the candidate is STATELESS between
calls and must reconstruct anything it wants to remember from the `history`
field it is handed).

  stdin (one call = one developmental step) : ONE JSON object
    { "name": str, "step": int, "steps_total": int, "budget_step": B (float),
      "ccost": float, "L": int, "n_tips": int,
      "tips": [ {"id": int, "pos": float, "sensed": float}, ... ],  # ACTIVE
               # tips only (pos < L); "sensed" = concentration AT the tip's
               # CURRENT position -- no lookahead into the future.
      "history": [ {"sensed": {tipid: val, ...}, "alloc": {tipid: amt, ...}},
                    ... ]   # one record per PAST step, in order
    }

  stdout: ONE JSON object   { "alloc": {tipid(str): amount, ...} }
    - `amount` for each active tip id: a non-negative finite number, carbon
      to spend growing that tip THIS step (it advances the tip's position by
      `min(amount, L - pos)`; overshoot is simply not credited).
    - Sum of amounts (over active ids) must be <= budget_step; unrecognized
      keys are ignored. Any negative / non-finite / non-numeric amount, or a
      total exceeding budget_step, makes THIS INSTANCE score 0.0 (all T steps
      voided). A crash, timeout, or non-JSON reply at ANY step also scores
      the whole instance 0.0.

SCORING (deterministic; no wall-time). Per instance the evaluator computes,
by DIRECT SIMULATION of the T-step interaction:
    obj  = sum over all steps and tips of (nutrient harvested on the newly
           grown stretch) - (ccost * carbon spent), i.e. net uptake minus
           construction cost, using the REAL hidden profile (only ever
           exposed to the candidate one currently-occupied cell at a time).
    weak = obj achieved by a FIXED "spread budget evenly across all active
           tips, every step" reference policy (the naive non-adaptive
           recipe) -- simulated directly by the evaluator, no candidate call.
    ub   = an OMNISCIENT offline upper bound: knowing the FULL profile of
           every tip in advance, and ignoring the per-step batching /
           partial-observability constraints (a strict relaxation), choose
           how much total carbon x_i in [0,L] to sink into each tip (a tip
           must be grown contiguously from the surface, so investing x_i
           always costs exactly x_i and nets integral_0^{x_i}(c_i-ccost))
           to maximize the sum, subject to sum x_i <= T*B. Solved EXACTLY
           by a small knapsack DP. ub >= any achievable online value.
  normalized with an affine anchor (reproduce the weak recipe -> 0.1, reach
  the offline oracle -> 1.0):
    r = clamp( 0.1 + 0.9 * (obj - weak) / max(ub - weak, 1e-6), 0, 1 )

ISOLATION. Every step's candidate call runs in a FRESH sandboxed subprocess
via `isorun.run_candidate`; it only ever sees that step's public JSON. The
hidden profiles, the weak baseline, and the oracle bound are computed only in
this parent process, so a frame-walking / introspecting candidate learns
nothing that helps it forage.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

CCOST = 1.0          # construction cost per unit carbon spent (fixed, global)
STEP_TIMEOUT = 8      # seconds per per-step isorun call


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = seed & ((1 << 64) - 1)

    def unit():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    return unit


def _gen_profile(seed, L, lo, hi, patches):
    """Deterministic length-L concentration profile: seeded background noise
    in [lo, hi] plus additive PATCHES. Each patch is (center, halfwidth, amp)
    for a tapered triangular bump, or (center, halfwidth, amp, "flat") for a
    flat-topped plateau bump (constant amp across the whole span)."""
    u = _rng(seed)
    base = [lo + (hi - lo) * u() for _ in range(L)]
    for p in patches:
        center, halfwidth, amp = p[0], p[1], p[2]
        shape = p[3] if len(p) > 3 else "tri"
        if halfwidth <= 0:
            if 0 <= center < L:
                base[int(center)] += amp
            continue
        for d in range(L):
            dist = abs(d - center)
            if dist <= halfwidth:
                if shape == "flat":
                    base[d] += amp
                else:
                    base[d] += amp * (1.0 - dist / halfwidth)
    return base


def _prefix(profile):
    cum = [0.0] * (len(profile) + 1)
    for i, v in enumerate(profile):
        cum[i + 1] = cum[i] + v
    return cum


# ----------------------------- instance family ------------------------------
def _build_instances():
    """Deterministic instance family. Each spec: name, K, L, T, B, a per-tip
    patch dict, and a background noise band (lo, hi). `seed` decorrelates
    each tip's background draw. Patches are (center, halfwidth, amp) for a
    tapered triangular bump, or (center, halfwidth, amp, "flat") for a
    flat-topped plateau.

    Instances bed301/bed302/bed303/bed321 are the TRAP cases: a small,
    flat-topped decoy patch touches the surface on one tip (so it is sensed
    immediately and fully dominates every background reading for the couple
    of steps it takes to cross it), while a substantially bigger flat-topped
    patch sits well below the reach of a fixed 1/K-per-tip share of the
    whole episode's budget on a DIFFERENT, never-yet-touched tip. Once the
    decoy is exhausted every remaining unexplored tip reads identical flat
    background, so a pure "chase today's best reading" policy provably
    never revisits a tip it has abandoned and never reaches the deep patch."""
    specs = [
        # --- TRAP: shallow decoy tip1, WIDE big deep patch tip3 (never touched by greedy;
        #     uniform's fixed 1/K share only nibbles the near edge and misses most of it) ---
        dict(name="bed301", seed=301, K=4, L=40, T=9, B=10, noise=(0.8, 0.8),
             patches={1: [(3, 3, 2.0, "flat")], 3: [(20, 12, 9.0, "flat")]}),
        # --- TRAP: shallow decoy tip2, WIDE big deep patch tip4 ---
        dict(name="bed302", seed=302, K=5, L=44, T=9, B=10, noise=(0.8, 0.8),
             patches={2: [(4, 4, 1.8, "flat")], 4: [(18, 10, 9.5, "flat")]}),
        # --- TRAP: two shallow decoys (tips1,2), one WIDE big deep patch (tip4) ---
        dict(name="bed303", seed=303, K=5, L=42, T=10, B=9, noise=(0.8, 0.8),
             patches={1: [(3, 3, 1.6, "flat")], 2: [(3, 3, 1.5, "flat")],
                      4: [(18, 10, 8.5, "flat")]}),
        # --- plain: one WIDE patch touching the surface on tip0 -- an uncontested,
        #     unambiguous best target from step 0; a fixed 1/K uniform share does not
        #     have the reach to fully cross it, so committing hard (greedy/strong) pays ---
        dict(name="bed311", seed=311, K=4, L=40, T=9, B=8, noise=(0.75, 0.85),
             patches={0: [(15, 15, 2.5, "flat")]}),
        # --- plain: one WIDE patch touching the surface on tip1 ---
        dict(name="bed312", seed=312, K=3, L=36, T=8, B=9, noise=(0.75, 0.85),
             patches={1: [(14, 14, 3.0, "flat")]}),
        # --- plain: no patches at all -- pure background noise (near-zero EV) ---
        dict(name="bed313", seed=313, K=4, L=30, T=8, B=8, noise=(0.75, 0.85),
             patches={}),
        # --- plain: one WIDE patch touching the surface on tip2 ---
        dict(name="bed314", seed=314, K=4, L=38, T=9, B=8, noise=(0.75, 0.85),
             patches={2: [(17, 17, 2.2, "flat")]}),
        # --- TRAP (held-out, larger): shallow decoy tip1, WIDE big deep patch tip4 ---
        dict(name="bed321", seed=321, K=5, L=48, T=10, B=10, noise=(0.8, 0.8),
             patches={1: [(4, 4, 2.0, "flat")], 4: [(22, 12, 9.5, "flat")]}),
        # --- plain (held-out, larger): one very wide patch touching the surface on tip1 ---
        dict(name="bed322", seed=322, K=4, L=46, T=10, B=10, noise=(0.75, 0.85),
             patches={1: [(25, 25, 4.0, "flat")]}),
        # --- plain (held-out): one wide patch touching the surface on tip3 ---
        dict(name="bed323", seed=323, K=4, L=40, T=9, B=9, noise=(0.75, 0.85),
             patches={3: [(20, 20, 3.5, "flat")]}),
    ]
    out = []
    for spec in specs:
        K, L = spec["K"], spec["L"]
        lo, hi = spec["noise"]
        profiles = []
        for t in range(K):
            pats = spec["patches"].get(t, [])
            profiles.append(_gen_profile(spec["seed"] * 100 + t, L, lo, hi, pats))
        cums = [_prefix(p) for p in profiles]
        out.append({"name": spec["name"], "K": K, "L": L, "T": spec["T"], "B": spec["B"],
                    "ccost": CCOST, "profiles": profiles, "cum": cums})
    return out


# ----------------------------- geometry helpers -----------------------------
def _integral(profile, cum, L, a, b):
    """integral of the piecewise-constant profile over [a, b), 0 <= a <= b <= L."""
    if b <= a:
        return 0.0
    a = max(0.0, min(a, L)); b = max(0.0, min(b, L))
    ai = int(a); bi = int(b)
    if ai >= L:
        return 0.0
    if ai == bi:
        return profile[ai] * (b - a)
    total = profile[ai] * ((ai + 1) - a)
    if bi - 1 > ai:
        total += cum[bi] - cum[ai + 1]
    if bi < L and b > bi:
        total += profile[bi] * (b - bi)
    return total


def _profile_at(inst, i, x):
    L = inst["L"]
    xi = int(x)
    if xi >= L:
        xi = L - 1
    if xi < 0:
        xi = 0
    return inst["profiles"][i][xi]


# ----------------------------- offline oracle (upper bound) -----------------
def _oracle_upper_bound(inst):
    K, L, ccost = inst["K"], inst["L"], inst["ccost"]
    total_budget = int(round(inst["T"] * inst["B"]))
    gs = []
    for i in range(K):
        cum = inst["cum"][i]
        gs.append([cum[x] - ccost * x for x in range(L + 1)])
    dp = [0.0] * (total_budget + 1)
    for i in range(K):
        g = gs[i]
        ndp = dp[:]
        for j in range(total_budget + 1):
            lim = min(L, j)
            best = dp[j]
            for x in range(1, lim + 1):
                v = dp[j - x] + g[x]
                if v > best:
                    best = v
            ndp[j] = best
        dp = ndp
    return dp[total_budget]


# ----------------------------- stepped simulation ---------------------------
def _make_public(inst, active, pos, history, t):
    return {"name": inst["name"], "step": t, "steps_total": inst["T"],
            "budget_step": inst["B"], "ccost": inst["ccost"], "L": inst["L"],
            "n_tips": inst["K"],
            "tips": [{"id": i, "pos": round(pos[i], 6),
                       "sensed": round(_profile_at(inst, i, pos[i]), 6)} for i in active],
            "history": history}


def _simulate(inst, get_answer):
    """Run the T-step interaction. `get_answer(public) -> (answer, ok)`.
    Returns the net objective, or None if the interaction was ever invalid."""
    K, T, B, L = inst["K"], inst["T"], inst["B"], inst["L"]
    pos = [0.0] * K
    history = []
    total = 0.0
    for t in range(T):
        active = [i for i in range(K) if pos[i] < L - 1e-9]
        if not active:
            break
        public = _make_public(inst, active, pos, history, t)
        answer, ok = get_answer(public)
        if not ok or not isinstance(answer, dict):
            return None
        alloc = answer.get("alloc")
        if not isinstance(alloc, dict):
            return None
        amt = {}
        s = 0.0
        for i in active:
            v = alloc.get(str(i), 0.0)
            if v is None:
                v = 0.0
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            v = float(v)
            if v != v or v in (float("inf"), float("-inf")):
                return None
            if v < -1e-9:
                return None
            v = max(v, 0.0)
            amt[i] = v
            s += v
        if s > B + 1e-6:
            return None
        step_hist = {"sensed": {}, "alloc": {}}
        for i in active:
            delta = min(amt[i], L - pos[i])
            profile, cum = inst["profiles"][i], inst["cum"][i]
            val = _integral(profile, cum, L, pos[i], pos[i] + delta)
            cost = inst["ccost"] * delta
            total += (val - cost)
            step_hist["sensed"][str(i)] = round(_profile_at(inst, i, pos[i]), 6)
            step_hist["alloc"][str(i)] = round(amt[i], 6)
            pos[i] += delta
        history.append(step_hist)
    return total


def _uniform_get_answer(public):
    tips = public["tips"]
    B = public["budget_step"]
    n = len(tips)
    share = B / n if n else 0.0
    return {"alloc": {str(t["id"]): share for t in tips}}, True


def _candidate_get_answer(cand):
    def fn(public):
        ans, st = isorun.run_candidate(cand, public, timeout=STEP_TIMEOUT)
        return ans, (st == "OK")
    return fn


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        weak = _simulate(inst, _uniform_get_answer)
        if weak is None:
            weak = 0.0
        ub = _oracle_upper_bound(inst)
        obj = _simulate(inst, _candidate_get_answer(cand))
        if obj is None:
            vec.append(0.0)
            continue
        denom = max(ub - weak, 1e-6)
        r = 0.1 + 0.9 * (obj - weak) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
