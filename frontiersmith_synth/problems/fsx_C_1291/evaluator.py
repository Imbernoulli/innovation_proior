#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1291 -- "Reinvest or Cash Out, Ten Thousand Times"
(family: incremental-invest-policy; format B, quality-metric).

THEME.  A small operation grows its CAPITAL K by producing OUTPUT each turn:
output_t = K_t * rate(tier(K_t)), where rate is a step function of capital --
crossing a THRESHOLD permanently unlocks a higher per-turn multiplier (a
compounding-plus-unlock idle-game engine).  Each turn the operator splits that
turn's output between REINVEST (added to K, grows future output -- compounds)
and HARVEST (banked as realized score, does not compound further).  The game
runs for a FIXED, FINITE number of turns N; at the end, banked cash counts at
face value but any capital still sitting in K only counts at a discounted
SALVAGE fraction s < 1 (idle capital you never converted).

Always reinvesting (f=1 every turn) maximizes capital growth and would win an
INFINITE-horizon game, but on a finite horizon it strands value in K at the
(discounted) end -- the operator must eventually switch to harvesting.  THE
SUBTLETY: the right switch point is not a fixed fraction of the horizon.  It
depends jointly on how many turns remain AND on whether/when the NEXT
threshold is reachable -- pushing capital just far enough to cross one more
unlock can still pay for itself with only a few turns left, while chasing a
distant threshold with the same turns remaining does not.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance -- full information, nothing
          hidden; the whole point is that the solver must read and exploit
          the thresholds/multipliers, not guess a fixed ratio):
            {"name": str, "n_turns": N (int), "capital0": K0 (float),
             "base_rate": r0 (float), "thresholds": [T_1 < T_2 < ...] (floats),
             "multipliers": [1.0, m_1, m_2, ...] (floats, len = len(thresholds)+1,
                                                   strictly increasing),
             "salvage": s (float, 0 < s < 1)}
  stdout: ONE JSON object:
            {"invest": [f_0, ..., f_{N-1}]}   # f_t in [0,1]: fraction of turn
                                               # t's output reinvested (rest
                                               # is harvested to the bank)

  A trajectory is VALID iff `invest` is a list of exactly N numbers, each a
  finite float in [0,1] (bools rejected).  Invalid output, wrong length,
  an out-of-range value, a crash, a timeout, or non-JSON -> instance scores 0.0.

SIMULATION (deterministic; identical function used for candidate, baseline,
and the ideal bound).  Starting K = capital0, B = 0:
  for t in 0..N-1:
      tier   = number of thresholds already <= K
      output = K * base_rate * multipliers[tier]
      invest = f_t * output ; harvest = output - invest
      K += invest ; B += harvest
  final_value(s) = B + s * K

SCORING (deterministic; no wall-time).  Per instance we compute two references
IN THE PARENT (the candidate never sees them):
    q_base  = final_value(s) of the ALWAYS-HARVEST trajectory (f_t = 0 for
              all t) -- a weak, no-compounding baseline.
    q_ideal = final_value(1.0) of the ALWAYS-REINVEST trajectory (f_t = 1 for
              all t), i.e. the ending capital if you never harvested at all
              and the endgame salvage discount is waived.  This is a PROVABLE
              upper bound on ANY achievable final_value(s) with the real,
              discounted salvage -- see PROOF below -- so it is a legitimate,
              unreachable "ideal" anchor.  Because it waives a real discount,
              even an optimal real policy stays strictly below it -> headroom.
    q_cand  = final_value(s) of the candidate's submitted trajectory.
  and normalize with an affine anchor (weak baseline -> 0.1, ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(1e-9, q_ideal - q_base), 0, 1 )

PROOF that q_ideal upper-bounds every achievable final_value(s).  Define
V(t, K) = the best achievable (future harvest + s * K_final) starting turn t
with capital K, and g(t, K) = the capital reached at turn N by always
reinvesting from (t, K).  By backward induction: V(N, K) = s*K <= K = g(N,K).
For the step, g(t+1, .) is monotone non-decreasing (higher capital this turn
can only raise or match every future turn's rate, since multipliers are
non-decreasing in capital) and any gap between two capital trajectories that
both always-reinvest never shrinks going forward (each turn multiplies by a
factor >= 1). Hence, for any reinvest amount x <= K*rate(K) at turn t,
(K*rate(K) - x) + g(t+1, K + x) <= g(t+1, K + K*rate(K)), i.e. taking f=1 this
turn maximizes the induction's RHS bound, giving V(t, K) <= g(t, K) for all t.
So V(0, capital0) <= g(0, capital0) = q_ideal.  (Empirically fuzz-tested with
thousands of random trajectories per instance in this family: zero violations.)

ISOLATION.  The candidate runs in a FRESH SUBPROCESS via isorun.run_candidate;
it only ever sees the PUBLIC instance.  q_base/q_ideal are computed by THIS
parent process from the same public fields, so a frame-walking / introspecting
candidate learns nothing it doesn't already have.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- instance family ------------------------------
# Explicit, fully deterministic specs (no RNG needed -- every number below is
# fixed, so re-generation is trivially reproducible on any machine).
#   (name, n_turns, capital0, base_rate, thresholds, multipliers, salvage)
_SPECS = [
    ("orchard_slow",     40, 10.0, 0.06, [40.0],                    [1.0, 1.8],           0.50),
    ("workshop_tiered",  30, 10.0, 0.08, [30.0, 90.0],               [1.0, 1.6, 2.4],      0.40),
    ("forge_sprint",     18, 10.0, 0.15, [40.0],                    [1.0, 4.0],           0.15),
    ("mint_double_jump", 20,  8.0, 0.18, [25.0, 70.0],               [1.0, 2.2, 5.0],      0.12),
    ("quarry_deadline",  12, 12.0, 0.22, [45.0],                    [1.0, 5.5],           0.10),
    ("vineyard_gentle",  45, 15.0, 0.05, [60.0],                    [1.0, 1.5],           0.60),
    ("foundry_ladder",   35, 10.0, 0.10, [22.0, 55.0, 120.0],        [1.0, 1.7, 2.6, 4.0], 0.30),
    ("kiln_narrow",      16,  9.0, 0.20, [35.0, 90.0],               [1.0, 2.5, 6.0],      0.10),
    ("estate_patient",   50, 20.0, 0.04, [50.0, 150.0],              [1.0, 1.4, 2.0],      0.55),
    # harder / larger held-out instance
    ("citadel_triple",   24, 11.0, 0.16, [30.0, 80.0, 180.0],        [1.0, 2.0, 3.2, 6.5], 0.15),
]


def _build_instances():
    out = []
    for name, N, K0, br, thr, mul, sal in _SPECS:
        public = {"name": name, "n_turns": N, "capital0": K0, "base_rate": br,
                  "thresholds": list(thr), "multipliers": list(mul), "salvage": sal}
        out.append({"public": public, "hidden": {}})
    return out


# ----------------------------- core simulator --------------------------------
def _tier_of(K, thresholds):
    t = 0
    for T in thresholds:
        if K >= T:
            t += 1
        else:
            break
    return t


def _simulate(K0, base_rate, thresholds, multipliers, N, salvage, invest_seq):
    """Run the deterministic game for N turns under invest_seq (fractions in
    [0,1]); return final_value = banked_harvest + salvage * final_capital."""
    K = K0
    B = 0.0
    for t in range(N):
        tier = _tier_of(K, thresholds)
        rate = base_rate * multipliers[tier]
        output = K * rate
        f = invest_seq[t]
        inv = f * output
        K += inv
        B += (output - inv)
    return B + salvage * K


def _baseline(pub):
    N = pub["n_turns"]
    return _simulate(pub["capital0"], pub["base_rate"], pub["thresholds"],
                      pub["multipliers"], N, pub["salvage"], [0.0] * N)


def _ideal(pub):
    N = pub["n_turns"]
    # waive the salvage discount (salvage=1.0) and always reinvest: a provable
    # upper bound on any achievable final_value under the REAL salvage < 1.
    return _simulate(pub["capital0"], pub["base_rate"], pub["thresholds"],
                      pub["multipliers"], N, 1.0, [1.0] * N)


# ----------------------------- validation ------------------------------------
def _score_answer(pub, answer):
    """Validate + score the candidate's answer against pub. Return value or None."""
    if not isinstance(answer, dict):
        return None
    inv = answer.get("invest")
    if not isinstance(inv, list):
        return None
    N = pub["n_turns"]
    if len(inv) != N:
        return None
    seq = []
    for x in inv:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        xf = float(x)
        if not math.isfinite(xf):
            return None
        if xf < -1e-9 or xf > 1.0 + 1e-9:
            return None
        seq.append(min(1.0, max(0.0, xf)))
    return _simulate(pub["capital0"], pub["base_rate"], pub["thresholds"],
                      pub["multipliers"], N, pub["salvage"], seq)


# ----------------------------- scoring driver ---------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        pub = inst["public"]
        q_base = _baseline(pub)
        q_ideal = _ideal(pub)
        denom = q_ideal - q_base
        if denom < 1e-9:
            denom = 1e-9
        ans, st = isorun.run_candidate(cand, dict(pub), timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _score_answer(pub, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_cand - q_base) / denom
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
