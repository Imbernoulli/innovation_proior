#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_N_0830 -- "The Calibrated Recall"
(family: exploratory-secretary; format B, quality-metric).

THEME. A hiring desk processes a queue of N candidates arriving one at a
time. Each candidate carries a public INTERVIEW SCORE in [0,1]. On arrival
the desk must ACCEPT (irreversible, spends one of K slots, full value) or
PASS (the candidate enters a RECALL POOL for up to W further arrivals --
recallable later at a decayed value, gone forever once the window expires).

TWO COMPOSED MECHANISMS drive the objective:

 (1) irreversible-recall-window. Accepting is a one-way door; passing is not
     quite final either -- there is a short, decaying grace period (W
     rounds, multiplicative decay per round of delay) during which a passed
     candidate can still be grabbed, after which they are gone for good.

 (2) sequential-observation-info-value. Each candidate's TRUE hidden value is
     v[i] = max(0, v_base(score[i]) * drift_factor(i) + noise[i]), where
     v_base is ONE of two possible regimes -- "Rising" (v_base increasing in
     score) or "Fading" (v_base DECREASING in score) -- fixed for the whole
     instance but never stated in the input. The regime is only inferable
     from the STATISTICAL SHAPE of the score stream itself (Rising instances
     draw scores clustered high; Fading instances draw scores clustered
     low), so watching more of the stream sharpens every later value
     estimate. drift_factor(i) = 1 + drift*i/(N-1) is PUBLIC (drift is given)
     and rewards patience: later arrivals are systematically richer.

 THE TRAP. A policy that treats the raw interview score as value itself
 ("higher score is better", the natural secretary-problem assumption) is
 exactly backwards on every Fading instance -- it will chase the worst
 candidates while ignoring the best ones -- and, regardless of regime, it
 also has no reason to hold slots back for the high-drift tail, so it burns
 its budget on low-drift early arrivals. The insight is to spend a modest
 early prefix inferring which regime is active from the score distribution's
 shape, convert every score to a regime- and drift-corrected value estimate,
 and only then commit slots -- using the recall window to rescue promising
 candidates passed on while still calibrating.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
    {"name": str, "N": int, "K": int, "W": int, "decay": float, "drift": float,
     "score": [s_0, ..., s_{N-1}]}     # N floats in [0,1]
  stdout: ONE JSON object:
    {"actions": [a_0, ..., a_{N-1}],        # a_i in {0,1}; 1 = accept now
     "recalls": [[r_0, j_0], [r_1, j_1], ...]}  # recall candidate j at round r
  A recall [r, j] is valid iff 0 <= j < r < N, r - j <= W, actions[j] == 0,
  and each j appears in at most one recall. sum(actions) + len(recalls) must
  not exceed K.

  Any malformed output (wrong shape/types, an invalid recall, exceeding K
  hires, non-finite values, a crash, a timeout, or non-JSON) -> that
  instance scores 0.0.

SCORING (deterministic; no wall-time). The evaluator REPLAYS the submitted
answer against the TRUE hidden v[] array (never shown to the candidate) to
get q_cand, the total realized value. Reference:
    q_base = 0                         (reject everyone)
    UB     = K * (BASE_MAX * (1+drift) + NOISE_MARGIN)   (loose per-instance
                                                            ceiling)
    r = clamp(0.1 + 0.9 * (q_cand - q_base) / (UB - q_base), 0, 1)
UB deliberately ignores the recall/window friction and the regime-inference
error any real policy incurs, so even a strong policy stays under 1.0 --
real headroom remains.

ISOLATION. The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the public instance. The hidden
regime, per-candidate noise, and true v[] array live only in this parent
process.

CLI: python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / (1 << 53)

    return nxt


def _gauss(nx):
    u1 = max(nx(), 1e-12)
    u2 = nx()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# ----------------------------- shared constants -----------------------------
BASE_MAX = 1.35        # max of v_base(o) over o in [0,1], both regimes
NOISE_SIGMA = 0.08
NOISE_MARGIN = 0.30    # slack folded into UB for noise headroom


def _drift_factor(i, N, drift):
    return 1.0 + drift * (i / max(1, N - 1))


# ----------------------------- instance family ------------------------------
def _build_instance(seed, N, K, W, decay, drift, regime):
    nx = _rng(seed)
    if regime == "rising":
        score = [nx() ** 0.5 for _ in range(N)]   # clusters high
        A, B = 1.2, 0.15
    else:
        score = [nx() ** 2.0 for _ in range(N)]   # clusters low
        A, B = -1.2, 1.35
    v_true = []
    for i in range(N):
        base = A * score[i] + B
        drifted = base * _drift_factor(i, N, drift)
        eps = _gauss(nx) * NOISE_SIGMA
        v_true.append(max(0.0, drifted + eps))
    return dict(name=f"desk{seed}", N=N, K=K, W=W, decay=decay, drift=drift,
                score=score, v_true=v_true, regime=regime)


def _build_instances():
    specs = [
        # seed,  N,   K,  W,  decay, drift, regime,   note
        (2001, 160, 14, 6, 0.90, 0.35, "rising", "easy-rising"),
        (2002, 160, 14, 6, 0.90, 0.35, "fading", "trap-fading"),
        (2003, 180, 16, 5, 0.88, 0.45, "fading", "trap-fading-highdrift"),
        (2004, 140, 12, 8, 0.92, 0.25, "rising", "medium-rising"),
        (2005, 200, 18, 6, 0.90, 0.50, "fading", "trap-fading-large"),
        (2006, 150, 13, 4, 0.85, 0.30, "rising", "short-window-rising"),
        (2007, 170, 15, 7, 0.90, 0.40, "fading", "trap-fading2"),
        (2008, 160, 14, 6, 0.90, 0.20, "rising", "low-drift-rising"),
        (2009, 190, 16, 5, 0.87, 0.55, "fading", "trap-fading-hard"),
        (2010, 175, 15, 6, 0.90, 0.35, "rising", "held-out-rising"),
    ]
    out = []
    for seed, N, K, W, decay, drift, regime, note in specs:
        inst = _build_instance(seed, N, K, W, decay, drift, regime)
        inst["note"] = note
        out.append(inst)
    return out


# ----------------------------- baseline / bound -----------------------------
def baseline(inst):
    """Reject everyone: zero risk, zero value. The weak (0.1) anchor."""
    return 0.0


def _upper_bound(inst):
    N, K, drift = inst["N"], inst["K"], inst["drift"]
    return K * (BASE_MAX * _drift_factor(N - 1, N, drift) + NOISE_MARGIN)


# ----------------------------- validation + true replay ---------------------
def score(inst, answer):
    """Validate `answer` and replay against the TRUE hidden v[] array.
    Returns (ok: bool, total_value: float)."""
    if not isinstance(answer, dict):
        return False, 0.0
    actions = answer.get("actions")
    recalls = answer.get("recalls")
    N, K, W, decay = inst["N"], inst["K"], inst["W"], inst["decay"]

    if not isinstance(actions, list) or len(actions) != N:
        return False, 0.0
    for a in actions:
        if isinstance(a, bool) or not isinstance(a, int) or a not in (0, 1):
            return False, 0.0

    if not isinstance(recalls, list):
        return False, 0.0
    seen_j = set()
    parsed_recalls = []
    for entry in recalls:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            return False, 0.0
        r, j = entry
        if isinstance(r, bool) or isinstance(j, bool):
            return False, 0.0
        if not isinstance(r, int) or not isinstance(j, int):
            return False, 0.0
        if not (0 <= j < N) or not (0 <= r < N):
            return False, 0.0
        if not (r > j) or (r - j) > W:
            return False, 0.0
        if actions[j] != 0:
            return False, 0.0
        if j in seen_j:
            return False, 0.0
        seen_j.add(j)
        parsed_recalls.append((r, j))

    n_hires = sum(actions) + len(parsed_recalls)
    if n_hires > K:
        return False, 0.0

    v = inst["v_true"]
    total = 0.0
    for i in range(N):
        if actions[i] == 1:
            total += v[i]
    for (r, j) in parsed_recalls:
        total += v[j] * (decay ** (r - j))

    if not (total == total) or total in (float("inf"), float("-inf")):
        return False, 0.0
    return True, total


# ----------------------------- scoring driver --------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = {"name": inst["name"], "N": inst["N"], "K": inst["K"], "W": inst["W"],
                  "decay": inst["decay"], "drift": inst["drift"], "score": list(inst["score"])}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, total = score(inst, ans)
        except Exception:
            ok, total = False, 0.0
        if not ok:
            vec.append(0.0)
            continue
        base = baseline(inst)
        ub = _upper_bound(inst)
        denom = ub - base
        if denom < 1e-9:
            denom = 1e-9
        r = 0.1 + 0.9 * (total - base) / denom
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
