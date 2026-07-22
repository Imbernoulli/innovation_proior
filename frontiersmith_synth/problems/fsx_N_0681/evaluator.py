#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_N_0681 -- "The Gatekeeper's Ledger"
(family: arrival-mix-gatekeeping; format B, quality-metric).

THEME. A marketplace's seller-approval desk processes a queue of applicants in
T rounds of B applicants each. Every applicant carries a public APPLICATION
SCORE (a noisy profile signal: documents, off-platform reviews, etc.) in
[0, 1]. The desk approves or rejects each applicant. An approved applicant is
either a genuinely LEGITIMATE seller (contributes revenue) or a BAD ACTOR
(fraud / scam -- contributes a cost). Rejected applicants contribute nothing
either way.

TWO COMPOSED MECHANISMS drive the objective:

 (1) arrival-composition-feedback. The platform tracks a MIX HEALTH state
     h_t in [0,1] = the expected fraction of genuinely legitimate sellers in
     THIS round's pool. The realized legitimacy of applicant i is a coin
     flip with P(legit) = sigmoid((score_i - (1-h_t)) / temp) -- i.e. the
     SAME observable score is more often backed by a real business when the
     platform's overall mix is healthy, and more often a front for a bad
     actor when the mix has decayed. Approving bad actors this round lowers
     h for the NEXT round (the pool arriving tomorrow is worse because today
     you let scammers in); staying clean lets h recover slowly.

 (2) trust-reputation-decay. The platform also tracks TRUST r_t in [0,1].
     Approved-legit revenue scales with r_t (a distrusted platform earns
     less per legitimate seller); approved-bad cost is AMPLIFIED as r_t
     drops (fraud that happens on an already-damaged platform hurts more).
     r_t decays on bad approvals and recovers slowly when clean.

 A THIRD state, application VOLUME, shrinks if the round's approval rate
 falls below a threshold (word spreads that the desk is too strict, fewer
 sellers bother applying) and grows back toward full size otherwise --
 "over-rejecting shrinks volume."

 THE TRAP (compounding). If h_t ever dips below a per-instance
 cascade_threshold, the platform enters a PERMANENT "reputation spiral" for
 the REST of the run: the cost of any bad approval jumps sharply and organic
 recovery of h and r becomes far slower. Nothing in a single round's
 immediate expected value warns you this is about to happen -- a purely
 per-applicant EV-maximizing rule keeps approving marginally-profitable
 applicants right up to (and through) the cliff. The insight is to look
 AHEAD, notice when continuing the naive rule would push h below the
 threshold within the next few rounds, and TIGHTEN (raise the bar, trading
 away some immediate expected value) exactly then -- loosening back up once
 the danger has passed.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
    {"name": str, "T": int, "B": int, "N": int (=T*B),
     "score": [s_0, ..., s_{N-1}],   # N floats in [0,1], applicant i's public score
     "h0": float, "r0": float,
     "params": {   # ALL constants a solver needs; see statement.md for the exact formulas
        "h_decay_bad","h_recover","h_floor","h_ceiling","cascade_threshold",
        "r_decay_bad","r_recover","r_floor",
        "v_good","c_bad","bad_amp","temp",
        "vol_shrink_threshold","vol_shrink_step","vol_grow_step","vol_min_frac",
        "spiral_c_bad_mult","spiral_h_recover_mult","spiral_r_recover_mult"
     }}
  stdout: ONE JSON object: {"decisions": [d_0, ..., d_{N-1}]}  d_i in {0,1}
    (applicant i's slot may be INACTIVE this round if the round's volume has
    shrunk below B; inactive slots are decided over the batch in *local*
    index order B*t..B*t+B-1, and any decision for an inactive slot is
    simply ignored -- it costs nothing to submit 0 or 1 there.)

  Any malformed output (wrong length, non-0/1 element, non-finite, a crash,
  a timeout, or non-JSON) -> that instance scores 0.0.

SCORING (deterministic; no wall-time). The evaluator REPLAYS the submitted
decision sequence round by round using the TRUE hidden legitimacy draws
(never shown to the candidate) to get q_cand, the platform's total net
value. Two references, computed by the evaluator itself:
    q_base = 0                (reject everyone: zero risk, zero value)
    q_ub   = T * B * v_good   (loose ceiling: every slot legit & approved
                                at full trust, every round at full volume)
  r = clamp(0.1 + 0.9 * (q_cand - q_base) / (q_ub - q_base), 0, 1)
q_ub is deliberately loose (it ignores the trust ramp-up, the shrink/growth
dynamics and the very possibility of any bad approval at all), so even a
near-optimal policy stays well under 1.0 -- real headroom remains.

ISOLATION. The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the public instance. The hidden
legitimacy draws and the true-replay logic live only in this parent process.

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


def _sigmoid(x):
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


# ----------------------------- shared dynamics ------------------------------
GLOBAL_PARAMS = dict(
    h_decay_bad=0.5, h_recover=0.05, h_floor=0.05, h_ceiling=1.0,
    r_decay_bad=0.3, r_recover=0.05, r_floor=0.15,
    v_good=1.0, c_bad=0.75, bad_amp=0.5,
    vol_shrink_threshold=0.3, vol_shrink_step=0.1, vol_grow_step=0.06, vol_min_frac=0.45,
    spiral_c_bad_mult=2.4, spiral_h_recover_mult=0.15, spiral_r_recover_mult=0.2,
)


def _p_good(score, h, temp):
    return _sigmoid((score - (1.0 - h)) / temp)


def _step_update(h, r, vol_mult, bad_frac, approval_rate, in_spiral, p):
    h_recover = p["h_recover"] * (p["spiral_h_recover_mult"] if in_spiral else 1.0)
    r_recover = p["r_recover"] * (p["spiral_r_recover_mult"] if in_spiral else 1.0)
    h_new = h + h_recover * (1 - h) * (1 - bad_frac) - p["h_decay_bad"] * bad_frac
    h_new = min(p["h_ceiling"], max(p["h_floor"], h_new))
    r_new = r + r_recover * (1 - r) * (1 - bad_frac) - p["r_decay_bad"] * bad_frac
    r_new = min(1.0, max(p["r_floor"], r_new))
    new_spiral = in_spiral or (h_new < p["cascade_threshold"])
    if approval_rate < p["vol_shrink_threshold"]:
        vol_mult = max(p["vol_min_frac"], vol_mult - p["vol_shrink_step"])
    else:
        vol_mult = min(1.0, vol_mult + p["vol_grow_step"])
    return h_new, r_new, vol_mult, new_spiral


# ----------------------------- instance family ------------------------------
def _build_instance(seed, T, B, h0, r0, cascade_threshold, temp):
    nx = _rng(seed)
    N = T * B
    score = [nx() for _ in range(N)]
    q_hidden = [nx() for _ in range(N)]     # HIDDEN: never sent to the candidate
    params = dict(GLOBAL_PARAMS)
    params["cascade_threshold"] = cascade_threshold
    params["temp"] = temp
    return dict(name=f"desk{seed}", T=T, B=B, N=N, score=score, q_hidden=q_hidden,
                h0=h0, r0=r0, params=params)


def _build_instances():
    specs = [
        # seed,  T,  B, h0,   r0,   cascade_threshold, temp,   note
        (1001, 12, 30, 0.80, 0.85, 0.40, 0.16, "easy"),
        (1002, 12, 30, 0.75, 0.80, 0.42, 0.18, "easy"),
        (1003, 14, 30, 0.55, 0.65, 0.50, 0.16, "trap"),
        (1004, 14, 30, 0.52, 0.60, 0.48, 0.16, "trap"),
        (1005, 15, 30, 0.50, 0.55, 0.46, 0.15, "trap"),
        (1006, 12, 30, 0.85, 0.85, 0.35, 0.16, "easy"),
        (1007, 16, 30, 0.58, 0.62, 0.52, 0.16, "trap"),
        (1008, 14, 32, 0.65, 0.70, 0.55, 0.15, "medium"),
        (1009, 18, 30, 0.56, 0.60, 0.50, 0.18, "trap-hard"),
        (1010, 18, 32, 0.55, 0.58, 0.50, 0.18, "trap-hard"),
    ]
    out = []
    for seed, T, B, h0, r0, ct, temp, note in specs:
        inst = _build_instance(seed, T, B, h0, r0, ct, temp)
        inst["note"] = note
        out.append(inst)
    return out


# ----------------------------- baseline / bound -----------------------------
def baseline(inst):
    """Reject-everyone: zero risk, zero value. The weak (0.1) anchor."""
    return 0.0


def _upper_bound(inst):
    return inst["T"] * inst["B"] * inst["params"]["v_good"]


# ----------------------------- validation + true replay ---------------------
def score(inst, answer):
    """Validate `answer` and replay the TRUE dynamics using hidden draws.
    Returns (ok: bool, total_value: float)."""
    if not isinstance(answer, dict):
        return False, 0.0
    decisions = answer.get("decisions")
    N = inst["N"]
    if not isinstance(decisions, list) or len(decisions) != N:
        return False, 0.0
    for d in decisions:
        if isinstance(d, bool) or not isinstance(d, int) or d not in (0, 1):
            return False, 0.0

    p = inst["params"]; T = inst["T"]; B = inst["B"]
    score_arr = inst["score"]; qh = inst["q_hidden"]
    h = inst["h0"]; r = inst["r0"]; vol_mult = 1.0; in_spiral = False
    total = 0.0
    for t in range(T):
        vol_t = max(1, round(B * vol_mult))
        approved = 0; bad_approved = 0
        c_eff = p["c_bad"] * (p["spiral_c_bad_mult"] if in_spiral else 1.0)
        for j in range(B):
            i = t * B + j
            if j >= vol_t:
                continue
            a = decisions[i]
            pg = _p_good(score_arr[i], h, p["temp"])
            good = 1 if qh[i] < pg else 0
            if a:
                approved += 1
                if good:
                    total += p["v_good"] * r
                else:
                    bad_approved += 1
                    total -= c_eff * (1 + p["bad_amp"] * (1 - r))
        approval_rate = approved / vol_t if vol_t else 0.0
        bad_frac = bad_approved / vol_t if vol_t else 0.0
        h, r, vol_mult, in_spiral = _step_update(h, r, vol_mult, bad_frac, approval_rate, in_spiral, p)

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
        public = {"name": inst["name"], "T": inst["T"], "B": inst["B"], "N": inst["N"],
                  "score": list(inst["score"]), "h0": inst["h0"], "r0": inst["r0"],
                  "params": dict(inst["params"])}
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
