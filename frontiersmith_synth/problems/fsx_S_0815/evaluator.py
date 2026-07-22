#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0815 -- "Beads on a Wire: Loop-Line Holding Control"
(family: bus-bunching-stabilizer; format B, quality-metric).

THEME.  N buses share a circular loop line.  Passengers accumulate at a constant
rate at every stop, so a bus's dwell time (time spent boarding) grows linearly
with the headway since the PRECEDING bus visited that stop.  A bus that falls
even slightly behind picks up more passengers, dwells longer, and falls further
behind: an unstable positive-feedback (dwell-feedback-instability) that, left
alone, makes buses drift together into clumps ("bunching").  The operator's only
lever is HOLDING: a dispatcher may make a bus wait extra time at a stop before
it may depart (holding-control) -- it can only ADD delay, never subtract it.
Because the line is a LOOP, "ahead" and "behind" wrap around (loop-headway): bus
i's headway is measured against its immediate predecessor, and bus i's own
holding decision also reshapes the gap experienced by the bus immediately
FOLLOWING it.  This is a closed ring of N coupled oscillators.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance) -- see statement.md for the exact
          schema (n_buses, beta, rounds, nominal_headway, max_hold,
          capacity_weight, initial_headways).
  stdout: ONE JSON object -- a FIXED linear holding-control law, applied by
          THIS evaluator at every stop for the whole horizon:
            {"gain_back": gb, "gain_fwd": gf, "target_frac": tf, "cap_frac": cf}
          At round k, bus i is held for
            u_i(k) = clip( gb*(Ht - H_i(k)) + gf*(Ht - H_{(i+1)%N}(k)),
                            0, cap_frac * max_hold )
          where Ht = tf * nominal_headway, H_i(k) is bus i's OWN current
          headway (gap since its predecessor last visited this stop), and
          H_{(i+1)%N}(k) is the headway of the bus immediately BEHIND bus i
          (i.e. the gap bus i is about to leave in its wake).  gain_back reacts
          to your own gap; gain_fwd additionally reacts to your follower's gap
          -- a "two-way looking" law.  Bad shape / out-of-range / non-finite ->
          that instance scores 0.0.

DYNAMICS (deterministic; this evaluator computes it, never the candidate).
  H_i(k+1) = (1+beta)*H_i(k) - beta*H_{(i-1)%N}(k) + u_i(k) - u_{(i-1)%N}(k)
  (dwell at stop = beta * H_i(k); the added hold u_i(k) shows up in the NEXT
  headway of bus i and, with the opposite sign, in the next headway of the bus
  behind it -- holding a bus never destroys or creates total loop time, it only
  reshuffles it around the ring.)

SCORING (deterministic; no wall-time).  Per instance:
    wait_term  = mean over k in [0,rounds) , i in [0,N) of (H_i(k) - Hnom)^2
    hold_term  = mean over k in [0,rounds) , i in [0,N) of u_i(k)
    obj        = wait_term + capacity_weight * hold_term       (MINIMIZE)
  Reference q_base = obj of the internal NEVER-HOLD operator (gb=gf=0, weak
  baseline; this is exactly what solutions/trivial.py also submits).
  Normalize with an affine anchor (weak baseline -> 0.1, a perfect-spacing
  ideal (obj=0, generally unreachable given the seeded initial perturbation
  and only-nonnegative holds) -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(q_base, 1e-9), 0, 1 )

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the PUBLIC instance.  All simulation
(including the never-hold reference) happens in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

NORM_POWER = 0.5   # sqrt-compression exponent for the score normalization (see main())


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u = (state >> 11) / float(1 << 53)
        return lo + u * (hi - lo)

    return nxt


# ----------------------------- initial headway patterns ---------------------
def _pattern(seed, n, hnom, amp, kind):
    nx = _rng(seed)
    raw = []
    if kind == "alt":
        # dominant alternating (antisymmetric) component + a smaller mixed-mode
        # jitter, so the perturbation is not a single pure eigenmode
        for i in range(n):
            base = hnom * (1 + amp) if i % 2 == 0 else hnom * (1 - amp)
            base += hnom * nx(-0.22, 0.22)
            raw.append(base)
    elif kind == "spike":
        hit = int(nx(0, n))
        hit = max(0, min(n - 1, hit))
        for i in range(n):
            raw.append(hnom * (1 - amp) if i == hit else hnom * (1 + amp / (n - 1)))
    else:  # "mild_random"
        for i in range(n):
            raw.append(hnom * (1 + nx(-amp, amp)))
    s = sum(raw)
    resid = (s - n * hnom) / n
    return [max(1e-6, v - resid) for v in raw]


# ----------------------------- instance family -------------------------------
def _build_instances():
    """(name, seed, N, beta, K, Hnom, max_hold_frac, gamma, pattern, amp)."""
    specs = [
        ("loopA", 101, 5, 0.03, 10, 60, 0.50, 0.15, "mild_random", 0.15),
        ("loopB", 202, 6, 0.05, 10, 60, 0.60, 0.20, "alt",         0.35),
        ("loopC", 303, 8, 0.03, 10, 50, 0.35, 0.15, "alt",         0.55),  # trap: low beta, tight cap, big antisym
        ("loopD", 404, 7, 0.05, 10, 70, 0.70, 0.50, "mild_random", 0.20),  # trap: high capacity weight
        ("loopE", 505, 6, 0.06, 10, 55, 0.40, 0.25, "spike",       0.60),  # trap: spike + tight cap
        ("loopF", 606, 4, 0.02, 8,  45, 0.50, 0.10, "mild_random", 0.10),
        ("loopG", 707, 9, 0.04, 12, 65, 0.55, 0.30, "alt",         0.40),
        ("loopH", 808, 10, 0.05, 12, 75, 0.45, 0.35, "alt",        0.50),  # held-out harder
        ("loopI", 909, 8, 0.03, 10, 60, 0.30, 0.20, "spike",       0.70),  # held-out harder trap
        ("loopJ", 111, 6, 0.04, 14, 50, 0.60, 0.40, "mild_random", 0.25),  # held-out harder (long horizon)
    ]
    out = []
    for name, seed, n, beta, k, hnom, mhf, gamma, pattern, amp in specs:
        h0 = _pattern(seed + 1, n, hnom, amp, pattern)
        out.append({
            "name": name, "n_buses": n, "beta": beta, "rounds": k,
            "nominal_headway": hnom, "max_hold": mhf * hnom,
            "capacity_weight": gamma, "initial_headways": h0,
        })
    return out


# ----------------------------- simulation -------------------------------------
_CAP = 1e9


def simulate(inst, gb, gf, tf, cf):
    n = inst["n_buses"]
    beta = inst["beta"]
    k_rounds = inst["rounds"]
    hnom = inst["nominal_headway"]
    max_hold = inst["max_hold"]
    ht = tf * hnom
    cap = max(0.0, min(1.0, cf)) * max_hold
    H = list(inst["initial_headways"])
    wait_acc = 0.0
    hold_acc = 0.0
    for _k in range(k_rounds):
        u = [0.0] * n
        for i in range(n):
            j = (i + 1) % n
            raw = gb * (ht - H[i]) + gf * (ht - H[j])
            if raw < 0.0:
                raw = 0.0
            elif raw > cap:
                raw = cap
            u[i] = raw
        for i in range(n):
            d = H[i] - hnom
            wait_acc += d * d
            hold_acc += u[i]
        newH = [0.0] * n
        for i in range(n):
            p = (i - 1) % n
            v = (1 + beta) * H[i] - beta * H[p] + u[i] - u[p]
            if v != v:  # nan guard
                v = _CAP
            if v > _CAP:
                v = _CAP
            elif v < -_CAP:
                v = -_CAP
            newH[i] = v
        H = newH
    denom = float(k_rounds * n)
    wait_term = wait_acc / denom
    hold_term = hold_acc / denom
    obj = wait_term + inst["capacity_weight"] * hold_term
    if not (obj == obj):
        obj = _CAP
    if obj > _CAP:
        obj = _CAP
    return obj


def baseline(inst):
    return simulate(inst, 0.0, 0.0, 1.0, 0.0)


def _validate_answer(answer):
    if not isinstance(answer, dict):
        return None
    keys = ("gain_back", "gain_fwd", "target_frac", "cap_frac")
    vals = {}
    for kname in keys:
        v = answer.get(kname)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        v = float(v)
        if not (v == v) or v in (float("inf"), float("-inf")):
            return None
        vals[kname] = v
    if not (-3.0 <= vals["gain_back"] <= 3.0):
        return None
    if not (-3.0 <= vals["gain_fwd"] <= 3.0):
        return None
    if not (0.5 <= vals["target_frac"] <= 1.5):
        return None
    if not (0.0 <= vals["cap_frac"] <= 1.0):
        return None
    return vals


def score(inst, answer):
    vals = _validate_answer(answer)
    if vals is None:
        return False, None
    obj = simulate(inst, vals["gain_back"], vals["gain_fwd"],
                    vals["target_frac"], vals["cap_frac"])
    return True, obj


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        q_base = baseline(inst)
        public = {
            "name": inst["name"], "n_buses": inst["n_buses"], "beta": inst["beta"],
            "rounds": inst["rounds"], "nominal_headway": inst["nominal_headway"],
            "max_hold": inst["max_hold"], "capacity_weight": inst["capacity_weight"],
            "initial_headways": list(inst["initial_headways"]),
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, q_cand = score(inst, ans)
        except Exception:
            ok, q_cand = False, None
        if not ok or q_cand is None:
            vec.append(0.0)
            continue
        denom = q_base if q_base > 1e-9 else 1e-9
        ratio = q_cand / denom
        if ratio < 0.0:
            ratio = 0.0
        # sqrt-compressed affine anchor: matching the never-hold baseline -> 0.1,
        # the (unreachable) zero-objective ideal -> 1.0. The dwell-feedback
        # instability makes the RAW baseline-vs-candidate gap swing over many
        # orders of magnitude across instances (beta/N/horizon vary); a linear
        # anchor would make almost every stabilizing policy saturate near 1.0
        # and erase the gap between a mediocre and a good controller, so the
        # ratio is compressed with a sqrt (NORM_POWER) before the affine map.
        r = 0.1 + 0.9 * (1.0 - ratio ** NORM_POWER)
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
