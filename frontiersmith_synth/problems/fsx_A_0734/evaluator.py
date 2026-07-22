#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0734 -- "Loyalty Ledger: A Year of Grocery-App Coupons"
(family: reference-price-coupon; format B, quality-metric).

THEME.  A grocery app schedules coupons for one product across a fixed 52-week year,
independently for several customer SEGMENTS.  Every segment silently tracks a
REFERENCE PRICE -- an exponential moving average (EMA) of the net prices it has
actually been charged -- and that reference price is exactly what "feels like a fair
price" to that segment.  A coupon widens the gap between today's net price and the
segment's current reference price, which lifts THIS week's demand -- but the EMA also
folds today's price into the reference for next week.  Repeated discounting drags the
reference price down, and once it catches up to your price the "bargain" feeling is
gone -- you are just selling at a permanently thinner margin.  How fast a segment's
reference price chases the price it is shown (and how fast it drifts back toward full
price when you stop discounting) is a per-segment MEMORY CONSTANT `alpha` that is
HIDDEN from the candidate.  Before the year starts, a short controlled PILOT (one
discount shock followed by four full-price weeks) was already run per segment and its
realized sales are handed to the candidate as public data -- from that data alone,
`alpha` (memory speed) and `beta` (price sensitivity) are IDENTIFIABLE in closed form.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
    {"name": str, "n_weeks": 52, "price": P, "cost": C, "max_discount": D,
     "pilot_depths": [d0..d4],           # the 5 fixed pilot-week discount depths
     "segments": [ {"id": i, "size": int, "base_rate": float,
                     "pilot_demand": [u0..u4]},  ... ]}
  stdout: ONE JSON object:
    {"schedule": [[d_{i,0}, ..., d_{i,51}], ...]}   # one length-52 list of discount
                                                     # depths per segment, same order
                                                     # as `segments`, each in [0, D]

  A schedule is VALID iff it has exactly one row per segment, each row has exactly 52
  finite numeric (non-bool) entries in [0, max_discount] (small float tolerance).
  Wrong shape, out-of-range depth, non-finite value, a crash, a timeout, or non-JSON
  output makes that instance score 0.0.

SCORING (deterministic; no wall-time).  Per instance the evaluator simulates every
segment's true dynamics (hidden `alpha`, `beta`) under three schedules:
    weak  = "never discount" (every depth 0) -- achievable from PUBLIC data alone.
    oracle = the profit-MAXIMIZING schedule computed by exact per-segment dynamic
             programming WITH the true hidden alpha/beta (the evaluator's own
             privileged reference; never seen by the candidate) -- an upper anchor
             that is not perfectly reachable by a candidate who must first identify
             alpha/beta from noisy-free but limited pilot data and a restricted
             (non-adaptive) 52-week commitment.
    cand  = the candidate's realized profit under its submitted schedule.
  Summed over segments, with an affine anchor (never-discount -> 0.1, oracle -> 1.0):
    r = clamp( 0.1 + 0.9 * (cand - weak) / max(oracle - weak, 1.0), 0, 1 )

ISOLATION.  The candidate runs in a FRESH SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  Hidden alpha/beta and
the oracle DP live only in this parent process.

CLI:  python3 evaluator.py <solution.py>
Prints "Ratio: <mean r>" and "Vector: [r_1, ...]" on their own final lines.
"""
import sys, json, math
import numpy as np
import isorun

# ----------------------------- fixed economics ------------------------------
PRICE = 10.0
COST = 4.0
MAX_DISCOUNT = 0.50
N_WEEKS = 52
PILOT_DEPTHS = [0.30, 0.0, 0.0, 0.0, 0.0]
REF_BINS = 121
N_ACTIONS = 19
STRUCT_FRAC = 0.25   # share of elasticity that is permanent (price-vs-list), not erodible


# ----------------------------- deterministic RNG ----------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


def _unit(ni):
    return ni(1, 1000000) / 1000000.0


# ----------------------------- core dynamics --------------------------------
def simulate(depths, base_rate, size, alpha, beta, price=PRICE, cost=COST):
    """Replay the EMA reference-price law over `depths`. Returns (total_profit, units_list)."""
    ref = price
    total = 0.0
    units = []
    for d in depths:
        p = price * (1.0 - d)
        struct_gap = price - p
        ref_gap = ref - p
        frac = (base_rate + beta * STRUCT_FRAC * (struct_gap / price)
                + beta * (1.0 - STRUCT_FRAC) * (ref_gap / price))
        if frac < 0.0:
            frac = 0.0
        elif frac > 1.0:
            frac = 1.0
        u = size * frac
        total += u * (p - cost)
        units.append(u)
        ref = alpha * p + (1.0 - alpha) * ref
    return total, units


def _oracle_profit(base_rate, size, alpha, beta, price=PRICE, cost=COST):
    """Exact per-segment optimal 52-week profit via backward-induction DP over a
    discretized reference-price state, WITH true (hidden) alpha/beta. Vectorized
    over the ref-bin axis with numpy for speed."""
    lo = price * (1.0 - MAX_DISCOUNT)
    ref_grid = np.linspace(lo, price, REF_BINS)
    actions = np.linspace(0.0, MAX_DISCOUNT, N_ACTIONS)
    V = np.zeros(REF_BINS)
    span = price - lo
    for _ in range(N_WEEKS):
        best = np.full(REF_BINS, -np.inf)
        for a in actions:
            p = price * (1.0 - a)
            struct_gap = price - p
            ref_gap = ref_grid - p
            frac = np.clip(base_rate + beta * STRUCT_FRAC * (struct_gap / price)
                            + beta * (1.0 - STRUCT_FRAC) * (ref_gap / price), 0.0, 1.0)
            profit = size * frac * (p - cost)
            next_ref = alpha * p + (1.0 - alpha) * ref_grid
            idx = np.rint((next_ref - lo) / span * (REF_BINS - 1)).astype(int)
            idx = np.clip(idx, 0, REF_BINS - 1)
            total = profit + V[idx]
            best = np.maximum(best, total)
        V = best
    return float(V[-1])  # ref0 == price == ref_grid[-1] exactly


# ----------------------------- instance family -------------------------------
def _gen_segment(ni, alpha_lo, alpha_hi):
    size = ni(500, 3000)
    base_rate = 0.05 + 0.15 * _unit(ni)   # [0.05, 0.20]
    beta = 1.0 + 1.0 * _unit(ni)          # [1.0, 2.0]
    alpha = alpha_lo + (alpha_hi - alpha_lo) * _unit(ni)
    return size, base_rate, beta, alpha


def _build_instances():
    specs = [
        # (seed, m, alpha_lo, alpha_hi, label)
        (301, 5, 0.35, 0.60, "fast-high-1"),
        (302, 5, 0.35, 0.60, "fast-high-2"),
        (303, 6, 0.40, 0.60, "fast-high-3"),
        (304, 5, 0.05, 0.15, "slow-low-1"),
        (305, 5, 0.05, 0.15, "slow-low-2"),
        (306, 6, 0.05, 0.15, "slow-low-3"),
        (307, 6, 0.05, 0.60, "mixed-1"),
        (308, 6, 0.05, 0.60, "mixed-2"),
        (401, 8, 0.05, 0.60, "heldout-large-1"),
        (402, 8, 0.05, 0.60, "heldout-large-2"),
    ]
    out = []
    for seed, m, alo, ahi, label in specs:
        ni = _rng(seed)
        segs = []
        for i in range(m):
            size, base_rate, beta, alpha = _gen_segment(ni, alo, ahi)
            _, pilot_units = simulate(PILOT_DEPTHS, base_rate, size, alpha, beta)
            segs.append({
                "id": i, "size": size, "base_rate": round(base_rate, 6),
                "pilot_demand": [round(u, 4) for u in pilot_units],
                "_alpha": alpha, "_beta": beta,
            })
        out.append({"name": f"year{seed}", "label": label, "segments": segs})
    return out


# ----------------------------- validation -------------------------------------
def _validate(inst, answer):
    if not isinstance(answer, dict):
        return None
    sched = answer.get("schedule")
    segs = inst["segments"]
    m = len(segs)
    if not isinstance(sched, list) or len(sched) != m:
        return None
    rows = []
    for row in sched:
        if not isinstance(row, list) or len(row) != N_WEEKS:
            return None
        clean = []
        for x in row:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                return None
            xf = float(x)
            if not (xf == xf) or xf in (float("inf"), float("-inf")):
                return None
            if xf < -1e-9 or xf > MAX_DISCOUNT + 1e-6:
                return None
            clean.append(max(0.0, min(MAX_DISCOUNT, xf)))
        rows.append(clean)
    return rows


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        segs = inst["segments"]
        public = {
            "name": inst["name"], "n_weeks": N_WEEKS, "price": PRICE, "cost": COST,
            "max_discount": MAX_DISCOUNT, "pilot_depths": list(PILOT_DEPTHS),
            "segments": [
                {"id": s["id"], "size": s["size"], "base_rate": s["base_rate"],
                 "pilot_demand": list(s["pilot_demand"])}
                for s in segs
            ],
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            rows = _validate(inst, ans)
        except Exception:
            rows = None
        if rows is None:
            vec.append(0.0)
            continue

        weak = 0.0
        oracle = 0.0
        cand_profit = 0.0
        for s, row in zip(segs, rows):
            alpha, beta, base_rate, size = s["_alpha"], s["_beta"], s["base_rate"], s["size"]
            w, _ = simulate([0.0] * N_WEEKS, base_rate, size, alpha, beta)
            weak += w
            oracle += _oracle_profit(base_rate, size, alpha, beta)
            c, _ = simulate(row, base_rate, size, alpha, beta)
            cand_profit += c

        denom = max(oracle - weak, 1.0)
        r = 0.1 + 0.9 * (cand_profit - weak) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
