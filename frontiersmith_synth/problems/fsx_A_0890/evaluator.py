#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0890 -- "One Shopkeeper Brain: Reorder Policies Across
Demand Families" (family: reorder-policy-demand-families; format B, quality-metric).

THEME.  A shopkeeper runs ONE reorder brain that must simultaneously manage SIX
product lines whose weekly demand is drawn from six structurally distinct families:
seasonal, bursty, trending, intermittent, high-variance-flat, and a regime-shift
line that looks calm during the observation window and then changes character.  The
shopkeeper sees only H=30 weeks of realized demand history per line (the PUBLIC
instance) and must commit, once, to a reorder policy for the following Wf=70 weeks
-- no further feedback loop with the evaluator, no peeking at future demand.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
    {"name": str, "period": 12, "history_weeks": 30, "future_weeks": 70,
     "traces": [
        {"trace_id": 0..5,
         "history": [30 non-negative ints],      # realized demand, weeks 0..29
         "initial_on_hand": int,                 # on-hand stock at start of week 30
         "price": float, "unit_cost": float,
         "holding_cost": float, "stockout_cost": float},
        ... exactly 6 entries ...
     ]}
  stdout: ONE JSON object:
    {"policies": [
        {"trace_id": 0..5, "level": [12 floats], "trend": float, "react": float},
        ... exactly 6 entries, one per trace_id, trace_ids form the set {0..5} ...
     ]}

POLICY SEMANTICS (evaluator-executed, NOT candidate-executed).  For each trace the
evaluator simulates weeks t = H .. H+Wf-1 with lead time 0 (an order placed this
week arrives this week, before demand is realized).  Let j = t-H be the future-week
index, phase = t % period, and prev_demand the REALIZED demand of week t-1 (already
known -- causal).  The order-up-to target is

    S(t) = clamp( level[phase] + trend*j + react*(prev_demand - hist_mean), 0, 1e9 )
    order_t = max(0, round(S(t) - on_hand))

on_hand increases by order_t, then sales_t = min(on_hand, demand_t), lost_t =
demand_t - sales_t, end-of-week on_hand = on_hand - sales_t (carried to t+1), and

    profit_t = price*sales_t - holding_cost*end_on_hand - stockout_cost*lost_t
               - unit_cost*order_t

The candidate must therefore READ each trace's 30-week history, run some cheap
online test to recognize which family it is looking at (flat mean? high variance?
drift? mostly zeros? a shape that repeats every 12 weeks?), and only THEN choose
level/trend/react -- estimation before control. A single fixed "safety stock sized
to the historical mean" recipe, applied identically to all six lines, is a trap:
it is fine for the calm lines but goes badly wrong on the bursty, trending, and
regime-shift lines.

SCORING (deterministic; no wall-time).  Per trace we compute two references, both
INSIDE the evaluator (the candidate never sees them):
    profit_base = simulate() with level=[hist_mean]*period, trend=0, react=0
                  (flat order-up-to-the-mean, no adaptation -- the weak baseline).
    profit_ub   = clairvoyant bound: sum_t (price-unit_cost)*demand_t, i.e. zero
                  holding/stockout cost and an order that exactly matches every
                  week's demand. Loose (no real policy can see next week's demand),
                  so even a strong policy stays below 1.0 -- headroom.
    profit_cand = simulate() with the candidate's level/trend/react.
    r_trace = clamp(0.1 + 0.9*(profit_cand-profit_base)/max(1.0, profit_ub-profit_base), 0, 1)
Reproducing the flat-mean baseline scores ~0.1 on that trace; doing worse scores
below 0.1 (floor 0); adapting to the trace's real shape scores higher.

Each of the 10 instances then scores the GEOMETRIC MEAN of its 6 r_trace values --
NOT the arithmetic mean. This turns "do well on average" into "do not let any ONE
of the six lines collapse": a policy that nails five lines and lets the bursty (or
regime-shift) line go bankrupt is punished far harder under a geometric mean than
under an average, because a near-zero factor drags the 6th root of the product
toward zero regardless of the other five. The final Ratio is the plain mean of the
10 instances' geometric means.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the PUBLIC instance (history + costs).
Every reference (baseline, clairvoyant bound) and all validation happen in THIS
parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean of the 10 instances' geometric-mean scores, in [0,1]>
  Vector: [g_1, ..., g_10]
"""
import sys, json, math
import isorun

PERIOD = 12
H = 30      # history (calibration) weeks
WF = 70     # scored future weeks
N_TRACES = 6


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = seed & ((1 << 64) - 1)

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return state

    def nxt_float():
        return (nxt() >> 11) / float(1 << 53)

    def nxt_int(lo, hi):
        return lo + (nxt() >> 17) % (hi - lo + 1)

    return nxt_float, nxt_int


def _noise(nxt_float, scale):
    # Irwin-Hall(3)-based, roughly bell-shaped, zero-mean, deterministic.
    u = nxt_float() + nxt_float() + nxt_float() - 1.5
    return u * scale


# ----------------------------- demand family generators ---------------------
def _gen_seasonal(seed, base, amp, noise_scale):
    nxt_float, _ = _rng(seed)
    shape = [math.sin(2.0 * math.pi * p / PERIOD) for p in range(PERIOD)]
    T = H + WF
    out = []
    for t in range(T):
        v = base + amp * shape[t % PERIOD] + _noise(nxt_float, noise_scale)
        out.append(max(0, round(v)))
    return out


def _gen_bursty(seed, base, burst_prob, burst_lo, burst_hi, noise_scale):
    nxt_float, nxt_int = _rng(seed)
    T = H + WF
    out = []
    for _t in range(T):
        if nxt_float() < burst_prob:
            v = base + nxt_int(burst_lo, burst_hi)
        else:
            v = base + _noise(nxt_float, noise_scale)
        out.append(max(0, round(v)))
    return out


def _gen_trending(seed, base, slope, noise_scale):
    nxt_float, _ = _rng(seed)
    T = H + WF
    out = []
    for t in range(T):
        v = base + slope * t + _noise(nxt_float, noise_scale)
        out.append(max(0, round(v)))
    return out


def _gen_intermittent(seed, nonzero_prob, lo, hi):
    nxt_float, nxt_int = _rng(seed)
    T = H + WF
    out = []
    for _t in range(T):
        if nxt_float() < nonzero_prob:
            v = nxt_int(lo, hi)
        else:
            v = 0
        out.append(v)
    return out


def _gen_volatile(seed, base, noise_scale):
    nxt_float, _ = _rng(seed)
    T = H + WF
    out = []
    for _t in range(T):
        v = base + _noise(nxt_float, noise_scale)
        out.append(max(0, round(v)))
    return out


def _gen_regime_shift(seed, base_calm, noise_calm, shift_base, shift_lo, shift_hi,
                       noise_shift, shift_burst_prob):
    nxt_float, nxt_int = _rng(seed)
    T = H + WF
    shift_week = H + WF // 4     # shift happens shortly INTO the unseen future
    out = []
    for t in range(T):
        if t < shift_week:
            v = base_calm + _noise(nxt_float, noise_calm)
        else:
            if nxt_float() < shift_burst_prob:
                v = shift_base + nxt_int(shift_lo, shift_hi)
            else:
                v = shift_base + _noise(nxt_float, noise_shift)
        out.append(max(0, round(v)))
    return out


# ----------------------------- instance family -----------------------------
def _build_instance(idx):
    seed0 = 42013 + idx * 733
    scale = 18 + 3 * idx                      # overall demand-magnitude scale
    price = 9.0 + 0.3 * (idx % 5)
    unit_cost = 4.0 + 0.15 * (idx % 4)
    holding_cost = 0.35 + 0.05 * (idx % 4)
    stockout_cost = 5.5 + 0.6 * (idx % 5)

    demands = [
        _gen_seasonal(seed0 + 1, base=scale, amp=0.65 * scale, noise_scale=0.10 * scale),
        _gen_bursty(seed0 + 2, base=max(1, round(0.12 * scale)), burst_prob=0.12,
                    burst_lo=round(3.0 * scale), burst_hi=round(6.0 * scale),
                    noise_scale=0.06 * scale),
        _gen_trending(seed0 + 3, base=0.4 * scale, slope=0.045 * scale + 0.01 * idx,
                      noise_scale=0.10 * scale),
        _gen_intermittent(seed0 + 4, nonzero_prob=0.22, lo=max(3, round(0.25 * scale)),
                           hi=max(5, round(0.9 * scale))),
        _gen_volatile(seed0 + 5, base=scale, noise_scale=0.55 * scale),
        _gen_regime_shift(seed0 + 6, base_calm=0.35 * scale, noise_calm=0.08 * scale,
                           shift_base=1.9 * scale,
                           shift_lo=round(1.0 * scale), shift_hi=round(2.8 * scale),
                           noise_shift=0.30 * scale, shift_burst_prob=0.20),
    ]

    traces = []
    for tid, demand in enumerate(demands):
        hist = demand[:H]
        hist_mean = sum(hist) / float(H)
        init_on_hand = max(0, round(hist_mean * 1.4))
        traces.append({
            "trace_id": tid,
            "demand_full": demand,           # HIDDEN: full H+WF realization
            "history": hist,                 # PUBLIC
            "hist_mean": hist_mean,
            "initial_on_hand": init_on_hand,
            "price": round(price, 4),
            "unit_cost": round(unit_cost, 4),
            "holding_cost": round(holding_cost, 4),
            "stockout_cost": round(stockout_cost, 4),
        })
    return {"name": f"shop{idx:02d}", "traces": traces}


def _build_instances():
    return [_build_instance(i) for i in range(10)]


def _public_view(inst):
    return {
        "name": inst["name"], "period": PERIOD,
        "history_weeks": H, "future_weeks": WF,
        "traces": [
            {"trace_id": tr["trace_id"], "history": list(tr["history"]),
             "initial_on_hand": tr["initial_on_hand"], "price": tr["price"],
             "unit_cost": tr["unit_cost"], "holding_cost": tr["holding_cost"],
             "stockout_cost": tr["stockout_cost"]}
            for tr in inst["traces"]
        ],
    }


# ----------------------------- simulation / references ----------------------
def _simulate(tr, level, trend, react):
    demand_full = tr["demand_full"]
    hist_mean = tr["hist_mean"]
    on_hand = float(tr["initial_on_hand"])
    prev_demand = float(demand_full[H - 1])
    price = tr["price"]; unit_cost = tr["unit_cost"]
    holding_cost = tr["holding_cost"]; stockout_cost = tr["stockout_cost"]
    total = 0.0
    for j in range(WF):
        t = H + j
        phase = t % PERIOD
        S = level[phase] + trend * j + react * (prev_demand - hist_mean)
        if S < 0.0: S = 0.0
        elif S > 1e9: S = 1e9
        order = S - on_hand
        if order < 0.0: order = 0.0
        order = float(round(order))
        avail = on_hand + order
        d = float(demand_full[t])
        sales = min(avail, d)
        lost = d - sales
        end_on_hand = avail - sales
        profit = (price * sales - holding_cost * end_on_hand
                  - stockout_cost * lost - unit_cost * order)
        total += profit
        on_hand = end_on_hand
        prev_demand = d
    return total


def _baseline_profit(tr):
    return _simulate(tr, [tr["hist_mean"]] * PERIOD, 0.0, 0.0)


def _clairvoyant_profit(tr):
    demand_full = tr["demand_full"]
    price = tr["price"]; unit_cost = tr["unit_cost"]
    return sum((price - unit_cost) * demand_full[H + j] for j in range(WF))


# ----------------------------- answer validation -----------------------------
def _validate(answer):
    if not isinstance(answer, dict):
        return None
    pols = answer.get("policies")
    if not isinstance(pols, list) or len(pols) != N_TRACES:
        return None
    out = {}
    for p in pols:
        if not isinstance(p, dict):
            return None
        tid = p.get("trace_id")
        if isinstance(tid, bool) or not isinstance(tid, int):
            return None
        if tid < 0 or tid >= N_TRACES or tid in out:
            return None
        lvl = p.get("level")
        if not isinstance(lvl, list) or len(lvl) != PERIOD:
            return None
        lvl_f = []
        for x in lvl:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                return None
            xf = float(x)
            if xf != xf or xf in (float("inf"), float("-inf")) or abs(xf) > 1e6:
                return None
            lvl_f.append(xf)
        trend = p.get("trend"); react = p.get("react")
        if isinstance(trend, bool) or not isinstance(trend, (int, float)):
            return None
        if isinstance(react, bool) or not isinstance(react, (int, float)):
            return None
        trend = float(trend); react = float(react)
        if trend != trend or trend in (float("inf"), float("-inf")) or abs(trend) > 1e5:
            return None
        if react != react or react in (float("inf"), float("-inf")) or abs(react) > 1e5:
            return None
        out[tid] = {"level": lvl_f, "trend": trend, "react": react}
    if len(out) != N_TRACES:
        return None
    return out


# ----------------------------------- main ------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = _public_view(inst)
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            pols = _validate(ans)
        except Exception:
            pols = None
        if pols is None:
            vec.append(0.0)
            continue

        r_traces = []
        ok_all = True
        for tr in inst["traces"]:
            pol = pols.get(tr["trace_id"])
            if pol is None:
                ok_all = False
                break
            try:
                cand_profit = _simulate(tr, pol["level"], pol["trend"], pol["react"])
            except Exception:
                ok_all = False
                break
            if not (cand_profit == cand_profit) or cand_profit in (float("inf"), float("-inf")):
                ok_all = False
                break
            base_profit = _baseline_profit(tr)
            ub_profit = _clairvoyant_profit(tr)
            denom = ub_profit - base_profit
            if denom < 1.0:
                denom = 1.0
            r = 0.1 + 0.9 * (cand_profit - base_profit) / denom
            if r < 0.0: r = 0.0
            elif r > 1.0: r = 1.0
            r_traces.append(r)
        if not ok_all or len(r_traces) != N_TRACES:
            vec.append(0.0)
            continue

        prod = 1.0
        for r in r_traces:
            prod *= r
        gm = prod ** (1.0 / N_TRACES)
        if not (gm == gm) or gm in (float("inf"), float("-inf")):
            gm = 0.0
        vec.append(max(0.0, min(1.0, gm)))

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
