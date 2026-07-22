#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0898 -- "Shopkeeper Survives Every Embargo in the Almanac"
(family: embargo-shock-ordering-policy; format B, quality-metric).

THEME.  A shopkeeper runs one store for H days.  Every day the almanac already tells
her the day's customer demand (revealed in full, for every timeline, in the public
instance -- demand is exogenous, it does not depend on any ordering decision).  What
the almanac canNOT tell her is which future days will fall inside a trade EMBARGO: on
an embargoed day, zero supply crosses the border, so ANY shipment that was scheduled to
arrive that day is lost outright (no goods, no charge).  Before an embargo the caravans
carry rumours -- a numeric "precursor signal" (also revealed for every day, for every
timeline) that tends to ramp up in the days just before an embargo starts and fade back
to background chatter partway through it.  The shopkeeper submits ONE ordering POLICY
(shared across every timeline in the instance) that must (a) stay lean -- and profitable
-- through timelines that never see an embargo, while (b) building a buffer big enough,
early enough, to survive every embargo timeline.  The instance's score is the MINIMUM
profit ratio across all of that instance's timelines: a policy is only as good as its
worst timeline, so hoarding permanently (which starves the calm timelines with holding
cost) is exactly as much of a loss as never hoarding at all (which bankrupts the
embargo timelines).  The only way to win the minimum is to fund the embargo insurance
out of calm-timeline efficiency: hoard ONLY when the sweep's own precursor signal says
to.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
    {"name": str, "n_timelines": T, "horizon": H, "lead_time": L, "price": float,
     "unit_cost": float, "holding_rate": float, "stockout_penalty": float,
     "init_stock": float,
     "timelines": [ {"demand": [d_0..d_{H-1}], "precursor_signal": [p_0..p_{H-1}]}, ... ]}
  stdout: ONE JSON object -- a single POLICY applied to every timeline in the instance:
    {"base_target": float>=0, "trigger": float, "hoard_target": float>=0,
     "cooldown_days": int>=0}
  Each day t of each timeline, the (evaluator-simulated) shopkeeper reviews her
  inventory position (on-hand + outstanding orders) and orders up to a TARGET:
    target(t) = hoard_target   if precursor_signal[t] >= trigger, or a trigger fired
                                 within the last `cooldown_days` days (refreshed on
                                 every new firing)
              = base_target    otherwise
  Orders placed on day t arrive on day t+lead_time UNLESS that arrival day falls
  inside an embargo window (hidden from the candidate), in which case the shipment is
  lost (no goods delivered, no unit_cost charged).  Demand not covered by on-hand stock
  is a lost sale (stockout_penalty per unit, on top of the forgone revenue).  Holding
  cost is charged on end-of-day on-hand stock.

  VALID iff: answer is a dict with the four keys above; base_target, hoard_target are
  finite non-negative numbers <= 1e6; trigger is a finite number; cooldown_days is a
  non-negative integer (or integral float) <= 60.  Any violation, a crash, a timeout,
  or non-JSON output -> that instance scores 0.0.

SCORING (deterministic, no wall-time).  Per instance the evaluator computes, itself,
TWO references using the SAME simulator as the candidate's policy:
    BASE = the min-over-timelines profit of the "obvious" textbook recipe: a constant
           order-up-to level sized from the sample mean/std of the REVEALED demand
           (standard base-stock formula), NEVER hoarding -- i.e. completely blind to
           the precursor signal and to embargoes.
    UB   = the min-over-timelines profit of an oracle policy that is handed the TRUE
           (hidden) embargo windows and pre-builds an exactly-sized buffer starting
           lead_time days before each embargo and holding it through the embargo's end.
           Because it uses hidden information no causal policy can access, UB is a
           valid, generally-unreachable ceiling for any candidate using only the
           public precursor signal.
    r = clamp( 0.1 + 0.9 * (MIN_cand - BASE) / max(eps, UB - BASE), 0, 1 )
Matching BASE scores ~0.1; approaching UB (rarely fully reachable) scores near 1.0;
doing worse than BASE scores below 0.1.  Ratio is the mean of r over 10 fixed
instances; Vector lists the per-instance scores.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the PUBLIC instance.  The hidden embargo
windows and both references (BASE, UB) are computed by THIS parent process, so a
frame-walking / introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math, random
import isorun


# ----------------------------- shared simulator ------------------------------
def simulate(H, L, price, unit_cost, holding_rate, stockout_penalty, init_stock,
             demand, blocked, target_fn):
    """Order-up-to simulation. target_fn(t) -> today's order-up-to level.
    Returns total profit over the horizon."""
    on_hand = float(init_stock)
    pipeline = {}
    profit = 0.0
    for t in range(H):
        qty_arr = pipeline.pop(t, 0.0)
        if qty_arr > 0.0 and not blocked[t]:
            on_hand += qty_arr
            profit -= unit_cost * qty_arr
        d = demand[t]
        sold = min(on_hand, d)
        unmet = d - sold
        on_hand -= sold
        profit += price * sold - stockout_penalty * unmet - holding_rate * on_hand
        inv_pos = on_hand + sum(pipeline.values())
        tgt = target_fn(t)
        if tgt != tgt or tgt in (float("inf"), float("-inf")):
            tgt = 0.0
        order_qty = max(0.0, tgt - inv_pos)
        arrive = t + L
        if arrive < H:
            pipeline[arrive] = pipeline.get(arrive, 0.0) + order_qty
    return profit


def _sample_mean_std(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    var = sum((x - m) ** 2 for x in values) / n
    return m, math.sqrt(max(0.0, var))


def naive_base_target(all_demand, L, safety_z=1.2):
    """The textbook recipe: order-up-to sized from sample mean/std of demand,
    completely blind to any hoarding signal."""
    mu, sd = _sample_mean_std(all_demand)
    return mu * (L + 1) + safety_z * sd * math.sqrt(L + 1)


def make_reactive_target_fn(base_target, trigger, hoard_target, cooldown_days, signal):
    state = {"cd": 0}

    def target_fn(t):
        if signal[t] >= trigger:
            state["cd"] = cooldown_days
        active = state["cd"] > 0
        if state["cd"] > 0:
            state["cd"] -= 1
        return hoard_target if active else base_target

    return target_fn


def make_oracle_target_fn(H, L, base_target_opt, embargo_windows, demand,
                           buffer_frac=1.15, prebuild_days=2):
    """Oracle reference: pre-build a right-sized buffer in the narrow window of
    days whose order is GUARANTEED to arrive before the embargo starts, then
    drop straight back to the lean base target. Crucially the target must NOT
    stay elevated once further orders would arrive during (or right after) the
    embargo -- those orders are either wasted or land in a pile-up right after
    the embargo ends, which would make the "oracle" reference an artificially
    bad (rather than a valid ceiling) policy."""
    tgt = [base_target_opt] * H
    for (s, e) in embargo_windows:
        # an order placed on day t arrives on day t+L; to land BEFORE the
        # embargo starts (day s) it must satisfy t+L < s, i.e. t <= s-L-1.
        safe_last = s - L - 1
        if safe_last < 0:
            continue
        lead_start = max(0, safe_last - prebuild_days + 1)
        needed = sum(demand[s:e]) * buffer_frac
        elevated = base_target_opt + needed
        for t in range(lead_start, safe_last + 1):
            if elevated > tgt[t]:
                tgt[t] = elevated
        # target reverts to base_target_opt for [safe_last+1, e-1] (embargo and
        # the doomed-to-fail tail) -- no further ordering attempted there.
    return lambda t: tgt[t]


# ----------------------------- instance generation ----------------------------
def _gen_timeline(rng, H, mu, std, embargoes, K_pre=9, decoy=None):
    demand = [max(0.0, round(rng.gauss(mu, std), 2)) for _ in range(H)]
    if decoy is not None:
        ds, de, mult = decoy
        for t in range(ds, min(de, H)):
            demand[t] = round(demand[t] * mult + rng.uniform(0.0, 2.0), 2)
    signal = [round(max(0.0, rng.gauss(0.4, 0.5)), 3) for _ in range(H)]
    for (s, e) in embargoes:
        pre_start = max(0, s - K_pre)
        span = max(1, s - pre_start)
        for i, t in enumerate(range(pre_start, s)):
            frac = (i + 1) / span
            boost = 2.0 + 4.0 * frac + rng.uniform(-0.3, 0.3)
            if boost > signal[t]:
                signal[t] = round(boost, 3)
        decay_len = min(3, e - s)
        for i, t in enumerate(range(s, min(s + decay_len, H))):
            frac = 1.0 - (i + 1) / (decay_len + 1)
            boost = 2.0 * frac + rng.uniform(-0.2, 0.2)
            if boost > signal[t]:
                signal[t] = round(max(0.0, boost), 3)
    blocked = [False] * H
    for (s, e) in embargoes:
        for t in range(s, min(e, H)):
            blocked[t] = True
    return demand, signal, blocked


def _build_timeline(seed, H, mu, std, embargoes, decoy=None):
    rng = random.Random(seed)
    demand, signal, blocked = _gen_timeline(rng, H, mu, std, embargoes, decoy=decoy)
    return {"demand": demand, "signal": signal, "blocked": blocked, "embargoes": list(embargoes)}


def _instance_specs():
    """10 fixed instances. Each mixes calm timelines (no embargo) with shock
    timelines (embargo windows, sometimes doubled / early / long / with a
    demand-only decoy spike carrying NO precursor warning)."""
    L = 3
    specs = []

    def calm(seed, H=36, mu=8.0, std=1.5):
        return ("calm", seed, H, mu, std, [], None)

    def shock(seed, H=36, mu=8.0, std=1.5, embargoes=(), decoy=None):
        return ("shock", seed, H, mu, std, list(embargoes), decoy)

    # idx0: baseline mix, one mid-horizon embargo
    specs.append([calm(101), calm(102), shock(103, embargoes=[(14, 20)]),
                  shock(104, embargoes=[(16, 22)]), shock(105, embargoes=[(13, 19)])])
    # idx1: same shape, different seeds (independent draw)
    specs.append([calm(111), calm(112), shock(113, embargoes=[(15, 21)]),
                  shock(114, embargoes=[(12, 18)]), shock(115, embargoes=[(17, 23)])])
    # idx2: harsher mix (more shock timelines)
    specs.append([calm(121), shock(122, embargoes=[(14, 20)]), shock(123, embargoes=[(15, 21)]),
                  shock(124, embargoes=[(13, 19)]), shock(125, embargoes=[(16, 22)])])
    # idx3: gentler mix (more calm timelines)
    specs.append([calm(131), calm(132), calm(133), shock(134, embargoes=[(15, 21)]),
                  shock(135, embargoes=[(14, 20)])])
    # idx4: decoy demand spike with NO precursor warning, alongside a real embargo
    specs.append([calm(141), shock(142, embargoes=[(15, 21)]),
                  shock(143, embargoes=[(16, 22)], decoy=(24, 28, 3.0)),
                  calm(144), shock(145, embargoes=[(13, 19)])])
    # idx5: double embargo within one timeline
    specs.append([calm(151), calm(152), shock(153, embargoes=[(10, 15), (24, 29)]),
                  shock(154, embargoes=[(11, 16)]), shock(155, embargoes=[(23, 28)])])
    # idx6: calm-heavy sanity instance, one short mild embargo -- mostly a
    # holding-cost-discipline test, but keeps a non-degenerate UB-BASE gap
    specs.append([calm(161), calm(162), calm(163), calm(164),
                  shock(165, embargoes=[(17, 20)])])
    # idx7: long embargo (stress buffer sizing)
    specs.append([calm(171), calm(172), shock(173, embargoes=[(14, 26)]),
                  shock(174, embargoes=[(15, 27)]), shock(175, embargoes=[(13, 25)])])
    # idx8: early embargo (little runway to react from day 0)
    specs.append([calm(181), calm(182), shock(183, embargoes=[(4, 10)]),
                  shock(184, embargoes=[(5, 11)]), shock(185, embargoes=[(4, 10)])])
    # idx9: held-out generalization: bigger scale (higher mean demand, longer horizon)
    specs.append([calm(191, H=48, mu=13.0, std=2.2), calm(192, H=48, mu=13.0, std=2.2),
                  shock(193, H=48, mu=13.0, std=2.2, embargoes=[(20, 28)]),
                  shock(194, H=48, mu=13.0, std=2.2, embargoes=[(18, 26)]),
                  shock(195, H=48, mu=13.0, std=2.2, embargoes=[(22, 31)])])
    return specs, L


def _build_instances():
    specs, L = _instance_specs()
    instances = []
    for idx, tl_specs in enumerate(specs):
        H = tl_specs[0][2]
        timelines = []
        for (_kind, seed, h, mu, std, embargoes, decoy) in tl_specs:
            timelines.append(_build_timeline(seed, h, mu, std, embargoes, decoy=decoy))
        price = 6.0
        unit_cost = 3.0
        holding_rate = 0.12
        stockout_penalty = 4.0
        init_stock = round(tl_specs[0][3] * L, 2)
        public = {
            "name": f"embargo{idx:02d}",
            "n_timelines": len(timelines),
            "horizon": H,
            "lead_time": L,
            "price": price, "unit_cost": unit_cost,
            "holding_rate": holding_rate, "stockout_penalty": stockout_penalty,
            "init_stock": init_stock,
            "timelines": [{"demand": list(tl["demand"]), "precursor_signal": list(tl["signal"])}
                          for tl in timelines],
        }
        instances.append({
            "public": public,
            "H": H, "L": L, "price": price, "unit_cost": unit_cost,
            "holding_rate": holding_rate, "stockout_penalty": stockout_penalty,
            "init_stock": init_stock, "timelines": timelines,
        })
    return instances


# ----------------------------- validation ------------------------------------
def _validate(ans):
    if not isinstance(ans, dict):
        return None
    req = ("base_target", "trigger", "hoard_target", "cooldown_days")
    if not all(k in ans for k in req):
        return None

    def _num(x):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        xf = float(x)
        if xf != xf or xf in (float("inf"), float("-inf")):
            return None
        return xf

    bt = _num(ans["base_target"])
    tr = _num(ans["trigger"])
    ht = _num(ans["hoard_target"])
    cd = _num(ans["cooldown_days"])
    if bt is None or tr is None or ht is None or cd is None:
        return None
    if bt < 0 or bt > 1e6 or ht < 0 or ht > 1e6:
        return None
    if cd < -1e-9 or cd > 60 + 1e-9:
        return None
    if abs(cd - round(cd)) > 1e-6:
        return None
    return {"base_target": bt, "trigger": tr, "hoard_target": ht, "cooldown_days": int(round(cd))}


# ----------------------------- scoring driver --------------------------------
def score_instance(inst, ans):
    H, L = inst["H"], inst["L"]
    price, unit_cost = inst["price"], inst["unit_cost"]
    holding_rate, stockout_penalty = inst["holding_rate"], inst["stockout_penalty"]
    init_stock = inst["init_stock"]
    timelines = inst["timelines"]

    all_demand = []
    for tl in timelines:
        all_demand.extend(tl["demand"])
    base_target_naive = naive_base_target(all_demand, L)

    cand_profits = []
    base_profits = []
    ub_profits = []
    for tl in timelines:
        demand, blocked = tl["demand"], tl["blocked"]
        signal = tl["signal"]

        base_fn = lambda t: base_target_naive
        base_profits.append(simulate(H, L, price, unit_cost, holding_rate, stockout_penalty,
                                      init_stock, demand, blocked, base_fn))

        ub_fn = make_oracle_target_fn(H, L, base_target_naive, tl["embargoes"], demand)
        ub_profits.append(simulate(H, L, price, unit_cost, holding_rate, stockout_penalty,
                                    init_stock, demand, blocked, ub_fn))

        cand_fn = make_reactive_target_fn(ans["base_target"], ans["trigger"],
                                           ans["hoard_target"], ans["cooldown_days"], signal)
        cand_profits.append(simulate(H, L, price, unit_cost, holding_rate, stockout_penalty,
                                      init_stock, demand, blocked, cand_fn))

    min_cand = min(cand_profits)
    min_base = min(base_profits)
    min_ub = min(ub_profits)
    # numerical-stability floor tied to the instance's own profit scale, so a
    # near-degenerate (mostly-calm) instance can't blow up into a hair-trigger
    # denominator that clamps any tiny deviation straight to 0 or 1.
    scale_floor = max(5.0, 0.01 * abs(min_base))
    denom = max(scale_floor, min_ub - min_base)
    r = 0.1 + 0.9 * (min_cand - min_base) / denom
    if r != r or r in (float("inf"), float("-inf")):
        return 0.0
    return max(0.0, min(1.0, r))


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            valid = _validate(ans)
        except Exception:
            valid = None
        if valid is None:
            vec.append(0.0)
            continue
        try:
            r = score_instance(inst, valid)
        except Exception:
            r = 0.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
