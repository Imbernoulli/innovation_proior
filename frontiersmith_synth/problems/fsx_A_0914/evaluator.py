#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0914 -- "Glass Furnace Product Wheel: Dilution-Routed
Changeovers" (family: dilution-wheel-lot-sizing; format B, quality-metric).

THEME.  A glass furnace runs a repeating "wheel" of colour campaigns to satisfy K
steady demand streams.  Switching the melt from colour i to colour j is NOT a fixed
matrix lookup: the previous colour's pigment is a residual contamination that must be
flushed out with fresh melt, and each flush TON dilutes the residual multiplicatively
(exponential decay).  How many flush tons are needed depends on (a) how far apart the
two colours sit on the tint line and (b) how strict the TARGET colour's own purity
spec is.  Because that target-dependent threshold enters the formula, the direct
i->j changeover cost does NOT behave like a metric closed under shortest paths of its
own edges: routing i -> k -> j (actually PRODUCING an intermediate colour k as a real
campaign) can flush far fewer tons in total than going i -> j directly, whenever k's
own spec is loose (cheap to reach) even though k sits far from i.  Colour k becomes a
catalyst that rewrites the effective changeover graph -- but inserting it costs a real
minimum-size campaign, which perturbs every colour's inventory (holding/stockout)
because it changes the wheel's total cycle time.  This composes three mechanisms:

  (1) cyclic-sequence-design      -- the "wheel" is a cyclic list of campaigns which
                                      repeats forever; the solver chooses the order,
                                      the SET of campaigns (colours may repeat), and
                                      how many times each colour is visited.
  (2) exponential-dilution-transitions -- changeover cost between two campaigns is
                                      computed procedurally from a continuous dilution
                                      process, not read out of a fixed matrix.
  (3) lot-size-inventory-tradeoff -- each campaign's lot size trades off holding cost
                                      (excess inventory) against stockout cost
                                      (backlog), and the WHOLE wheel's cycle time
                                      (driven by total changeover waste) sets the
                                      review period for every colour's inventory.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance) -- see statement.md for the full
          schema: k, colors[] (id, tint, tau, demand, hold, back, min_lot), lambda,
          flush_cost, waste_price, cycles, max_campaigns, max_lot.
  stdout: ONE JSON object: {"wheel": [{"color": c0, "lot": L0}, ...]}
          1 <= len(wheel) <= max_campaigns; each "color" in [0,k); each "lot" an
          integer with colors[c].min_lot <= lot <= max_lot; EVERY colour id 0..k-1
          must appear at least once.  Anything else (wrong types, out-of-range
          ids/lots, a missing colour, a crash, a timeout, non-JSON) -> score 0.0.

SCORING (deterministic, no wall-time).  For each instance the evaluator itself
builds a WEAK reference wheel (visit every colour once, in the given input order,
each lot sized to its own demand share of a fixed generic cycle length -- exactly
"ignore the routing question").  It also computes the candidate's total cost the
same way: build the cyclic timeline of campaigns (changeover waste computed by the
dilution formula between every consecutive pair, including wrap-around), then
integrate, colour by colour, the EXACT (closed-form, no time-stepping) holding /
backlog cost over `cycles` repeats of the wheel, plus the direct material cost of
wasted tons.  Minimization is normalized against the weak reference:
    r = clamp( 0.1 * cost_baseline / max(cost_candidate, eps), 0, 1 )
so reproducing the naive wheel scores ~0.1, and every ton of wasted material or
period of avoidable backlog you shave off increases the score, with headroom left
above the strong reference (see solutions/strong.py).

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  All references,
the dilution formula's ground truth, and validation happen in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# --------------------------- deterministic RNG ------------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    def nxt_float(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u = ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)
        return lo + u * (hi - lo)

    return nxt_int, nxt_float


# --------------------------- dilution formula --------------------------------
def direct_waste_steps(tint_i, tau_j, lam, diff=None):
    """Flush steps to dilute the |tint_i - tint_j| pigment mismatch below the
    TARGET colour j's own purity threshold tau_j, with per-step multiplicative
    decay `lam` (0 < lam < 1).  0 steps if already within spec."""
    if diff <= tau_j:
        return 0
    ratio = diff / float(tau_j)
    steps = math.log(ratio) / math.log(1.0 / lam)
    return max(0, int(math.ceil(steps - 1e-9)))


def build_waste_fn(colors, lam, flush_cost):
    tints = [c["tint"] for c in colors]
    taus = [c["tau"] for c in colors]

    def waste(i, j):
        if i == j:
            return 0
        diff = abs(tints[i] - tints[j])
        steps = direct_waste_steps(tints[i], taus[j], lam, diff)
        return flush_cost * steps

    return waste


# --------------------------- instance family ---------------------------------
def _make_colors(seed, k, tint_span, tau_lo, tau_hi, dem_lo, dem_hi,
                  hold_lo, hold_hi, back_lo, back_hi, minlot_lo, minlot_hi,
                  special=None):
    ni, nf = _rng(seed)
    colors = []
    tints = sorted(ni(0, tint_span) for _ in range(k))
    # avoid exact duplicate tints (would make some pairs trivially free forever)
    for idx in range(1, k):
        if tints[idx] <= tints[idx - 1]:
            tints[idx] = tints[idx - 1] + ni(3, 9)
    for i in range(k):
        tau = ni(tau_lo, tau_hi)
        dem = round(nf(dem_lo, dem_hi), 3)
        hold = round(nf(hold_lo, hold_hi), 2)
        back = round(nf(back_lo, back_hi), 2)
        minlot = ni(minlot_lo, minlot_hi)
        colors.append({"id": i, "tint": tints[i], "tau": tau, "demand": dem,
                        "hold": hold, "back": back, "min_lot": minlot})
    if special:
        special(colors, ni, nf)
    return colors


def _plant_shortcut(colors, a, b, hub, tau_b_strict, tau_hub_loose, gap):
    """Force: colour `a` and colour `b` sit `gap` tint units apart with b's own
    threshold STRICT (expensive direct a->b); colour `hub` sits close to b but
    has a LOOSE threshold. This plants the trap: direct a->b is expensive, but
    a->hub->b is much cheaper (loose hub threshold absorbs the long leg cheaply).
    A myopic nearest-neighbour construction on the DIRECT matrix has no reason to
    discover this route (it never looks at target-dependent thresholds two hops
    ahead), so it frequently misses it or finds it only partially; how much any
    given single-visit tour benefits from it still depends on the instance's full
    geometry, which is why the acceptance gate is checked on the AGGREGATE mean
    over all 10 instances, not any one instance in isolation."""
    lo_tint = min(c["tint"] for c in colors)
    colors[a]["tint"] = lo_tint
    colors[b]["tint"] = lo_tint + gap
    colors[b]["tau"] = tau_b_strict
    colors[hub]["tint"] = lo_tint + gap - max(3, gap // 12)
    colors[hub]["tau"] = tau_hub_loose


def _build_instances():
    insts = []

    # ---- instances 0-2: HAND-PLANTED trap cases (routing beats direct) ----
    def spec0(colors, ni, nf):
        _plant_shortcut(colors, a=2, b=3, hub=4, tau_b_strict=3,
                         tau_hub_loose=70, gap=150)

    c0 = _make_colors(9101, 6, 190, 8, 28, 0.06, 0.14, 1.0, 3.0, 6.0, 16.0, 7, 14,
                       special=spec0)
    insts.append({"name": "wheel_trap_A", "colors": c0, "lam": 0.5,
                  "flush_cost": 5, "waste_price": 3, "cycles": 12,
                  "max_campaigns": 14, "max_lot": 500})

    def spec1(colors, ni, nf):
        _plant_shortcut(colors, a=1, b=4, hub=0, tau_b_strict=2,
                         tau_hub_loose=80, gap=170)

    c1 = _make_colors(9102, 7, 195, 6, 24, 0.07, 0.15, 1.0, 3.5, 7.0, 18.0, 8, 16,
                       special=spec1)
    insts.append({"name": "wheel_trap_B", "colors": c1, "lam": 0.45,
                  "flush_cost": 6, "waste_price": 2, "cycles": 14,
                  "max_campaigns": 16, "max_lot": 520})

    def spec2(colors, ni, nf):
        _plant_shortcut(colors, a=2, b=5, hub=6, tau_b_strict=2,
                         tau_hub_loose=75, gap=180)
        # a SECOND shortcut in the same instance to stack the effect
        _plant_shortcut(colors, a=0, b=3, hub=1, tau_b_strict=3,
                         tau_hub_loose=65, gap=140)

    c2 = _make_colors(9103, 8, 200, 6, 22, 0.06, 0.13, 1.2, 3.5, 8.0, 20.0, 7, 15,
                       special=spec2)
    insts.append({"name": "wheel_trap_C", "colors": c2, "lam": 0.5,
                  "flush_cost": 6, "waste_price": 3, "cycles": 12,
                  "max_campaigns": 18, "max_lot": 480})

    # ---- instances 3-4: high-utilization trap variants (cycle time matters
    #      a lot because backlog cost amplifies any wasted-tonnage difference) ----
    def spec3(colors, ni, nf):
        _plant_shortcut(colors, a=1, b=3, hub=5, tau_b_strict=2,
                         tau_hub_loose=85, gap=175)

    c3 = _make_colors(9104, 7, 195, 7, 24, 0.09, 0.16, 1.0, 3.0, 9.0, 22.0, 8, 16,
                       special=spec3)
    insts.append({"name": "wheel_trap_hiutil_A", "colors": c3, "lam": 0.5,
                  "flush_cost": 5, "waste_price": 3, "cycles": 16,
                  "max_campaigns": 15, "max_lot": 420})

    def spec4(colors, ni, nf):
        _plant_shortcut(colors, a=1, b=6, hub=3, tau_b_strict=2,
                         tau_hub_loose=90, gap=185)

    c4 = _make_colors(9105, 8, 198, 7, 22, 0.08, 0.15, 1.0, 3.2, 9.0, 24.0, 8, 16,
                       special=spec4)
    insts.append({"name": "wheel_trap_hiutil_B", "colors": c4, "lam": 0.55,
                  "flush_cost": 6, "waste_price": 2, "cycles": 15,
                  "max_campaigns": 16, "max_lot": 440})

    # ---- instances 5-6: CONTROL cases, no strong shortcut planted (uniformly
    #      spread tints / mild thresholds) so the ladder isn't artificially wide
    #      everywhere ----
    c5 = _make_colors(9210, 9, 200, 10, 45, 0.06, 0.13, 1.0, 3.0, 6.0, 15.0, 7, 16)
    insts.append({"name": "wheel_control_A", "colors": c5, "lam": 0.5,
                  "flush_cost": 4, "waste_price": 2, "cycles": 12,
                  "max_campaigns": 12, "max_lot": 500})

    c6 = _make_colors(9211, 10, 210, 12, 50, 0.06, 0.12, 1.0, 3.0, 6.0, 16.0, 7, 16)
    insts.append({"name": "wheel_control_B", "colors": c6, "lam": 0.5,
                  "flush_cost": 4, "waste_price": 2, "cycles": 12,
                  "max_campaigns": 13, "max_lot": 500})

    # ---- instances 7-9: larger HELD-OUT instances, randomized but seeded, with
    #      shortcuts planted probabilistically (still deterministic) ----
    for idx, seed in enumerate((9301, 9302, 9303)):
        ni, nf = _rng(seed)
        k = ni(8, 10)

        def spec_rand(colors, ni2, nf2, _k=k):
            n_shortcuts = 2 if _k >= 9 else 1
            used = set()
            for _ in range(n_shortcuts):
                a = ni2(0, _k - 1)
                b = ni2(0, _k - 1)
                while b == a or (a, b) in used:
                    b = ni2(0, _k - 1)
                hub = ni2(0, _k - 1)
                while hub in (a, b):
                    hub = ni2(0, _k - 1)
                used.add((a, b))
                gap = ni2(140, 190)
                _plant_shortcut(colors, a, b, hub, tau_b_strict=ni2(2, 4),
                                 tau_hub_loose=ni2(70, 95), gap=gap)

        cN = _make_colors(seed, k, 200, 6, 26, 0.05, 0.10, 1.0, 3.4, 7.0, 20.0,
                           14, 34, special=spec_rand)
        insts.append({"name": f"wheel_heldout_{idx}", "colors": cN,
                      "lam": round(_rng(seed + 1)[1](0.4, 0.6), 2),
                      "flush_cost": ni(4, 7), "waste_price": ni(2, 4),
                      "cycles": ni(11, 17), "max_campaigns": k + 8,
                      "max_lot": ni(400, 550)})

    return insts


# --------------------------- cyclic timeline / cost --------------------------
def _timeline(campaigns, waste_fn):
    """Given [(color, lot), ...] in cyclic order, return (T_cyc, events_per_color,
    total_waste_tons) where events_per_color[c] = [(local_offset, lot), ...]."""
    m = len(campaigns)
    t = 0
    prev_color = campaigns[-1][0]
    events = {}
    total_waste = 0
    for (color, lot) in campaigns:
        w = waste_fn(prev_color, color)
        total_waste += w
        t += w
        events.setdefault(color, []).append((t, lot))
        t += lot
        prev_color = color
    return t, events, total_waste


def _integrate_linear(level0, rate, length, h, p):
    """Integral over tau in [0,length) of h*max(0,level) + p*max(0,-level),
    where level(tau) = level0 - rate*tau, rate >= 0."""
    if length <= 0:
        return 0.0
    if rate <= 1e-15:
        lvl = level0
        return length * (h * max(0.0, lvl) + p * max(0.0, -lvl))
    tstar = level0 / rate  # time of zero crossing (may be <0 or >length)
    if level0 >= 0:
        t1 = min(max(tstar, 0.0), length)
        # positive part: integral_0^t1 h*(level0 - rate*tau) dtau
        pos = h * (level0 * t1 - 0.5 * rate * t1 * t1)
        neg = 0.0
        if t1 < length:
            t2 = length
            # integral_t1^t2 p*(rate*tau - level0) dtau
            neg = p * (0.5 * rate * (t2 * t2 - t1 * t1) - level0 * (t2 - t1))
        return pos + neg
    else:
        # already negative throughout (rate>0 keeps decreasing it further)
        t2 = length
        return p * (0.5 * rate * (t2 * t2 - 0.0) - level0 * (t2 - 0.0))


def _total_cost(colors, campaigns, waste_fn, waste_price, cycles):
    """Total cost over `cycles` REAL repeats of the wheel.  Inventory starts at 0
    and is tracked continuously across every repeat (no artificial reset): a
    colour whose per-cycle production Q_i falls short of its per-cycle consumption
    d_i*T_cyc genuinely falls further and further behind on every lap, exactly as
    a real backlog would -- there is no "long-run steady state" discount for
    chronic under-production. Small, honest rounding slack (lot sizes are
    integers) costs little because it does not accumulate a large per-cycle
    deficit; deliberately under-producing every colour (e.g. always shipping the
    legal minimum lot regardless of demand) accumulates a real, escalating
    backlog and is priced accordingly."""
    T_cyc, events, waste_per_cycle = _timeline(campaigns, waste_fn)
    if T_cyc <= 0:
        return float("inf")
    horizon_end = cycles * T_cyc
    cost = waste_price * waste_per_cycle * cycles
    for c in colors:
        cid = c["id"]
        local = events.get(cid, [])
        d = c["demand"]; h = c["hold"]; p = c["back"]
        level = 0.0
        cur_t = 0.0
        cc = 0.0
        for r in range(cycles):
            base = r * T_cyc
            for (loc_t, lot) in local:
                et = base + loc_t
                seg = et - cur_t
                if seg > 0:
                    cc += _integrate_linear(level, d, seg, h, p)
                    level -= d * seg
                level += lot
                cur_t = et
        seg = horizon_end - cur_t
        if seg > 0:
            cc += _integrate_linear(level, d, seg, h, p)
        cost += cc
    return cost


def _baseline_cost(inst):
    """Weak reference: visit every colour once, in the GIVEN input order, ignoring
    routing entirely (uses whatever direct changeover the input order happens to
    incur) with lot sized to a fixed generic per-cycle demand share."""
    colors = inst["colors"]
    waste_fn = build_waste_fn(colors, inst["lam"], inst["flush_cost"])
    k = len(colors)
    generic_T = 260.0 + 40.0 * k
    campaigns = []
    for c in colors:
        lot = max(c["min_lot"], int(round(c["demand"] * generic_T)))
        lot = min(lot, inst["max_lot"])
        campaigns.append((c["id"], lot))
    return _total_cost(colors, campaigns, waste_fn, inst["waste_price"], inst["cycles"])


# --------------------------- answer validation --------------------------------
def _validate(inst, answer):
    if not isinstance(answer, dict):
        return None
    wheel = answer.get("wheel")
    if not isinstance(wheel, list):
        return None
    m = len(wheel)
    if m < 1 or m > inst["max_campaigns"]:
        return None
    k = len(inst["colors"])
    colors = inst["colors"]
    out = []
    for entry in wheel:
        if not isinstance(entry, dict):
            return None
        color = entry.get("color")
        lot = entry.get("lot")
        if isinstance(color, bool) or not isinstance(color, int):
            return None
        if color < 0 or color >= k:
            return None
        if isinstance(lot, bool) or not isinstance(lot, int):
            return None
        minlot = colors[color]["min_lot"]
        if lot < minlot or lot > inst["max_lot"]:
            return None
        out.append((color, lot))
    # every colour's demand stream must actually be served: a colour missing from
    # the wheel is not "produced zero" in any physical sense -- the furnace never
    # revisits it, so its backlog grows forever.  Rather than model that unbounded
    # divergence explicitly, we require every colour id to appear >=1 time; this
    # also closes an exploit where an all-but-one-colour-omitting wheel would get
    # the missing colours' cost "smoothed" by the symmetric-drift accounting below
    # as if they were being served on average (they are not).
    if {c for (c, _) in out} != set(range(k)):
        return None
    return out


# --------------------------- main ---------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        colors = inst["colors"]
        waste_fn = build_waste_fn(colors, inst["lam"], inst["flush_cost"])
        b_cost = _baseline_cost(inst)

        public = {
            "name": inst["name"],
            "k": len(colors),
            "colors": [dict(c) for c in colors],
            "lambda": inst["lam"],
            "flush_cost": inst["flush_cost"],
            "waste_price": inst["waste_price"],
            "cycles": inst["cycles"],
            "max_campaigns": inst["max_campaigns"],
            "max_lot": inst["max_lot"],
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            campaigns = _validate(inst, ans)
        except Exception:
            campaigns = None
        if campaigns is None:
            vec.append(0.0)
            continue
        try:
            cand_cost = _total_cost(colors, campaigns, waste_fn,
                                     inst["waste_price"], inst["cycles"])
        except Exception:
            cand_cost = float("inf")
        if not (cand_cost == cand_cost) or cand_cost in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = 0.1 * b_cost / max(cand_cost, 1e-9)
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
