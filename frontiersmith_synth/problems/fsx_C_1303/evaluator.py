#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1303 -- "Rebalancing a Line While It Is Running"
(family: factory-line-balance-policy; eval_form: quality-metric).

A serial assembly line of K stations (station 0 .. K-1, in series, connected by
K-1 finite WIP buffers) runs for T product-mix "epochs" (epoch t lasts L[t]
ticks). In epoch t, station i's *base* cycle time is base_cycle[t][i] (time to
process one unit with no boost). A shared pool of P "boost units" (extra
labor/tooling) can be redistributed across stations each epoch; assigning
`u` boost units to station i in epoch t divides its cycle time by
(1 + k_eff[i] * u). Because the product mix changes every epoch, WHICH station
is slowest (the bottleneck) *migrates* across the horizon -- sometimes smoothly
drifting, sometimes oscillating between two or three stations.

Re-pointing boost units at a station costs a CHANGEOVER: every station whose
boost allocation changes from the previous epoch is offline (produces zero,
neither accepts input nor emits output) for the first D ticks of the new
epoch (D = d0[i] + d1[i]*|delta|, capped so >=1 tick of real production
remains), and a booking/logistics cost m0[i] + m1[i]*|delta| is charged
against the score. WIP buffers between stations are sized ONCE, from a fixed
total buffer budget, at the start of the horizon (they are physical shelving,
not something you re-plan every epoch) -- their sizes determine how much of a
transient imbalance (a changeover stall, or a station suddenly becoming the
bottleneck) the line can absorb without a full stop-the-line stall.

The candidate is run as an ISOLATED subprocess (isorun): it reads ONE JSON
"public instance" (stdin) -- the FULL demand-mix schedule for the whole
horizon plus all cost/effectiveness constants; nothing is held back -- and
writes ONE JSON policy (stdout): a boost-allocation schedule for every epoch
plus a one-time buffer sizing. It never touches this evaluator's memory.

Score per instance: `shipped units over the horizon - money_weight*changeover
cost`, re-simulated tick-by-tick here (deterministic fluid flow-line model,
downstream-to-upstream sweep per tick -- see `simulate`), affine-normalized
against two evaluator-internal reference policies computed the SAME way:

    obj_base = "do nothing" policy: keep the given initial allocation and a
               uniform buffer split for the whole horizon (zero changeover)
    obj_ref  = idealized upper bound: re-optimize the boost allocation
               EVERY epoch for FREE (no changeover, infinite buffers) --
               strictly better than anything a real (single, paid-for-moves,
               finite-buffer) policy can achieve

    r = clamp( 0.1 + 0.9 * (obj_cand - obj_base) / (obj_ref - obj_base), 0, 1 )

Final score is the arithmetic mean of the 10 per-instance `r`.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r, in [0,1]>
  Vector: [r_1, ..., r_10]
"""
import sys, json, math, random
import isorun

MONEY_WEIGHT = 1.0
CAND_TIMEOUT = 20
VALID_FLOOR = 0.02
MIN_DENOM_FRAC = 0.03   # denom floor as a fraction of obj_ref, so a near-saturated ref still leaves headroom


# ============================ flow-line simulation ==========================
def eff_cycle(base, k, units):
    return base / (1.0 + k * units)


def simulate(K, T, L, base_cycle, k_eff, buffers, alloc, d0, d1, m0, m1, initial_alloc):
    """Deterministic fluid-flow simulation of the serial line.
    buffers: length K-1 ints (buffer[j] sits between station j and station j+1).
    alloc:   T x K ints (boost units assigned to each station each epoch).
    Returns (total_shipped, total_changeover_money)."""
    levels = [0.0] * (K - 1)
    shipped = 0.0
    money = 0.0
    prev = list(initial_alloc)
    for t in range(T):
        row = alloc[t]
        downtime = [0] * K
        for i in range(K):
            d = abs(row[i] - prev[i])
            if d > 0:
                dt = d0[i] + d1[i] * d
                downtime[i] = min(L[t] - 1, int(round(dt)))
                money += m0[i] + m1[i] * d
        prev = row
        cyc = [eff_cycle(base_cycle[t][i], k_eff[i], row[i]) for i in range(K)]
        rate = [(1.0 / c) if c > 1e-9 else 0.0 for c in cyc]
        for tick in range(L[t]):
            for i in range(K - 1, -1, -1):
                r = 0.0 if tick < downtime[i] else rate[i]
                inp = levels[i - 1] if i > 0 else float("inf")
                if i < K - 1:
                    space = buffers[i] - levels[i]
                else:
                    space = float("inf")
                produce = min(r, inp, space)
                if produce < 0.0:
                    produce = 0.0
                if i > 0:
                    levels[i - 1] -= produce
                if i < K - 1:
                    levels[i] += produce
                else:
                    shipped += produce
    return shipped, money


def compositions(total, parts):
    """All ways to write `total` as an ordered sum of `parts` nonneg ints."""
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def ideal_shipped(K, T, L, base_cycle, k_eff, P):
    """Upper bound: re-optimize the allocation every epoch for FREE
    (no changeover, infinite buffers)."""
    total = 0.0
    for t in range(T):
        best = 0.0
        for row in compositions(P, K):
            cyc = [eff_cycle(base_cycle[t][i], k_eff[i], row[i]) for i in range(K)]
            bottleneck_rate = min(1.0 / c for c in cyc)
            if bottleneck_rate > best:
                best = bottleneck_rate
        total += L[t] * best
    return total


def uniform_buffers(K, budget):
    n = K - 1
    base = budget // n
    rem = budget % n
    return [base + (1 if j < rem else 0) for j in range(n)]


# ============================ instance generation ===========================
def make_profile(K, T, kind, rng, base_low, base_high, jitter, **kw):
    mat = [[base_low for _ in range(K)] for _ in range(T)]
    if kind == "stable":
        b = kw["station"]
        for t in range(T):
            mat[t][b] = base_high
    elif kind == "two_phase":
        a, b, sw = kw["a"], kw["b"], kw["switch"]
        for t in range(T):
            mat[t][a if t < sw else b] = base_high
    elif kind == "oscillate2":
        a, b = kw["a"], kw["b"]
        for t in range(T):
            mat[t][a if t % 2 == 0 else b] = base_high
    elif kind == "oscillate3":
        cyc = kw["stations"]
        for t in range(T):
            mat[t][cyc[t % len(cyc)]] = base_high
    elif kind == "drift":
        sigma = kw.get("sigma", 0.85)
        start, end = kw["start"], kw["end"]
        for t in range(T):
            center = start + (end - start) * (t / max(T - 1, 1))
            for i in range(K):
                bump = (base_high - base_low) * math.exp(-((i - center) ** 2) / (2 * sigma * sigma))
                mat[t][i] = base_low + bump
    else:
        raise ValueError(kind)
    for t in range(T):
        for i in range(K):
            mat[t][i] = max(0.25, mat[t][i] + rng.uniform(-jitter, jitter))
    return mat


def build_instance(name, seed, K, T, L, kind, kparams, P, buffer_budget,
                    k_eff, d0, d1, m0, m1, base_low=1.0, base_high=2.7, jitter=0.05):
    rng = random.Random(seed)
    base_cycle = make_profile(K, T, kind, rng, base_low, base_high, jitter, **kparams)
    initial_alloc = [P // K + (1 if i < P % K else 0) for i in range(K)]
    return {
        "name": name, "K": K, "T": T, "L": list(L), "base_cycle": base_cycle,
        "k_eff": list(k_eff), "P": P, "buffer_budget": buffer_budget,
        "d0": list(d0), "d1": list(d1), "m0": list(m0), "m1": list(m1),
        "initial_alloc": initial_alloc, "seed": seed,
    }


def _build_instances():
    specs = []

    # 1: stable bottleneck -- sanity control (greedy should NOT be badly hurt here)
    specs.append(build_instance(
        "stable3", 130301, K=3, T=5, L=[70] * 5, kind="stable", kparams=dict(station=1),
        P=4, buffer_budget=8, k_eff=[0.28, 0.28, 0.28], d0=[1, 1, 1], d1=[1.2, 1.2, 1.2],
        m0=[1.2, 1.2, 1.2], m1=[0.7, 0.7, 0.7]))

    # 2: gentle noise around a stable bottleneck -- another control
    specs.append(build_instance(
        "noise3", 130302, K=3, T=6, L=[60] * 6, kind="stable", kparams=dict(station=2),
        P=4, buffer_budget=9, k_eff=[0.25, 0.30, 0.22], d0=[1, 1, 1], d1=[1.0, 1.0, 1.0],
        m0=[1.0, 1.0, 1.0], m1=[0.6, 0.6, 0.6], jitter=0.35))

    # 3: two-station oscillation -- TRAP: chases every epoch, pays full changeover every time
    specs.append(build_instance(
        "osc2_a", 130303, K=3, T=8, L=[55] * 8, kind="oscillate2", kparams=dict(a=0, b=2),
        P=4, buffer_budget=8, k_eff=[0.28, 0.28, 0.28], d0=[3, 3, 3], d1=[2.2, 2.2, 2.2],
        m0=[2.0, 2.0, 2.0], m1=[1.1, 1.1, 1.1]))

    # 4: three-station round-robin oscillation, larger line -- TRAP
    specs.append(build_instance(
        "osc3_a", 130304, K=4, T=9, L=[50] * 9, kind="oscillate3", kparams=dict(stations=[0, 2, 3]),
        P=5, buffer_budget=9, k_eff=[0.26, 0.26, 0.26, 0.26], d0=[3, 3, 3, 3], d1=[2.0, 2.0, 2.0, 2.0],
        m0=[1.8, 1.8, 1.8, 1.8], m1=[1.0, 1.0, 1.0, 1.0]))

    # 5: oscillation where the "obvious" bottleneck station has LOW capacity effectiveness
    #    -- TRAP: greedy dumps boost where it barely helps
    specs.append(build_instance(
        "osc2_lowk", 130305, K=4, T=8, L=[55] * 8, kind="oscillate2", kparams=dict(a=1, b=3),
        P=5, buffer_budget=10, k_eff=[0.42, 0.10, 0.40, 0.12], d0=[2, 2, 2, 2], d1=[1.6, 1.6, 1.6, 1.6],
        m0=[1.5, 1.5, 1.5, 1.5], m1=[0.9, 0.9, 0.9, 0.9]))

    # 6: sharp single regime switch with a tight post-switch epoch -- buffer-sizing TRAP
    specs.append(build_instance(
        "sharp_switch", 130306, K=5, T=6, L=[70, 70, 22, 70, 70, 70], kind="two_phase",
        kparams=dict(a=0, b=4, switch=2), P=6, buffer_budget=8,
        k_eff=[0.24] * 5, d0=[2] * 5, d1=[1.4] * 5, m0=[1.3] * 5, m1=[0.8] * 5,
        base_high=3.1))

    # 7: two-phase with mild jitter -- greedy handles the ONE real switch fine but
    #    jitter tempts it into extra spurious changeovers
    specs.append(build_instance(
        "two_phase_noisy", 130307, K=4, T=7, L=[60] * 7, kind="two_phase",
        kparams=dict(a=0, b=3, switch=4), P=5, buffer_budget=9,
        k_eff=[0.24, 0.30, 0.24, 0.30], d0=[2] * 4, d1=[1.3] * 4, m0=[1.2] * 4, m1=[0.8] * 4,
        jitter=0.30))

    # 8: smooth drift across the whole line -- migration TRAP (repeated re-chasing along the path)
    specs.append(build_instance(
        "drift5", 130308, K=5, T=8, L=[55] * 8, kind="drift", kparams=dict(start=0.0, end=4.0, sigma=0.8),
        P=6, buffer_budget=10, k_eff=[0.26] * 5, d0=[2] * 5, d1=[1.3] * 5, m0=[1.3] * 5, m1=[0.8] * 5))

    # 9: dense oscillation with a TIGHT buffer budget -- TRAP: uniform split cannot survive it,
    #    buffer allocation must concentrate at the hot gap
    specs.append(build_instance(
        "osc2_tightbuf", 130309, K=4, T=8, L=[50] * 8, kind="oscillate2", kparams=dict(a=0, b=1),
        P=4, buffer_budget=4, k_eff=[0.30, 0.30, 0.20, 0.20], d0=[2] * 4, d1=[1.5] * 4,
        m0=[1.4] * 4, m1=[0.8] * 4))

    # 10: held-out generalization -- bigger line, oscillation + differential effectiveness combined
    specs.append(build_instance(
        "combo_holdout", 130310, K=5, T=9, L=[48] * 9, kind="oscillate3",
        kparams=dict(stations=[0, 2, 4]), P=6, buffer_budget=11,
        k_eff=[0.40, 0.15, 0.38, 0.15, 0.20], d0=[2] * 5, d1=[1.5] * 5, m0=[1.5] * 5, m1=[0.9] * 5,
        jitter=0.20))

    return specs


# ============================ candidate answer handling =====================
def _parse_answer(inst, answer):
    K, T, P, budget = inst["K"], inst["T"], inst["P"], inst["buffer_budget"]
    if not isinstance(answer, dict):
        return None
    alloc_raw = answer.get("alloc")
    buf_raw = answer.get("buffers")
    if not isinstance(alloc_raw, list) or len(alloc_raw) != T:
        return None
    alloc = []
    for row in alloc_raw:
        if not isinstance(row, list) or len(row) != K:
            return None
        r = []
        for x in row:
            if isinstance(x, bool) or not isinstance(x, (int, float)):
                return None
            if not math.isfinite(x):
                return None
            xi = round(x)
            if abs(x - xi) > 1e-6 or xi < 0:
                return None
            r.append(int(xi))
        if sum(r) > P:
            return None
        alloc.append(r)
    if not isinstance(buf_raw, list) or len(buf_raw) != K - 1:
        return None
    bufs = []
    for x in buf_raw:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        if not math.isfinite(x):
            return None
        xi = round(x)
        if abs(x - xi) > 1e-6 or xi < 1:
            return None
        bufs.append(int(xi))
    if sum(bufs) > budget:
        return None
    return alloc, bufs


def baseline(inst):
    bufs = uniform_buffers(inst["K"], inst["buffer_budget"])
    alloc = [list(inst["initial_alloc"]) for _ in range(inst["T"])]
    shipped, money = simulate(inst["K"], inst["T"], inst["L"], inst["base_cycle"], inst["k_eff"],
                               bufs, alloc, inst["d0"], inst["d1"], inst["m0"], inst["m1"],
                               inst["initial_alloc"])
    return shipped - MONEY_WEIGHT * money


def reference(inst):
    return ideal_shipped(inst["K"], inst["T"], inst["L"], inst["base_cycle"], inst["k_eff"], inst["P"])


def score(inst, answer):
    parsed = _parse_answer(inst, answer)
    if parsed is None:
        return False, 0.0
    alloc, bufs = parsed
    shipped, money = simulate(inst["K"], inst["T"], inst["L"], inst["base_cycle"], inst["k_eff"],
                               bufs, alloc, inst["d0"], inst["d1"], inst["m0"], inst["m1"],
                               inst["initial_alloc"])
    return True, shipped - MONEY_WEIGHT * money


def public_view(inst):
    return {
        "K": inst["K"], "T": inst["T"], "P": inst["P"], "buffer_budget": inst["buffer_budget"],
        "L": inst["L"], "base_cycle": inst["base_cycle"], "k_eff": inst["k_eff"],
        "changeover_downtime_fixed": inst["d0"], "changeover_downtime_per_unit": inst["d1"],
        "changeover_money_fixed": inst["m0"], "changeover_money_per_unit": inst["m1"],
        "initial_alloc": inst["initial_alloc"], "money_weight": MONEY_WEIGHT, "seed": inst["seed"],
    }


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        obj_base = baseline(inst)
        obj_ref = reference(inst)
        denom = max(obj_ref - obj_base, MIN_DENOM_FRAC * max(obj_ref, 1e-9))

        ans, st = isorun.run_candidate(cand, public_view(inst), timeout=CAND_TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, 0.0
        if not ok:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (obj - obj_base) / denom
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        if r < VALID_FLOOR:
            r = VALID_FLOOR
        vec.append(float(r))

    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(v, 6) for v in vec]))


if __name__ == "__main__":
    main()
