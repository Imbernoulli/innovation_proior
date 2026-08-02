#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1302 -- "Bidding Against Bots That Watch You Back"
(family: auction-bidding-policy; format B, quality-metric).

THEME.  A demand-side bidder faces a SEQUENCE of T second-price-style auctions
(an ad-exchange / procurement style feed).  Each round i has a PUBLIC "signal"
`base[i]` (an estimate of how valuable the item is) but the REALIZED true value
`v[i] = base[i] * mult[i]` depends on a HIDDEN per-round multiplier `mult[i]`
that the candidate never sees (value uncertainty).  The candidate has a single
HARD BUDGET `B` that never replenishes (budget-pacing).  A single reactive
competitor bids `comp_base[i] * (1 + adapt_rate * recent_avg_aggression)`,
where `recent_avg_aggression` is the average, over the last `mem_k` rounds,
of the candidate's OWN `bid / base` ratio -- i.e. the competitor watches how
aggressively the candidate has been bidding (relative to the public signal)
and ratchets its own bid up in response (competitor-response). Because it is
a second-price rule, the WINNER of round i pays the competitor's bid `c_i`
(not their own bid) -- so a high bid never costs extra THIS round, but it
DOES train the competitor for future rounds. A bid can never exceed the
remaining budget (auto-clipped).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "T": int, "budget": float,
             "base": [b_0, ..., b_{T-1}],          # public per-round signal
             "comp_base": [c_0, ..., c_{T-1}],      # public competitor baseline
             "adapt_rate": float, "mem_k": int}
  stdout: ONE JSON object:
            {"bids": [x_0, ..., x_{T-1}]}           # x_i >= 0, finite
          A layout is VALID iff `bids` is a list of exactly T finite numbers,
          each >= 0. Invalid output, wrong length, negative/non-finite value,
          a crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time). For each instance the checker replays
the EXACT round-by-round mechanics described above using the candidate's bid
vector and the HIDDEN true values, producing total realized surplus F_cand =
sum over won rounds of (v_i - price_i). Two internal references anchor the
score (candidate never sees either):
    F_weak  = surplus from bidding 0 every round (always 0, by construction)
    F_ideal = surplus from an internal oracle that sees the TRUE values (not
              just the noisy public signal) and paces/shades bids accordingly
Score per instance:
    r = clamp( 0.1 + 0.9 * (F_cand - F_weak) / max(1e-9, F_ideal - F_weak), 0, 1)
so doing nothing scores ~0.1 and matching the informed oracle approaches 1.0
(the oracle has strictly more information than any candidate can have, so it
is generally unreachable -> headroom is preserved).

ISOLATION. The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance. All hidden
data (mult / true values) and the oracle computation live in THIS parent
process only.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = seed & ((1 << 64) - 1)

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return state

    def nxt_int(lo, hi):
        return lo + (nxt() >> 17) % (hi - lo + 1)

    def nxt_float(lo, hi):
        return lo + (nxt() >> 11) / float(1 << 53) * (hi - lo)

    return nxt_int, nxt_float


# ----------------------------- instance construction ------------------------
def _build_base(nxt_float, T, lo, hi, windows, jitter_frac=0.12):
    base = [float(lo)] * T
    for (s, ln) in windows:
        for i in range(s, min(T, s + ln)):
            base[i] = float(hi)
    out = []
    for i in range(T):
        j = base[i] * jitter_frac
        val = base[i] + nxt_float(-j, j)
        if val < 0.5:
            val = 0.5
        out.append(round(val, 3))
    return out


def _build_comp_base(base, comp_frac, nxt_float, jitter=0.10):
    out = []
    for b in base:
        j = b * comp_frac * jitter
        c = b * comp_frac + nxt_float(-j, j)
        if c < 0.1:
            c = 0.1
        out.append(round(c, 3))
    return out


def _build_mult(nxt_float, T, mrange):
    return [round(nxt_float(mrange[0], mrange[1]), 4) for _ in range(T)]


_SPECS = [
    # (name, seed, T, lo, hi, windows, comp_frac, adapt_rate, mem_k, budget_frac, mrange)
    ("late_spike_tight", 9001, 40, 4.0, 60.0, [(32, 6)], 0.45, 0.55, 6, 0.30, (0.6, 1.6)),
    ("late_spike_tight2", 9002, 44, 5.0, 70.0, [(35, 7)], 0.42, 0.50, 6, 0.28, (0.6, 1.6)),
    ("double_spike_late_heavy", 9003, 50, 4.0, 55.0, [(14, 4), (40, 6)], 0.45, 0.50, 6, 0.32, (0.6, 1.6)),
    ("early_spike_generous", 9004, 40, 4.0, 55.0, [(2, 6)], 0.45, 0.35, 5, 0.55, (0.6, 1.6)),
    ("mid_spike_moderate", 9005, 42, 4.5, 50.0, [(18, 6)], 0.45, 0.45, 6, 0.38, (0.6, 1.6)),
    ("late_narrow_window", 9006, 46, 4.0, 80.0, [(40, 3)], 0.40, 0.60, 7, 0.24, (0.6, 1.6)),
    ("long_horizon_late", 9007, 64, 3.5, 65.0, [(54, 5)], 0.45, 0.55, 6, 0.26, (0.6, 1.6)),
    ("multi_window_scatter", 9008, 48, 4.0, 45.0, [(6, 3), (20, 3), (44, 4)], 0.45, 0.50, 6, 0.30, (0.6, 1.6)),
    ("late_spike_high_adapt", 9009, 40, 4.0, 60.0, [(33, 5)], 0.42, 0.80, 5, 0.30, (0.6, 1.6)),
    ("held_out_large_late", 9010, 70, 4.0, 75.0, [(58, 7)], 0.44, 0.55, 6, 0.27, (0.55, 1.7)),
]


def _build_instance(spec):
    name, seed, T, lo, hi, windows, comp_frac, adapt_rate, mem_k, budget_frac, mrange = spec
    nxt_int, nxt_float = _rng(seed)
    base = _build_base(nxt_float, T, lo, hi, windows)
    comp_base = _build_comp_base(base, comp_frac, nxt_float)
    mult = _build_mult(nxt_float, T, mrange)
    budget = round(budget_frac * sum(base), 2)
    public = {"name": name, "T": T, "budget": budget, "base": base,
              "comp_base": comp_base, "adapt_rate": adapt_rate, "mem_k": mem_k}
    return {"public": public, "hidden": {"mult": mult}}


def make_instances():
    return [_build_instance(spec) for spec in _SPECS]


# ----------------------------- simulation engine -----------------------------
def simulate(public, mult, bids):
    """Replay the round-by-round mechanics. `bids` = list of T non-negative
    floats (candidate's plan). `mult` = hidden per-round true-value multiplier
    (None allowed only when bids are all-zero, since it is never referenced).
    """
    T = public["T"]; base = public["base"]; comp_base = public["comp_base"]
    adapt_rate = public["adapt_rate"]; mem_k = public["mem_k"]
    remaining = public["budget"]
    hist = []
    total = 0.0
    for i in range(T):
        bi = bids[i]
        b_eff = bi if bi < remaining else remaining
        if b_eff < 0:
            b_eff = 0.0
        recent = hist[-mem_k:] if hist else []
        avg_agg = sum(recent) / len(recent) if recent else 0.0
        c_i = comp_base[i] * (1.0 + adapt_rate * avg_agg)
        if b_eff >= c_i and b_eff > 1e-12:
            price = c_i
            remaining -= price
            v_i = base[i] * (mult[i] if mult is not None else 1.0)
            total += (v_i - price)
        agg_i = b_eff / base[i] if base[i] > 1e-9 else 0.0
        hist.append(agg_i)
    return total


def _oracle_ideal(public, mult):
    """Reference with FULL knowledge of the true values (never given to the
    candidate). Prices each round at its worst-case (fully escalated)
    competitor cost, then greedily fills the budget by SURPLUS DENSITY
    (net surplus per dollar of worst-case cost) -- a budgeted-knapsack
    selection over the true values, which a candidate that only sees the
    noisy public signal cannot replicate exactly."""
    T = public["T"]; base = public["base"]; comp_base = public["comp_base"]
    adapt_rate = public["adapt_rate"]; budget = public["budget"]
    v = [base[i] * mult[i] for i in range(T)]
    costs = [comp_base[i] * (1.0 + adapt_rate) for i in range(T)]
    idx = [i for i in range(T) if v[i] > costs[i] and costs[i] > 1e-9]
    idx.sort(key=lambda i: (v[i] - costs[i]) / costs[i], reverse=True)
    bids = [0.0] * T
    spent = 0.0
    for i in idx:
        c = costs[i] * 1.12
        if spent + c <= budget:
            bids[i] = c
            spent += c
    return simulate(public, mult, bids)


# ----------------------------- validation ----------------------------------
def _extract_bids(inst, answer):
    if not isinstance(answer, dict):
        return None
    bids = answer.get("bids")
    T = inst["public"]["T"]
    if not isinstance(bids, list) or len(bids) != T:
        return None
    out = []
    for x in bids:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        xf = float(x)
        if xf != xf or xf in (float("inf"), float("-inf")) or xf < 0:
            return None
        out.append(xf)
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = make_instances()

    vec = []
    for inst in instances:
        public = inst["public"]
        mult = inst["hidden"]["mult"]
        f_weak = 0.0  # bidding 0 everywhere always yields 0 surplus
        f_ideal = _oracle_ideal(public, mult)
        denom = f_ideal - f_weak
        if denom < 1e-9:
            denom = 1e-9

        ans, st = isorun.run_candidate(cand, public, timeout=5)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            bids = _extract_bids(inst, ans)
        except Exception:
            bids = None
        if bids is None:
            vec.append(0.0)
            continue
        try:
            f_cand = simulate(public, mult, bids)
        except Exception:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (f_cand - f_weak) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
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
