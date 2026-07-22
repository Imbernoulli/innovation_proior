#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0791 -- "Last Curtain Call: Dynamic Ticket Pricing"
(family: perishable-price-shaping; format B, quality-metric).

THEME.  A venue is selling tickets for a one-night event.  There are C0 tickets
in stock and T pricing periods (days) counting down to showtime.  Any ticket
still unsold when the curtain rises is worthless -- it PERISHES.  Each period
the seller posts one price; the number of buyers who show up that period is a
deterministic, PRICE-ELASTIC function of that price.  Buyers also remember
recent prices: if the seller has been discounting, buyers expect a bargain and
arrive more slowly at any given price the next period -- a "reference price"
that drifts toward whatever was actually charged.  So today's price shapes not
only today's sales but tomorrow's willingness to pay.

MECHANISMS COMPOSED.
  (1) perishable-inventory-carryover: stock carries over period to period but
      is worth exactly 0 once the horizon (showtime) is reached.
  (2) price-elastic-arrival: the number of arrivals this period is a (seeded,
      per-period, per-instance) affine function of the price posted THIS period.
Interaction term: a price-driven reference level r_t that persists across
periods (mechanism 2 acting through time), which the seller's OWN pricing
history moves, which then modulates future periods' demand (mechanism 1's
"future still has to sell out of this same pool" makes that persistence bite).

INNOVATION HOOK.  The obvious move is to post, each period, the price that
maximizes THAT period's own revenue (the classic monopolist price for the
period's live demand curve).  That is myopic: it ignores that inventory is
FIXED and TIME-LIMITED, so unconstrained per-period optimization routinely
sells out the whole stock early at bargain prices, leaving a rich late-period
crowd (or a long unsold remainder) on the table.  The insight the strong
reference exploits: price to the demand THE REMAINING HORIZON CAN STILL
ABSORB, not to today's willingness-to-pay -- i.e. track the implicit shadow
value of one more unit of remaining stock against remaining time, and let
that shadow value (approximated here via inventory-pacing: sell at the rate
remaining_stock / remaining_periods) pull the price up when stock is scarce
relative to time left, and down when stock is abundant relative to time left.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "T": int, "C0": float, "p_max": float,
             "alpha": float, "r1": float,
             "a": [T floats], "b": [T floats], "g": [T floats]}
          Period t (0-indexed) demand-if-unconstrained, given price p and the
          CURRENT reference level r:
              A_t(p) = max(0, (a[t] + g[t]*r) - (b[t] + g[t]) * p)
          Realized sales this period = min(A_t(p), stock currently remaining).
          Reference update after the period's price p is posted:
              r_next = alpha * r + (1 - alpha) * p
          (r starts at r1 before period 0.)
  stdout: ONE JSON object:
            {"prices": [p_0, ..., p_{T-1}]}
          T real numbers, each in [0, p_max].

  A layout is VALID iff `prices` is a list of exactly T finite numbers (not
  bool), each within [0, p_max] (small float tolerance).  Invalid output,
  wrong length, an out-of-range price, a crash, a timeout, or non-JSON output
  -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  For each instance we forward-simulate
the candidate's full price schedule with the EXACT rules above (parent
process only) to get total revenue `obj`.  We also compute, ourselves,
`base` = the revenue of a flat, non-adaptive, curve-blind reference price
(the single-period monopolist price implied by the AVERAGE demand parameters
across the whole horizon, held constant, forward-simulated with the same real
dynamics).  We normalize:
    r = clamp( SCALE * obj / max(base, eps), 0, 1 )     with SCALE = 0.42
A perfectly-blind flat guess scores well under 0.1; myopic per-period revenue
maximization (the obvious "recipe") typically lands in the 0.3-0.5 range
because it depletes stock too fast on several instances; the pacing-aware
strong reference lands in the 0.6-0.9 range without ever saturating 1.0 --
headroom remains for a genuinely better shadow-price estimate.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  All references
(base price, simulation) are computed by THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

SCALE = 0.42
EPS = 1e-9


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    return nxt


# ----------------------------- instance family -----------------------------
def _build_schedule(seed, T, pattern, a_base, b_base, g_base):
    nx = _rng(seed)
    a, b, g = [], [], []
    for t in range(T):
        frac = t / max(1, T - 1)
        jitter = 1.0 + 0.06 * (nx() - 0.5)
        if pattern == "flat":
            mult = 1.0
        elif pattern == "late_surge":
            mult = 0.55 + 1.4 * (frac ** 3)
        elif pattern == "early_rush":
            mult = 1.9 * math.exp(-3.0 * frac) + 0.35
        elif pattern == "double_hump":
            mult = (0.5 + 0.9 * math.exp(-((frac - 0.2) ** 2) / 0.01)
                    + 0.9 * math.exp(-((frac - 0.8) ** 2) / 0.01))
        elif pattern == "b_rise":
            mult = 1.0
        else:
            mult = 1.0
        av = a_base * mult * jitter
        bv = b_base * (1.0 + (0.9 * frac if pattern == "b_rise" else 0.0)) * (1.0 + 0.04 * (nx() - 0.5))
        gv = g_base * (1.0 + 0.05 * (nx() - 0.5))
        a.append(av)
        b.append(bv)
        g.append(gv)
    return a, b, g


def _build_instances():
    # name, seed, T, C0, p_max, pattern, a_base, b_base, g_base, alpha, r1_frac
    specs = [
        ("balanced_flat",    201, 20, 140.0, 100.0, "flat",       60.0, 0.50, 0.15, 0.70, 0.50),
        ("late_surge",       202, 24, 160.0, 120.0, "late_surge", 50.0, 0.45, 0.10, 0.75, 0.40),
        ("early_rush",       203, 22, 150.0, 110.0, "early_rush", 55.0, 0.50, 0.12, 0.70, 0.55),
        ("elastic_late",     204, 24, 150.0, 100.0, "b_rise",     55.0, 0.35, 0.15, 0.70, 0.50),
        ("scarce_stock",     205, 24,  60.0, 130.0, "flat",       60.0, 0.45, 0.12, 0.70, 0.50),
        ("abundant_stock",   206, 24, 260.0, 100.0, "flat",       60.0, 0.50, 0.12, 0.70, 0.50),
        ("anchor_sensitive", 207, 22, 150.0, 110.0, "flat",       55.0, 0.40, 0.35, 0.85, 0.55),
        ("double_hump",      208, 26, 170.0, 120.0, "double_hump",50.0, 0.45, 0.12, 0.70, 0.45),
        ("short_horizon",    209, 10,  90.0, 100.0, "flat",       65.0, 0.50, 0.15, 0.65, 0.50),
        # larger held-out instance
        ("large_scale",      210, 30, 320.0, 140.0, "late_surge", 70.0, 0.40, 0.10, 0.75, 0.45),
    ]
    out = []
    for name, seed, T, C0, p_max, pattern, ab, bb, gb, alpha, r1f in specs:
        a, b, g = _build_schedule(seed, T, pattern, ab, bb, gb)
        out.append({
            "name": name, "T": T, "C0": C0, "p_max": p_max,
            "alpha": alpha, "r1": r1f * p_max,
            "a": a, "b": b, "g": g,
        })
    return out


# ----------------------------- simulation -----------------------------------
def _simulate(inst, prices):
    """Forward-simulate the REAL dynamics with a full price schedule. Returns total revenue."""
    T = inst["T"]; a = inst["a"]; b = inst["b"]; g = inst["g"]; alpha = inst["alpha"]
    inv = inst["C0"]; r = inst["r1"]
    rev = 0.0
    for t in range(T):
        p = prices[t]
        A = (a[t] + g[t] * r) - (b[t] + g[t]) * p
        if A < 0.0:
            A = 0.0
        sales = A if A < inv else inv
        rev += p * sales
        inv -= sales
        r = alpha * r + (1.0 - alpha) * p
    return rev


def _baseline_revenue(inst):
    """Flat, curve-blind reference: the monopolist price implied by the AVERAGE
    demand parameters over the whole horizon, held constant, forward-simulated
    with the real dynamics (this is the internal normalization anchor -- NOT
    one of the four solution tiers)."""
    T = inst["T"]
    avg_a = sum(inst["a"]) / T
    avg_b = sum(inst["b"]) / T
    p = avg_a / (2.0 * avg_b) if avg_b > 0 else 0.0
    p = max(0.0, min(inst["p_max"], p))
    return _simulate(inst, [p] * T)


# ----------------------------- validation ------------------------------------
def _score(inst, answer):
    """Validate + score a candidate answer. Return (ok, revenue)."""
    if not isinstance(answer, dict):
        return False, 0.0
    prices = answer.get("prices")
    T = inst["T"]
    if not isinstance(prices, list) or len(prices) != T:
        return False, 0.0
    p_max = inst["p_max"]
    clean = []
    for p in prices:
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            return False, 0.0
        pf = float(p)
        if not math.isfinite(pf):
            return False, 0.0
        if pf < -1e-6 or pf > p_max + 1e-6:
            return False, 0.0
        clean.append(max(0.0, min(p_max, pf)))
    rev = _simulate(inst, clean)
    if not math.isfinite(rev):
        return False, 0.0
    return True, rev


# ----------------------------- scoring driver ---------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        base = _baseline_revenue(inst)
        public = {
            "name": inst["name"], "T": inst["T"], "C0": inst["C0"], "p_max": inst["p_max"],
            "alpha": inst["alpha"], "r1": inst["r1"],
            "a": list(inst["a"]), "b": list(inst["b"]), "g": list(inst["g"]),
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = _score(inst, ans)
        except Exception:
            ok, obj = False, 0.0
        if not ok:
            vec.append(0.0)
            continue
        r = SCALE * obj / max(base, EPS)
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
