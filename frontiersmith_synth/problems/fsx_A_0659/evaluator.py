#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0659 -- "Ridgeline Grid: Aging-Aware Battery Arbitrage"
(family: aging-aware-arbitrage; format B, quality-metric).

THEME.  A grid-scale battery trades energy against a seeded day-ahead price
series.  Every step it may CHARGE (buy grid energy) or DISCHARGE (sell stored
energy) up to a power limit.  Two frictions compose into one objective:

  (1) state-of-charge round-trip loss: a charging efficiency eta_c and a
      discharging efficiency eta_d each shave energy off every trade, so a
      round trip only nets money once the sell price clears
      buy_price / (eta_c * eta_d) -- pure noise-chasing on narrow spreads is
      a guaranteed loser before aging is even considered.
  (2) cycling-depth aging: EVERY discharge step of size d, taken while the
      battery's current usable capacity is `cap`, fades that capacity by
      aging_coeff * capacity0 * (d/cap)^2 -- QUADRATIC in the per-step depth
      of discharge (dod).  Because the penalty is convex in depth, splitting
      the same total energy across several shallower discharge steps costs
      strictly less aging than dumping it in one deep step (compare k*(d/k)^2
      < d^2 for k>1).  The dollar cost of the fade (degradation_price per
      unit of capacity lost) is subtracted from profit immediately, AND the
      capacity loss itself shrinks all FUTURE headroom.

The seeded price series is mostly small noise (many tempting but marginal
local spreads) punctuated by a handful of large, wide swings.  The intended
strategy: read the instance's own eta/aging/degradation fields to compute
the true marginal breakeven spread, ignore anything narrower than that
(hold through the noise), and when a genuine wide swing arrives, LADDER the
trade across its multi-step plateau (shallow-cycle) instead of dumping the
full power limit in one step -- preserving capacity for the next swing.
A naive trader that fires on every attractive-looking local spread and always
trades at full power over-cycles the battery: efficiency tax plus quadratic
aging erodes its gains, and it is often capacity-starved by the time the
real swings arrive.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
    {"name": str, "T": int, "prices": [T floats],
     "capacity0": float, "power_max": float, "eta_c": float, "eta_d": float,
     "soc0": float, "aging_coeff": float, "degradation_price": float}
  stdout: ONE JSON object:
    {"actions": [x_0, ..., x_{T-1}]}   # x_t > 0 charge (buy), x_t < 0 discharge (sell)

  Feasibility (checked by the evaluator, in order, on the FULL instance):
    - `actions` is a list of exactly T finite numbers.
    - |x_t| <= power_max (+1e-6 tolerance) for every t.
    - Charging: soc + x_t*eta_c <= current capacity.
    - Discharging: -x_t <= current soc.
  Any violation, wrong shape/length, non-finite value, a crash, a timeout, or
  non-JSON output makes that instance score 0.0.

SCORING (deterministic; no wall-time).  The evaluator itself simulates the
candidate's action trace (soc, capacity, profit evolve exactly as described
above and in statement.md) to get `cand_profit`.  It also computes, itself,
a LOOSE unreachable upper bound `q_ideal` (unlimited capacity, no aging, one
power-limit's worth of energy captured on every consecutive price increase)
and anchors:
    r = clamp( 0.1 + 0.9 * cand_profit / q_ideal, 0, 1 )
Doing nothing scores exactly 0.1 (q_ideal>0, cand_profit=0). Because q_ideal
ignores capacity/soc/aging entirely it is never reachable, so even a very
good trader stays below 1.0 -- headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  All
references (ideal bound, simulation/validation) are computed by THIS parent
process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    def nxt_float(lo, hi):
        raw = nxt_int(0, 1_000_000)
        return lo + (hi - lo) * raw / 1_000_000.0

    return nxt_int, nxt_float


# ----------------------------- price-series family --------------------------
def _build_prices(seed, T, P0, noise_amp, n_swings, mag_lo, mag_hi, dur_lo, dur_hi):
    ni, nf = _rng(seed)
    prices = [P0 for _ in range(T)]
    for t in range(T):
        prices[t] += nf(-noise_amp, noise_amp)

    placed = []
    tries = 0
    made = 0
    while made < n_swings and tries < 400:
        tries += 1
        dur = ni(dur_lo, dur_hi)
        if T - dur - 1 <= 0:
            break
        start = ni(0, T - dur - 1)
        end = start + dur
        if any(not (end + 1 <= s or start >= e + 1) for (s, e) in placed):
            continue
        placed.append((start, end))
        made += 1
        sign = 1 if (made % 2 == 1) else -1
        mag = nf(mag_lo, mag_hi) * sign
        ramp = max(1, dur // 3)
        hold = max(0, dur - 2 * ramp)
        if ramp * 2 + hold != dur:
            ramp = max(1, dur // 2)
            hold = dur - 2 * ramp
            if hold < 0:
                hold = 0
        for i in range(dur):
            if i < ramp:
                frac = (i + 1) / ramp
            elif i < ramp + hold:
                frac = 1.0
            else:
                j = i - ramp - hold + 1
                frac = max(0.0, 1.0 - j / ramp)
            prices[start + i] += mag * frac

    prices = [round(max(1.0, p), 3) for p in prices]
    return prices


def _build_instances():
    """Deterministic instance family: (seed, T, cap0, pmax, eta_c, eta_d,
    aging_coeff, degr_price, noise_amp, n_swings, mag_lo, mag_hi, dur_lo, dur_hi)."""
    specs = [
        (901, 60, 60.0, 14.0, 0.93, 0.93, 0.030, 6.0, 3.0, 3, 18, 32, 4, 8, "std"),
        (902, 60, 60.0, 12.0, 0.90, 0.90, 0.045, 9.0, 4.0, 3, 20, 34, 4, 7, "harsh_aging"),
        (903, 65, 55.0, 16.0, 0.95, 0.95, 0.015, 4.0, 2.5, 3, 16, 28, 4, 8, "gentle"),
        (904, 70, 60.0, 13.0, 0.88, 0.90, 0.035, 8.0, 5.0, 2, 22, 36, 5, 9, "noisy_few_swings"),
        (905, 60, 50.0, 15.0, 0.92, 0.92, 0.025, 6.0, 1.5, 4, 15, 26, 4, 7, "low_noise_many_swings"),
        (906, 75, 65.0, 12.0, 0.90, 0.93, 0.040, 10.0, 4.5, 3, 24, 38, 5, 9, "harsh_tight_power"),
        (907, 60, 60.0, 20.0, 0.94, 0.94, 0.020, 5.0, 3.5, 3, 17, 30, 3, 6, "generous_power"),
        (908, 68, 58.0, 14.0, 0.89, 0.91, 0.038, 8.5, 4.0, 3, 19, 33, 4, 8, "mixed_harsh"),
        # held-out / larger, harder instances
        (911, 85, 70.0, 15.0, 0.91, 0.91, 0.028, 7.0, 4.0, 4, 20, 34, 4, 8, "held_out_long"),
        (912, 80, 62.0, 13.0, 0.88, 0.90, 0.042, 9.5, 4.5, 3, 23, 37, 4, 7, "held_out_harsh"),
    ]
    out = []
    for seed, T, cap0, pmax, eta_c, eta_d, ac, dp, namp, ns, mlo, mhi, dlo, dhi, label in specs:
        prices = _build_prices(seed, T, 50.0, namp, ns, mlo, mhi, dlo, dhi)
        out.append({
            "name": f"grid{seed}", "T": T, "prices": prices,
            "capacity0": cap0, "power_max": pmax, "eta_c": eta_c, "eta_d": eta_d,
            "soc0": round(cap0 / 2.0, 3), "aging_coeff": ac, "degradation_price": dp,
            "label": label,
        })
    return out


# ----------------------------- simulation / scoring -------------------------
EPS = 1e-6


def _simulate(inst, actions):
    """Replay actions against the instance's dynamics. Return (feasible, profit)."""
    T = inst["T"]
    prices = inst["prices"]
    cap0 = float(inst["capacity0"])
    pmax = float(inst["power_max"])
    eta_c = float(inst["eta_c"])
    eta_d = float(inst["eta_d"])
    ac = float(inst["aging_coeff"])
    dp = float(inst["degradation_price"])

    if not isinstance(actions, list) or len(actions) != T:
        return False, None
    xs = []
    for v in actions:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False, None
        fv = float(v)
        if fv != fv or fv in (float("inf"), float("-inf")):
            return False, None
        xs.append(fv)

    soc = float(inst["soc0"])
    cap = cap0
    profit = 0.0
    floor = 0.05 * cap0

    for t in range(T):
        x = xs[t]
        if x > EPS:
            if x > pmax + EPS:
                return False, None
            new_soc = soc + x * eta_c
            if new_soc > cap + EPS:
                return False, None
            profit -= x * prices[t]
            soc = min(new_soc, cap)
        elif x < -EPS:
            d = -x
            if d > pmax + EPS:
                return False, None
            if d > soc + EPS:
                return False, None
            denom = cap if cap > EPS else EPS
            dod = d / denom
            fade = ac * cap0 * dod * dod
            cap = cap - fade
            if cap < floor:
                cap = floor
            soc = soc - d
            revenue = d * eta_d * prices[t]
            aging_cost = dp * fade
            profit += revenue - aging_cost
        # hold: no state change
    return True, profit


def _ideal_bound(inst):
    """Loose, deliberately-unreachable upper bound: an integer-level DP that
    respects the REAL capacity0/soc0/power_max constraints and the real
    efficiency losses, but pays ZERO aging cost (capacity never fades).
    Since every feasible candidate trace pays >=0 aging cost on every
    discharge, this DP value strictly dominates any real (aging-penalized)
    profit -- a legitimate, deliberately-unreachable ceiling, tight enough
    to avoid being a vacuous scale."""
    T = inst["T"]
    prices = inst["prices"]
    cap0 = int(round(inst["capacity0"]))
    pmax = int(round(inst["power_max"]))
    soc0 = int(round(inst["soc0"]))
    eta_c = float(inst["eta_c"])
    eta_d = float(inst["eta_d"])
    NEG = float("-inf")
    dparr = [NEG] * (cap0 + 1)
    dparr[soc0] = 0.0
    for t in range(T):
        p = prices[t]
        ndp = [NEG] * (cap0 + 1)
        for s in range(cap0 + 1):
            v = dparr[s]
            if v == NEG:
                continue
            if v > ndp[s]:
                ndp[s] = v                      # hold
            for a in range(1, pmax + 1):        # charge a grid-side units
                ns = s + int(round(a * eta_c))
                if ns > cap0:
                    continue
                nv = v - a * p
                if nv > ndp[ns]:
                    ndp[ns] = nv
            for d in range(1, pmax + 1):        # discharge d battery-side units
                if d > s:
                    break
                ns = s - d
                nv = v + d * eta_d * p
                if nv > ndp[ns]:
                    ndp[ns] = nv
        dparr = ndp
    best = max((x for x in dparr if x != NEG), default=0.0)
    return max(best, 1e-6)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = {
            "name": inst["name"], "T": inst["T"], "prices": list(inst["prices"]),
            "capacity0": inst["capacity0"], "power_max": inst["power_max"],
            "eta_c": inst["eta_c"], "eta_d": inst["eta_d"], "soc0": inst["soc0"],
            "aging_coeff": inst["aging_coeff"], "degradation_price": inst["degradation_price"],
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        actions = ans.get("actions") if isinstance(ans, dict) else None
        try:
            feasible, profit = _simulate(inst, actions)
        except Exception:
            feasible, profit = False, None
        if not feasible or profit is None:
            vec.append(0.0)
            continue
        q_ideal = _ideal_bound(inst)
        r = 0.1 + 0.9 * profit / q_ideal
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
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
