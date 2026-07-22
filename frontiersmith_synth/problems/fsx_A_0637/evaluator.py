#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0637 -- "Tideline Commons: Multi-Zone Fishery Quota Setting"
(family: stock-surplus-harvest; format B, quality-metric).

THEME.  A fishing cooperative manages several offshore ZONES.  Each zone holds a
stock of fish that regenerates according to a logistic-style law: growth per step
is largest when the stock sits near HALF of the zone's carrying capacity K, and
shrinks toward zero as the stock nears either 0 or K.  Every step the cooperative
sets a harvest quota for every zone; the harvested fish are removed, and *then*
the remaining stock regenerates for the next step.  Over a long horizon (dozens
of steps) the cooperative wants to MAXIMIZE total fish landed.

THE CATCH (mechanism 1: nonlinear regeneration).  Because growth is maximized
near S = K/2, a policy that mines a zone down to a very low residual stock
harvests far less regrowth per step than one that holds the stock near K/2 and
skims only the surplus growth.

THE TRAP (mechanism 2: depletion-collapse tipping).  Every zone also has a
COLLAPSE THRESHOLD: an invisible floor (never revealed to the candidate).  If,
after harvesting, a zone's remaining stock ever drops below its own threshold,
that zone COLLAPSES for the rest of the episode -- its regeneration multiplier
drops to a small residual (2% of normal) forever after, destroying essentially
all future yield from that zone.  The candidate is never told the exact
threshold, but is guaranteed (globally, for every zone in every instance) that
no threshold exceeds 45% of that zone's own carrying capacity K.  A policy that
always keeps the post-harvest stock at or above K/2 is therefore PROVABLY safe
against collapse in every zone, regardless of the hidden threshold's exact value.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "T": T (int, steps), "n_zones": Z (int),
             "collapse_growth_mult": 0.02, "max_threshold_frac": 0.45,
             "zones": [{"K": float, "r": float, "S0": float}, ...]}   # length Z
          K = carrying capacity, r = intrinsic growth rate, S0 = starting stock.
  stdout: ONE JSON object:
            {"harvest": [[h_00, h_01, ..., h_0(Z-1)],
                         [h_10, h_11, ..., h_1(Z-1)],
                         ...                                   # T rows
                         [h_(T-1)0, ..., h_(T-1)(Z-1)]]}
          h_tz >= 0 is the harvest requested from zone z at step t.  A request
          exceeding the zone's currently-available stock is silently CLIPPED to
          what is available (you cannot harvest fish that are not there).

  A layout is VALID iff `harvest` is a list of exactly T rows, each a list of
  exactly Z finite, non-negative numbers (ints or floats; no bool/str/nan/inf).
  Any shape/type violation, a crash, a timeout, or non-JSON output makes that
  instance score 0.0.

DYNAMICS (per zone z, per step t; evaluator-side, uses the TRUE hidden threshold).
    applied      = min(h_tz, S)                       # clip to available stock
    S_after      = S - applied
    collapsed    |= (S_after < theta_z)                # sticky, once true stays true
    mult         = collapse_growth_mult if collapsed else 1.0
    growth       = mult * r_z * S_after * (1 - S_after / K_z)
    S            = clip(S_after + growth, 0, K_z)
  Total catch = sum of `applied` over all zones and all steps.

SCORING (deterministic; no wall-time).  Per instance the evaluator computes,
itself, two references:
    q_base  = 0                          # "harvest nothing" trivial reference
    q_ideal = sum_z [ T * r_z*K_z/4 + 0.3*K_z ]   # generous UNREACHABLE ideal:
              as if every zone sat exactly at its max-growth point K/2 for the
              WHOLE horizon (rK/4 per step) *and* also yielded a fictional bonus
              windfall of 0.3*K -- no real policy can capture both simultaneously
              (real trajectories pay a warm-up cost before first reaching K/2 and
              never get the fictional bonus), so q_ideal is a loose, strictly
              unreachable ceiling that keeps headroom above the reference "strong"
              solution.
    q_cand  = candidate's realized total catch
and normalizes with an affine anchor (do-nothing -> 0.1, unreachable ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(1e-9, q_ideal - q_base), 0, 1 )

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance (never theta_z,
never q_ideal).  All references and the true dynamics are computed by THIS
parent process, so a frame-walking / introspecting candidate learns nothing.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        x = (state >> 11) / float(1 << 53)          # in [0, 1)
        return lo + x * (hi - lo)

    return nxt


# ----------------------------- constants ------------------------------------
EPS_COLLAPSE = 0.02          # residual growth multiplier once a zone has collapsed
MAX_THETA_FRAC = 0.45        # disclosed global bound: theta_z <= 0.45 * K_z (never violated below)
NORMAL_THETA = (0.02, 0.08)  # hidden threshold range for "normal" (non-trap) zones, as frac of K
TRAP_THETA = (0.28, 0.42)    # hidden threshold range for "trap" zones, as frac of K (< MAX_THETA_FRAC)
IDEAL_BONUS_FRAC = 0.3       # unreachable-ideal fictional bonus, as frac of K per zone


# ----------------------------- instance family -------------------------------
def _build_zones(seed, Z, trap_zones):
    ni = _rng(seed)
    zones = []
    for zi in range(Z):
        K = ni(80.0, 400.0)
        r = ni(0.15, 0.55)
        S0 = ni(0.10, 0.90) * K
        if zi in trap_zones:
            theta = ni(TRAP_THETA[0], TRAP_THETA[1]) * K
        else:
            theta = ni(NORMAL_THETA[0], NORMAL_THETA[1]) * K
        zones.append({"K": K, "r": r, "S0": S0, "theta": theta})
    return zones


def _build_instances():
    """Deterministic instance family: (seed, Z, T, trap_zone_indices)."""
    specs = [
        (101, 4, 40, []),
        (102, 4, 40, [1]),
        (103, 5, 45, [0, 2]),
        (104, 3, 35, []),
        (105, 5, 50, [3]),
        (106, 4, 40, [0, 1, 2]),
        (107, 6, 55, [4]),
        (108, 3, 30, []),
        # harder / larger held-out instances
        (211, 6, 60, [1, 3, 5]),
        (212, 5, 50, [0]),
    ]
    out = []
    for seed, Z, T, trap in specs:
        zones = _build_zones(seed, Z, trap)
        out.append({"name": f"zone{seed}", "T": T, "n_zones": Z, "zones": zones})
    return out


# ----------------------------- references -------------------------------------
def _ideal_upper(inst):
    tot = 0.0
    for z in inst["zones"]:
        tot += inst["T"] * (z["r"] * z["K"] / 4.0) + IDEAL_BONUS_FRAC * z["K"]
    return tot


# ----------------------------- validation + simulation -------------------------
def _simulate(inst, answer):
    """Validate answer strictly and simulate TRUE dynamics. Return total catch or None."""
    if not isinstance(answer, dict):
        return None
    harvest = answer.get("harvest")
    T = inst["T"]
    Z = inst["n_zones"]
    zones = inst["zones"]
    if not isinstance(harvest, list) or len(harvest) != T:
        return None
    S = [z["S0"] for z in zones]
    collapsed = [False] * Z
    total = 0.0
    for t in range(T):
        row = harvest[t]
        if not isinstance(row, list) or len(row) != Z:
            return None
        for zi in range(Z):
            h = row[zi]
            if isinstance(h, bool) or not isinstance(h, (int, float)):
                return None
            h = float(h)
            if not (h == h) or h in (float("inf"), float("-inf")):
                return None
            if h < 0:
                return None
            K = zones[zi]["K"]; r = zones[zi]["r"]; theta = zones[zi]["theta"]
            applied = min(h, S[zi])
            S_after = S[zi] - applied
            if (not collapsed[zi]) and S_after < theta:
                collapsed[zi] = True
            mult = EPS_COLLAPSE if collapsed[zi] else 1.0
            growth = mult * r * S_after * (1.0 - S_after / K)
            S_next = S_after + growth
            if S_next < 0.0:
                S_next = 0.0
            if S_next > K:
                S_next = K
            S[zi] = S_next
            total += applied
    return total


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        q_base = 0.0                         # trivial "harvest nothing" reference
        q_ideal = _ideal_upper(inst)
        denom = q_ideal - q_base
        if denom < 1e-9:
            denom = 1e-9

        public = {
            "name": inst["name"], "T": inst["T"], "n_zones": inst["n_zones"],
            "collapse_growth_mult": EPS_COLLAPSE, "max_threshold_frac": MAX_THETA_FRAC,
            "zones": [{"K": z["K"], "r": z["r"], "S0": z["S0"]} for z in inst["zones"]],
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _simulate(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_cand - q_base) / denom
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
