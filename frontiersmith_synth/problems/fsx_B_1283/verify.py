#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for fsx_B_1283
   Family: staffing-service-level (format C, minimize expected cost).

Instance (see gen.py / statement.md):
  T, n_starts, starts[], MAX_PER_SLOT,
  cost_base_J cost_base_S,
  ot_num_J ot_den_J ot_num_S ot_den_S,
  cost_ot_J cost_ot_S,
  cost_agency_J cost_agency_S,
  K, then K days of T lines "L H".

Artifact (participant stdout): n_starts lines "j_count s_count" (one per
shift-block, same order as `starts`).

The checker allocates the FIXED roster's capacity against each held-out
day's hourly (low, high) acuity demand via an exact cheapest-tier-first fill
(provably optimal for this nested-eligibility substitution structure: senior
capacity is eligible for both low and high, junior only for low, and all
capacity within a tier has identical marginal cost, so exhausting tiers in
increasing cost order is optimal by a standard exchange argument). This is a
deterministic subroutine -- the OPEN decision is how the roster itself is
built; that is what participants are scored on.
"""
import sys, math

MAX_TOKENS = 2000


def read_instance(path):
    toks = open(path).read().split()
    p = 0
    def nxt():
        nonlocal p
        v = toks[p]; p += 1
        return v
    T = int(nxt())
    n_starts = int(nxt())
    starts = [int(nxt()) for _ in range(n_starts)]
    max_per_slot = int(nxt())
    cost_base_J = int(nxt()); cost_base_S = int(nxt())
    ot_num_J = int(nxt()); ot_den_J = int(nxt())
    ot_num_S = int(nxt()); ot_den_S = int(nxt())
    cost_ot_J = int(nxt()); cost_ot_S = int(nxt())
    cost_agency_J = int(nxt()); cost_agency_S = int(nxt())
    K = int(nxt())
    days = []
    for _d in range(K):
        L = [0] * T
        H = [0] * T
        for t in range(T):
            L[t] = int(nxt())
            H[t] = int(nxt())
        days.append((L, H))
    inst = dict(T=T, n_starts=n_starts, starts=starts, max_per_slot=max_per_slot,
                cost_base_J=cost_base_J, cost_base_S=cost_base_S,
                ot_num_J=ot_num_J, ot_den_J=ot_den_J, ot_num_S=ot_num_S, ot_den_S=ot_den_S,
                cost_ot_J=cost_ot_J, cost_ot_S=cost_ot_S,
                cost_agency_J=cost_agency_J, cost_agency_S=cost_agency_S,
                K=K, days=days)
    return inst


def parse_roster(text, inst):
    """Return (roster_J, roster_S, reason). Each a list of length n_starts."""
    n_starts = inst["n_starts"]
    max_per_slot = inst["max_per_slot"]
    toks = text.split()
    if len(toks) == 0:
        return None, None, "empty output"
    if len(toks) > MAX_TOKENS:
        return None, None, "too many tokens"
    try:
        vals = [int(t) for t in toks]
    except ValueError:
        return None, None, "non-integer token (nan/inf/garbage)"
    if len(vals) != 2 * n_starts:
        return None, None, f"expected {2*n_starts} tokens, got {len(vals)}"
    roster_J = []
    roster_S = []
    for b in range(n_starts):
        j = vals[2 * b]; s = vals[2 * b + 1]
        if j < 0 or j > max_per_slot or s < 0 or s > max_per_slot:
            return None, None, f"block {b} count out of range [0,{max_per_slot}]"
        roster_J.append(j)
        roster_S.append(s)
    return roster_J, roster_S, "ok"


def block_of(hour, starts, block_len):
    return hour // block_len


def day_cost(roster_J, roster_S, L, H, T, block_len, inst):
    """Exact optimal allocation cost for one day, given a fixed roster.
    Cheapest-eligible-tier-first fill (see module docstring)."""
    cost_ot_J = inst["cost_ot_J"]; cost_ot_S = inst["cost_ot_S"]
    cost_agency_J = inst["cost_agency_J"]; cost_agency_S = inst["cost_agency_S"]
    ot_num_J = inst["ot_num_J"]; ot_den_J = inst["ot_den_J"]
    ot_num_S = inst["ot_num_S"]; ot_den_S = inst["ot_den_S"]
    total = 0
    for t in range(T):
        b = block_of(t, inst["starts"], block_len)
        baseJ = roster_J[b]
        baseS = roster_S[b]
        otJ_cap = (ot_num_J * baseJ) // ot_den_J
        otS_cap = (ot_num_S * baseS) // ot_den_S
        h = H[t]; l = L[t]
        # ---- high acuity: only senior-eligible tiers, cheapest first ----
        use_base_S_high = min(h, baseS)
        rem_h = h - use_base_S_high
        use_ot_S_high = min(rem_h, otS_cap)
        rem_h -= use_ot_S_high
        agency_S_high = rem_h  # unlimited, most expensive
        leftover_baseS = baseS - use_base_S_high
        leftover_otS = otS_cap - use_ot_S_high
        # ---- low acuity: junior- or senior-eligible tiers, cheapest first ----
        rem_l = l
        use_baseJ_low = min(rem_l, baseJ); rem_l -= use_baseJ_low
        use_leftover_baseS_low = min(rem_l, leftover_baseS); rem_l -= use_leftover_baseS_low
        use_otJ_low = min(rem_l, otJ_cap); rem_l -= use_otJ_low
        use_leftover_otS_low = min(rem_l, leftover_otS); rem_l -= use_leftover_otS_low
        agency_J_low = rem_l  # unlimited
        total += (use_ot_S_high * cost_ot_S + agency_S_high * cost_agency_S
                  + use_otJ_low * cost_ot_J + use_leftover_otS_low * cost_ot_S
                  + agency_J_low * cost_agency_J)
    return total


def total_cost(roster_J, roster_S, inst):
    T = inst["T"]; K = inst["K"]; n_starts = inst["n_starts"]
    block_len = T // n_starts
    base_cost = sum(roster_J[b] * inst["cost_base_J"] + roster_S[b] * inst["cost_base_S"]
                     for b in range(n_starts))
    day_total = 0
    for (L, H) in inst["days"]:
        day_total += day_cost(roster_J, roster_S, L, H, T, block_len, inst)
    return base_cost + day_total / float(K)


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return 0
    inf, outf = sys.argv[1], sys.argv[2]
    inst = read_instance(inf)

    text = open(outf).read()
    roster_J, roster_S, reason = parse_roster(text, inst)
    if roster_J is None:
        print(f"infeasible: {reason}")
        print("Ratio: 0.0")
        return 0

    F = total_cost(roster_J, roster_S, inst)
    if not math.isfinite(F):
        print("non-finite objective")
        print("Ratio: 0.0")
        return 0

    # checker's own trivial feasible construction: the empty roster (always
    # feasible: 0 is within [0, max_per_slot]) -- everything covered by agency.
    zero_J = [0] * inst["n_starts"]
    zero_S = [0] * inst["n_starts"]
    B = total_cost(zero_J, zero_S, inst)
    B = max(B, 1e-6)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print(f"F={F:.4f} baseline={B:.4f} roster_J={roster_J} roster_S={roster_S}")
    print("Ratio: %.6f" % (sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
