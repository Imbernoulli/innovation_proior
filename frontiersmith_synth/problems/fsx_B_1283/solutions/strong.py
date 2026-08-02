# TIER: strong
"""The insight: flexible senior (skill-substitutable) capacity is worth more
than raw headcount when arrivals are non-stationary.

Two ideas the greedy silo recipe misses entirely:

1. Robust sizing, not mean sizing. A recurring surge that hits only SOME of
   the held-out days gets averaged away by a plain mean. We size off a
   day-distribution PERCENTILE per hour instead, so the recurring surge
   shape (not diluted by the quieter days mixed into a mean) drives the
   seed roster.

2. Marginal-value exchange. A unit of senior capacity is a superset resource
   -- it can cover high-acuity demand (which junior cannot touch at all) and
   fall back to covering low-acuity overflow for free whenever it is not
   needed for high. So after seeding, we run a steepest-improvement local
   search over the roster that explicitly evaluates J<->S CONVERSION moves
   (not just raising/lowering headcount) against the checker's own exact
   cost function -- i.e. we price flexibility directly by its effect on
   overtime/agency spend, the classic shadow-price exchange argument for
   substitutable resources, instead of sizing each skill in its own silo.
"""
import sys, math


def read_input():
    data = sys.stdin.read().split()
    p = 0
    def nxt():
        nonlocal p
        v = data[p]; p += 1
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
        L = [0] * T; H = [0] * T
        for t in range(T):
            L[t] = int(nxt()); H[t] = int(nxt())
        days.append((L, H))
    return dict(T=T, n_starts=n_starts, starts=starts, max_per_slot=max_per_slot,
                cost_base_J=cost_base_J, cost_base_S=cost_base_S,
                ot_num_J=ot_num_J, ot_den_J=ot_den_J, ot_num_S=ot_num_S, ot_den_S=ot_den_S,
                cost_ot_J=cost_ot_J, cost_ot_S=cost_ot_S,
                cost_agency_J=cost_agency_J, cost_agency_S=cost_agency_S,
                K=K, days=days)


def day_cost(roster_J, roster_S, L, H, T, block_len, inst):
    cost_ot_J = inst["cost_ot_J"]; cost_ot_S = inst["cost_ot_S"]
    cost_agency_J = inst["cost_agency_J"]; cost_agency_S = inst["cost_agency_S"]
    ot_num_J = inst["ot_num_J"]; ot_den_J = inst["ot_den_J"]
    ot_num_S = inst["ot_num_S"]; ot_den_S = inst["ot_den_S"]
    total = 0
    for t in range(T):
        b = t // block_len
        baseJ = roster_J[b]; baseS = roster_S[b]
        otJ_cap = (ot_num_J * baseJ) // ot_den_J
        otS_cap = (ot_num_S * baseS) // ot_den_S
        h = H[t]; l = L[t]
        use_base_S_high = min(h, baseS)
        rem_h = h - use_base_S_high
        use_ot_S_high = min(rem_h, otS_cap)
        rem_h -= use_ot_S_high
        agency_S_high = rem_h
        leftover_baseS = baseS - use_base_S_high
        leftover_otS = otS_cap - use_ot_S_high
        rem_l = l
        use_baseJ_low = min(rem_l, baseJ); rem_l -= use_baseJ_low
        use_leftover_baseS_low = min(rem_l, leftover_baseS); rem_l -= use_leftover_baseS_low
        use_otJ_low = min(rem_l, otJ_cap); rem_l -= use_otJ_low
        use_leftover_otS_low = min(rem_l, leftover_otS); rem_l -= use_leftover_otS_low
        agency_J_low = rem_l
        total += (use_ot_S_high * cost_ot_S + agency_S_high * cost_agency_S
                  + use_otJ_low * cost_ot_J + use_leftover_otS_low * cost_ot_S
                  + agency_J_low * cost_agency_J)
    return total


def total_cost(roster_J, roster_S, inst, block_len):
    T = inst["T"]; K = inst["K"]; n_starts = inst["n_starts"]
    base_cost = sum(roster_J[b] * inst["cost_base_J"] + roster_S[b] * inst["cost_base_S"]
                     for b in range(n_starts))
    day_total = 0
    for (L, H) in inst["days"]:
        day_total += day_cost(roster_J, roster_S, L, H, T, block_len, inst)
    return base_cost + day_total / float(K)


def percentile(sorted_vals, q):
    n = len(sorted_vals)
    idx = min(n - 1, max(0, int(math.floor(q * (n - 1) + 1e-9))))
    return sorted_vals[idx]


def seed_roster(inst, block_len):
    T = inst["T"]; K = inst["K"]; n_starts = inst["n_starts"]
    max_per_slot = inst["max_per_slot"]
    robL = [0] * T; robH = [0] * T
    for t in range(T):
        ls = sorted(inst["days"][d][0][t] for d in range(K))
        hs = sorted(inst["days"][d][1][t] for d in range(K))
        # 80th percentile: robust to a surge that hits roughly half the days
        # without being swamped by a single one-off outlier, and (unlike a
        # mean) not diluted away by the quieter days mixed into the set.
        robL[t] = percentile(ls, 0.80)
        robH[t] = percentile(hs, 0.80)
    roster_J = []
    roster_S = []
    for b in range(n_starts):
        hrs = range(b * block_len, (b + 1) * block_len)
        needJ = max(robL[t] for t in hrs)
        needS = max(robH[t] for t in hrs)
        roster_J.append(min(max_per_slot, max(0, needJ)))
        roster_S.append(min(max_per_slot, max(0, needS)))
    return roster_J, roster_S


def local_search(roster_J, roster_S, inst, block_len, max_iters=250):
    n_starts = inst["n_starts"]
    max_per_slot = inst["max_per_slot"]
    cur_cost = total_cost(roster_J, roster_S, inst, block_len)
    for _ in range(max_iters):
        best_delta = 0.0
        best_move = None
        # unit +/- moves on each (block, skill)
        for b in range(n_starts):
            for skill in ("J", "S"):
                for d in (+1, -1):
                    rj = list(roster_J); rs = list(roster_S)
                    if skill == "J":
                        rj[b] += d
                        if rj[b] < 0 or rj[b] > max_per_slot:
                            continue
                    else:
                        rs[b] += d
                        if rs[b] < 0 or rs[b] > max_per_slot:
                            continue
                    c = total_cost(rj, rs, inst, block_len)
                    delta = c - cur_cost
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_move = (rj, rs, c)
        # explicit substitution/exchange moves: convert one junior slot to
        # senior or vice versa within a block (prices flexibility directly)
        for b in range(n_starts):
            if roster_J[b] > 0 and roster_S[b] < max_per_slot:
                rj = list(roster_J); rs = list(roster_S)
                rj[b] -= 1; rs[b] += 1
                c = total_cost(rj, rs, inst, block_len)
                delta = c - cur_cost
                if delta < best_delta - 1e-9:
                    best_delta = delta; best_move = (rj, rs, c)
            if roster_S[b] > 0 and roster_J[b] < max_per_slot:
                rj = list(roster_J); rs = list(roster_S)
                rs[b] -= 1; rj[b] += 1
                c = total_cost(rj, rs, inst, block_len)
                delta = c - cur_cost
                if delta < best_delta - 1e-9:
                    best_delta = delta; best_move = (rj, rs, c)
        if best_move is None:
            break
        roster_J, roster_S, cur_cost = best_move
    return roster_J, roster_S


def main():
    inst = read_input()
    block_len = inst["T"] // inst["n_starts"]
    roster_J, roster_S = seed_roster(inst, block_len)
    roster_J, roster_S = local_search(roster_J, roster_S, inst, block_len)
    for b in range(inst["n_starts"]):
        print(roster_J[b], roster_S[b])


if __name__ == "__main__":
    main()
