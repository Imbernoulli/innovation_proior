#!/usr/bin/env python3
"""
gen.py <testId> -- DRAM refresh-schedule instance generator (family: dram-refresh-schedule).
Deterministic: all randomness seeded ONLY from testId.

Model
-----
T time slots [0..T-1], Bnum independent banks. Bank b has Rb rows; row i in bank b has a
retention bound rho[b][i] (slots): the gap between consecutive refreshes of that row (with
virtual boundaries at -1 and T) must never exceed rho[b][i]. A bank can perform at most one
refresh command per slot (all rows in a bank share one refresh port -> "bank-parallelism":
different banks never contend with each other, but rows *within* a bank do).

An access trace lists (slot, bank, weight) read/write requests. If a refresh command occupies
`bank` at exactly `slot`, that request stalls, costing `weight`. Objective: minimize total
stall weight while respecting every row's retention bound.

Generator invariant (keeps a trivial round-robin schedule always feasible, so the instance is
never accidentally unsatisfiable): every rho[b][i] >= Rb (that bank's own row count). Weak rows
sit at rho = c1*Rb (c1=2, the tightest legal value); the remaining rows are "strong" with
rho = c2*Rb for c2 drawn from a wide range (sometimes so large no in-window refresh is needed
at all) -- this is the planted retention-time-variation skew.

Trap cases (testId in TRAP_IDS) additionally load extra weight onto the exact slot-band a
naive *uniform worst-case* refresh policy is forced to occupy (slots with slot % P < Rb,
P = weak retention), while leaving the rest of the period lightly loaded -- punishing any
schedule that refreshes every row at the bank's single worst-case cadence.
"""
import random
import sys

TRAP_IDS = {3, 5, 7, 9, 10}


def gen(test_id: int):
    rnd = random.Random(900001 + 97 * test_id)

    Bnum = 2 + (test_id - 1) % 4  # 2..5
    trap = test_id in TRAP_IDS

    banks = []  # list of (Rb, rho_list, P, weak_count)
    for b in range(Bnum):
        base = 10 + 6 * test_id + 4 * b
        Rb = max(6, base + rnd.randint(-3, 4))
        # P (== weak retention) controls how much "cold" residue capacity (P - Rb)
        # exists beyond the row count Rb. On trap tests we deliberately keep this
        # tight (P close to Rb) so the cold capacity is well BELOW the row count:
        # no schedule, however smart, can push every row out of the hot band --
        # some rows are unavoidably forced to share it with the naive uniform
        # policy, which keeps genuine headroom above `strong`.
        p_ratio = 1.3 if trap else 2.2
        P = max(Rb + 2, round(p_ratio * Rb))
        frac_weak = 0.30 + 0.05 * rnd.random()
        weak_count = max(2, round(frac_weak * Rb))
        weak_count = min(weak_count, Rb - 1) if Rb > 1 else weak_count
        weak_idx = set(rnd.sample(range(Rb), weak_count))
        rho_list = []
        for i in range(Rb):
            if i in weak_idx:
                rho_list.append(P)
            else:
                c2 = rnd.randint(6, 22)
                rho_list.append(c2 * Rb)
        banks.append((Rb, rho_list, P, weak_count))

    # Horizon: enough periods for the skew to matter.
    maxP = max(P for (_, _, P, _) in banks)
    T = maxP * rnd.randint(7, 12)
    T = max(T, 120)
    T = min(T, 4200)

    # Access trace.
    p_base = 0.42 + 0.10 * rnd.random()
    reqs = []  # (slot, bank, weight)
    for b, (Rb, rho_list, P, weak_count) in enumerate(banks):
        for s in range(T):
            w = 0
            if rnd.random() < p_base:
                w = rnd.randint(1, 4)
            if trap and (s % P) < Rb:
                if rnd.random() < 0.88:
                    w = max(w, rnd.randint(12, 26))
            if w > 0:
                reqs.append((s, b, w))
    rnd.shuffle(reqs)

    out = []
    out.append(f"{T} {Bnum}")
    for (Rb, rho_list, P, weak_count) in banks:
        out.append(str(Rb))
        out.append(" ".join(str(x) for x in rho_list))
    out.append(str(len(reqs)))
    for (s, b, w) in reqs:
        out.append(f"{s} {b} {w}")
    sys.stdout.write("\n".join(out) + "\n")


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    gen(int(sys.argv[1]))


if __name__ == "__main__":
    main()
