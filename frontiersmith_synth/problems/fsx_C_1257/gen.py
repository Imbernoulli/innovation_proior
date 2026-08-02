#!/usr/bin/env python3
"""
gen.py <testId> -- sensor-sampling-schedule instance generator (family: sensor-sampling-schedule).
Deterministic: all randomness seeded ONLY from testId.

Model
-----
A battery-powered sensor watches K independent deployments ("streams"), each spanning the same
horizon of T discrete slots and starting with the same fresh energy budget E. Two channels:
  - a FULL sample (cost e_full) checks a slot for an event: hits count only if a full sample
    lands EXACTLY on an event slot.
  - a CHEAP precursor check (cost e_cheap << e_full) reads a binary "precursor" flag; it never
    detects an event by itself, but every event CLUSTER (a burst of consecutive event slots) is
    preceded by a `lead`-slot-long window in which the precursor flag is 1 (a real early-warning
    signal). Isolated ("background") events carry NO precursor warning at all.

The output artifact is ONE fixed policy (P0, Pc, P1, W) applied identically, independently, to
every stream:
  - sample fully every P0 slots by default;
  - every Pc slots (while not escalated) spend e_cheap on a precursor check;
  - if that check reads 1, escalate: for the next W slots sample fully every P1 slots instead.
Energy is a hard per-stream cap -- a channel access is simply skipped once the budget for that
stream is exhausted (no violation, just silent starvation), so no output can ever be "energy
infeasible"; the only feasibility gate is the artifact's own schema (four in-range integers).

Trap construction
------------------
testId in {1,2,3}: purely "warm-up" streams -- only scattered, precursor-less background events
(no clusters at all, Cmax=0). Any fixed rate that spends the whole budget performs about as well
as any other policy here (nothing to escalate on).

testId in {4..10} (7 of the 10 -- the seed's planted trap): each stream additionally packs up to
Cmax event clusters, each of length <= Lmax, each preceded by a full `lead`-slot precursor
window. Clusters are spaced with a wide buffer (>= 3*(lead+Lmax)) so no two clusters' precursor
windows / escalation tails can ever collide. A fixed-rate policy's grid essentially never aligns
with a whole cluster (it visits O(cluster_len/period) of a cluster's slots); a policy that reacts
to the cheap precursor can guarantee catching a whole cluster (pigeonhole: any period
Pc <= lead is guaranteed to fire at least once inside every lead-long precursor window),
recovering nearly all clustered events for a small, cheap, reserved slice of the same budget.
"""
import random
import sys

TRAP_IDS = {4, 5, 6, 7, 8, 9, 10}


def make_stream(rnd, T, lead, Lmax, Cmax, bg_gap_lo, bg_gap_hi):
    """Return (sorted event slots, sorted precursor-active slots) for one stream."""
    events = set()
    precursors = set()

    # ---- cluster bursts (only when Cmax > 0) ----
    n_clusters = rnd.randint(1, Cmax) if Cmax > 0 else 0
    reserved = []  # (lo, hi) blocked spans (precursor-window..escalation-tail) incl margin
    margin = 3 * (lead + Lmax)
    attempts = 0
    placed = 0
    while placed < n_clusters and attempts < 200:
        attempts += 1
        onset = rnd.randint(lead, T - Lmax - 1)
        length = rnd.randint(max(1, Lmax // 2), Lmax)
        lo, hi = onset - lead - margin, onset + length + margin
        if any(not (hi < r_lo or lo > r_hi) for (r_lo, r_hi) in reserved):
            continue
        reserved.append((lo, hi))
        for s in range(onset, onset + length):
            events.add(s)
        for s in range(onset - lead, onset):
            precursors.add(s)
        placed += 1

    # ---- scattered, precursor-less background events (jittered near-periodic) ----
    gap = rnd.randint(bg_gap_lo, bg_gap_hi)
    t = rnd.randint(0, gap)
    while t < T:
        jit = t + rnd.randint(-gap // 3, gap // 3)
        if 0 <= jit < T and not any(lo <= jit <= hi for (lo, hi) in reserved):
            events.add(jit)
        t += gap

    return sorted(events), sorted(precursors)


def gen(test_id: int):
    rnd = random.Random(730001 + 149 * test_id)
    trap = test_id in TRAP_IDS

    T = 300 + 140 * test_id  # 440 .. 1700, small -> large ladder
    e_full = rnd.randint(10, 18)
    e_cheap = rnd.randint(1, 2)
    lead = rnd.randint(8, 14)
    Lmax = rnd.randint(6, 12)
    Cmax = rnd.randint(1, 3) if trap else 0
    K = rnd.randint(6, 10)

    n_afford = max(4, T // rnd.randint(20, 35))
    base_budget = n_afford * e_full
    worst_escal_cost = Cmax * (lead + Lmax) * e_full
    worst_prec_cost = -(-T // max(1, lead)) * e_cheap  # ceil(T/lead)*e_cheap
    slack = (base_budget + worst_escal_cost + worst_prec_cost) // 10
    E = base_budget + worst_escal_cost + worst_prec_cost + slack
    E = max(E, e_full * 4)

    if trap:
        bg_lo, bg_hi = rnd.randint(35, 50), rnd.randint(55, 80)
    else:
        bg_lo, bg_hi = rnd.randint(18, 28), rnd.randint(30, 45)

    streams = [make_stream(rnd, T, lead, Lmax, Cmax, bg_lo, bg_hi) for _ in range(K)]

    out = [f"{T} {E} {e_full} {e_cheap} {lead} {Lmax} {Cmax} {K}"]
    for (evs, prs) in streams:
        out.append(str(len(evs)))
        out.append(" ".join(map(str, evs)) if evs else "")
        out.append(str(len(prs)))
        out.append(" ".join(map(str, prs)) if prs else "")
    sys.stdout.write("\n".join(out) + "\n")


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    gen(int(sys.argv[1]))


if __name__ == "__main__":
    main()
