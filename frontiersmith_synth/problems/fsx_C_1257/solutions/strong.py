# TIER: strong
"""
The insight: don't spend the WHOLE budget on one blind rate -- carve out a small, cheap,
PROVABLY-sufficient reserve for the precursor channel and a burst-response window, and only
spend the rest on routine coverage.

  - Pigeonhole guarantee: every event cluster is preceded by a `lead`-slot-long window during
    which the precursor flag reads 1. A periodic precursor check with period Pc <= lead is
    therefore GUARANTEED to land inside every such window (consecutive checks are <= lead
    slots apart, so no window of length `lead` can be skipped) -- not a heuristic, a covering
    argument. We use Pc = lead, the cheapest period that still keeps the guarantee.
  - On a hit, escalate to P1=1 (sample every slot) for W = lead + Lmax slots: long enough that
    the escalated window still covers the LATEST possible cluster end (onset can arrive up to
    `lead` slots after the precursor first fires, and the cluster itself is <= Lmax slots), so
    every clustered event is caught, not just the ones near the front of the burst.
  - Worst-case-bound the escalation + precursor cost (Cmax clusters, each capped at W*e_full,
    plus ceil(T/Pc) cheap checks) and spend ONLY the remainder of the budget on a base rate --
    the same fixed-rate recipe `greedy` uses, just sized to what's left after the reserve.

If the instance has no clusters at all (Cmax=0, the "warm-up" cases), there is nothing to
escalate on: reserving budget for the precursor channel would only waste energy, so we fall
back to exactly `greedy`'s full-budget fixed rate -- matching the fact that a fixed rate IS the
right answer under pure uniform arrival.
"""
import sys


def main():
    data = sys.stdin.read().split()
    T = int(data[0])
    E = int(data[1])
    e_full = int(data[2])
    e_cheap = int(data[3])
    lead = int(data[4])
    Lmax = int(data[5])
    Cmax = int(data[6])

    if Cmax <= 0 or lead <= 0:
        n_full = max(1, E // e_full)
        P0 = max(1, T // n_full)
        print(f"{P0} 0 {P0} 0")
        return

    Pc = max(1, lead)
    P1 = 1
    W = lead + Lmax

    worst_escal_cost = Cmax * W * e_full
    worst_prec_cost = -(-T // Pc) * e_cheap  # ceil(T/Pc)*e_cheap
    remaining = E - worst_escal_cost - worst_prec_cost

    n_base = max(1, remaining // e_full) if remaining > 0 else 1
    P0 = max(1, T // n_base)

    print(f"{P0} {Pc} {P1} {W}")


if __name__ == "__main__":
    main()
