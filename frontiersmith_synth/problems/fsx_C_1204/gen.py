#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training instance to stdout.

Family: supply-lead-time-forecast (bullwhip / order-inflation feedback / capacity
queueing / held-out demand shock).  testId 1..10 is the difficulty ladder: later
ids give more training rows AND a (checker-side, not visible here) more severe
held-out shock.

Output:
    line 1:  n t Cap L0
    n lines: D O L      (period demand estimate, orders placed, observed lead time)

Only Cap (declared facility capacity) and L0 (declared baseline/free-flow lead
time) are exposed as known constants.  The congestion/feedback gain that governs
how lead time compounds as orders approach capacity is NEVER printed -- it must
be estimated from the data.
"""
import sys, random, math

SEED_BASE = 20260726
SEED_MULT = 104729


def params(t):
    """Hidden per-instance parameters. Cap, L0 get exposed in the header; g, D0,
    sigma, h never appear in gen.py's stdout -- only their FOOTPRINT in the data."""
    rng = random.Random(SEED_BASE + t * SEED_MULT)
    Cap = rng.uniform(90.0, 170.0)          # declared capacity (orders/period)
    L0 = rng.uniform(1.8, 3.4)              # declared free-flow lead time
    g = rng.uniform(1.5, 3.5)               # HIDDEN congestion/feedback gain
    util0 = rng.uniform(0.20, 0.30)         # baseline utilization (sub-critical)
    D0 = util0 * Cap                        # HIDDEN baseline demand level
    sigma = D0 * rng.uniform(0.05, 0.08)    # HIDDEN demand noise scale
    h = rng.uniform(0.008, 0.018)           # HIDDEN order-inflation sensitivity
    return Cap, L0, g, D0, sigma, h


def n_train(t):
    return 150 + 10 * (t - 1)


def gen_train(t):
    """Simulate the STABLE observation window: demand fluctuates mildly around a
    sub-critical baseline; buyers inflate orders when last period's observed lead
    time ran above baseline (order-inflation-feedback); lead time follows a
    capacity-queue law L = L0 + g*O/(Cap-O) (bullwhip shows up as Var(O)>Var(D))."""
    Cap, L0, g, D0, sigma, h = params(t)
    rng = random.Random(1000 + t * 7919)
    N = n_train(t)
    L_prev = L0
    rows = []
    for _ in range(N):
        Dt = max(1e-3, D0 + rng.gauss(0.0, sigma))
        Ot = max(1e-3, Dt * (1.0 + h * (L_prev - L0)))
        Ot = min(Ot, 0.97 * Cap)
        denom = Cap - Ot
        Lt_true = L0 + g * Ot / denom
        Lt = Lt_true * (1.0 + rng.gauss(0.0, 0.008))
        L_prev = Lt
        rows.append((Dt, Ot, Lt))
    return Cap, L0, rows


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    t = int(sys.argv[1])
    Cap, L0, rows = gen_train(t)
    out = [f"{len(rows)} {t} {Cap:.6f} {L0:.6f}"]
    for Dt, Ot, Lt in rows:
        out.append(f"{Dt:.6f} {Ot:.6f} {Lt:.6f}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
