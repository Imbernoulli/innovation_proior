#!/usr/bin/env python3
"""
gen.py <testId> -- tsv-thermal-placement instance generator
(family: tsv-thermal-placement, theme: "Stacking chips without cooking the middle one").

Deterministic: all randomness seeded ONLY from testId (random.Random(seed)).

Model
-----
M dies are stacked directly on top of a heat sink. Die 1 sits immediately on the sink; die M
is the farthest die (on top of the stack). Each die is discretized into N columns (shared XY
grid across the whole stack). Die m dissipates power p[m][c] at column c.

A thermal via (TSV) may be drilled straight down through column c, replacing the per-layer
thermal resistance in that column, for EVERY layer, from R0 (no via) to Rv < R0 (via). Because
the via runs through the WHOLE stack, it is an all-or-nothing decision per column: x[c] in
{0,1}. Each via costs a[c] units of the shared area budget A (0 <= sum of costs of chosen
columns <= A) -- placing vias densely everywhere is never affordable.

Heat generated on die m must cross every layer boundary between die m and the sink, so it
crosses the resistance of column c a total of m times (once per boundary below it). Heat from
different dies sharing the same column stacks additively (a via, or the lack of one, is a
SHARED bottleneck for every die above it in that column). The peak temperature realized at
column c is therefore:
    T[c] = R(c) * W[c],   W[c] = sum_{m=1}^{M} m * p[m][c],   R(c) = Rv if x[c]=1 else R0.
Minimize max_c T[c]. W[c] is the depth-weighted, STACKED hotspot profile -- the quantity that
actually matters -- and it is generally NOT maximized at the same column as any single die's
own power map.

Trap construction (>= 3 of the 10 cases)
-----------------------------------------
Several tests plant a "spike vs. stack" pair of columns: one column carries a single huge
spike on the TOPMOST die alone (large per-die value, looks like the obvious hotspot if you
only look at one die -- and is deliberately made EXPENSIVE to via), while a different column
carries moderate, well-ALIGNED power across several of the lower/middle dies (individually
unremarkable, cumulatively -- once depth-weighted -- the true peak of W[c] -- and deliberately
made CHEAP to via). A per-die-only or value-density-only view spends the scarce budget on the
flashy single-die spike (or on whatever has the best raw value/cost ratio) and never affords
the true coupled hotspot; reading W[c] directly finds it immediately.

Determinism: random.Random(seed) seeded from testId only; no wall clock, no OS entropy.
"""
import random
import sys

# testId -> (M, N, mode, extra)
#   mode "aligned"     : easy warm-up, hotspots line up across dies, uniform cheap costs.
#   mode "trap"        : hand-placed spike-vs-stack trap column pair (small/medium scale).
#   mode "trap_random" : same trap pattern injected into a larger randomized background,
#                        with adversarial (value-density-inverted) costs.
TESTS = {
    1: (2, 5, "aligned", {}),
    2: (3, 6, "aligned", {}),
    3: (3, 8, "mild_mis", {}),
    4: (4, 10, "trap", {}),
    5: (4, 12, "trap", {"adv_cost": True}),
    6: (5, 15, "trap_random", {"n_traps": 1}),
    7: (6, 20, "trap_random", {"n_traps": 2}),
    8: (6, 26, "trap_random", {"n_traps": 2, "adv_cost": True}),
    9: (7, 34, "trap_random", {"n_traps": 3, "adv_cost": True}),
    10: (8, 44, "trap_random", {"n_traps": 3, "adv_cost": True}),
}


def emit(M, N, A, R0, Rv, a, P):
    out = [f"{M} {N} {A}", f"{R0} {Rv}", " ".join(map(str, a))]
    for m in range(M):
        out.append(" ".join(map(str, P[m])))
    return "\n".join(out) + "\n"


def gen_aligned(rng, M, N):
    """All dies share the same hot column(s) (per-die view and stacked view agree); the area
    budget is sized EXACTLY off those hot columns' own costs so both a per-die and a stacked
    reader can actually afford to cover them."""
    R0, Rv = 100, 25
    P = [[rng.randint(0, 5) for _ in range(N)] for _ in range(M)]
    a = [rng.randint(4, 8) for _ in range(N)]
    hot_cols = rng.sample(range(N), k=min(2, N))
    for c in hot_cols:
        for m in range(M):
            P[m][c] = rng.randint(60, 90)
    A = sum(a[c] for c in hot_cols)  # exactly enough to cover every hot column
    return M, N, A, R0, Rv, a, P


def gen_mild_mis(rng, M, N):
    """Each die gets its OWN exclusive hot column (no two dies share one); the budget only
    affords a minority of them, forcing a choice among mismatched single-die spikes."""
    R0, Rv = 100, 25
    P = [[rng.randint(0, 4) for _ in range(N)] for _ in range(M)]
    a = [rng.randint(3, 6) for _ in range(N)]
    cols = rng.sample(range(N), k=min(M, N))
    for i, m in enumerate(range(M)):
        c = cols[i]
        P[m][c] = rng.randint(40, 70)
    # Budget: exactly enough for the SINGLE truly-highest-W hot column (computed directly here
    # from the depth-weighted formula, same as the checker) -- never enough for all of them.
    W = [sum((m + 1) * P[m][c] for m in range(M)) for c in range(N)]
    best_col = max(cols, key=lambda c: W[c])
    A = a[best_col]
    return M, N, A, R0, Rv, a, P


def plant_trap(rng, M, N, P, a, used_cols, spike_val=(30, 50), stack_val=(60, 90),
               spike_cost=(15, 20), stack_cost=(5, 8), decoy_val=(10, 25), decoy_cost=(2, 4)):
    """Plant one spike/stack/decoy trap TRIPLE into P (in place) and return
    (stack_col, stack_cost_val, decoy_cost_val).

    - spike: only the TOPMOST die is hot here (large single-die reading) -- expensive to via,
      and NOT the true hotspot once depth-weighting is accounted for.
    - stack: every die EXCEPT the topmost carries a moderate, well-aligned reading here -- the
      true highest depth-weighted column (W[stack] is guaranteed, by construction, to exceed
      W[spike] for any M >= 4) -- and it is cheap to via, with a near-zero reading on the
      topmost die (so a per-die-only view sees essentially nothing there).
    - decoy: only the topmost die is hot here too, moderately (well below the spike, but with
      a MUCH better raw value-per-cost ratio than the stack column can ever show under a
      per-die-only value): a value-density ranking spends the leftover budget here instead of
      on the stack column, even though the decoy's true depth-weighted W is minor.
    """
    remaining = [c for c in range(N) if c not in used_cols]
    spike_col, stack_col, decoy_col = rng.sample(remaining, 3)
    used_cols.update((spike_col, stack_col, decoy_col))
    P[M - 1][spike_col] = rng.randint(*spike_val)
    for m in range(M - 1):
        P[m][spike_col] = rng.randint(0, 2)
    for m in range(M - 1):
        P[m][stack_col] = rng.randint(*stack_val)
    P[M - 1][stack_col] = 0
    P[M - 1][decoy_col] = rng.randint(*decoy_val)
    for m in range(M - 1):
        P[m][decoy_col] = rng.randint(0, 2)
    sc = rng.randint(*spike_cost)
    kc = rng.randint(*stack_cost)
    dc = rng.randint(*decoy_cost)
    a[spike_col] = sc
    a[stack_col] = kc
    a[decoy_col] = dc
    return stack_col, kc, dc


def gen_trap(rng, M, N, adv_cost=False):
    R0, Rv = 100, 24
    P = [[rng.randint(0, 2) for _ in range(N)] for _ in range(M)]
    a = [rng.randint(4, 8) for _ in range(N)]
    used = set()
    stack_col, kc, dc = plant_trap(rng, M, N, P, a, used)
    if adv_cost:
        # make every OTHER background column artificially cheap-looking too, so a pure
        # value-density ranking has plenty of low-value-but-affordable noise to fall into.
        for c in range(N):
            if c not in used:
                a[c] = rng.randint(1, 2)
    # Budget: exactly enough to afford the true (cheap) stack column alone -- but the decoy's
    # value-density is strictly better than the stack column's (which reads ~0 on the topmost
    # die), so a value-density ranking spends on the decoy first and is left short for the
    # stack column (dc < kc is enforced by the cost ranges above).
    A = kc
    return M, N, A, R0, Rv, a, P


def gen_trap_random(rng, M, N, n_traps, adv_cost=False):
    R0, Rv = 100, 22
    P = [[rng.randint(0, 2) for _ in range(N)] for _ in range(M)]
    a = [rng.randint(4, 9) for _ in range(N)]
    used = set()
    kcs = []
    for _ in range(n_traps):
        stack_col, kc, dc = plant_trap(rng, M, N, P, a, used)
        kcs.append(kc)
    if adv_cost:
        for c in range(N):
            if c not in used:
                a[c] = rng.randint(1, 2)
    # Budget: exactly enough to afford every true stack column across all planted traps, and
    # nothing more -- any budget a value-density ranking diverts to a decoy or a spike directly
    # steals from what strong spends on the real hotspots.
    A = sum(kcs)
    return M, N, A, R0, Rv, a, P


def gen(test_id: int):
    M, N, mode, extra = TESTS[test_id]
    rng = random.Random(2_000_003 + 131 * test_id)
    if mode == "aligned":
        args = gen_aligned(rng, M, N)
    elif mode == "mild_mis":
        args = gen_mild_mis(rng, M, N)
    elif mode == "trap":
        args = gen_trap(rng, M, N, adv_cost=extra.get("adv_cost", False))
    elif mode == "trap_random":
        args = gen_trap_random(rng, M, N, extra["n_traps"], adv_cost=extra.get("adv_cost", False))
    else:
        raise ValueError(mode)
    return emit(*args)


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    sys.stdout.write(gen(test_id))


if __name__ == "__main__":
    main()
