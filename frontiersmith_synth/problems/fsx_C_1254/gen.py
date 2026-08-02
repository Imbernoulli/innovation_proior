#!/usr/bin/env python3
"""
gen.py <testId> -- power-gating-partition instance generator
(family: power-gating-partition, theme: "Turning off the parts nobody is using").

Deterministic: all randomness seeded ONLY from testId.

Model
-----
N hardware blocks. K independently-observed activity traces, each over a shared time window
of T timesteps: trace[k][i][t] in {0,1} says whether block i is doing useful work at timestep
t in trace k. You must partition the N blocks into at most D POWER DOMAINS and pick one
non-negative integer GATING THRESHOLD theta_d per domain d. A domain must be powered ON at
any timestep when any block assigned to it is active (that is non-negotiable). Between active
periods a domain has idle runs; an idle run of length L is powered OFF (paying a one-time
wakeup energy W right before the next active period, or nothing if the run runs off the end
of the trace) iff L >= theta_d, otherwise the domain is kept ON through it (paying leakage L*
per idle step instead). One partition + one theta per domain is chosen ONCE and replayed
against all K traces.

Trace construction
-------------------
Each block is given a persistent per-instance "profile" (drawn from a small library of
run-length regimes) that is reused, freshly re-sampled, across all K traces:
  - "warmup"  : idle runs ~U[10,22], active runs ~U[1,2]  (long idle -> gating amortizes)
  - "burst"   : idle runs ~U[1,2],   active runs ~U[1,2]  (short idle -> gating a trap)
  - "mixed"   : idle runs ~U[5,9],   active runs ~U[1,2]  (straddles the breakeven length)
On several tests a CORRELATED CLUSTER of blocks shares one common "warmup" base pattern per
trace (each member = the base pattern with ~2% of bits flipped): grouping the cluster into ONE
domain keeps its idle windows long (little coverage lost), while scattering it round-robin
across many domains (as a granularity-blind partition would) destroys that structure because
any domain containing even one independent "burst"/"mixed" block loses its long idle runs.

Determinism: `random.Random(seed)` seeded from testId only; no wall clock, no OS entropy.
"""
import random
import sys

REGIME_PARAMS = {
    "warmup": (10, 22, 1, 2),
    "burst": (1, 2, 1, 2),
    "mixed": (5, 9, 1, 2),
}

# testId -> (N, D, T, K, L, W, cluster_size, cluster_regime, noncluster_regime_pattern)
# noncluster_regime_pattern: a list of regime names, cycled over the non-cluster blocks.
# L=10, W=35 everywhere except test 10 (W=75) keeps the singleton breakeven
# theta*(size=1) = floor(W/L)+1 = 4 (resp. 8) comfortably between "burst" idle runs
# (max 2, never worth gating) and "warmup" idle runs (min 10, always worth gating).
# EVERY test forces real grouping decisions (D << N, with a correlated "warmup" cluster
# whose members share one base pattern) on top of a genuine ~50/50 mix of gate-able
# "warmup" blocks and trap "burst"/"mixed" blocks, so the naive fine-grained-round-robin
# recipe pays BOTH mistakes it is prone to: it shatters the cluster's shared idle windows
# across many domains, and it gates every short burst idle run no matter how brief.
TESTS = {
    1: (8, 3, 40, 2, 10, 35, 4, "warmup", ["warmup", "burst"]),
    2: (10, 3, 50, 4, 10, 35, 5, "warmup", ["warmup", "burst"]),
    3: (12, 4, 60, 6, 10, 35, 6, "warmup", ["mixed", "burst"]),
    4: (12, 4, 60, 10, 10, 35, 6, "warmup", ["warmup", "burst"]),
    5: (14, 4, 70, 10, 10, 35, 7, "warmup", ["warmup", "burst"]),
    6: (16, 5, 80, 10, 10, 35, 8, "warmup", ["warmup", "burst"]),
    7: (20, 6, 90, 10, 10, 35, 10, "warmup", ["warmup", "burst"]),
    8: (26, 7, 100, 10, 10, 35, 13, "warmup", ["warmup", "burst"]),
    9: (34, 9, 130, 10, 10, 35, 17, "warmup", ["warmup", "burst"]),
    10: (40, 10, 160, 10, 10, 75, 20, "warmup", ["warmup", "burst"]),
}


def gen_alternating(rng, T, idle_lo, idle_hi, act_lo, act_hi):
    """One block's activity string of length T: alternating idle/active runs."""
    chars = []
    t = 0
    state = rng.randint(0, 1)
    while t < T:
        if state == 0:
            length = rng.randint(idle_lo, idle_hi)
        else:
            length = rng.randint(act_lo, act_hi)
        length = min(length, T - t)
        chars.append(chr(48 + state) * length)
        t += length
        state = 1 - state
    return "".join(chars)


def jitter(rng, base, p):
    """Flip each bit of `base` independently with probability p (small correlated noise)."""
    out = list(base)
    for idx in range(len(out)):
        if rng.random() < p:
            out[idx] = "1" if out[idx] == "0" else "0"
    return "".join(out)


def gen(test_id: int):
    N, D, T, K, L, W, csize, cregime, nc_pattern = TESTS[test_id]
    rng = random.Random(1000003 + 97 * test_id)

    cluster = set(range(csize))
    nc_blocks = list(range(csize, N))
    nc_regime_of = {}
    for j, blk in enumerate(nc_blocks):
        nc_regime_of[blk] = nc_pattern[j % len(nc_pattern)]

    trace_rows = []  # list of K lists, each of N strings length T
    for _k in range(K):
        rows = [None] * N
        if cluster:
            base = gen_alternating(rng, T, *REGIME_PARAMS[cregime])
            for i in cluster:
                rows[i] = jitter(rng, base, 0.02)
        for i in nc_blocks:
            reg = nc_regime_of[i]
            rows[i] = gen_alternating(rng, T, *REGIME_PARAMS[reg])
        trace_rows.append(rows)

    out = [f"{N} {D} {K} {T}", f"{L} {W}"]
    for rows in trace_rows:
        out.extend(rows)
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    test_id = int(sys.argv[1])
    sys.stdout.write(gen(test_id))


if __name__ == "__main__":
    main()
