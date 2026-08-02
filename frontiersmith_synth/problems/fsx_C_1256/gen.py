#!/usr/bin/env python3
"""
gen.py <testId> -- prints ONE checkpoint-interval-choose instance to stdout.

The failure process is specified as a sequence of ACTIVE-COMPUTE-TIME gaps g_1..g_m:
the machine fails after g_1 units of active computation (whether that work is ultimately
kept or later thrown away by an even earlier failure does not matter -- only genuine
"the CPU was busy" time counts, checkpoint/restart pauses do not), then after another g_2
units of active computation counted fresh from the moment failure 1 fired, and so on.
This is the standard MTBF-in-compute-time formulation, and it keeps each failure's
consequences fully controlled by the checkpoint schedule (no wall-clock drift artifacts).

testId 1..3   : "warm-up" instances -- gaps are drawn i.i.d. from a single memoryless
                (roughly exponential) distribution. The classic closed-form Young/Daly
                interval (fixed, computed from the global average gap) is a good fit.
testId 4..10  : "bad node" instances -- the same sparse background gap process, PLUS 1-3
                short bursts of several tiny consecutive gaps (a node that keeps crashing
                for a while, then goes quiet). A single global-rate interval is tuned to
                the calm majority of the sequence and is far too coarse inside a burst.

All randomness is seeded ONLY from testId (deterministic, reproducible).
"""
import random
import sys


def build_instance(test_id):
    rng = random.Random(1000 + test_id)
    is_burst = test_id > 3

    scale = 1.0 + 0.35 * (test_id - 1)          # gentle size ramp across the ladder
    W = int(3000 * scale)
    C = rng.randint(20, 60)
    R = rng.randint(150, 400)

    gaps = []

    if not is_burst:
        # Warm-up: a single homogeneous memoryless (exponential-gap) process at a
        # realistic failure rate -- the classic Young/Daly closed-form interval, built
        # from the global average, is close to optimal here.
        mean_gap = W / rng.uniform(9.0, 13.0)
        n_bg = rng.randint(9, 14)
        for _ in range(n_bg):
            g = max(1, int(round(rng.expovariate(1.0 / mean_gap))))
            gaps.append(g)
    else:
        # Bad-node instances mix TWO very different populations:
        #  - many long, calm gaps (a comfortable fraction of W each -- large enough,
        #    and numerous enough, to badly skew a naive ARITHMETIC-MEAN MTBF estimate,
        #    while still individually safely below W so every gap keeps a real chance
        #    to fire regardless of where it lands in the sequence);
        #  - a handful of short, tightly-clustered burst gaps (a bad node crashing
        #    repeatedly), each still checkpoint-trackable (a controlled multiple of C)
        #    but far faster than what the skewed global average would suggest.
        # A schedule built from the (outlier-inflated) global mean ends up checkpointing
        # far too coarsely for the burst and pays close to the full burst-gap cost on
        # every one of its hits; a schedule that reacts only to gaps it has actually
        # SEEN fire recently is never fooled by the numerous large calm ones.
        n_bg = rng.randint(7, 9)
        n_bursts = rng.randint(2, 4)
        candidates = list(range(1, n_bg + 1))
        burst_after = set(rng.sample(candidates, min(n_bursts, len(candidates))))
        for i in range(1, n_bg + 1):
            g = int(round(W * rng.uniform(0.70, 0.98)))
            gaps.append(g)
            if i in burst_after:
                n_hits = rng.randint(10, 15)
                burst_mean = C * rng.uniform(4.0, 7.0)
                for _ in range(n_hits):
                    gg = max(1, int(round(burst_mean * rng.uniform(0.6, 1.1))))
                    gaps.append(gg)
        rng.shuffle(gaps)  # no structural artifact from generation order

    return W, C, R, gaps


def main():
    test_id = int(sys.argv[1])
    W, C, R, gaps = build_instance(test_id)
    m = len(gaps)
    lines = [f"{W} {C} {R}", str(m), " ".join(map(str, gaps))]
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
