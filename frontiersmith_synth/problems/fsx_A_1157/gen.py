import sys, random

KINDS = 5

# machine catalog: (name, cost-per-work-unit for each of the 5 kinds, price)
# types 0..4 are single-kind specialists (cost=1 at their own kind, cost=14 elsewhere);
# type 5 is a generalist (cost=2 at every kind -- never best, but always a solid
# fallback the frozen scheduler can fall back on instead of a wrong specialist).
TYPES = [
    ("SpecA", [1, 14, 14, 14, 14], 30),
    ("SpecB", [14, 1, 14, 14, 14], 30),
    ("SpecC", [14, 14, 1, 14, 14], 30),
    ("SpecD", [14, 14, 14, 1, 14], 30),
    ("SpecE", [14, 14, 14, 14, 1], 30),
    ("Gen",   [2, 2, 2, 2, 2],     48),
]
T = len(TYPES)

# difficulty ladder (n_jobs, budget) per testId, tuned so kind-coverage genuinely
# matters (neither starved nor so flush that raw unit-count alone suffices)
LADDER = {
    1: (6, 129), 2: (6, 150), 3: (7, 150), 4: (7, 172), 5: (8, 172),
    6: (8, 193), 7: (9, 193), 8: (9, 215), 9: (10, 215), 10: (10, 236),
}


def gen_job(rng, spike_kinds, spike_weight, op_lo, op_hi, w_lo, w_hi):
    L = rng.randint(op_lo, op_hi)
    ops = []
    for _ in range(L):
        if spike_kinds and rng.random() < spike_weight:
            k = rng.choice(spike_kinds)
        else:
            k = rng.choice([1, 2, 3, 4])  # background kinds never include kind 0
        w = rng.randint(w_lo, w_hi)
        ops.append((k, w))
    return ops


def gen_scenario(rng, spike_kinds, spike_weight, n_jobs, op_lo, op_hi, w_lo, w_hi):
    jobs = [gen_job(rng, spike_kinds, spike_weight, op_lo, op_hi, w_lo, w_hi) for _ in range(n_jobs)]
    oracle = max(sum(w for k, w in job) for job in jobs)
    return jobs, oracle


def main():
    tid = int(sys.argv[1])
    rng = random.Random(20260726 + 97 * tid)

    n_jobs, budget = LADDER.get(tid, LADDER[10])
    w_lo, w_hi = 3, 8                       # work magnitude

    # 9 scenarios: a thin kind-0 spike, 4 real single-kind spikes (kinds1-4),
    # 2 balanced (background over kinds1-4), 2 disjoint dual-spikes.
    specs = [
        ([0], 0.30, max(3, n_jobs - 2)),   # s0: thin kind-0 spike (kind0 is deliberately weak/rare)
        ([1], 0.75, n_jobs),               # s1
        ([2], 0.75, n_jobs),               # s2
        ([3], 0.75, n_jobs),               # s3
        ([4], 0.75, n_jobs),               # s4
        ([], 0.0, n_jobs),                 # s5 balanced (kinds1-4)
        ([], 0.0, n_jobs),                 # s6 balanced (kinds1-4)
        ([1, 3], 0.75, n_jobs),            # s7 dual spike
        ([2, 4], 0.75, n_jobs),            # s8 dual spike
    ]

    scenarios = []
    for spike_kinds, weight, nj in specs:
        srng = random.Random(rng.randint(0, 10 ** 9))
        jobs, oracle = gen_scenario(srng, spike_kinds, weight, nj, 2, 3, w_lo, w_hi)
        scenarios.append((jobs, oracle))

    lines = [f"{T} {KINDS} {budget}"]
    for name, cost, price in TYPES:
        lines.append(" ".join(map(str, cost)) + " " + str(price))
    lines.append(str(len(scenarios)))
    for jobs, oracle in scenarios:
        lines.append(f"{len(jobs)} {oracle}")
        for job in jobs:
            toks = [str(len(job))]
            for k, w in job:
                toks.append(str(k)); toks.append(str(w))
            lines.append(" ".join(toks))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
