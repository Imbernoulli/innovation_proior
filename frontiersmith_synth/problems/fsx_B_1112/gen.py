import sys
import random

# gen.py <testId>  -- prints ONE batched-polynomial-evaluation instance.
#
# One fixed degree-n integer polynomial P(x) = sum_k a[k] x^k must be evaluated at
# every query point in THREE query batches that arrive together each duty cycle:
#   SWEEP  (m1 pts) -- an arithmetic progression x0, x0+h, ..., x0+(m1-1)h
#   PROBE  (m2 pts) -- repeated calibration probes: only D DISTINCT relative offsets
#                      around a moving reference c2, each probed R times (shuffled order)
#   ADHOC  (m3 pts) -- scattered one-off points, no shared structure
#
# STRUCTURE (n, h, m1, D, R, m2, m3, the offset pattern, the probe repeat/shuffle
# pattern) depends ONLY on testId.  The actual NUMBERS (coefficients, the sweep
# start, the probe center, the scattered points) depend on a separate `value_seed`.
# The checker re-derives shadow instances with the SAME structure but different
# value_seed to defend against numeric-coincidence cheating -- build() is written
# so that is always possible (import this module from counter.py).

OFFSET_POOL = list(range(-6, 7))  # candidate small relative probe offsets


def build(test_id, value_seed):
    struct_rng = random.Random(1000003 * test_id + 7)
    n = 8 + (test_id % 4)             # 8..11
    h = 2 + (test_id % 3)             # 2..4
    m1 = 6 * n                        # sweep batch size
    D, R = 4, 5                       # 4 distinct probe offsets, each repeated 5x
    m2 = D * R                        # 20
    m3 = 10                           # ad-hoc batch size

    pool = OFFSET_POOL[:]
    struct_rng.shuffle(pool)
    offsets = pool[:D]                        # structural: same offsets every value_seed
    slot_seq = []
    for i in range(D):
        slot_seq += [i] * R
    struct_rng.shuffle(slot_seq)              # structural: same repeat/shuffle pattern

    vrng = random.Random(value_seed)
    a = [vrng.randint(-97, 97) for _ in range(n + 1)]
    while a[n] == 0:
        a[n] = vrng.choice([v for v in range(-97, 98) if v != 0])

    x0 = vrng.randint(200, 400)
    sweep_pts = [x0 + j * h for j in range(m1)]

    c2 = vrng.randint(-950, -650)
    distinct_probe_vals = [c2 + offsets[i] for i in range(D)]
    probe_pts = [distinct_probe_vals[i] for i in slot_seq]

    adhoc_pts = vrng.sample(range(3000, 8000), m3)

    all_vals = list(dict.fromkeys(sweep_pts + probe_pts + adhoc_pts))
    qidx = {v: i for i, v in enumerate(all_vals)}
    idx_sweep = [qidx[v] for v in sweep_pts]
    idx_probe = [qidx[v] for v in probe_pts]
    idx_adhoc = [qidx[v] for v in adhoc_pts]

    return {
        "n": n, "a": a, "q": all_vals, "Q": len(all_vals),
        "m1": m1, "idx1": idx_sweep,
        "m2": m2, "idx2": idx_probe,
        "m3": m3, "idx3": idx_adhoc,
    }


def render(test_id, inst):
    out = []
    out.append(str(test_id))
    out.append(str(inst["n"]))
    out.append(" ".join(str(v) for v in inst["a"]))
    out.append(str(inst["Q"]))
    out.append(" ".join(str(v) for v in inst["q"]))
    out.append(str(inst["m1"]))
    out.append(" ".join(str(v) for v in inst["idx1"]))
    out.append(str(inst["m2"]))
    out.append(" ".join(str(v) for v in inst["idx2"]))
    out.append(str(inst["m3"]))
    out.append(" ".join(str(v) for v in inst["idx3"]))
    return "\n".join(out) + "\n"


def main():
    tid = int(sys.argv[1])
    if tid < 1:
        tid = ((tid - 1) % 10) + 1
    inst = build(tid, value_seed=tid)
    sys.stdout.write(render(tid, inst))


if __name__ == "__main__":
    main()
