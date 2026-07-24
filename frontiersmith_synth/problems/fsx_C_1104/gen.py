#!/usr/bin/env python3
"""
gen.py <testId> -> prints ONE instance of "Beacon Hill" (crt-schedule-packing).

A hilltop relay hub repeats forever with period M (a multiple of L0 = 3360, so
every generated period divides M).  n periodic beacon jobs share the single
channel; job i starts a transmission of length d_i at times o_i + k*p_i.  Every
integer time t in [0, M) is an audit instant carrying a posted interference
cost w[t].  The graded waste is F = sum_t w[t] * max(0, load(t)-1).

Planted structure (the trap): the cheapest audit instants form a single
residue class mod 4 (t == 3 mod 4 costs ~1, everything else costs `heavy`).
Two job families are planted:
  F family ("flexible"): d=1, p divisible by 4, SHORT periods -> HIGH duty
      cycle (so any duty-cycle-sorted greedy places them FIRST).  They can
      live entirely inside one residue class mod 4, but the greedy never
      looks ahead and scatters them over the expensive classes as singles.
  R/X families ("rigid"): long periods / odd periods, longer runs, LOW duty
      cycle -> placed last, when the cheap class is already consumed as
      singles and their unavoidable overlap lands on expensive instants.
The CRT insight: jobs with 4 | p_i and d_i = 1 can be routed wholesale into
the cheap residue class (their unavoidable mutual excess is then priced ~1
instead of ~heavy), and rigid jobs should claim the expensive real estate
first while it is still free.

Determinism: all randomness comes from random.Random(testId * 100003 + 1104).
"""
import sys
import random

L0 = 3360  # 2^5 * 3 * 5 * 7; every generated period divides L0, hence M

REPS  = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4]
LOAD  = [1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.90, 2.00]
HEAVY = [3, 4, 4, 5, 5, 6, 6, 7, 7, 8]

L_POOL = [48, 60, 80, 96]                     # 4 | p, short period, duty >= 1/96
R_POOL = [240, 280, 336, 420, 480, 560, 672, 840, 1680, 3360]  # even, long
X_POOL = [105]                                # odd: anchors cycle all classes mod 4


def gen(test_id):
    rng = random.Random(test_id * 100003 + 1104)
    i = (test_id - 1) % 10
    M = L0 * REPS[i]
    target = LOAD[i]
    heavy = HEAVY[i]

    jobs = []

    # ---- F family: flexible (d=1, 4|p), high duty cycle, ~50% of load ----
    acc = 0.0
    budget = 0.50 * target
    while acc < budget:
        p = rng.choice(L_POOL)
        jobs.append((p, 1))
        acc += 1.0 / p

    # ---- R family: rigid even, long period, runs of 2..10, duty <= 0.0095 ----
    acc = 0.0
    budget = 0.42 * target
    while acc < budget:
        p = rng.choice(R_POOL)
        dmax = max(2, min(10, p // 105))
        d = rng.randint(2, dmax)
        jobs.append((p, d))
        acc += d / p

    # ---- X family: odd period 105, duty 1/105 < 1/96 (placed after F) ----
    acc = 0.0
    budget = 0.08 * target
    while acc < budget:
        p = rng.choice(X_POOL)
        jobs.append((p, 1))
        acc += 1.0 / p

    # ---- audit costs: residue class 3 mod 4 is cheap, others heavy ----
    extra = [1, 0, 2]  # mild asymmetry among the expensive classes
    w = []
    for t in range(M):
        r = t & 3
        if r == 3:
            w.append(1 + rng.randint(0, 1))
        else:
            w.append(heavy + extra[r] + rng.randint(0, 2))
    return M, jobs, w


def main():
    test_id = int(sys.argv[1])
    M, jobs, w = gen(test_id)
    out = [f"{M} {len(jobs)}"]
    out += [f"{p} {d}" for (p, d) in jobs]
    out.append(" ".join(map(str, w)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
