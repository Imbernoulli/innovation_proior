#!/usr/bin/env python3
"""gen.py <testId> -- prints one scheduler-priority-inversion instance to stdout.
Deterministic: seeded purely by testId (no wall-clock / external entropy).

Instance = a fixed job list run under strict fixed-priority preemptive scheduling,
plus a set of shared locks each with a static "ceiling priority". The instance is
built from two reusable, hand-verified building blocks:

  lockA_case -- a lock whose critical-section holder is briefly preempted by
                medium-priority filler jobs *before* the real high-priority
                waiter ever arrives.  A reactive (priority-inheritance) holder
                is unprotected during that early gap and gets pushed back;
                a proactive (priority-ceiling) holder is immune from the first
                tick, so it releases sooner and the real waiter meets its
                deadline. => favors CEILING.

  lockB_case -- a lock that a low-priority job holds for a while but that
                NO high-priority job ever actually contends for; its ceiling
                is nonetheless high (some rarely-present high-priority task
                is a nominal user of the resource). A ceiling holder gets
                boosted the entire time regardless, needlessly blocking
                unrelated medium-priority jobs that never wanted the lock at
                all. A priority-inheritance holder never boosts (nobody is
                ever actually waiting), so those medium jobs run on time.
                => favors INHERIT.

A test case chains several such blocks (one per lock) at well-separated
arrival-time offsets so they cannot accidentally interact, plus one
"floor" job with an unmeetable deadline (arrival 0, deadline 0) that
guarantees a strictly positive baseline cost on every case (so the checker's
B/F normalization never divides degenerate zeros).
"""
import sys

def lockA_case(jobs, ceilings, lock_id, base_t, scale=1):
    # Lo: long critical-section holder, low base priority.
    jobs.append((9, base_t + 0, base_t + 400, 1, [(10 * scale, lock_id)]))
    # Ma, Mb: medium-priority filler that arrives *before* the real waiter H
    # -- steals cycles from an unboosted holder under both "none" and
    # "inherit" (inherit only starts protecting once H is actually waiting).
    jobs.append((6, base_t + 1, base_t + 400, 1, [(2 * scale, 0)]))
    jobs.append((6, base_t + 4 * scale, base_t + 400, 1, [(2 * scale, 0)]))
    # H: the real high(er)-priority waiter, tight deadline.
    h_arr = base_t + 8 * scale
    jobs.append((4, h_arr, h_arr + 3 * scale, 6, [(1, lock_id)]))
    # Mc, Md: medium-priority filler arriving *after* H starts waiting --
    # "inherit" already protects the holder by then (boosted to H's
    # priority), "none" does not, so Mc/Md still steal cycles under "none"
    # (this is the gap that makes plain inheritance a real, visible
    # improvement over doing nothing -- not just a tie).
    jobs.append((6, base_t + 9 * scale, base_t + 400, 1, [(2 * scale, 0)]))
    jobs.append((6, base_t + 12 * scale, base_t + 400, 1, [(2 * scale, 0)]))
    ceilings[lock_id] = 1
    return base_t + 26 * scale  # end of this block's window


def lockB_case(jobs, ceilings, lock_id, base_t, scale=1):
    # Lo: low-priority holder of a moderate critical section, generous own deadline.
    jobs.append((9, base_t + 0, base_t + 400, 1, [(6 * scale, lock_id)]))
    # M1..M3: medium-priority jobs that never touch the lock, tight deadlines.
    jobs.append((5, base_t + 1, base_t + 4 * scale, 4, [(2 * scale, 0)]))
    jobs.append((5, base_t + 2, base_t + 5 * scale, 4, [(2 * scale, 0)]))
    jobs.append((5, base_t + 3, base_t + 6 * scale, 4, [(2 * scale, 0)]))
    ceilings[lock_id] = 1
    return base_t + 20 * scale


def calm_case(jobs, ceilings, lock_id, base_t):
    # A lone user of the lock: nobody else ever contends -> protocol-invariant.
    jobs.append((9, base_t + 0, base_t + 20, 1, [(3, lock_id)]))
    ceilings[lock_id] = 1
    return base_t + 12


BLOCK_SPACING = 40


def build(test_id):
    jobs = []
    ceilings = {}
    # floor job: impossible deadline, priority far below everything else ->
    # guarantees a strictly positive baseline cost on every instance.
    jobs.append((99, 0, 0, 1, [(1, 0)]))

    # Every plan includes >=1 lockA block (the ceiling-favoring trap that a
    # uniform-inheritance greedy cannot see coming), difficulty rising via
    # more locks and a bigger scale multiplier.
    plans = {
        1: [("A", 1)],
        2: [("A", 1), ("B", 2)],
        3: [("B", 1), ("A", 2)],
        4: [("A", 1), ("A", 2)],
        5: [("A", 1), ("B", 2), ("A", 3)],
        6: [("B", 1), ("A", 2), ("B", 3)],
        7: [("A", 1), ("A", 2), ("B", 3)],
        8: [("A", 1), ("B", 2), ("A", 3), ("B", 4)],
        9: [("A", 1), ("A", 2), ("B", 3), ("B", 4)],
        10: [("A", 1), ("B", 2), ("A", 3), ("B", 4)],
    }
    scales = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 2, 10: 2}
    scale = scales[test_id]
    plan = plans[test_id]
    base_t = 0
    for kind, lock_id in plan:
        if kind == "A":
            base_t = lockA_case(jobs, ceilings, lock_id, base_t, scale)
        elif kind == "B":
            base_t = lockB_case(jobs, ceilings, lock_id, base_t, scale)
        else:
            base_t = calm_case(jobs, ceilings, lock_id, base_t)
        base_t += BLOCK_SPACING
    L = max(ceilings.keys())
    return L, jobs, ceilings


def main():
    test_id = int(sys.argv[1])
    L, jobs, ceilings = build(test_id)
    J = len(jobs)
    total_work = sum(l for j in jobs for (l, lk) in j[4])
    max_arr = max(j[1] for j in jobs)
    Tmax = max_arr + total_work + 10

    out = []
    out.append(f"{L} {J} {Tmax}")
    out.append(" ".join(str(ceilings.get(l, 1)) for l in range(1, L + 1)))
    for (pri, arr, dl, w, segs) in jobs:
        toks = [str(pri), str(arr), str(dl), str(w), str(len(segs))]
        for (ln, lk) in segs:
            toks.append(str(ln))
            toks.append(str(lk))
        out.append(" ".join(toks))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
