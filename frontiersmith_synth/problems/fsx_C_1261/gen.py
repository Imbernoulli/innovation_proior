import sys

# gen.py <testId> -- prints ONE DVFS deadline-schedule instance to stdout.
#
# A processor executes J deadline-bound jobs over T discrete time slots. At
# each slot the operator picks one of m DVFS levels (level 0 = idle,
# level m-1 = max); level k executes at most s[k] cycles that slot and
# burns p[k] energy (p is convex/cubic-ish in s). Switching levels between
# consecutive slots pays a transition-energy tax trans[i][j] AND a ramp
# penalty: the slot right after a switch loses `ramp` cycles of capacity
# (warm-up). Jobs are (release, deadline, work); feasibility is a
# preemptive earliest-deadline-first packing against the per-slot capacity
# sequence induced by the level choice.
#
# Cases 1-3: sparse warm-ups -- isolated jobs, each genuinely needing
# near-max speed to make its own short window, separated by long idle
# gaps. "Race to a burst, then idle" is close to the best you can do here.
#
# Cases 4-5: pure ramp-trap -- an isolated job preceded immediately by an
# idle gap, sized so a transition landing exactly on its first serviced
# slot (ramp loss right there) misses the deadline by exactly 1 cycle,
# while pre-warming during the idle gap (where the lost cycle is free)
# does not.
#
# Cases 6-7: pure energy-waste -- concurrent job pairs sharing a window,
# sized so a modest *sustained* level clears them comfortably; racing to
# the max level for that whole window is feasible but burns far more
# (convex) energy than necessary.
#
# Cases 8-10: combined -- both a ramp-trap job and an energy-waste overlap
# block in the same instance, growing in size/complexity.

M = 4
S = [0, 3, 6, 10]
P = [0, 27, 216, 1000]
RAMP = 2
TRANS = [
    [0, 4, 8, 12],
    [4, 0, 4, 8],
    [8, 4, 0, 4],
    [12, 8, 4, 0],
]
IDLE, LOW, MID, MAX = 0, 1, 2, 3


def simulate_feasible(levels, T, s, ramp, jobs):
    import heapq
    cap = [0] * T
    prev = None
    for t in range(T):
        lvl = levels[t]
        c = s[lvl]
        if prev is not None and lvl != prev:
            c = max(0, c - ramp)
        cap[t] = c
        prev = lvl
    by_release = {}
    by_deadline = {}
    for idx, (r, d, w) in enumerate(jobs):
        by_release.setdefault(r, []).append(idx)
        by_deadline.setdefault(d, []).append(idx)
    remaining = [w for (r, d, w) in jobs]
    heap = []
    feasible = True
    for t in range(T):
        for idx in by_release.get(t, []):
            heapq.heappush(heap, (jobs[idx][1], idx))
        avail = cap[t]
        tmp = []
        while avail > 0 and heap:
            dl, idx = heapq.heappop(heap)
            if remaining[idx] <= 0:
                continue
            use = min(avail, remaining[idx])
            remaining[idx] -= use
            avail -= use
            if remaining[idx] > 0:
                tmp.append((dl, idx))
        for item in tmp:
            heapq.heappush(heap, item)
        for idx in by_deadline.get(t + 1, []):
            if remaining[idx] > 0:
                feasible = False
    return feasible


class Builder:
    def __init__(self):
        self.t = 0
        self.jobs = []

    def gap(self, length):
        self.t += length

    def solo(self, length, tight):
        start = self.t
        if tight:
            work = S[MAX] * length - RAMP + 1
        else:
            work = 9 * length
        self.jobs.append((start, start + length, work))
        self.t += length

    def overlap(self, length, target_level, njobs=2):
        start = self.t
        rate = S[target_level]
        agg = int(rate * length * 0.85)
        per = agg // njobs
        rem = agg - per * (njobs - 1)
        for i in range(njobs):
            w = per if i < njobs - 1 else rem
            self.jobs.append((start, start + length, max(1, w)))
        self.t += length

    def finish(self):
        return self.t, self.jobs


def build(tid):
    b = Builder()
    if tid == 1:
        b.gap(4); b.solo(3, False); b.gap(6); b.solo(3, False); b.gap(4)
    elif tid == 2:
        b.gap(5); b.solo(4, False); b.gap(5); b.solo(3, False); b.gap(5)
    elif tid == 3:
        b.gap(3); b.solo(3, False); b.gap(4); b.solo(4, False); b.gap(3); b.solo(3, False); b.gap(3)
    elif tid == 4:
        b.gap(3); b.solo(6, True); b.gap(10)
    elif tid == 5:
        b.gap(2); b.solo(4, True); b.gap(4); b.solo(5, True); b.gap(8)
    elif tid == 6:
        b.gap(2); b.overlap(12, MID); b.gap(2)
    elif tid == 7:
        b.gap(2); b.overlap(8, MID); b.gap(2); b.overlap(8, LOW); b.gap(2)
    elif tid == 8:
        b.gap(2); b.overlap(9, MID); b.gap(3); b.overlap(10, MID); b.gap(2)
    elif tid == 9:
        b.gap(2); b.overlap(8, MID); b.gap(2); b.solo(5, True); b.gap(2); b.overlap(6, LOW); b.gap(2)
    elif tid == 10:
        b.gap(2); b.solo(4, True); b.gap(2); b.overlap(8, MID); b.gap(2)
        b.overlap(10, MID); b.gap(2); b.overlap(6, LOW); b.gap(4)
    else:
        # generic fallback for any extra testId: alternate solo/overlap growing with tid
        for k in range(3 + (tid % 4)):
            if k % 2 == 0:
                b.gap(2); b.solo(3 + (k % 3), tid % 2 == 0)
            else:
                b.gap(2); b.overlap(6 + (k % 3), MID)
        b.gap(4)
    # uniform trailing idle padding: costs nothing for any schedule that
    # correctly idles once all work is done (greedy/strong both do), but
    # directly penalizes the "never idle" trivial baseline -- keeps the
    # trivial/greedy separation calibrated without touching the traps.
    b.gap(4)
    T, jobs = b.finish()
    return T, jobs


def main():
    tid = int(sys.argv[1])
    T, jobs = build(tid)
    J = len(jobs)

    # sanity: constant-max for the whole horizon must be feasible (the
    # generator's own certificate that the instance is solvable at all).
    assert simulate_feasible([MAX] * T, T, S, RAMP, jobs), f"tid={tid} infeasible even at max"

    out = []
    out.append(f"{T} {M} {J}")
    for k in range(M):
        out.append(f"{S[k]} {P[k]}")
    out.append(str(RAMP))
    for row in TRANS:
        out.append(" ".join(str(x) for x in row))
    for (r, d, w) in jobs:
        out.append(f"{r} {d} {w}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
