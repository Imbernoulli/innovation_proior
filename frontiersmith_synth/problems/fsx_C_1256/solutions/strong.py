# TIER: strong
"""
Insight: the interval should be conditioned on RECENT failure history, not the run's
global average -- and, crucially, that history must be re-read after EVERY single
failure, not just after a checkpoint finally succeeds. A schedule that commits to one
target and blindly retries it through several failures in a row (as a per-segment
recipe would) cannot react mid-burst; this solution re-derives its next target after
every individual event (success OR failure), one step at a time.

At each decision point we look ONLY BACKWARD at the last few gaps that have already
fired (never at gaps that have not happened yet) and size the next interval off their
WORST (smallest) value (sqrt(2*C*local_MTBF)), not their mean -- a single recent short
gap is treated as a live warning that a bad node may still be misbehaving. Before any
failure has occurred we bootstrap from the single global mean (exactly the classic
formula). The moment a burst starts, the very first short gap immediately shrinks the
NEXT decision's interval.

A trailing window alone would stay stuck in "panic mode" until the NEXT failure fires to
refresh it -- even deep into a long calm stretch after the burst ended. To relax properly
without waiting on another failure, the local estimate is also floored by how much
compute time has ALREADY been safely survived since the last failure (`since_fail`): the
still-unfired current gap is provably at least that large, so this lower bound keeps
rising every successful step and pulls the interval back open the moment real evidence
of calm accumulates -- no extra failure required to license relaxing. This is the same
Young/Daly formula applied to a defensive, continuously-refreshed local statistic instead
of a single whole-run average -- a reformulation, not a bigger search.
"""
import math
import sys

WIN = 2  # trailing window size (in gap-index units, most-recently-fired gaps only)


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    W = int(next(it))
    C = int(next(it))
    R = int(next(it))
    m = int(next(it))
    gaps = [int(next(it)) for _ in range(m)]

    global_mtbf = max(1.0, (sum(gaps) / m) if m > 0 else float(W))

    wall = 0
    cur = 0
    saved = 0
    gi = 0
    since_fail = 0
    schedule = []
    target = None  # currently committed next checkpoint mark; None => must (re)decide

    while cur < W:
        if target is None:
            lo = max(0, gi - WIN)
            window = gaps[lo:gi]           # only ALREADY-FIRED gaps -- trailing history
            trailing_min = min(window) if window else global_mtbf  # worst recent gap
            # the still-unfired current gap is provably > since_fail already survived;
            # use that as a rising floor so we relax without needing a fresh failure
            local_mtbf = max(trailing_min, since_fail + 1.0)
            local_mtbf = max(local_mtbf, 1.0)
            T_local = max(1, int(round(math.sqrt(2.0 * C * local_mtbf))))
            target = min(W, cur + T_local)
            if target <= cur:
                target = cur + 1

        to_target = target - cur
        to_failure = (gaps[gi] - since_fail) if gi < m else None

        if to_failure is not None and to_failure < to_target:
            # a failure fires before we reach the committed target: take the hit, then
            # immediately re-decide the NEXT target using the freshly updated history
            wall += to_failure
            since_fail = 0
            cur = saved
            gi += 1
            wall += R
            target = None
        else:
            wall += to_target
            cur = target
            since_fail += to_target
            if target < W:
                saved = cur
                wall += C
                schedule.append(target)
            target = None

    print(len(schedule))
    print(" ".join(map(str, schedule)))


if __name__ == "__main__":
    main()
