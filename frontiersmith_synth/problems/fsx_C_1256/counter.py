#!/usr/bin/env python3
"""
counter.py <in> <out> <ans> -- deterministic scorer for checkpoint-interval-choose.

Instance: total useful work W, checkpoint write cost C, restart cost R, and a fixed
sequence of `m` ACTIVE-COMPUTE-TIME failure gaps g_1..g_m (MTBF-in-compute-time
formulation: the machine fails after g_1 units of active computation from the start,
then after another g_2 units of active computation counted fresh from the moment
failure 1 fired, and so on; checkpoint/restart pauses do not count as active computation).

Participant artifact: a strictly increasing list of checkpoint PROGRESS marks
p_1 < p_2 < ... < p_k, each in (0, W) -- "take a checkpoint after this much useful work
has been completed".

Deterministic replay (`simulate`): the job computes at rate 1 progress/time, which is
also rate 1 active-compute-time/time. If it reaches a checkpoint mark before the next
scheduled failure gap is exhausted, it pauses and pays cost C, then keeps going. If a
failure gap is exhausted first, ALL progress since the last checkpoint (or start) is
lost, the compute-time-since-last-failure counter resets to 0, cost R is paid, and the
job resumes from the last checkpoint. Objective: minimize the wall-clock time to reach W.

Feasibility: output must be well-formed integers, 0 <= k <= 200000, each mark strictly
inside (0, W), strictly increasing. Any violation -> Ratio: 0.0.

Baseline B: the checker's own reference construction, NEVER checkpointing at all (redo
everything from scratch on every failure). Minimization ratio:
sc = min(1000, 100*B/max(1e-9,F)); print(sc/1000).
"""
import sys

MAX_K = 200_000
MAX_TOKEN_LEN = 20


def fail(reason):
    print(f"Ratio: 0.0  # {reason}")
    sys.exit(0)


def parse_int(tok):
    if len(tok) > MAX_TOKEN_LEN:
        raise ValueError("token too long")
    return int(tok)


def simulate(thresholds, W, C, R, gaps):
    """Replay the job against the fixed failure-gap sequence; return total wall-clock
    time to accumulate W units of saved progress. `thresholds` must already be validated
    as strictly increasing values in (0, W)."""
    saved = 0            # last checkpointed (safe) progress
    cur = 0               # current progress (== saved right after start/failure)
    wall = 0
    m = len(gaps)
    gi = 0                # index of the next failure gap to exhaust
    since_fail = 0         # active compute time since the last failure (or start)
    marks = thresholds + [W]
    mi = 0
    while cur < W:
        target = marks[mi]
        to_target = target - cur
        to_failure = (gaps[gi] - since_fail) if gi < m else None
        if to_failure is not None and to_failure < to_target:
            wall += to_failure
            since_fail = 0
            cur = saved          # lose unsaved progress
            gi += 1
            wall += R
        else:
            wall += to_target
            cur = target
            since_fail += to_target
            if target < W:
                saved = cur
                wall += C
                mi += 1
            # else: cur == W, job complete
    return wall


def main():
    if len(sys.argv) != 4:
        fail("bad invocation")
    in_path, out_path, _ans_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(in_path) as f:
        itoks = f.read().split()
    ip = iter(itoks)

    def inext():
        return next(ip)

    try:
        W = parse_int(inext())
        C = parse_int(inext())
        R = parse_int(inext())
        if W <= 0 or C <= 0 or R <= 0:
            fail("bad instance header")
        m = parse_int(inext())
        if m < 0:
            fail("bad m")
        gaps = [parse_int(inext()) for _ in range(m)]
        for g in gaps:
            if g <= 0:
                fail("bad instance failure gap (should not happen)")
    except (StopIteration, ValueError):
        fail("malformed instance (should not happen)")

    # ---- parse participant output (untrusted) ----
    try:
        with open(out_path) as f:
            otoks = f.read().split()
    except OSError:
        fail("cannot read output")

    if not otoks:
        fail("empty output")

    op = iter(otoks)
    try:
        k = parse_int(next(op))
    except (StopIteration, ValueError):
        fail("bad checkpoint count")
    if k < 0 or k > MAX_K:
        fail("checkpoint count out of range")

    thresholds = []
    try:
        for _ in range(k):
            p = parse_int(next(op))
            thresholds.append(p)
    except (StopIteration, ValueError):
        fail("malformed checkpoint mark (non-finite/garbage/truncated)")

    prev = 0
    for p in thresholds:
        if not (0 < p < W):
            fail("checkpoint mark out of range (0, W)")
        if p <= prev:
            fail("checkpoint marks not strictly increasing")
        prev = p

    # ---- score ----
    F = simulate(thresholds, W, C, R, gaps)
    if F <= 0:
        fail("degenerate replay (should not happen)")

    # Baseline: never checkpoint (redo everything from scratch on every failure). This is
    # the checker's own trivial feasible construction -- deliberately not schedule-aware.
    B = simulate([], W, C, R, gaps)
    if B <= 0:
        fail("degenerate baseline (should not happen)")

    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    ratio = sc / 1000.0
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
