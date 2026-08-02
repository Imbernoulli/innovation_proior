# TIER: strong
"""Global admission budgeting instead of per-request exponential backoff.

The textbook per-request recipe (`greedy`) treats every request as its own
isolated optimization and ignores two things that are fully visible in the
input: (1) the shared capacity C -- the resource all N requests actually
compete for -- and (2) each endpoint's own outage/ambiguous-ack-loss
schedule.

1. TOKEN-BUCKET STAGGERING (exploits retry-storm-feedback): rather than
   giving every request the identical backoff clock (which resynchronizes
   an entire correlated-outage cohort onto the same recovery tick), spread
   the N requests over L = ceil(N / C) "lanes" (lane = request_id % L) and
   add the lane index to every backoff step. Two requests that fail
   together no longer retry together -- their combined instantaneous demand
   stays within the shared budget C instead of spiking far above it, so the
   collapse feedback is never triggered in the first place.

2. IDEMPOTENCY-AWARE DODGING (exploits idempotency-classification): for a
   NON-idempotent endpoint a second server-side success is a harmful
   duplicate, so its retry ticks are pushed forward (still using the given,
   fully-visible outage/ambiguous arrays) past any tick flagged outage OR
   ambiguous for that endpoint -- it only ever retries onto a tick that is
   known to be clean. An idempotent endpoint has no duplicate risk, so it
   is left on the plain staggered clock (no need to spend extra ticks
   dodging).

Both ingredients are needed together: staggering alone still walks straight
through ambiguous windows (duplicates on non-idempotent endpoints);
dodging alone (with everyone still on the same clock) still resynchronizes
the herd and re-triggers the capacity collapse.
"""
import sys
import math


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); N = int(next(it)); E = int(next(it)); C = int(next(it))
    idem = [int(next(it)) for _ in range(E)]
    outage = [[int(next(it)) for _ in range(T)] for _ in range(E)]
    ambiguous = [[int(next(it)) for _ in range(T)] for _ in range(E)]
    arrivals = []
    for _ in range(N):
        t0 = int(next(it)); e0 = int(next(it))
        arrivals.append((t0, e0))

    L = max(1, math.ceil(N / max(1, C)))
    STEP = 2
    lines = []
    for i in range(N):
        t0, e = arrivals[i]
        lane = i % L
        max_att = 5
        backoffs = []
        cur = t0
        for k in range(max_att - 1):
            cand = cur + STEP * (k + 1) + lane
            if idem[e] == 0:
                while cand < T and (outage[e][cand] == 1 or ambiguous[e][cand] == 1):
                    cand += 1
            b = max(1, cand - cur)
            backoffs.append(b)
            cur = cand
        lines.append(str(max_att) + " " + " ".join(str(x) for x in backoffs))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
