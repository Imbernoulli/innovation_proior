# TIER: greedy
"""The obvious first approach: textbook exponential backoff, applied
independently per request, ignoring both the shared capacity and each
endpoint's idempotency classification. Every request gets the SAME
schedule: up to 5 attempts, backoff 2, 4, 8, 16 ticks after each failure.

In isolation this maximizes any ONE request's own success chance (it keeps
trying through an outage and gives itself several chances to land on a
healthy tick). The trap: because every client uses the identical
un-staggered schedule, a correlated outage that hits many requests at once
makes them all fail in lockstep and therefore all retry in lockstep --
their un-jittered exponential clocks resynchronize right when the outage
ends, producing a demand spike that overwhelms the shared capacity and
triggers the collapse feedback (which this policy has no way to see or
avoid, since it never looks at the capacity or the endpoint's own outage/
ambiguous schedule). It also ignores idempotency, so it retries
non-idempotent endpoints exactly as aggressively as idempotent ones,
walking straight through the post-outage ambiguous-ack-loss window and
racking up avoidable duplicate side effects.
"""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); N = int(next(it)); E = int(next(it)); C = int(next(it))
    # consume the rest of the instance -- greedy does not use any of it.
    for _ in range(E):
        for _ in range(T):
            next(it)
    for _ in range(E):
        for _ in range(T):
            next(it)
    for _ in range(N):
        next(it); next(it)

    for _ in range(N):
        print("5 2 4 8 16")


if __name__ == "__main__":
    main()
