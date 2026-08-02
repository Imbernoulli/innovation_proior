#!/usr/bin/env python3
"""
gen.py <testId> -> prints ONE instance to stdout.

Theme: "retrying without making the outage worse."

A fleet of N client requests must each reach one of E backend endpoints
through a shared backend with a deterministic per-tick health/outage
schedule. Endpoints are classified idempotent (safe to re-execute) or
non-idempotent (a second server-side execution after an already-successful
but unacknowledged attempt is a harmful DUPLICATE). A short "ack-loss"
window right after every correlated outage recovers makes some successes
AMBIGUOUS to the client (it looks like a failure, so the client retries a
request the server already completed).

Determinism: all randomness comes from Python's `random.Random(testId)`.

Difficulty ladder (testId 1..10):
  - testId 1-3 ("calm"): generous shared capacity, staggered/uncorrelated
    per-endpoint outages -- retries help, no storm is even possible.
  - testId 4-10 ("storm trap", 7 of 10 cases): tight shared capacity, ALL
    endpoints share ONE correlated outage window, and most requests arrive
    while it is active. A same-schedule-for-everyone backoff (no jitter,
    no shared budget) resynchronizes retries right when the outage ends,
    overwhelming the shared capacity and triggering the collapse feedback.
"""
import sys
import random


def build_instance(test_id):
    rng = random.Random(test_id)

    N_schedule = [8, 10, 14, 18, 24, 30, 36, 44, 54, 66]
    T_schedule = [22, 24, 28, 32, 38, 44, 50, 56, 64, 74]
    N = N_schedule[(test_id - 1) % len(N_schedule)]
    T = T_schedule[(test_id - 1) % len(T_schedule)]
    E = 4

    idem = [1, 0, 1, 0]
    rng.shuffle(idem)

    is_trap = test_id >= 4

    outage = [[0] * T for _ in range(E)]
    ambiguous = [[0] * T for _ in range(E)]

    if is_trap:
        C = max(2, round(0.16 * N))
        O = round(T * 0.30)
        for e in range(E):
            for t in range(O):
                outage[e][t] = 1
            for t in range(O, min(T, O + 4)):
                ambiguous[e][t] = 1
        arrivals = []
        n_pre = round(0.70 * N)
        for i in range(N):
            if i < n_pre:
                t0 = rng.randrange(0, O) if O > 0 else 0
            else:
                t0 = rng.randrange(O, T) if O < T else T - 1
            e0 = rng.randrange(E)
            arrivals.append((t0, e0))
    else:
        C = max(4, round(0.9 * N))
        O = 0
        win = max(2, T // (E + 2))
        for e in range(E):
            start = e * win
            for t in range(start, min(T, start + max(1, win // 2))):
                outage[e][t] = 1
        for e in range(E):
            for _ in range(2):
                t = rng.randrange(0, T)
                ambiguous[e][t] = 1
        arrivals = []
        for i in range(N):
            t0 = rng.randrange(0, T)
            e0 = rng.randrange(E)
            arrivals.append((t0, e0))

    return T, N, E, C, idem, outage, ambiguous, arrivals


def main():
    test_id = int(sys.argv[1])
    T, N, E, C, idem, outage, ambiguous, arrivals = build_instance(test_id)

    out = []
    out.append(f"{T} {N} {E} {C}")
    out.append(" ".join(str(x) for x in idem))
    for e in range(E):
        out.append(" ".join(str(x) for x in outage[e]))
    for e in range(E):
        out.append(" ".join(str(x) for x in ambiguous[e]))
    for i in range(N):
        t0, e0 = arrivals[i]
        out.append(f"{t0} {e0}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
