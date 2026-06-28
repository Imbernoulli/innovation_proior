#!/usr/bin/env python3
"""Instance generator for "Time-Indexed Crew Rostering" (ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout. The format is (whitespace separated, line breaks
are only for readability -- the solver reads tokens):

    W D S
    HOURS[1] HOURS[2] ... HOURS[S]          # length of each shift, in hours
    START[1] START[2] ... START[S]          # clock start hour of each shift (0..23)
    MIN_REST  MAXCONS  MAXH  HARDH  LAMBDA   # rest hours, max consecutive days,
                                             # soft / hard weekly-hour caps, overtime rate
    then for each day d = 0..D-1:
        DEMAND[d][1] ... DEMAND[d][S]        # workers needed on shift s of day d
        VALUE[d][1]  ... VALUE[d][S]         # value of one covered unit of shift s, day d
    then for each worker w = 0..W-1:
        for each day d = 0..D-1:
            AVAIL[w][d][1] ... AVAIL[w][d][S]   # 1 if w may work shift s on day d, else 0

Shifts are indexed 1..S; shift 0 means "day off" and is not represented in the
demand / value / availability arrays. END[s] = START[s] + HOURS[s] (the clock end,
which may exceed 24 to denote crossing midnight). The rest gap between a shift s on
day d and a shift s' on day d+1 is 24 + START[s'] - END[s] hours.

Workers have heterogeneous availability (each worker is unavailable for a random
subset of (day, shift) slots) and demands exceed total available supply on the
busy shifts, so the rostering decision is genuinely contended -- which is the
regime where the per-worker pattern abstraction + local search pays off.
"""
import sys
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0xC0FFEE ^ (seed * 2654435761 & 0xFFFFFFFF))

    # Problem size: a week-ish grid, deterministic from the seed.
    D = 7
    S = 3  # 1 = morning, 2 = day, 3 = night (canonical three-shift roster)
    W = rng.randint(18, 40)

    # Shift lengths and clock starts (hours). Night shift is last and starts late,
    # so it constrains the next-morning rest rule -- the back-to-back lever.
    HOURS = [8, 8, 10]            # morning 8h, day 8h, night 10h
    START = [6, 14, 22]          # 06:00, 14:00, 22:00  (night crosses midnight)
    # END = [14, 22, 32]; rest gap night(d)->morning(d+1) = 24 + 6 - 32 = -2 (illegal)

    MIN_REST = 11                # legally required rest between consecutive shifts
    MAXCONS = 5                  # at most 5 consecutive working days
    MAXH = 40                    # soft weekly-hour cap; hours beyond this are overtime
    HARDH = 52                   # hard weekly-hour cap; exceeding it is INFEASIBLE
    LAMBDA = 3                   # overtime penalty per hour over MAXH

    # Per-day demand and value for each shift. Demands are tuned so total demand
    # somewhat exceeds the supply a single feasible roster can cover -> contention.
    demand = [[0] * (S + 1) for _ in range(D)]
    value = [[0] * (S + 1) for _ in range(D)]
    for d in range(D):
        weekend = d >= 5
        for s in range(1, S + 1):
            base = rng.randint(4, 9)
            if weekend:
                base = max(2, base - rng.randint(0, 3))
            if s == 3:  # night shifts tend to need fewer but are valued higher
                base = max(2, base - rng.randint(0, 2))
            demand[d][s] = base
            v = rng.randint(5, 12)
            if s == 3:
                v += rng.randint(2, 5)   # night premium
            value[d][s] = v

    # Worker availability: each worker is available for most slots but blocks a
    # random fraction (personal constraints, requested days off, qualifications).
    avail = [[[0] * (S + 1) for _ in range(D)] for _ in range(W)]
    for w in range(W):
        block_frac = rng.uniform(0.10, 0.35)
        for d in range(D):
            for s in range(1, S + 1):
                avail[w][d][s] = 0 if rng.random() < block_frac else 1
        # Guarantee every worker can work at least a couple of slots (never a
        # fully-dead worker, which would be a trivial degenerate input).
        if sum(avail[w][d][s] for d in range(D) for s in range(1, S + 1)) < 3:
            for _ in range(3):
                d = rng.randrange(D)
                s = rng.randint(1, S)
                avail[w][d][s] = 1

    out = []
    out.append(f"{W} {D} {S}")
    out.append(" ".join(str(HOURS[s - 1]) for s in range(1, S + 1)))
    out.append(" ".join(str(START[s - 1]) for s in range(1, S + 1)))
    out.append(f"{MIN_REST} {MAXCONS} {MAXH} {HARDH} {LAMBDA}")
    for d in range(D):
        out.append(" ".join(str(demand[d][s]) for s in range(1, S + 1)))
        out.append(" ".join(str(value[d][s]) for s in range(1, S + 1)))
    for w in range(W):
        for d in range(D):
            out.append(" ".join(str(avail[w][d][s]) for s in range(1, S + 1)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
