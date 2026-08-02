#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of fsx_B_1176 to stdout.

Family: load-disaggregate-meter (format C, maximize).
Builds a whole-house aggregate power trace from A hidden appliance state
machines (2-state ON/OFF, but each state has a LEGAL DWELL-TIME window --
that is the "state machine" part: an appliance is not a constant load, it is
a load that must stay ON/OFF for a bounded number of steps before it may
switch again). Only the AGGREGATE trace and the appliance library (power
level + dwell bounds per appliance) are printed to stdout; the hidden true
per-appliance state sequence is NOT printed (it lives only inside the
checker, re-derived by the identical deterministic construction below).

Cases 3..10 (of 10) are TRAP cases: they include two or more appliances
that share the exact same ON power level but have different legal dwell
windows (e.g. a short-burst 1200W load and a long-run 1200W load), so the
instantaneous aggregate value alone cannot tell them apart -- only the
edge (step-change) timing and each appliance's own legal-duration window
can. Cases 1-2 have no shared power levels (control cases).
"""
import sys
import random

# appliance archetype catalog: name -> (P, min_on, max_on, min_off, max_off)
# dwell windows are kept small relative to T so every appliance in a case
# cycles ON/OFF several times over the trace (rich edge signal, not a single
# lucky activation). Within each shared-power TWIN pair the on/off dwell
# WINDOWS are deliberately DISJOINT (never overlap) -- so an edge's exact
# magnitude never determines identity, but the elapsed dwell time at that
# edge almost always does. That is the whole point: read only the level and
# the two are indistinguishable; read the legal timing and they usually are
# not. (Residual overlap-free ambiguity can still occur when both members of
# a pair happen to be simultaneously mid-cycle at a boundary, which is what
# keeps a perfect reconstruction out of reach.)
ARCH = {
    "FRIDGE":    (150,  5, 8,   8, 14),
    "KETTLE":    (2000, 2, 4,  10, 18),
    "WASHER":    (500,  7, 11, 12, 20),
    "DRYER":     (700,  6, 9,   7, 11),
    "MICROWAVE": (1200, 2, 4,  16, 24),   # twin group 1200W: fast/short bursts
    "HEATER":    (1200, 11, 15, 3, 5),    # twin group 1200W: slow/long runs (disjoint on/off windows)
    "POOLPUMP":  (900,  2, 4,  10, 15),   # twin group 900W: fast/short bursts
    "ACCOMP":    (900,  10, 14, 3, 5),    # twin group 900W: slow/long runs (disjoint on/off windows)
}

# testId -> (T, [archetype names], wT, wA). Every case includes at least one
# shared-power twin pair -- there is no "no ambiguity at all" control case,
# so no case admits a trivially-reachable, fully-certain optimum.
TABLE = {
    1:  (36, ["FRIDGE", "MICROWAVE", "HEATER"], 2.0, 1.0),
    2:  (40, ["FRIDGE", "POOLPUMP", "ACCOMP"], 1.8, 1.2),
    3:  (42, ["FRIDGE", "MICROWAVE", "HEATER", "WASHER"], 2.0, 1.3),
    4:  (44, ["FRIDGE", "MICROWAVE", "HEATER", "KETTLE"], 1.8, 1.4),
    5:  (46, ["MICROWAVE", "HEATER", "POOLPUMP", "ACCOMP", "FRIDGE"], 2.0, 1.5),
    6:  (48, ["MICROWAVE", "HEATER", "POOLPUMP", "ACCOMP", "KETTLE"], 1.7, 1.6),
    7:  (50, ["MICROWAVE", "HEATER", "POOLPUMP", "ACCOMP", "WASHER"], 1.6, 1.7),
    8:  (54, ["MICROWAVE", "HEATER", "DRYER", "POOLPUMP", "ACCOMP", "FRIDGE"], 1.8, 1.8),
    9:  (58, ["MICROWAVE", "HEATER", "DRYER", "POOLPUMP", "ACCOMP", "KETTLE"], 1.6, 2.0),
    10: (62, ["MICROWAVE", "HEATER", "DRYER", "POOLPUMP", "ACCOMP", "WASHER"], 1.5, 2.2),
}


def build_case(test_id):
    """Deterministically build (T, archs, hidden, aggregate, wT, wA) for a testId.
    archs = list of (P, min_on, max_on, min_off, max_off).
    hidden = list (len A) of 0/1 lists (len T): the TRUE state sequence.
    Regenerated identically (same code) inside verify.py -- never shipped to
    the solver. A small deterministic seed-retry avoids two appliances
    transitioning on the exact same timestep (keeps every observed edge
    attributable to a single appliance's own dwell-legal transition)."""
    T, names, wT, wA = TABLE[test_id]
    archs = [ARCH[nm] for nm in names]
    attempt = 0
    hidden = None
    while True:
        seed = 20000 + 137 * test_id + 991 * attempt
        rng = random.Random(seed)
        cand = []
        for (P, mon, mxon, moff, mxoff) in archs:
            seq = []
            state = 0
            while len(seq) < T:
                d = rng.randint(moff, mxoff) if state == 0 else rng.randint(mon, mxon)
                d = min(d, T - len(seq))
                seq.extend([state] * d)
                state = 1 - state
            cand.append(seq[:T])
        seen = set()
        collide = False
        for seq in cand:
            for t in range(1, T):
                if seq[t] != seq[t - 1]:
                    if t in seen:
                        collide = True
                        break
                    seen.add(t)
            if collide:
                break
        if not collide:
            hidden = cand
            break
        attempt += 1
        if attempt > 300:
            hidden = cand
            break
    aggregate = [0] * T
    for (P, *_rest), seq in zip(archs, hidden):
        for t in range(T):
            if seq[t] == 1:
                aggregate[t] += P
    return T, archs, hidden, aggregate, wT, wA


def main():
    test_id = int(sys.argv[1])
    T, archs, hidden, aggregate, wT, wA = build_case(test_id)
    A = len(archs)
    print(T, A)
    print(f"{wT:.4f} {wA:.4f}")
    for (P, mon, mxon, moff, mxoff) in archs:
        print(P, mon, mxon, moff, mxoff)
    print(" ".join(str(x) for x in aggregate))


if __name__ == "__main__":
    main()
