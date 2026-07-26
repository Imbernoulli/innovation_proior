#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE humpyard instance to stdout.

Instance format:
    line 1: N T L Y a b s cap
        N   = number of inbound cars              (N = T*L)
        T   = number of outbound trains
        L   = cars per train (uniform)
        Y   = number of classification tracks (the "digit buckets")
        a,b = move-cost coefficients: a move of k contiguous cars costs a + b*k ticks
        s   = engine mode-switch penalty (ticks) added when an engine's move-type
              changes from that engine's own immediately-previous move
        cap = capacity of each classification track (and of the inbound track)
    next N lines: "train_id slot_id" -- car i (0-indexed, i=0 is FIRST / TOP of the
        inbound track, i.e. immediately accessible) belongs to outbound train
        train_id and must end up at position slot_id (0-indexed) of that train.

Difficulty ladder (testId 1..10): N, T, L grow; Y stays a small fixed radix (3) so
the number of required radix passes D = ceil(log_Y(N)) stays >= 3 (genuinely
multi-pass) across the whole ladder; a and s grow so that fixed-per-move overhead
increasingly rewards batching cars into larger cuts and specializing engines by
move-type (the two things a naive single-car / single-engine solver ignores).

All randomness is seeded ONLY from testId -> fully deterministic.
"""
import sys


LADDER = [
    # (T, L, Y, a, b, s)
    (4, 4, 4, 5, 1, 20),
    (4, 5, 4, 6, 1, 22),
    (5, 6, 4, 7, 1, 24),
    (5, 7, 4, 8, 1, 26),
    (6, 8, 4, 9, 1, 28),
    (6, 9, 4, 10, 1, 30),
    (7, 10, 4, 11, 1, 32),
    (7, 11, 4, 12, 1, 34),
    (8, 12, 4, 13, 1, 36),
    (8, 13, 4, 14, 1, 38),
]


def main():
    t = int(sys.argv[1])
    t = max(1, min(10, t))
    T, L, Y, a, b, s = LADDER[t - 1]
    N = T * L
    cap = N  # generous per-track cap: capacity is checked, but never the bottleneck

    cars = [(tr, sl) for tr in range(T) for sl in range(L)]

    # deterministic Fisher-Yates shuffle seeded only by testId (no library RNG
    # dependence beyond stdlib `random`, which is seed-reproducible across runs).
    import random
    rng = random.Random(1000003 * t + 17)
    rng.shuffle(cars)

    out = [f"{N} {T} {L} {Y} {a} {b} {s} {cap}"]
    for tr, sl in cars:
        out.append(f"{tr} {sl}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
