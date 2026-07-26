#!/usr/bin/env python3
"""gen.py <testId> -> prints one instance of "The Foresighted Referee" to stdout.

Deterministic: all randomness is seeded from testId only. Builds the registration/cut
trace as a sequence of ROUNDS: each round registers W players (W = a bounded "window",
much smaller than n) in RANDOM order ("random" cases), then cuts all W of them before
the next round starts. This keeps the live working set bounded (a real heap only ever
sees ~W elements at a time) while the *cumulative* roster still grows to n -- exactly
the regime that separates "pay for what a live container needs" (greedy) from "pay for
what THIS script needs to certify" (strong).

"trap" cases additionally plant a genuine regime change a live heap cannot see coming:
before the round-robin starts, a single MEGA-BLOCK of R >> W players registers, in
strictly monotone (ascending/descending) order, with ZERO cuts interleaved -- so a
real heap's internal live set balloons to size R for that stretch (each of its R
sift-ups/downs then costs O(log R), not O(log W)) before the block is fully drained.
An offline certificate is unaffected: linking a new player to its two immediate rating
neighbors costs the same O(1) regardless of how big the current live set is or how the
block is ordered. This is the trap: a textbook live-heap replay cannot "see" that the
mega-block is coming and cannot avoid paying its structural cost, while the certificate
approach never notices the block size at all.
"""
import sys, random

# (n, window, mode) ladder: small/sane -> large/adversarial. mode in {'random','trap'}.
CASES = [
    (30,   6,  'random'),
    (45,   7,  'trap'),
    (70,   7,  'random'),
    (100,  8,  'trap'),
    (160,  9,  'random'),
    (260,  10, 'trap'),
    (420,  11, 'random'),
    (650,  12, 'trap'),
    (900,  13, 'random'),
    (1300, 14, 'trap'),
]


def build(n, W, mode, seed):
    rnd = random.Random(seed)
    ratings_pool = rnd.sample(range(1, 50 * n + 1), n)  # distinct ratings
    ratings_order = []  # rating of the k-th player to register (index k-1)
    events = []
    idx = 0

    if mode == 'trap':
        R = min(n, max(2 * W, n // 8))
        mega = sorted(ratings_pool[:R])          # one strictly monotone mega-block
        ratings_order.extend(mega)
        idx = R
        events.extend(['E'] * R)
        events.extend(['C'] * R)                 # no cuts interleaved -- live set hits R

    toggle = (mode == 'trap')  # keep alternating monotone direction going for trap rounds
    while idx < n:
        w = min(W, n - idx)
        block = ratings_pool[idx: idx + w]
        if mode == 'trap':
            block = sorted(block, reverse=toggle)
            toggle = not toggle
        else:
            rnd.shuffle(block)
        ratings_order.extend(block)
        idx += w
        events.extend(['E'] * w)
        events.extend(['C'] * w)
    return ratings_order, events


def main():
    test_id = int(sys.argv[1])
    n, W, mode = CASES[(test_id - 1) % len(CASES)]
    seed = 1_000_003 * test_id + 97 * n + W
    ratings_order, events = build(n, W, mode, seed)
    q = events.count('C')
    out = []
    out.append(f"{n} {q}")
    out.append(' '.join(map(str, ratings_order)))
    out.append(' '.join(events))
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
