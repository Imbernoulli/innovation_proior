#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training log to stdout.

Decoding a retired traffic engineer's intersection controller.

A hidden deterministic controller picks, every cycle, ONE approach at a
junction to receive the green light.  Each approach has a queue length `q`
(vehicles waiting), a static weight `w` (lane-count / arterial importance),
and an age `a` (cycles since it last got green).  `cw` is how many steps
clockwise the approach sits from the approach that got the PREVIOUS green
(0 = immediately next in rotation).  The controller's rule -- how q, w, a and
the clockwise order combine, and whether there is any special-case override
to stop an approach waiting forever -- is NOT printed anywhere; it must be
induced from the traces below.

Training traces are logged ONLY at 3-way and 4-way junctions.  The graded
extrapolation split (regenerated only inside the checker) uses 5-way and
6-way junctions that never appear here -- a rule keyed to features of a
specific junction degree will not transfer.

STDOUT prints ONLY: a header, then a stream of STATE blocks (junction
snapshot -> which approach actually got the green).  The hidden law, its
constants, and the RNG seed derivation are never printed -- data rows only.
"""
import sys, random

STARVE_LIMIT = 6      # cycles of age at which an approach is force-served
W_MIN, W_MAX = 1, 3
ARRIVAL_MAX = 3
TESTID_SALT = 900001


def decide(q, w, a, last_green, N):
    """Hidden controller (identical logic lives in verify.py, never imported)."""
    def cw(i):
        return (i - last_green - 1) % N
    overriders = [i for i in range(N) if a[i] >= STARVE_LIMIT]
    if overriders:
        maxage = max(a[i] for i in overriders)
        cands = [i for i in overriders if a[i] == maxage]
    else:
        maxscore = max(w[i] * q[i] for i in range(N))
        cands = [i for i in range(N) if w[i] * q[i] == maxscore]
    winner = min(cands, key=cw)
    return winner, [cw(i) for i in range(N)]


def bulk_episode(rng, N, T):
    """Simulate one running junction: arrivals each cycle, then a decision."""
    w = [rng.randint(W_MIN, W_MAX) for _ in range(N)]
    q = [0] * N
    a = [0] * N
    lg = -1
    rows = []
    for _ in range(T):
        for i in range(N):
            q[i] += rng.randint(0, ARRIVAL_MAX)
        winner, cwlist = decide(q, w, a, lg, N)
        rows.append((N, lg, winner, list(q), list(w), list(a), cwlist))
        q[winner] = 0
        a[winner] = 0
        for i in range(N):
            if i != winner:
                a[i] += 1
        lg = winner
    return rows


def age_probe(rng, N):
    """Matched pair: IDENTICAL (q,w); only one approach's age crosses the
    starvation threshold.  Isolates the override, independent of q,w."""
    q = [rng.randint(6, 10)] + [rng.randint(1, 3) for _ in range(N - 1)]
    w = [1] * N
    lg = rng.randint(-1, N - 1)
    j = rng.randint(1, N - 1)
    a_a = [0] * N
    a_a[j] = STARVE_LIMIT - 1
    win_a, cw_a = decide(q, w, a_a, lg, N)
    a_b = list(a_a)
    a_b[j] = STARVE_LIMIT
    win_b, cw_b = decide(q, w, a_b, lg, N)
    return [(N, lg, win_a, list(q), list(w), a_a, cw_a),
            (N, lg, win_b, list(q), list(w), a_b, cw_b)]


def weight_probe(rng, N):
    """Matched pair: IDENTICAL q, a=0 (no override); only one approach's
    weight is boosted.  Isolates the weight multiplier on the queue score."""
    base = rng.randint(1, 3)
    q = [base * (N - i) for i in range(N)]   # strictly descending
    a = [0] * N
    lg = rng.randint(-1, N - 1)
    k = rng.randint(1, N - 1)
    w_flat = [1] * N
    win_flat, cw_flat = decide(q, w_flat, a, lg, N)
    boost = q[0] // q[k] + 2
    w_boost = [1] * N
    w_boost[k] = boost
    win_boost, cw_boost = decide(q, w_boost, a, lg, N)
    return [(N, lg, win_flat, list(q), w_flat, list(a), cw_flat),
            (N, lg, win_boost, list(q), w_boost, list(a), cw_boost)]


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rng = random.Random(TESTID_SALT + t * 7919)

    n_bulk = 3 + t
    T_bulk = 12 + t
    n_age = 2 + t // 3
    n_wt = 2 + t // 3

    rows = []
    for _ in range(n_bulk):
        N = rng.choice([3, 4])
        rows.extend(bulk_episode(rng, N, T_bulk))
    for _ in range(n_age):
        N = rng.choice([3, 4])
        rows.extend(age_probe(rng, N))
    for _ in range(n_wt):
        N = rng.choice([3, 4])
        rows.extend(weight_probe(rng, N))

    rng.shuffle(rows)

    out = ["TESTID %d" % t, "NSTATES %d" % len(rows)]
    for (N, lg, winner, q, w, a, cw) in rows:
        out.append("STATE N=%d LASTGREEN=%d WINNER=%d" % (N, lg, winner))
        out.append("Q " + " ".join(str(x) for x in q))
        out.append("W " + " ".join(str(x) for x in w))
        out.append("A " + " ".join(str(x) for x in a))
        out.append("CW " + " ".join(str(x) for x in cw))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
