#!/usr/bin/env python3
"""Deterministic local scorer for "Dynamic Bin Packing with Rebalancing".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score (higher is better).

Scoring rule (see context.md "Evaluation settings")
----------------------------------------------------
  * The instance has N items and a per-bin capacity C.  Item i is alive during the
    half-open interval [a_i, d_i) and consumes size s_i while alive.
  * A SOLUTION is N integers b_0..b_{N-1}: b_i is the (non-negative) bin index that
    item i is placed into for its whole lifetime.
  * FEASIBILITY.  The output is INFEASIBLE (score 0) if any of these hold:
      - the file does not contain exactly N integers, or a token is not an integer;
      - some b_i < 0;
      - at some instant t some bin's alive load exceeds C (a capacity violation).
    Capacity is verified by a sweep line over arrival/departure events per bin: it
    suffices to check the load at every item's arrival time, because the load in a
    bin only increases at arrivals.
  * SCORE.  Let K = number of DISTINCT bin indices that the solution actually uses.
    Let B = the number of bins used by the deterministic FIRST-FIT-BY-ARRIVAL
    baseline (recomputed here, independent of the solver).  For a feasible solution
    with K >= 1, SCORE = round(1_000_000 * B / K).  The baseline scores exactly
    1_000_000; using fewer bins scores strictly more; using more scores less but
    stays positive.  Infeasible -> 0.

The scorer is self-contained and deterministic: it does not trust the solver and
recomputes the baseline B itself.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    N = int(next(it))
    C = int(next(it))
    a = [0] * N
    d = [0] * N
    s = [0] * N
    for i in range(N):
        a[i] = int(next(it))
        d[i] = int(next(it))
        s[i] = int(next(it))
    return N, C, a, d, s


def read_solution(path, N):
    """Return a list of N non-negative ints, or None if malformed."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if len(toks) != N:
        return None
    b = []
    for t in toks:
        try:
            v = int(t)
        except ValueError:
            return None
        if v < 0:
            return None
        b.append(v)
    return b


def feasible(N, C, a, d, s, b):
    """True iff no bin ever exceeds capacity C at any instant.

    For each bin we build the list of items assigned to it and run a sweep line
    over (+s at a_i, -s at d_i) events; the peak load is the max prefix sum.
    """
    from collections import defaultdict
    by_bin = defaultdict(list)
    for i in range(N):
        by_bin[b[i]].append(i)
    for items in by_bin.values():
        events = []  # (time, delta); process departures (-) before arrivals (+) at equal time
        for i in items:
            events.append((a[i], 1, s[i]))   # arrival
            events.append((d[i], 0, s[i]))   # departure (sort key 0 < 1 -> processed first)
        # sort by time, then departures (kind 0) before arrivals (kind 1) at the same time
        events.sort(key=lambda e: (e[0], e[1]))
        load = 0
        for (_, kind, sz) in events:
            if kind == 0:
                load -= sz
            else:
                load += sz
                if load > C:
                    return False
    return True


def first_fit_baseline(N, C, a, d, s):
    """Deterministic first-fit by arrival order.

    Process items in order of (arrival, departure, size, index).  Maintain, per
    bin, the list of (departure, size) of items placed there; an item fits in a bin
    iff at its arrival instant the bin's alive load + its size <= C.  Place it into
    the lowest-indexed bin where it fits, opening a new bin if none fits.
    Returns the number of bins used.
    """
    order = sorted(range(N), key=lambda i: (a[i], d[i], s[i], i))
    bins = []  # bins[k] = list of (departure, size)
    for i in order:
        placed = False
        ai, di, si = a[i], d[i], s[i]
        for k in range(len(bins)):
            load = 0
            for (dd, ss) in bins[k]:
                if dd > ai:  # still alive at time ai (half-open [arr, dep))
                    load += ss
            if load + si <= C:
                bins[k].append((di, si))
                placed = True
                break
        if not placed:
            bins.append([(di, si)])
    return len(bins)


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    N, C, a, d, s = read_instance(sys.argv[1])

    b = read_solution(sys.argv[2], N)
    if b is None:
        print(0)  # INFEASIBLE -> floored to 0
        return

    if N == 0:
        print(1_000_000)
        return

    if not feasible(N, C, a, d, s, b):
        print(0)  # capacity violation -> infeasible
        return

    K = len(set(b))
    if K < 1:
        print(0)
        return

    B = first_fit_baseline(N, C, a, d, s)
    score = int(round(1_000_000.0 * B / K))
    print(score)


if __name__ == "__main__":
    main()
