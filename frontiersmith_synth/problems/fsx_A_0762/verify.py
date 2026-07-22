#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for tug-escort-chaining.

Feasibility gate + minimization score. Prints exactly one final `Ratio: <float in [0,1]>`
line and exits 0. The internal baseline B is a fixed round-robin team assignment (no
persistent-team insight, no idle-awareness) -- the SAME construction 'trivial.py' emits, so
a trivial submission scores Ratio ~= 0.1.
"""
import math
import sys
from fractions import Fraction

MAX_WAYPOINTS = 400
MAX_TIME = 10 ** 7
MAX_TOKENS_SAFETY = 200000


def parse_input(path):
    toks = open(path).read().split()
    it = iter(toks)
    T = int(next(it)); N = int(next(it)); L = int(next(it))
    pos = [int(next(it)) for _ in range(T)]
    coeff = int(next(it)); pen = int(next(it))
    jobs = []
    for _ in range(N):
        a = int(next(it)); b = int(next(it)); k = int(next(it))
        rel = int(next(it)); w = int(next(it)); nw = int(next(it))
        windows = [(int(next(it)), int(next(it))) for _ in range(nw)]
        jobs.append({"a": a, "b": b, "k": k, "release": rel, "weight": w,
                     "windows": windows, "dur": abs(b - a)})
    return T, N, L, pos, coeff, pen, jobs


def round_robin_baseline(T, N, L, pos, coeff, pen, jobs):
    """Checker's own reference plan: process jobs in LISTED order, assign a fixed
    round-robin block of tug ids (no geometry, no idle-awareness) each time."""
    free_time = [0] * T
    free_pos = list(pos)
    cursor = 0
    total_travel = 0
    total_delay = 0
    total_pen = 0
    for j in range(N):
        job = jobs[j]
        k = job["k"]
        chosen = [(cursor + i) % T for i in range(k)]
        cursor = (cursor + k) % T
        a, b, dur, rel = job["a"], job["b"], job["dur"], job["release"]
        arrival = max(free_time[t] + abs(free_pos[t] - a) for t in chosen)
        s_cand = max(rel, arrival)
        s_final = None
        for (o, c) in job["windows"]:
            s_try = max(s_cand, o)
            if s_try + dur <= c:
                s_final = s_try
                break
        if s_final is None:
            total_pen += job["weight"] * pen
            continue
        for t in chosen:
            total_travel += abs(free_pos[t] - a) + dur
            free_pos[t] = b
            free_time[t] = s_final + dur
        total_delay += job["weight"] * (s_final - rel)
    return coeff * total_travel + total_delay + total_pen


def fail(reason):
    print("INVALID: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    T, N, L, pos, coeff, pen, jobs = parse_input(inf)
    B = round_robin_baseline(T, N, L, pos, coeff, pen, jobs)

    try:
        out_toks = open(outf).read().split()
    except Exception:
        fail("cannot read output")
        return
    if len(out_toks) > MAX_TOKENS_SAFETY:
        fail("output too large")
        return

    it = iter(out_toks)

    def nxt_int(lo=None, hi=None):
        s = next(it)
        # reject non-finite / non-integer tokens explicitly (nan, inf, floats, garbage)
        if not (s.lstrip("+-").isdigit()):
            raise ValueError("not an integer token: %r" % s)
        v = int(s)
        if lo is not None and v < lo:
            raise ValueError("value %d below %d" % (v, lo))
        if hi is not None and v > hi:
            raise ValueError("value %d above %d" % (v, hi))
        return v

    try:
        itineraries = []
        for t in range(T):
            m = nxt_int(1, MAX_WAYPOINTS)
            wps = []
            prev_t = -1
            for i in range(m):
                tt = nxt_int(0, MAX_TIME)
                xx = nxt_int(0, L)
                if tt <= prev_t:
                    raise ValueError("waypoint times not strictly increasing")
                prev_t = tt
                wps.append((tt, xx))
            if wps[0][0] != 0:
                raise ValueError("itinerary must start at time 0")
            if wps[0][1] != pos[t]:
                raise ValueError("itinerary start position mismatch for tug %d" % t)
            for i in range(len(wps) - 1):
                dt = wps[i + 1][0] - wps[i][0]
                dx = abs(wps[i + 1][1] - wps[i][1])
                if dx > dt:
                    raise ValueError("tug %d exceeds max speed" % t)
            itineraries.append(wps)

        S = nxt_int(0, N)
        claims = {}
        for _ in range(S):
            j = nxt_int(1, N)
            if j in claims:
                raise ValueError("ship %d claimed twice" % j)
            s_j = nxt_int(0, MAX_TIME)
            k = jobs[j - 1]["k"]
            tug_ids = []
            for _ in range(k):
                tid = nxt_int(0, T - 1)
                tug_ids.append(tid)
            if len(set(tug_ids)) != k:
                raise ValueError("ship %d team has duplicate tugs" % j)
            claims[j] = (s_j, tug_ids)
    except (StopIteration, ValueError) as e:
        fail(str(e))
        return

    def pos_at(wps, t):
        if t <= wps[0][0]:
            return Fraction(wps[0][1])
        if t >= wps[-1][0]:
            return Fraction(wps[-1][1])
        for i in range(len(wps) - 1):
            t0, x0 = wps[i]
            t1, x1 = wps[i + 1]
            if t0 <= t <= t1:
                if t1 == t0:
                    return Fraction(x0)
                return Fraction(x0) + Fraction(x1 - x0, t1 - t0) * Fraction(t - t0)
        raise AssertionError("unreachable")

    # per-tug list of (start,end,job) intervals for overlap checking
    per_tug_intervals = [[] for _ in range(T)]
    total_delay = 0
    served = set()
    for j, (s_j, tug_ids) in claims.items():
        job = jobs[j - 1]
        a, b, dur, rel, w = job["a"], job["b"], job["dur"], job["release"], job["weight"]
        end = s_j + dur
        if s_j < rel:
            fail("ship %d starts before release" % j)
            return
        if not any(o <= s_j and end <= c for (o, c) in job["windows"]):
            fail("ship %d transit not inside a single tide window" % j)
            return
        for t in tug_ids:
            wps = itineraries[t]
            if pos_at(wps, s_j) != Fraction(a):
                fail("tug %d not at rendezvous %d at start of ship %d" % (t, a, j))
                return
            if pos_at(wps, end) != Fraction(b):
                fail("tug %d not at exit %d at end of ship %d" % (t, b, j))
                return
            per_tug_intervals[t].append((s_j, end, j))
        total_delay += w * (s_j - rel)
        served.add(j)

    for t in range(T):
        ivs = sorted(per_tug_intervals[t])
        for i in range(len(ivs) - 1):
            if ivs[i][1] > ivs[i + 1][0]:
                fail("tug %d double-booked on ships %d and %d" % (t, ivs[i][2], ivs[i + 1][2]))
                return

    total_travel = 0
    for t in range(T):
        wps = itineraries[t]
        for i in range(len(wps) - 1):
            total_travel += abs(wps[i + 1][1] - wps[i][1])

    total_pen = 0
    for j in range(1, N + 1):
        if j not in served:
            total_pen += jobs[j - 1]["weight"] * pen

    F = coeff * total_travel + total_delay + total_pen
    if F != F or F in (float("inf"), float("-inf")):
        fail("non-finite cost")
        return

    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    ratio = sc / 1000.0
    print("B=%s F=%s served=%d/%d" % (B, F, len(served), N))
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
