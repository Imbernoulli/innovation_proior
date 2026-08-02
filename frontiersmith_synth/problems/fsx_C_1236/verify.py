#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for the circuit-breaker
tuning problem (family: circuit-breaker-tuning).

Input: a fixed timeline of T ticks, each tick t has a (pre-decided, given)
outcome o_t in {0,1} -- whether a call attempted at tick t would succeed.

Output (the artifact): 7 integers
    w_trip k_trip w_recover k_recover probe_base probe_num probe_den
defining a breaker policy:
  - failure-threshold: while CLOSED or HALF_OPEN, track the last w_trip call
    outcomes; if k_trip of them are failures, TRIP to OPEN.
  - half-open-probing: while OPEN, no traffic is sent except single spaced
    probes every `probe_interval` ticks; a successful probe moves to
    HALF_OPEN, a failed one keeps the same schedule.
  - flapping-suppression via asymmetric hysteresis: HALF_OPEN uses the SAME
    short trip window to bail back to OPEN fast on renewed failures, but
    needs a SEPARATE, independently-sized w_recover window with k_recover
    successes to fully CLOSE -- recovering is deliberately harder to satisfy
    than tripping (a brief lucky streak can satisfy a short window; it can't
    sustain a long one). If `probe_num>0`: every time a FRESH trip happens
    (from CLOSED), `probe_interval` for that whole episode is re-derived from
    the "signal gap" of the MOST RECENTLY fully-closed episode -- the number
    of ticks between that episode's trip and the first probe that showed any
    sign of life -- so an episode that tends to come back quickly gets probed
    soon, and one that tends to stay down a long time is not hammered with
    wasted probes (probe rate tied to observed recovery time). If
    `probe_num==0`, `probe_interval` is always `probe_base`, ignoring history.

Feasibility: exactly 7 finite-integer tokens, each in its declared range
(below). Any violation -> Ratio: 0.0.

Objective: F = sum over every attempted call (CLOSED, HALF_OPEN, or an
OPEN-state probe) of (+R if it succeeds, -CF if it fails). Skipped ticks
(OPEN, non-probe) contribute 0.

Baseline B: the checker's own reference is "always call, no breaker at all"
(equivalently: a breaker that never trips), computed directly from the
input timeline. sc = min(1000, 100*F/max(1e-9,B)); Ratio = sc/1000.
"""
import sys, math


def die0(reason):
    print(f"INFEASIBLE: {reason}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_input(path):
    with open(path) as f:
        lines = [ln for ln in f.read().split('\n') if ln.strip() != '']
    head = lines[0].split()
    T, R, CF = int(head[0]), int(head[1]), int(head[2])
    outcomes = list(map(int, lines[1].split()))
    return T, R, CF, outcomes


def _finite_int(tok):
    try:
        v = float(tok)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(v):
        return None
    if abs(v - round(v)) > 1e-9:
        return None
    return int(round(v))


def parse_output(path, T):
    with open(path) as f:
        toks = f.read().split()
    if len(toks) != 7:
        return None, f"expected 7 tokens, got {len(toks)}"
    vals = []
    for tok in toks:
        v = _finite_int(tok)
        if v is None:
            return None, "a token is not a finite integer"
        vals.append(v)
    w_trip, k_trip, w_recover, k_recover, probe_base, probe_num, probe_den = vals
    if not (1 <= w_trip <= T):
        return None, f"w_trip {w_trip} out of range [1,{T}]"
    if not (1 <= k_trip <= w_trip):
        return None, f"k_trip {k_trip} out of range [1,{w_trip}]"
    if not (1 <= w_recover <= T):
        return None, f"w_recover {w_recover} out of range [1,{T}]"
    if not (1 <= k_recover <= w_recover):
        return None, f"k_recover {k_recover} out of range [1,{w_recover}]"
    if not (1 <= probe_base <= T):
        return None, f"probe_base {probe_base} out of range [1,{T}]"
    if not (1 <= probe_den <= 1000):
        return None, f"probe_den {probe_den} out of range [1,1000]"
    if not (0 <= probe_num <= probe_den):
        return None, f"probe_num {probe_num} out of range [0,{probe_den}]"
    return (w_trip, k_trip, w_recover, k_recover, probe_base, probe_num, probe_den), None


def simulate(T, R, CF, outcomes, params):
    w_trip, k_trip, w_recover, k_recover, probe_base, probe_num, probe_den = params

    def scaled(duration):
        # round-half-up integer arithmetic only -- no floats, bit-exact.
        v = (probe_num * duration + probe_den // 2) // probe_den
        return max(1, min(T, v))

    state = 'CLOSED'
    trip_win = []          # list used as a small fixed-capacity FIFO
    recover_win = []
    open_start = None
    current_interval = probe_base
    next_probe_t = None
    last_signal_gap = None   # ticks-to-first-sign-of-life of the last CLOSED episode
    F = 0

    def push(win, cap, val):
        win.append(val)
        if len(win) > cap:
            del win[0]

    for t in range(1, T + 1):
        o = outcomes[t - 1]
        if state in ('CLOSED', 'HALF_OPEN'):
            F += R if o else -CF
            push(trip_win, w_trip, o)
            if state == 'HALF_OPEN':
                push(recover_win, w_recover, o)

            fails = trip_win.count(0)
            if fails >= k_trip:
                if state == 'CLOSED':
                    open_start = t
                    if probe_num > 0 and last_signal_gap is not None:
                        current_interval = scaled(last_signal_gap)
                    else:
                        current_interval = probe_base
                state = 'OPEN'
                next_probe_t = t + current_interval
                trip_win = []
                recover_win = []
            elif state == 'HALF_OPEN':
                if len(recover_win) >= w_recover and recover_win.count(1) >= k_recover:
                    state = 'CLOSED'
                    trip_win = []
                    recover_win = []
        else:  # OPEN
            if t == next_probe_t:
                F += R if o else -CF
                if o == 1:
                    last_signal_gap = t - open_start
                    state = 'HALF_OPEN'
                    trip_win = [1]
                    recover_win = [1]
                else:
                    next_probe_t = t + current_interval

    return F


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    T, R, CF, outcomes = read_input(in_path)

    parsed, err = parse_output(out_path, T)
    if parsed is None:
        die0(err)

    F = simulate(T, R, CF, outcomes, parsed)
    B = sum(R if o else -CF for o in outcomes)
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * F / B)
    print(f"F={F} B={B}")
    print(f"Ratio: {sc / 1000.0:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
