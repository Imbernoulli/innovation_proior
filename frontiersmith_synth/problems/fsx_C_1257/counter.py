#!/usr/bin/env python3
"""
counter.py <in> <out> <ans> -- deterministic scorer for the sensor-sampling-schedule problem.

1. Parses the instance: T, E, e_full, e_cheap, lead, Lmax, Cmax, K, then K streams of
   (event slots, precursor-active slots).
2. Parses the participant's policy line `P0 Pc P1 W` (four integers); validates STRICTLY:
   well-formed integer tokens (garbage/empty/huge/nan/inf/non-integer -> Ratio: 0.0),
   1<=P0<=10**6, 0<=Pc<=10**6, 1<=P1<=10**6, 0<=W<=10**6.
3. Replays the SAME fixed simulation engine, independently, on every one of the K streams
   (each stream starts with a fresh energy budget E): a channel access is simply skipped once
   that stream's remaining budget can't afford it (self-limiting, never "infeasible").
4. F = total events detected (summed over all K streams). Baseline B = what the checker's own
   trivial "spend half the budget on one fixed, precursor-blind rate" construction detects.
   Maximization ratio: sc = min(1000, 100*F/max(1e-9,B)); print("Ratio: %.6f" % (sc/1000)).
"""
import sys

MAXTOK = 10 ** 6


def fail(reason):
    print(f"Ratio: 0.0  # {reason}")
    sys.exit(0)


def simulate(P0, Pc, P1, W, T, E, e_full, e_cheap, events, precursors):
    """Fixed judge algorithm: run policy (P0,Pc,P1,W) against one stream. Returns detected count."""
    ev = events          # sorted tuple of event slots
    pr = precursors       # sorted tuple of precursor-active slots
    ev_set = ev if isinstance(ev, set) else set(ev)
    pr_set = pr if isinstance(pr, set) else set(pr)

    energy = E
    escalate_end = -1
    next_full = 0
    next_prec = 0
    detected = 0

    for t in range(T):
        escalated = t <= escalate_end
        if t >= next_full:
            if energy >= e_full:
                energy -= e_full
                if t in ev_set:
                    detected += 1
            period = P1 if escalated else P0
            next_full = t + max(1, period)
        if (not escalated) and Pc > 0 and t >= next_prec:
            if energy >= e_cheap:
                energy -= e_cheap
                if t in pr_set:
                    end = t + W
                    if end > escalate_end:
                        escalate_end = end
                    if next_full > t:
                        next_full = t + 1
            next_prec = t + Pc
    return detected


def main():
    if len(sys.argv) != 4:
        fail("bad invocation")
    in_path, out_path, _ans_path = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(in_path) as f:
        itoks = f.read().split()
    ip = iter(itoks)

    def inext():
        return next(ip)

    try:
        T = int(inext())
        E = int(inext())
        e_full = int(inext())
        e_cheap = int(inext())
        lead = int(inext())
        Lmax = int(inext())
        Cmax = int(inext())
        K = int(inext())
        if T <= 0 or E < 0 or e_full <= 0 or e_cheap <= 0 or lead < 0 or Lmax < 0 or Cmax < 0 or K <= 0:
            fail("bad instance header")
        streams = []
        for _ in range(K):
            ne = int(inext())
            evs = tuple(int(inext()) for _ in range(ne))
            npv = int(inext())
            prs = tuple(int(inext()) for _ in range(npv))
            streams.append((evs, prs))
    except (StopIteration, ValueError):
        fail("malformed instance (should not happen)")

    # ---- parse participant output (untrusted) ----
    try:
        with open(out_path) as f:
            otoks = f.read().split()
    except OSError:
        fail("cannot read output")

    if len(otoks) != 4:
        fail("expected exactly 4 tokens: P0 Pc P1 W")

    def parse_int(tok):
        # strict integer parse: rejects nan/inf/floats/garbage (no float() detour)
        s = tok.strip()
        if s[0] in "+-":
            body = s[1:]
        else:
            body = s
        if not body.isdigit():
            raise ValueError("not an integer token")
        return int(s)

    try:
        P0 = parse_int(otoks[0])
        Pc = parse_int(otoks[1])
        P1 = parse_int(otoks[2])
        W = parse_int(otoks[3])
    except ValueError:
        fail("non-integer/non-finite token in policy")

    if not (1 <= P0 <= MAXTOK):
        fail("P0 out of range")
    if not (0 <= Pc <= MAXTOK):
        fail("Pc out of range")
    if not (1 <= P1 <= MAXTOK):
        fail("P1 out of range")
    if not (0 <= W <= MAXTOK):
        fail("W out of range")

    # ---- F: submitted policy replayed on every stream ----
    F = 0
    for (evs, prs) in streams:
        F += simulate(P0, Pc, P1, W, T, E, e_full, e_cheap, evs, prs)

    # ---- B: checker's own trivial construction -- spend (up to) the WHOLE budget on one
    #         fixed rate spread over the horizon, precursor-blind, using CEILING spacing
    #         (an independent naive formula from anything a solution tier uses) ----
    n_full = max(1, E // e_full)
    P0_B = -(-T // n_full)  # ceil(T / n_full)
    B = 0
    for (evs, prs) in streams:
        B += simulate(P0_B, 0, P0_B, 0, T, E, e_full, e_cheap, evs, prs)

    if B <= 0:
        fail("degenerate instance (baseline detects nothing)")

    sc = min(1000.0, 100.0 * float(F) / max(1e-9, float(B)))
    ratio = sc / 1000.0
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


if __name__ == "__main__":
    main()
