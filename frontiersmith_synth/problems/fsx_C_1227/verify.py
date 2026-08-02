#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for the per-line adaptive
cache-coherence problem. Prints 'Ratio: <float in [0,1]>' on its own final line.
Minimization objective: fewer interconnect-message + miss-cost units is better.
"""
import sys
import math


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = toks[pos]
        pos += 1
        return v

    T = int(nxt())
    C = int(nxt())
    L = int(nxt())
    MISS = float(nxt())
    INVC = float(nxt())
    UPDC = float(nxt())
    events_by_line = [[] for _ in range(L)]
    for _ in range(T):
        c = int(nxt())
        l = int(nxt())
        op = nxt()
        events_by_line[l].append((c, op))
    return C, L, MISS, INVC, UPDC, events_by_line


def decide_mode(mode, W, Tt, idx, events):
    if mode != "ADAPT":
        return mode
    lo = max(0, idx - W)
    window = events[lo:idx]
    if not window:
        return "INV"
    reads = sum(1 for (_c, op) in window if op == "R")
    frac = reads / len(window)
    return "UPD" if frac >= Tt else "INV"


def simulate_line(events, MISS, INVC, UPDC, mode, W=None, Tt=None):
    valid = set()
    cost = 0.0
    for idx, (c, op) in enumerate(events):
        if op == "R":
            if c not in valid:
                cost += MISS
                valid.add(c)
        else:  # write
            m = decide_mode(mode, W, Tt, idx, events)
            others = valid - {c}
            if m == "INV":
                cost += INVC * len(others)
                valid = {c}
            else:  # UPD
                cost += UPDC * len(others)
                valid.add(c)
    return cost


def parse_output(path, L):
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception as e:
        fail("cannot read output: %s" % e)

    oi = 0

    def onext():
        nonlocal oi
        if oi >= len(toks):
            raise ValueError("unexpected end of output")
        v = toks[oi]
        oi += 1
        return v

    policies = []
    try:
        for _li in range(L):
            mode = onext()
            wtok = onext()
            ttok = onext()
            if mode not in ("INV", "UPD", "ADAPT"):
                raise ValueError("illegal policy token %r" % mode)
            # W and T are REQUIRED and validated on every line (even INV/UPD,
            # which merely ignore their values) -- there is no shorthand.
            try:
                W = int(wtok)
            except ValueError:
                raise ValueError("W is not an integer: %r" % wtok)
            try:
                Tt = float(ttok)
            except ValueError:
                raise ValueError("T is not a float: %r" % ttok)
            if not math.isfinite(Tt):
                raise ValueError("T is not finite: %r" % ttok)
            if W < 1 or W > 5000:
                raise ValueError("W out of range: %d" % W)
            if Tt < 0.0 or Tt > 1.0:
                raise ValueError("T out of range: %f" % Tt)
            policies.append((mode, W, Tt))
        if oi != len(toks):
            raise ValueError("trailing tokens in output (%d extra)" % (len(toks) - oi))
    except (ValueError, IndexError) as e:
        fail(str(e))
    return policies


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    C, L, MISS, INVC, UPDC, events_by_line = read_instance(inp)
    policies = parse_output(outp, L)

    F = 0.0
    for li in range(L):
        mode, W, Tt = policies[li]
        F += simulate_line(events_by_line[li], MISS, INVC, UPDC, mode, W, Tt)

    B = 0.0
    for li in range(L):
        B += simulate_line(events_by_line[li], MISS, INVC, UPDC, "INV")

    if not math.isfinite(F) or F < 0:
        fail("non-finite or negative total cost")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
