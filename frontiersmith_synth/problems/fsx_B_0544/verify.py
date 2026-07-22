#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>     (ans is ignored)

Deterministic grader for the pin-budget-drift-cache task.

INPUT (stdin instance):
    line1: testId
    line2: N E L p q Bmax
    line3: C_PIN C_L2 C_MISS C_SWAP
    line4: T
    line5..: T request keys (whitespace separated)

OUTPUT (participant, stdout):
    exactly E lines; line t = "m k1 ... km" = the set of bolted lamps for epoch t
    (0 <= m <= p, keys in [0,N), distinct within the line).

Feasibility (any violation -> Ratio: 0.0):
    * exactly E lines, every token a non-negative integer;
    * per line 0 <= m <= p, exactly m keys, keys distinct and in [0,N);
    * reconfiguration bandwidth: for t>=1, |pins[t] \\ pins[t-1]| <= Bmax
      (bolting a NEW lamp costs a slot of bandwidth; unbolting is free);
      epoch 0 is the free install (up to p lamps).

Cost replayed exactly:
    for each request in epoch t: pinned-hit = C_PIN; else L2-hit = C_L2 (LRU size q,
    managed only over NON-pinned accesses); else miss = C_MISS.
    plus C_SWAP per newly-bolted lamp (install at t=0 counts as |pins[0]| bolts).

Score (minimisation):  B = cost of the empty (bolt-nothing) schedule, replayed
identically.  F = participant total cost.  sc = min(1000, 100*B/F);
Ratio = sc/1000.  Bolt-nothing reproduces B exactly -> 0.1.
"""
import sys
from collections import OrderedDict


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    def nxt():
        return next(it)
    tid = int(nxt())
    N = int(nxt()); E = int(nxt()); L = int(nxt()); p = int(nxt()); q = int(nxt()); Bmax = int(nxt())
    cpin = int(nxt()); cl2 = int(nxt()); cmiss = int(nxt()); cswap = int(nxt())
    T = int(nxt())
    seq = [int(nxt()) for _ in range(T)]
    return dict(tid=tid, N=N, E=E, L=L, p=p, q=q, Bmax=Bmax,
                cpin=cpin, cl2=cl2, cmiss=cmiss, cswap=cswap, T=T, seq=seq)


def replay(inst, pins):
    """pins: list of E frozensets.  Returns total cost (access + swap)."""
    seq, L, q = inst["seq"], inst["L"], inst["q"]
    cpin, cl2, cmiss, cswap = inst["cpin"], inst["cl2"], inst["cmiss"], inst["cswap"]
    # swap charge
    cost = 0
    prev = frozenset()
    for t in range(inst["E"]):
        cost += cswap * len(pins[t] - prev)
        prev = pins[t]
    # access charge
    l2 = OrderedDict()
    for i, x in enumerate(seq):
        t = i // L
        if t >= inst["E"]:
            t = inst["E"] - 1
        if x in pins[t]:
            cost += cpin
        elif x in l2:
            cost += cl2
            l2.move_to_end(x)
        else:
            cost += cmiss
            l2[x] = 1
            if len(l2) > q:
                l2.popitem(last=False)
    return cost


def parse_output(path, inst):
    N, E, p, Bmax = inst["N"], inst["E"], inst["p"], inst["Bmax"]
    try:
        with open(path) as f:
            raw = f.read()
    except Exception:
        fail("cannot read output")
    lines = [ln for ln in raw.splitlines()]
    # strip trailing blank lines only
    while lines and lines[-1].strip() == "":
        lines.pop()
    if len(lines) != E:
        fail("expected %d lines, got %d" % (E, len(lines)))
    pins = []
    prev = frozenset()
    for t, ln in enumerate(lines):
        parts = ln.split()
        if not parts:
            fail("line %d empty (need count)" % t)
        toks = []
        for tk in parts:
            # strict integer parse -> rejects nan/inf/garbage
            try:
                v = int(tk)
            except ValueError:
                fail("non-integer token '%s' on line %d" % (tk[:20], t))
            toks.append(v)
        m = toks[0]
        keys = toks[1:]
        if m < 0 or m > p:
            fail("line %d: count %d out of [0,%d]" % (t, m, p))
        if len(keys) != m:
            fail("line %d: declared %d keys, found %d" % (t, m, len(keys)))
        s = set()
        for k in keys:
            if k < 0 or k >= N:
                fail("line %d: key %d out of [0,%d)" % (t, k, N))
            if k in s:
                fail("line %d: duplicate key %d" % (t, k))
            s.add(k)
        fs = frozenset(s)
        added = fs - prev
        if t >= 1 and len(added) > Bmax:
            fail("line %d: %d new bolts > bandwidth %d" % (t, len(added), Bmax))
        pins.append(fs)
        prev = fs
    return pins


def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    inst = read_instance(sys.argv[1])
    pins = parse_output(sys.argv[2], inst)

    F = replay(inst, pins)
    B = replay(inst, [frozenset() for _ in range(inst["E"])])
    if F <= 0:
        fail("non-positive cost")
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("cost=%d baseline=%d  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
