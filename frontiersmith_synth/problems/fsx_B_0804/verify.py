#!/usr/bin/env python3
"""Deterministic checker for shared-humidity-coat-staggering (fsx_B_0804).

Simulates the humidity-coupled drying process EXACTLY (rational arithmetic,
Fraction) given a fixed schedule of application times, checks feasibility
(precedence + range), computes the makespan F, and normalizes against the
checker's own fully-sequential baseline B (minimization convention).
"""
import sys
from fractions import Fraction as Fr

T_MAX = 10 ** 9
MAX_TOKLEN = 40


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def simulate(n, base, apply_time, prec, c):
    """Given fixed application times, resolve drying dynamics exactly.
    Drying rate of every currently-wet coat is divided by (1+c*W)^2, W =
    number of coats currently wet (applied but not yet dry).  Returns
    (finish_list, ok, msg)."""
    times_sorted = sorted(set(apply_time))
    ai = 0
    m = len(times_sorted)
    wet = {}
    finish = [None] * n
    cur = Fr(0)
    guard = 0
    guard_max = 4 * n + 20
    while wet or ai < m:
        guard += 1
        if guard > guard_max:
            return finish, False, "simulation did not terminate"
        w = len(wet)
        next_apply = times_sorted[ai] if ai < m else None
        if wet:
            mult = (1 + c * w) * (1 + c * w)
            next_complete_dt = min(r * mult for r in wet.values())
            next_complete = cur + next_complete_dt
        else:
            mult = None
            next_complete = None
        if next_apply is None and next_complete is None:
            break
        if next_complete is None or (next_apply is not None and next_apply <= next_complete):
            nxt = next_apply
        else:
            nxt = next_complete
        dt = nxt - cur
        if wet:
            for k in list(wet.keys()):
                wet[k] = wet[k] - dt / mult
        cur = nxt
        done = [k for k, r in wet.items() if r <= 0]
        for k in done:
            finish[k] = cur
            del wet[k]
        if next_apply is not None and cur == next_apply:
            for i in range(n):
                if apply_time[i] == cur and finish[i] is None and i not in wet:
                    wet[i] = Fr(base[i])
            ai += 1
    for i in range(n):
        if finish[i] is None:
            return finish, False, "coat %d never completes drying" % i
    for i in range(n):
        p = prec[i]
        if p != -1 and apply_time[i] < finish[p]:
            return finish, False, ("coat %d applied at %s before predecessor %d dries at %s"
                                    % (i, apply_time[i], p, finish[p]))
    return finish, True, ""


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    try:
        outtoks = open(sys.argv[2]).read().split()
    except Exception:
        outtoks = []

    it = iter(inp)
    try:
        P = int(next(it))
        cn = int(next(it))
        cd = int(next(it))
        if P <= 0 or cn <= 0 or cd <= 0:
            fail("bad header")
        c = Fr(cn, cd)
        base = []
        prec = []
        for _p in range(P):
            k = int(next(it))
            if k <= 0:
                fail("bad chain length")
            start = len(base)
            for j in range(k):
                b = int(next(it))
                if b <= 0:
                    fail("bad base time")
                base.append(b)
                prec.append(start + j - 1 if j > 0 else -1)
    except Exception:
        fail("bad input")
    n = len(base)
    if n == 0:
        fail("empty instance")

    # ---- internal baseline B: fully-sequential (one coat wet at a time) ----
    triv_apply = [None] * n
    cur = Fr(0)
    seq_mult = (1 + c) * (1 + c)
    for i in range(n):
        triv_apply[i] = cur
        cur = cur + Fr(base[i]) * seq_mult
    bfin, bok, bmsg = simulate(n, base, triv_apply, prec, c)
    if not bok:
        fail("internal baseline error: %s" % bmsg)
    B = max(bfin)
    if B <= 0:
        B = Fr(1)

    # ---- parse participant output ----
    if len(outtoks) != n:
        fail("expected %d application-time tokens, got %d" % (n, len(outtoks)))
    apply_time = []
    for tok in outtoks:
        if len(tok) > MAX_TOKLEN:
            fail("token too long")
        try:
            v = Fr(tok)
        except Exception:
            fail("unparsable token %r" % tok)
        if v < 0 or v > T_MAX:
            fail("out-of-range time %r" % tok)
        apply_time.append(v)

    fin, ok, msg = simulate(n, base, apply_time, prec, c)
    if not ok:
        fail(msg)
    F = max(fin)

    sc = min(1000.0, 100.0 * float(B) / max(1e-9, float(F)))
    print("B=%s F=%s Ratio: %.6f" % (str(B), str(F), sc / 1000.0))


if __name__ == "__main__":
    main()
