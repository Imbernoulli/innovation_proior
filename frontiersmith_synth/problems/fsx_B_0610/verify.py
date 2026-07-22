#!/usr/bin/env python3
# Deterministic checker for the catalyst reactor-farm schedule (format C, MAXIMIZE revenue).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored).
# Prints "... Ratio: <r>" with r in [0,1]; any feasibility violation -> Ratio: 0.0 .
import sys, math


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        lines = f.read().split("\n")
    hdr = lines[0].split()
    T = int(hdr[0]); R = int(hdr[1]); Q = int(hdr[2]); d = int(hdr[3]); L = int(hdr[4])
    e = [float(x) for x in lines[1].split()]
    p = [float(x) for x in lines[2].split()]
    cap = [float(x) for x in lines[3].split()]
    assert len(e) == L and len(p) == T and len(cap) == T
    return T, R, Q, d, L, e, p, cap


def marginal(e, w, L):
    return e[w] if w < L else e[L - 1]


def simulate(schedule, T, R, Q, d, L, e):
    """schedule: flat list of R*T integer tokens (reactor-major).
    Returns produced[t] pooled across reactors, or raises ValueError on infeasibility."""
    produced = [0.0] * T
    for r in range(R):
        row = schedule[r * T:(r + 1) * T]
        w = 0
        i = 0
        while i < T:
            tok = row[i]
            if tok == -1:
                j = i
                while j < T and row[j] == -1:
                    j += 1
                run = j - i
                if run % d != 0:
                    raise ValueError("regen run length %d not a multiple of d=%d" % (run, d))
                w = 0                       # catalyst reset after the offline block
                i = j
            else:
                if tok < 0 or tok > Q:
                    raise ValueError("throughput %d out of [0,%d]" % (tok, Q))
                s = 0.0
                for u in range(tok):
                    s += marginal(e, w + u, L)
                w += tok
                produced[i] += s
                i += 1
    return produced


def revenue(produced, p, cap):
    tot = 0.0
    for t in range(len(produced)):
        tot += p[t] * min(cap[t], produced[t])
    return tot


def main():
    try:
        T, R, Q, d, L, e, p, cap = read_instance(sys.argv[1])
    except Exception:
        fail("bad instance")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if len(otoks) != R * T:
        fail("expected %d tokens, got %d" % (R * T, len(otoks)))

    sched = []
    for tk in otoks:
        try:
            v = int(tk)
        except Exception:
            fail("non-integer token %r" % tk)
        if not math.isfinite(v):
            fail("non-finite token")
        if v < -1 or v > Q:
            fail("token %d out of range [-1,%d]" % (v, Q))
        sched.append(v)

    try:
        produced = simulate(sched, T, R, Q, d, L, e)
    except ValueError as ex:
        fail(str(ex))

    F = revenue(produced, p, cap)

    # internal trivial baseline B: every reactor runs at max throughput every step,
    # never regenerating (a valid, catalyst-agnostic construction).
    base_sched = [Q] * (R * T)
    B = revenue(simulate(base_sched, T, R, Q, d, L, e), p, cap)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
