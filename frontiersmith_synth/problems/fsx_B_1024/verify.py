#!/usr/bin/env python3
# Deterministic checker for coproduct-envelope-riding (format C, MINIMIZE cost).
# CLI: python3 verify.py <in> <out> <ans>   (ans ignored).
# Prints "... Ratio: <r>" with r in [0,1]; any feasibility breach -> Ratio: 0.0.
import sys, math

EPS = 2e-4


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def envelope(P, Qcap, alpha, beta, Gamma, delta, Epsilon):
    U = min(Qcap, (Gamma - alpha * P) / beta)
    if U < 0.0:
        U = 0.0
    L = delta * P - Epsilon
    if L < 0.0:
        L = 0.0
    if L > U:
        L = U
    return L, U


def is_finite(x):
    return isinstance(x, float) and math.isfinite(x)


def main():
    # ---- instance ------------------------------------------------------
    try:
        toks = open(sys.argv[1]).read().split()
    except Exception:
        fail("bad instance")
    p = 0
    try:
        T = int(toks[p]); p += 1
        Cap = float(toks[p]); p += 1
        S_init = float(toks[p]); p += 1
        Pmin = float(toks[p]); p += 1
        Pmax = float(toks[p]); p += 1
        Qcap = float(toks[p]); p += 1
        alpha = float(toks[p]); p += 1
        beta = float(toks[p]); p += 1
        Gamma = float(toks[p]); p += 1
        delta = float(toks[p]); p += 1
        Epsilon = float(toks[p]); p += 1
        b = float(toks[p]); p += 1
        kappa = float(toks[p]); p += 1
        dumpfee = float(toks[p]); p += 1
        DP = [0.0] * T
        DQ = [0.0] * T
        for t in range(T):
            DP[t] = float(toks[p]); p += 1
            DQ[t] = float(toks[p]); p += 1
    except Exception:
        fail("malformed instance")

    L_arr = [0.0] * T
    U_arr = [0.0] * T
    for t in range(T):
        L_arr[t], U_arr[t] = envelope(DP[t], Qcap, alpha, beta, Gamma, delta, Epsilon)

    # ---- participant output --------------------------------------------
    try:
        raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if len(raw) != 2 * T:
        fail("expected %d tokens (T lines of 'Q dump'), got %d" % (2 * T, len(raw)))
    Q = [0.0] * T
    Dump = [0.0] * T
    for t in range(T):
        try:
            qv = float(raw[2 * t])
            dv = float(raw[2 * t + 1])
        except Exception:
            fail("non-numeric token at period %d" % (t + 1))
        if not (is_finite(qv) and is_finite(dv)):
            fail("non-finite value at period %d" % (t + 1))
        Q[t] = qv
        Dump[t] = dv

    # ---- feasibility: envelope membership + tank chain + non-negative dump
    s = S_init
    if s < -EPS or s > Cap + EPS:
        fail("S_init outside tank bounds (instance bug)")
    total_fuel = 0.0
    for t in range(T):
        Lt, Ut = L_arr[t], U_arr[t]
        if Q[t] < Lt - EPS or Q[t] > Ut + EPS:
            fail("period %d: Q=%.6f outside envelope [%.6f, %.6f]" % (t + 1, Q[t], Lt, Ut))
        if Dump[t] < -EPS:
            fail("period %d: negative dump" % (t + 1))
        # Every raw value that passed a tolerance check above is CLAMPED to
        # its exact legal range before it is used in the balance/cost below.
        # Using the raw (still slightly out-of-range) value anywhere would
        # let a submission buy something for free with sub-tolerance noise:
        #   - a sub-zero dump would inject free heat into the tank balance
        #     while being credited zero cost;
        #   - a Q past U would flip the concave bump negative;
        #   - a tank level past Cap would silently drop the required dump.
        # Clamping first closes every one of those loopholes while still
        # tolerating genuine floating-point noise near a boundary.
        Q_eff = Lt if Q[t] < Lt else (Ut if Q[t] > Ut else Q[t])
        dump_eff = Dump[t] if Dump[t] > 0.0 else 0.0
        s_new = s + Q_eff - DQ[t] - dump_eff
        if s_new < -EPS or s_new > Cap + EPS:
            fail("period %d: tank level %.6f outside [0, %.6f]" % (t + 1, s_new, Cap))
        forced_extra_dump = max(0.0, s_new - Cap)  # charge any sub-tolerance overflow as dump
        s = min(Cap, max(0.0, s_new))
        bump = kappa * (Q_eff - Lt) * (Ut - Q_eff)
        total_fuel += b * Q_eff + bump + dumpfee * (dump_eff + forced_extra_dump)

    F = total_fuel

    # ---- checker's own baseline B: "always operate at the top envelope
    #      vertex U(t), dump forced overflow" -- always feasible because the
    #      generator guarantees DQ[t] <= U(t) for every t, so the running
    #      balance can never go negative under this policy.
    sb = S_init
    B = 0.0
    for t in range(T):
        Lt, Ut = L_arr[t], U_arr[t]
        qb = Ut
        raw = sb + qb - DQ[t]
        if raw < 0.0:
            raw = 0.0  # numerical guard; provably >=0 given DQ[t]<=Ut
        dumpb = max(0.0, raw - Cap)
        sb = min(Cap, raw)
        bumpb = kappa * (qb - Lt) * (Ut - qb)   # == 0 (qb is the top vertex)
        B += b * qb + bumpb + dumpfee * dumpb

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("coproduct-envelope-riding: F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
