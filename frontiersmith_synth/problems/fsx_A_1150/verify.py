#!/usr/bin/env python3
# Deterministic checker for "Hospital Utility Plant: Steam, Chill, and Power" (format C,
# minimize total primary-fuel input). CLI: python3 verify.py <in> <out> <ans>  (ans ignored)
# Prints "... Ratio: <r>" with r in [0,1] on the LAST line, exits 0.
import sys
import math

TOL = 1e-6
SAFETY = 2.1  # internal trivial baseline: dedicated converters, generously oversized (unoptimized)


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def is_finite(x):
    return x == x and x not in (float("inf"), float("-inf"))


def read_instance(path):
    try:
        toks = open(path).read().split()
    except Exception:
        fail("cannot read instance")
    p = 0
    T = int(toks[p]); p += 1
    a_b = float(toks[p]); p += 1
    c_b = float(toks[p]); p += 1
    Cap_b = float(toks[p]); p += 1
    eps_p = float(toks[p]); p += 1
    eps_s = float(toks[p]); p += 1
    cop_abs = float(toks[p]); p += 1
    cop_elec = float(toks[p]); p += 1
    a_g = float(toks[p]); p += 1
    c_g = float(toks[p]); p += 1
    S, Pw, Ch = [], [], []
    for t in range(T):
        S.append(int(toks[p])); p += 1
        Pw.append(int(toks[p])); p += 1
        Ch.append(int(toks[p])); p += 1
    return dict(T=T, a_b=a_b, c_b=c_b, Cap_b=Cap_b, eps_p=eps_p, eps_s=eps_s,
                cop_abs=cop_abs, cop_elec=cop_elec, a_g=a_g, c_g=c_g, S=S, Pw=Pw, Ch=Ch)


def fuel_cost(a_b, c_b, a_g, c_g, b, e_grid):
    return a_b * b + c_b * b * b + a_g * e_grid + c_g * e_grid * e_grid


def score_schedule(inst, rows):
    """rows: list of (b,x,z,e_chill,e_grid). Returns total fuel F, or None if infeasible
    (with a printed reason via fail(), which exits)."""
    eps_p, eps_s = inst['eps_p'], inst['eps_s']
    cop_abs, cop_elec = inst['cop_abs'], inst['cop_elec']
    a_b, c_b, a_g, c_g = inst['a_b'], inst['c_b'], inst['a_g'], inst['c_g']
    Cap_b = inst['Cap_b']
    S, Pw, Ch = inst['S'], inst['Pw'], inst['Ch']
    F = 0.0
    for t in range(inst['T']):
        b, x, z, e_chill, e_grid = rows[t]
        for name, v in (('b', b), ('x', x), ('z', z), ('e_chill', e_chill), ('e_grid', e_grid)):
            if not is_finite(v):
                fail("non-finite %s at t=%d" % (name, t))
        if b < -TOL or x < -TOL or z < -TOL or e_chill < -TOL or e_grid < -TOL:
            fail("negative value at t=%d" % t)
        if b > Cap_b + TOL:
            fail("boiler output exceeds capacity at t=%d" % t)
        if x > b + TOL:
            fail("turbine draw exceeds boiler output at t=%d" % t)
        if z > eps_s * x + TOL:
            fail("absorption chiller draws more LP steam than the turbine produced at t=%d" % t)
        b = max(0.0, b); x = max(0.0, min(x, b)); z = max(0.0, min(z, eps_s * x))
        e_chill = max(0.0, e_chill); e_grid = max(0.0, e_grid)

        steam_supplied = b - x
        if steam_supplied < S[t] - TOL:
            fail("steam demand unmet at t=%d (%.6f < %d)" % (t, steam_supplied, S[t]))
        w = eps_p * x
        elec_supplied = w + e_grid
        elec_needed = Pw[t] + e_chill
        if elec_supplied < elec_needed - TOL:
            fail("electricity demand unmet at t=%d (%.6f < %.6f)" % (t, elec_supplied, elec_needed))
        chill_supplied = cop_abs * z + cop_elec * e_chill
        if chill_supplied < Ch[t] - TOL:
            fail("chill demand unmet at t=%d (%.6f < %d)" % (t, chill_supplied, Ch[t]))

        F += fuel_cost(a_b, c_b, a_g, c_g, b, e_grid)
    return F


def internal_baseline(inst):
    """Fully-dedicated construction with a SAFETY margin: boiler exactly covers steam
    (no turbine use), electric chiller exactly covers chill, grid covers the rest --
    every quantity inflated by SAFETY. Always feasible; deliberately not tightly sized."""
    a_b, c_b, a_g, c_g = inst['a_b'], inst['c_b'], inst['a_g'], inst['c_g']
    cop_elec = inst['cop_elec']
    F = 0.0
    for t in range(inst['T']):
        b = SAFETY * inst['S'][t]
        e_chill = SAFETY * inst['Ch'][t] / cop_elec
        e_grid = SAFETY * (inst['Pw'][t] + e_chill)
        F += fuel_cost(a_b, c_b, a_g, c_g, b, e_grid)
    return F


def main():
    inst = read_instance(sys.argv[1])
    T = inst['T']
    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    need = 5 * T
    if len(otoks) != need:
        fail("expected exactly %d numbers (5 per timestep x %d timesteps), got %d" %
             (need, T, len(otoks)))
    rows = []
    for t in range(T):
        vals = []
        for k in range(5):
            try:
                v = float(otoks[5 * t + k])
            except Exception:
                fail("unparsable number at t=%d" % t)
            vals.append(v)
        rows.append(tuple(vals))

    F = score_schedule(inst, rows)
    if F is None or F != F or F <= 0.0:
        fail("degenerate objective")

    B = internal_baseline(inst)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
