#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for fsx_A_1171.

Rebuilds the FULL instance (public netlist + secret true resistances + secret held-out
excitation patterns) from testId alone via common_gen.build_instance -- the same function
gen.py used to print the public .in file, so this is bit-for-bit reproducible.

Participant output ("artifact"):
    k
    k lines: edge_index new_resistance_value

Scoring: forward-simulate the CLAIMED circuit (claimed edges at the claimed value, every
other edge at its nominal value) under the HELD-OUT excitation patterns (never shown to the
solver) and compare against the true held-out readings. F = (mean match fraction) /
max(1, k): reward explaining ALL held-out excitations with as FEW claimed components as
possible -- the minimum-cardinality-explanation mechanism. The checker's own trivial
baseline B is the same formula evaluated at the empty claim set (k=0, "nothing drifted").
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_gen import build_instance, make_readings, solve_nodal_float

TOL = 0.12
B_FLOOR = 0.05
MAX_VAL = 1.0e7


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        in_tokens = open(in_path).read().split()
        test_id = int(in_tokens[0])
    except Exception:
        fail("bad input file")

    try:
        inst = build_instance(test_id)
        _shown, held = make_readings(inst)
    except Exception:
        fail("instance rebuild failed")

    n = inst['n']
    edges = inst['edges']
    n_edges = inst['n_edges']

    try:
        out_tokens = open(out_path).read().split()
    except Exception:
        fail("no output file")
    if not out_tokens:
        fail("empty output")

    try:
        k = int(out_tokens[0])
    except Exception:
        fail("bad claim count")
    if k < 0 or k > n_edges:
        fail("claim count out of range")
    if len(out_tokens) != 1 + 2 * k:
        fail("token count mismatch (expected %d, got %d)" % (1 + 2 * k, len(out_tokens)))

    claims = {}
    for i in range(k):
        idx_tok = out_tokens[1 + 2 * i]
        val_tok = out_tokens[2 + 2 * i]
        try:
            idx = int(idx_tok)
            val = float(val_tok)
        except Exception:
            fail("unparsable claim token")
        if not math.isfinite(val):
            fail("non-finite claim value")
        if idx < 0 or idx >= n_edges:
            fail("edge index out of range: %d" % idx)
        if idx in claims:
            fail("duplicate edge index %d" % idx)
        if not (0.0 < val <= MAX_VAL):
            fail("claim value out of range: %r" % val)
        claims[idx] = val

    def predict(override, s, g, Q):
        R = [e[2] for e in edges]
        for idx, v in override.items():
            R[idx] = v
        V = solve_nodal_float(n, edges, R, s, g, Q)
        return V[s], V[g]

    vals = [Vs for (_s, _g, _Q, Vs, _Vg) in held]
    scale = max(math.sqrt(sum(v * v for v in vals) / max(1, len(vals))), 1e-6)

    def match_fraction(override):
        ms = []
        for (s, g, Q, Vs, Vg) in held:
            ps, pg = predict(override, s, g, Q)
            if not (math.isfinite(ps) and math.isfinite(pg)):
                ms.append(0.0)
                continue
            err = abs(ps - Vs) / scale
            ms.append(max(0.0, min(1.0, 1.0 - err / TOL)))
        return sum(ms) / len(ms)

    F_frac = match_fraction(claims)
    F = F_frac / max(1, k)
    B_frac = max(match_fraction({}), B_FLOOR)

    sc = min(1000.0, 100.0 * F / max(1e-9, B_frac))
    print("F=%.6f B=%.6f k=%d Ratio: %.6f" % (F_frac, B_frac, k, sc / 1000.0))


if __name__ == "__main__":
    main()
