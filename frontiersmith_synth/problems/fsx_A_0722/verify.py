#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for Ladder Filter Synthesis (fsx_A_0722).

Feasibility: participant output must be EXACTLY N whitespace-separated tokens, each
parseable as an integer index within the valid range for its slot type (odd slots index
into the L list, even slots into the C list). Any violation (wrong count, non-integer,
non-finite, out of range) -> Ratio: 0.0.

Objective (minimize): F = mean squared dB error between the participant's cascaded
response and the target curve, PLUS cost_per_component * (# populated components).
Baseline B = F of the all-unpopulated (bare resistor-divider) network, built by the
checker itself. Ratio = min(1.0, 0.1 * B / F)  (so reproducing the baseline -> ~0.1).
"""
import sys, math


def cascade_mag(components, Z0, freqs):
    out = []
    for f in freqs:
        w = 2.0 * math.pi * f
        A, B, C, D = complex(1, 0), complex(0, 0), complex(0, 0), complex(1, 0)
        for typ, val in components:
            if typ == 'L':
                a, b, c, d = complex(1, 0), complex(0, w * val), complex(0, 0), complex(1, 0)
            else:
                a, b, c, d = complex(1, 0), complex(0, 0), complex(0, w * val), complex(1, 0)
            A, B, C, D = A * a + B * c, A * b + B * d, C * a + D * c, C * b + D * d
        denom = A * Z0 + B + C * (Z0 * Z0) + D * Z0
        mag = 0.0 if abs(denom) < 1e-300 else abs(Z0 / denom)
        out.append(mag)
    return out


DB_ERR_CAP = 12.0  # a single wildly-off frequency point (deep stopband) can't swamp the score


def objective(indices, N, Z0, L_grid, C_grid, freqs, target_db, cost_per_component):
    components = []
    populated = 0
    for i in range(1, N + 1):
        idx = indices[i - 1]
        if i % 2 == 1:
            val = L_grid[idx]
            components.append(('L', val))
        else:
            val = C_grid[idx]
            components.append(('C', val))
        if idx != 0:
            populated += 1
    mags = cascade_mag(components, Z0, freqs)
    err = 0.0
    for m, td in zip(mags, target_db):
        db = 20.0 * math.log10(max(m, 1e-15))
        d = min(abs(db - td), DB_ERR_CAP)
        err += d * d
    err /= len(target_db)
    return err + cost_per_component * populated


def fail(msg):
    print("INVALID: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    with open(inf) as f:
        lines = f.read().split("\n")
    header = lines[0].split()
    N = int(header[0]); Z0 = float(header[1]); M = int(header[2]); cost_per_component = float(header[3])

    l_tok = lines[1].split()
    K_L = int(l_tok[0]); L_grid = [float(x) for x in l_tok[1:1 + K_L]]

    c_tok = lines[2].split()
    K_C = int(c_tok[0]); C_grid = [float(x) for x in c_tok[1:1 + K_C]]

    freqs = [float(x) for x in lines[3].split()]
    target_db = [float(x) for x in lines[4].split()]
    if len(freqs) != M or len(target_db) != M:
        fail("bad input freq/target length")

    try:
        with open(outf) as f:
            out_text = f.read()
    except Exception:
        fail("cannot read output")

    toks = out_text.split()
    if len(toks) != N:
        fail("expected exactly %d integer indices, got %d" % (N, len(toks)))

    indices = []
    for i, tok in enumerate(toks):
        try:
            v = int(tok)
        except ValueError:
            fail("token %d (%r) is not an integer" % (i + 1, tok))
        if v != v or v in (float('inf'), float('-inf')):
            fail("token %d is non-finite" % (i + 1,))
        slot_is_L = (i % 2 == 0)  # slot i+1: odd slots (1,3,5,...) are inductors
        K = K_L if slot_is_L else K_C
        if v < 0 or v >= K:
            fail("token %d = %d out of range [0,%d)" % (i + 1, v, K))
        indices.append(v)

    F = objective(indices, N, Z0, L_grid, C_grid, freqs, target_db, cost_per_component)
    if not math.isfinite(F):
        fail("non-finite objective")

    baseline_indices = [0] * N
    B = objective(baseline_indices, N, Z0, L_grid, C_grid, freqs, target_db, cost_per_component)
    if not math.isfinite(B) or B <= 0:
        B = 1e-9

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.8f B=%.8f" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
