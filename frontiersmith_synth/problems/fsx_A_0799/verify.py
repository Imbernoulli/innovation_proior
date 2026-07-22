#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>   (ans is an empty placeholder, ignored)

"Drum-corps bridge" comb-dodging-stiffness checker (format C, minimize).

Instance (<in>):
    S BUDGET
    eps
    F
    f_1 w_1
    ...
    f_F w_F
  S = number of stiffness segments (springs) in a fixed-fixed 1-D chain of S-1
  point masses (mass=1 each); BUDGET = total integer stiffness units that MUST be
  allocated across the S segments; eps = scoring resolution; (f_i, w_i) = the
  drum corps' forcing comb (frequency, integer weight).

Artifact (<out>): S integers k_1 .. k_S -- the per-segment stiffness allocation.

Feasibility (STRICT):
  * exactly S tokens, each a finite integer;
  * every k_j >= 1;
  * sum(k_j) == BUDGET exactly.
  ANY violation -> Ratio: 0.0

Physics: the stiffness allocation k_1..k_S defines the (S-1)x(S-1) tridiagonal
stiffness matrix K of the fixed-fixed mass-spring chain (unit masses):
    K[i][i]   = k_i + k_{i+1}
    K[i][i+1] = K[i+1][i] = -k_{i+1}
Eigenfrequencies omega_j = sqrt(eigenvalue_j(K)), computed deterministically via
symmetric eigendecomposition.

Objective (minimize): total resonant amplification
    F_obj = sum_f  w_f / ( min_j |f - omega_j| + eps )
Internal baseline B: the SAME formula evaluated at the perfectly UNIFORM
allocation k_j = BUDGET / S for all j (the "obvious" fix -- and, by
construction, the exact spectrum the comb was planted on, so B is a large,
easily-beaten "sitting duck" reference).
    sc = min(1000, 100 * B / max(1e-9, F_obj))   (minimization)
    Ratio = sc / 1000
"""
import sys
import numpy as np


def fail(reason):
    print("%s Ratio: 0.0" % reason)
    sys.exit(0)


def read_tokens(path):
    try:
        return open(path).read().split()
    except Exception:
        return None


def eigenfrequencies(ks):
    """ks: list of S positive spring constants. Returns sorted omega array (S-1,)."""
    S = len(ks)
    N = S - 1
    if N <= 0:
        return np.array([])
    K = np.zeros((N, N), dtype=float)
    for i in range(N):  # node i corresponds to spring i (k[i]) and spring i+1 (k[i+1])
        K[i, i] = ks[i] + ks[i + 1]
    for i in range(N - 1):
        K[i, i + 1] = -ks[i + 1]
        K[i + 1, i] = -ks[i + 1]
    ev = np.linalg.eigvalsh(K)
    ev = np.clip(ev, 0.0, None)
    return np.sqrt(ev)


def resonance_score(omegas, lines, eps):
    if len(omegas) == 0:
        return float("inf")
    total = 0.0
    for f, w in lines:
        d = float(np.min(np.abs(omegas - f)))
        total += w / (d + eps)
    return total


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    itoks = read_tokens(in_path)
    if not itoks:
        fail("bad instance")
    try:
        S = int(itoks[0]); BUDGET = int(itoks[1])
        eps = float(itoks[2])
        F = int(itoks[3])
        rest = itoks[4:]
        if len(rest) != 2 * F:
            fail("malformed instance")
        lines = []
        for i in range(F):
            f = float(rest[2 * i]); w = int(rest[2 * i + 1])
            lines.append((f, w))
    except Exception:
        fail("instance parse error")

    if S < 2 or BUDGET < S or eps <= 0:
        fail("degenerate instance")

    otoks = read_tokens(out_path)
    if otoks is None or len(otoks) == 0:
        fail("empty or unreadable output")
    if len(otoks) != S:
        fail("expected %d stiffness values, got %d" % (S, len(otoks)))

    ks = []
    for tok in otoks:
        try:
            v = float(tok)
        except ValueError:
            fail("non-numeric token '%s'" % tok)
        if not np.isfinite(v):
            fail("non-finite value")
        if abs(v - round(v)) > 1e-6:
            fail("non-integer stiffness '%s'" % tok)
        iv = int(round(v))
        if iv < 1:
            fail("segment stiffness must be >= 1")
        ks.append(iv)

    if sum(ks) != BUDGET:
        fail("stiffness budget not exactly used: sum=%d, budget=%d" % (sum(ks), BUDGET))

    omegas = eigenfrequencies(ks)
    F_obj = resonance_score(omegas, lines, eps)

    # internal baseline: the perfectly uniform allocation (BUDGET spread evenly)
    base = BUDGET // S
    rem = BUDGET - base * S
    ks_uniform = [base + (1 if i < rem else 0) for i in range(S)]
    om_uniform = eigenfrequencies(ks_uniform)
    B = resonance_score(om_uniform, lines, eps)

    if not np.isfinite(F_obj) or F_obj <= 0:
        fail("degenerate objective")

    sc = min(1000.0, 100.0 * B / max(1e-9, F_obj))
    print("S=%d F_obj=%.8f B=%.8f  Ratio: %.6f" % (S, F_obj, B, sc / 1000.0))


if __name__ == "__main__":
    main()
