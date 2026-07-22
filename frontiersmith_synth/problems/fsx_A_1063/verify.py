#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>  -- deterministic checker for "chord-of-masses".

Reads the instance (n, B, CAP, r, target ratios) from <in>, the participant's
mass-unit vector e_1..e_n from <out>, validates feasibility strictly, then
solves the EXACT generalized eigenproblem K v = lambda M v (K = fixed
tridiagonal(-1,2,-1) discrete-string stiffness matrix, M = diag(1+e_i)) and
scores how well the achieved frequency ratios omega_k/omega_1 (k=1..r)
match the requested target chord, versus the checker's own uniform-mass
baseline construction.
"""
import sys
import math
import numpy as np


def eigen_freq_ratios(n, m, r):
    """Return [omega_1..omega_r]/omega_1-normalizable raw omega_1..omega_r
    (ascending) for stiffness K (fixed tridiagonal(-1,2,-1)) and mass
    vector m (all m_i>0), via the symmetrized generalized eigenproblem
    Ksym = D^-1/2 K D^-1/2 (same nonzero spectrum as K v = lambda M v)."""
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = 2.0
    for i in range(n - 1):
        K[i, i + 1] = -1.0
        K[i + 1, i] = -1.0
    dinv = 1.0 / np.sqrt(m)
    Ksym = (dinv[:, None] * K) * dinv[None, :]
    ev = np.linalg.eigvalsh(Ksym)
    ev = np.clip(ev, 1e-12, None)
    ev = np.sort(ev)
    return np.sqrt(ev[:r])


def objective(n, e, r, targets):
    """Sum of squared log-ratio errors for modes k=2..r (cost; minimize)."""
    m = 1.0 + e
    w = eigen_freq_ratios(n, m, r)
    lt = [math.log(targets[k] / targets[0]) for k in range(r)]
    F = 0.0
    for k in range(1, r):
        la = math.log(w[k] / w[0])
        F += (la - lt[k]) ** 2
    return F


def uniform_baseline(n, B):
    """The checker's own trivial feasible construction: spread the budget
    as evenly as possible (deterministic, matches solutions/trivial.py)."""
    e = np.zeros(n)
    q, rem = divmod(B, n)
    for i in range(n):
        e[i] = q + (1 if i < rem else 0)
    return e


def fail(msg):
    print(msg + " Ratio: 0.0")
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        toks = f.read().split()
    p = 0
    n = int(toks[p]); p += 1
    B = int(toks[p]); p += 1
    CAP = int(toks[p]); p += 1
    r = int(toks[p]); p += 1
    targets = []
    for _ in range(r):
        num = int(toks[p]); p += 1
        den = int(toks[p]); p += 1
        targets.append(num / den)

    try:
        with open(out_path) as f:
            out_toks = f.read().split()
    except FileNotFoundError:
        fail("No output file.")

    if len(out_toks) != n:
        fail(f"Expected exactly {n} integers, got {len(out_toks)}.")

    e_vals = []
    for tok in out_toks:
        try:
            v = int(tok)
        except ValueError:
            fail(f"Non-integer token {tok!r}.")
        if not math.isfinite(v):
            fail("Non-finite token.")
        e_vals.append(v)

    for v in e_vals:
        if v < 0 or v > CAP:
            fail(f"Mass unit {v} out of range [0,{CAP}].")

    if sum(e_vals) != B:
        fail(f"Budget mismatch: sum={sum(e_vals)}, required B={B}.")

    e = np.array(e_vals, dtype=float)
    m = 1.0 + e
    if not np.all(np.isfinite(m)) or np.any(m <= 0):
        fail("Invalid mass vector.")

    try:
        F = objective(n, e, r, targets)
    except Exception as exc:
        fail(f"Evaluation error: {exc}")

    if not math.isfinite(F):
        fail("Non-finite objective.")

    e_base = uniform_baseline(n, B)
    F_base = objective(n, e_base, r, targets)
    if not math.isfinite(F_base) or F_base <= 0:
        F_base = 1e-6

    CAP_CONST = 900.0  # headroom cap: keeps strong solutions well below 1.0
    EPS = 1e-6
    sc_raw = 100.0 * F_base / max(EPS, F)
    sc = min(CAP_CONST, sc_raw)
    print("F=%.6f F_base=%.6f Ratio: %.6f" % (F, F_base, sc / 1000.0))


if __name__ == "__main__":
    main()
