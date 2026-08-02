#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for a graded adhesive
bond-line design under thermal cycling. Prints 'Ratio: <float in [0,1]>' on
its own final line and exits 0.

Input (<in>) format:
  N M
  Csub dAlpha
  C
  dT_1 ... dT_C
  k_0 s_0        (M lines: adhesive type library, sorted by increasing shear
  ...              stiffness k_j; s_j is that type's shear-strength capacity,
  k_{M-1} s_{M-1}  also increasing in k_j)

Output (<out>) format: exactly N whitespace-separated integers a_1..a_N (each
in [0, M-1]) -- the adhesive TYPE index used at bond-line segment i. This IS
the artifact: a compliant-layer grading profile along the bond line.

Objective (maximize): thermal cycling life. Two forward sweeps (1-indexed,
i=1..N) over the chosen per-segment stiffness k_i and strength s_i:

  Homogeneous unit-load sweep (pure math tool -- NOT a mechanical test --
  used only to satisfy the thermal free-free boundary condition below):
    H[0]=0, slip[0]=1
    shear[i] = k_i*slip[i-1]; H[i] = H[i-1]+shear[i]; slip[i]=slip[i-1]+Csub*H[i]

  Thermal unit-dT sweep (dAlpha = the CTE mismatch forcing term):
    T[0]=0, tslip[0]=0
    tshear[i]=k_i*tslip[i-1]; T[i]=T[i-1]+tshear[i]
    tslip[i]=tslip[i-1]+Csub*T[i]+dAlpha

  d0 = -T[N]/H[N] (enforces zero net force at both free bond-line edges).
  Per-segment thermal shear under unit dT: thermal_i = tshear[i] + d0*shear[i].
  R = max_i(|thermal_i| / s_i)  -- the worst normalized edge-concentrated
  stress ("interfacial-stress-concentration"); MORE uniform stiffness makes R
  worse via a shorter mismatch-strain decay length.
  Given the held-out cycling profile dT_1..dT_C and a fixed Basquin fatigue
  exponent p=3: Q = sum_c |dT_c|^p, and cycling life F = 1/(Q * R^p).

The checker's OWN reference construction is the uniform SOFTEST design (type
0 everywhere) -- always well-defined (R>0 whenever some k_i>0) and never
collapses, so it is always a valid, positive baseline B.
"""
import sys
import math

P_EXP = 3
EPS = 1e-6


def die0(reason):
    print(f"infeasible: {reason}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_in(path):
    toks = open(path).read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it))
    Csub = float(next(it)); dAlpha = float(next(it))
    C = int(next(it))
    dTs = [float(next(it)) for _ in range(C)]
    lib = []
    for _ in range(M):
        k = float(next(it)); s = float(next(it))
        lib.append((k, s))
    return N, M, Csub, dAlpha, dTs, lib


def sim_H(N, Csub, k_list):
    H = [0.0]*(N+1); slip = [0.0]*(N+1); shear = [0.0]*(N+1)
    slip[0] = 1.0
    for i in range(1, N+1):
        shear[i] = k_list[i-1]*slip[i-1]
        H[i] = H[i-1] + shear[i]
        slip[i] = slip[i-1] + Csub*H[i]
    return H, shear


def sim_T(N, Csub, dAlpha, k_list):
    T = [0.0]*(N+1); tslip = [0.0]*(N+1); tshear = [0.0]*(N+1)
    for i in range(1, N+1):
        tshear[i] = k_list[i-1]*tslip[i-1]
        T[i] = T[i-1] + tshear[i]
        tslip[i] = tslip[i-1] + Csub*T[i] + dAlpha
    return T, tshear


def cycling_life(N, Csub, dAlpha, k_list, s_list, dTs):
    H, shear = sim_H(N, Csub, k_list)
    if H[N] <= 0 or not math.isfinite(H[N]):
        return 0.0
    T, tshear = sim_T(N, Csub, dAlpha, k_list)
    if not math.isfinite(T[N]):
        return 0.0
    d0 = -T[N]/H[N]
    R = 0.0
    for i in range(1, N+1):
        th = tshear[i] + d0*shear[i]
        if not math.isfinite(th):
            return 0.0
        r = abs(th)/s_list[i-1]
        if r > R:
            R = r
    if R <= 0:
        return 0.0
    Q = sum(abs(dt)**P_EXP for dt in dTs)
    try:
        denom = Q*(R**P_EXP)
    except OverflowError:
        return 0.0
    if not math.isfinite(denom) or denom <= 0:
        return 0.0
    val = 1.0/denom
    return val if math.isfinite(val) else 0.0


def internal_baseline(N, Csub, dAlpha, dTs, lib):
    """Checker's own trivial-but-feasible construction: uniform SOFTEST
    adhesive type (index 0) on every segment."""
    k0, s0 = lib[0]
    return max(1e-12, cycling_life(N, Csub, dAlpha, [k0]*N, [s0]*N, dTs))


def main():
    if len(sys.argv) < 3:
        print("usage: verify.py <in> <out> <ans>")
        print("Ratio: 0.0")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, M, Csub, dAlpha, dTs, lib = read_in(in_path)

    try:
        out_txt = open(out_path).read()
    except Exception:
        die0("cannot read output")

    toks = out_txt.split()
    if len(toks) != N:
        die0(f"expected {N} tokens, got {len(toks)}")

    idxs = []
    for t in toks:
        try:
            v = float(t)
        except ValueError:
            die0(f"non-numeric token '{t}'")
        if not math.isfinite(v):
            die0(f"non-finite token '{t}'")
        if abs(v - round(v)) > 1e-6:
            die0(f"non-integer adhesive index '{t}'")
        iv = int(round(v))
        if iv < 0 or iv > M-1:
            die0(f"adhesive index {iv} out of range [0,{M-1}]")
        idxs.append(iv)

    k_list = [lib[a][0] for a in idxs]
    s_list = [lib[a][1] for a in idxs]

    F = cycling_life(N, Csub, dAlpha, k_list, s_list, dTs)
    B = internal_baseline(N, Csub, dAlpha, dTs, lib)

    sc = min(1000.0, 100.0*F/max(1e-12, B))
    print(f"F={F:.8e} B={B:.8e}")
    print("Ratio: %.6f" % (sc/1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
