#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for a battery-electrolyte
formulation. Prints 'Ratio: <float in [0,1]>' on its own final line and exits 0.

Input (<in>) format:
  N M
  A_max V_target cov_target Kconst
  eta_1 kappa_1 thr_1        (N lines: solvent viscosity, conductivity coeff,
  ...                         native anodic-stability threshold)
  p_1 etapen_1 kappapen_1 cap_1   (M lines: additive SEI strength/unit loading,
  ...                              viscosity penalty/unit, conductivity-dilution
                                    penalty/unit, useful-loading cap)

Output (<out>) format: exactly N+M whitespace-separated finite numbers.
  x_1..x_N  = solvent volume fractions (must sum to 1, each >= 0)
  a_1..a_M  = additive volume loadings (each >= 0, sum <= A_max)

Objective (maximize): a Walden-quotient conductivity F, gated to 0 whenever the
electrochemical window is not protected -- either natively (every solvent that
is actually used in the blend has thr_i >= V_target) or via enough sacrificial
SEI-additive coverage (cov_target reached). The checker's OWN reference
construction is the single native-safest solvent alone (max thr_i, no
additive) -- always window-protected by construction, since the generator
guarantees max_i(thr_i) >= V_target for every instance.
"""
import sys
import math

EPS = 1e-6


def die0(reason):
    print(f"infeasible: {reason}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_in(path):
    toks = open(path).read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it))
    A_max = float(next(it)); V_target = float(next(it))
    cov_target = float(next(it)); Kconst = float(next(it))
    solv = []
    for _ in range(N):
        eta = float(next(it)); kappa = float(next(it)); thr = float(next(it))
        solv.append((eta, kappa, thr))
    add = []
    for _ in range(M):
        p = float(next(it)); etapen = float(next(it))
        kappapen = float(next(it)); cap = float(next(it))
        add.append((p, etapen, kappapen, cap))
    return N, M, A_max, V_target, cov_target, Kconst, solv, add


def compute_F(x, a, Kconst, V_target, cov_target, solv, add):
    """x: list of N solvent fractions, a: list of M additive loadings.
    Returns the objective value (0.0 if the electrochemical window fails)."""
    N, M = len(solv), len(add)
    used_solv = [i for i in range(N) if x[i] > EPS]
    min_thr_used = min(solv[i][2] for i in used_solv)

    used_add = [min(a[j], add[j][3]) for j in range(M)]
    coverage = sum(used_add[j] * add[j][0] for j in range(M))

    # tolerance is 1e-6, not 1e-9: outputs are read back from text (typically
    # printed with ~6 decimal digits), so a coverage/threshold that is exactly
    # on target can legitimately be off by a few 1e-7 after the print/parse
    # round-trip -- the gate must not punish that.
    window_ok = (coverage >= cov_target - 1e-6) or (min_thr_used >= V_target - 1e-6)
    if not window_ok:
        return 0.0

    numer = sum(x[i] * solv[i][1] for i in range(N)) - sum(a[j] * add[j][2] for j in range(M))
    numer = max(numer, 0.0)
    denom = sum(x[i] * solv[i][0] for i in range(N)) + sum(a[j] * add[j][1] for j in range(M))
    if denom <= 0:
        return 0.0
    return Kconst * numer / denom


def internal_baseline(Kconst, V_target, solv):
    """Checker's own trivial-but-feasible construction: 100% of the solvent with
    the highest native threshold (ties -> smallest index), zero additive. The
    generator guarantees this always natively satisfies V_target."""
    N = len(solv)
    i_star = max(range(N), key=lambda i: (solv[i][2], -i))
    eta_s, kappa_s, thr_s = solv[i_star]
    return max(1e-9, Kconst * kappa_s / eta_s)


def main():
    if len(sys.argv) < 3:
        print("usage: verify.py <in> <out> <ans>")
        print("Ratio: 0.0")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, M, A_max, V_target, cov_target, Kconst, solv, add = read_in(in_path)

    try:
        out_txt = open(out_path).read()
    except Exception:
        die0("cannot read output")

    raw_toks = out_txt.split()
    if len(raw_toks) != N + M:
        die0(f"expected {N+M} tokens, got {len(raw_toks)}")

    vals = []
    for t in raw_toks:
        try:
            v = float(t)
        except ValueError:
            die0(f"non-numeric token '{t}'")
        if not math.isfinite(v):
            die0(f"non-finite token '{t}'")
        vals.append(v)

    x = vals[:N]
    a = vals[N:]

    for v in x:
        if v < -1e-9:
            die0("negative solvent fraction")
    for v in a:
        if v < -1e-9:
            die0("negative additive loading")
    x = [max(0.0, v) for v in x]
    a = [max(0.0, v) for v in a]

    sx = sum(x)
    if abs(sx - 1.0) > 1e-4:
        die0(f"solvent fractions sum to {sx}, must sum to 1")
    # renormalize away tiny float slop so downstream arithmetic is exact-ish
    x = [v / sx for v in x]

    sa = sum(a)
    if sa > A_max + 1e-6:
        die0(f"additive budget exceeded: {sa} > {A_max}")

    F = compute_F(x, a, Kconst, V_target, cov_target, solv, add)
    B = internal_baseline(Kconst, V_target, solv)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print(f"F={F:.6f} B={B:.6f}")
    print("Ratio: %.6f" % (sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
