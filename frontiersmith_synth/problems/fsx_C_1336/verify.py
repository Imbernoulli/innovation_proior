#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for cocrystal-former
selection (hydrogen-bond-synthon-match + lattice-energy-tradeoff +
regulatory-former-list).

Output contract: exactly two whitespace-separated integers "idx r" -- the
0-based index of the chosen regulatory-approved former and the stoichiometry
ratio (former molecules per 1 API molecule).

Feasibility (ANY violation -> Ratio: 0.0):
  * output parses to exactly two finite integers;
  * 0 <= idx < M;
  * r is present in that former's regulatory-approved ratio list;
  * the resulting lattice energy L = h(idx,r) + lc[idx]*r meets the
    stability threshold L >= L_min (the cocrystal must actually FORM).

Objective (maximize): the solubility improvement
    dSol(idx,r) = P_BONUS * p[idx]  -  DECAY * (L - L_min)
A stable cocrystal that does not improve solubility over the naked API
(dSol <= 0) earns Ratio: 0.0 too -- it is feasible but useless.

Baseline B (checker's own trivial feasible construction) = dSol(idx=0, r=1),
i.e. always selecting the given "reference" former at its only approved
ratio. B > 0 is guaranteed by the generator's construction.

Score:  sc = min(1000, 100*F/B);  print Ratio: sc/1000  (trivial ~= 0.1).
"""
import math
import sys


def read_tokens(path):
    with open(path) as f:
        return f.read().split()


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def h_score(W, Ad, Aa, fd, fa, r, K):
    tot = 0
    for t in range(K):
        tot += W[t] * (min(Ad[t], r * fa[t]) + min(Aa[t], r * fd[t]))
    return tot


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    itok = read_tokens(inf)
    if len(itok) < 1:
        fail("empty input")
    p = 0

    def nxt(cnt):
        nonlocal p
        vals = itok[p:p + cnt]
        p += cnt
        return vals

    K = int(itok[p]); p += 1
    donor_strength = list(map(int, nxt(K)))
    acceptor_strength = list(map(int, nxt(K)))
    Ad = list(map(int, nxt(K)))
    Aa = list(map(int, nxt(K)))
    P_BONUS, DECAY, L_min = map(int, nxt(3))
    M = int(itok[p]); p += 1

    W = [donor_strength[t] * acceptor_strength[t] for t in range(K)]

    formers = []
    for _ in range(M):
        fd = list(map(int, nxt(K)))
        fa = list(map(int, nxt(K)))
        lc = int(itok[p]); p += 1
        pol = int(itok[p]); p += 1
        R = int(itok[p]); p += 1
        ratios = list(map(int, nxt(R)))
        formers.append((fd, fa, lc, pol, ratios))

    if M <= 0 or len(formers) != M:
        fail("degenerate/truncated instance")

    def dsol_of(idx, r):
        fd, fa, lc, pol, ratios = formers[idx]
        h = h_score(W, Ad, Aa, fd, fa, r, K)
        L = h + lc * r
        if L < L_min:
            return None  # infeasible: unstable
        return P_BONUS * pol - DECAY * (L - L_min)

    # --- checker's own baseline construction: former 0 at ratio 1 ---
    fd0, fa0, lc0, pol0, ratios0 = formers[0]
    if 1 not in ratios0:
        fail("degenerate instance (baseline former missing ratio 1)")
    B = dsol_of(0, 1)
    if B is None or B <= 0:
        fail("degenerate instance (baseline infeasible or non-positive)")

    # --- parse participant output ---
    otok = read_tokens(outf)
    if len(otok) != 2:
        fail("output must be exactly 2 tokens 'idx r', got %d" % len(otok))
    try:
        idx = int(otok[0])
        r = int(otok[1])
    except ValueError:
        fail("non-integer token in output")
    if not (math.isfinite(idx) and math.isfinite(r)):
        fail("non-finite value")

    if idx < 0 or idx >= M:
        fail("former index %d out of range [0,%d]" % (idx, M - 1))
    fd, fa, lc, pol, ratios = formers[idx]
    if r not in ratios:
        fail("ratio %d not regulatory-approved for former %d (allowed %s)" % (r, idx, ratios))

    h = h_score(W, Ad, Aa, fd, fa, r, K)
    L = h + lc * r
    if L < L_min:
        fail("cocrystal unstable: lattice energy %d < threshold %d" % (L, L_min))

    dSol = P_BONUS * pol - DECAY * (L - L_min)
    if dSol <= 0:
        fail("stable cocrystal but no solubility improvement (dSol=%d)" % dSol)

    F = float(dSol)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("idx=%d r=%d h=%d L=%d L_min=%d dSol=%d B=%d  Ratio: %.6f"
          % (idx, r, h, L, L_min, dSol, B, sc / 1000.0))


if __name__ == "__main__":
    main()
