#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for the scaffold-hop problem.

Feasibility (any violation -> Ratio: 0.0):
  * output = an integer L on line 1, then exactly L integer fragment ids on line 2
    (no more, no fewer tokens; every token a plain base-10 integer -- 'nan'/'inf'
    tokens fail int() and are rejected);
  * 1 <= L <= L_max; every id in [0, M-1];
  * total synthetic cost (sum of the ids' library cost) <= BUDGET.

Objective (maximization):
  P = fraction of the K pharmacophore anchors matched by SOME atom of the candidate
      (same feature type, Euclidean distance <= tol).
  N = 1 - overlap(candidate multiset, reference multiset) / max(len(candidate), L_ref)
      where overlap = sum over fragment-id of min(count_candidate, count_reference)
      (a Counter-min multiset intersection).  Copying the reference exactly -> N = 0.
  eff = min(1, cost(reference) / cost(candidate))   (synthetic-accessibility term,
        capped at 1 so being cheaper than the reference gives no extra credit).
  F = P * N * eff.

Baseline B: the checker's own construction = reference sequence + append
  round(0.25*L_ref) (>=1) copies of the single cheapest library fragment. B is
  scored with the same F formula (its P is always 1, since it never touches the
  reference's original positions). B is always > 0 (padding with an off-reference
  fragment always yields nonzero N).

Score: sc = min(1000, 100*F/B); print Ratio: sc/1000.
"""
import sys
from collections import Counter


def read_tokens(path):
    with open(path) as f:
        return f.read().split()


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def dist(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    itok = read_tokens(inf)
    if len(itok) < 4:
        fail("truncated input header")
    p = 0
    M = int(itok[p]); p += 1
    L_max = int(itok[p]); p += 1
    STEP = int(itok[p]); p += 1
    BUDGET = int(itok[p]); p += 1

    library = []  # (cost, type, dx, dy, dz)
    for _ in range(M):
        fid = int(itok[p]); p += 1
        cost = int(itok[p]); p += 1
        typ = itok[p]; p += 1
        dx = float(itok[p]); p += 1
        dy = float(itok[p]); p += 1
        dz = float(itok[p]); p += 1
        library.append((cost, typ, dx, dy, dz))
    L_ref = int(itok[p]); p += 1
    ref_seq = [int(itok[p + i]) for i in range(L_ref)]; p += L_ref
    K = int(itok[p]); p += 1
    anchors = []
    for _ in range(K):
        x = float(itok[p]); p += 1
        y = float(itok[p]); p += 1
        z = float(itok[p]); p += 1
        typ = itok[p]; p += 1
        tol = float(itok[p]); p += 1
        anchors.append((x, y, z, typ, tol))

    if K <= 0 or L_ref <= 0:
        fail("degenerate instance")

    def cost_of(seq):
        return sum(library[fid][0] for fid in seq)

    S_ref = cost_of(ref_seq)
    if S_ref <= 0:
        fail("degenerate reference cost")

    def atoms_of(seq):
        pts = []
        for i, fid in enumerate(seq):
            cost, typ, dx, dy, dz = library[fid]
            pts.append((i * STEP + dx, dy, dz, typ))
        return pts

    def match_fraction(seq_atoms):
        hit = 0
        for (ax, ay, az, atyp, atol) in anchors:
            ok = False
            for (px, py, pz, ptyp) in seq_atoms:
                if ptyp == atyp and dist((px, py, pz), (ax, ay, az)) <= atol + 1e-9:
                    ok = True
                    break
            if ok:
                hit += 1
        return hit / K

    def novelty(seq):
        cc = Counter(seq)
        cr = Counter(ref_seq)
        overlap = sum(min(cc[k], cr.get(k, 0)) for k in cc)
        return 1.0 - overlap / max(len(seq), len(ref_seq))

    def score_F(seq):
        P = match_fraction(atoms_of(seq))
        N = novelty(seq)
        S = cost_of(seq)
        eff = min(1.0, S_ref / max(1e-9, S))
        return P * N * eff

    # ---- baseline B: reference + padding with the single cheapest fragment ----
    cheapest_id = min(range(M), key=lambda fid: (library[fid][0], fid))
    append_n = max(1, round(0.25 * L_ref))
    b_seq = ref_seq + [cheapest_id] * append_n
    F_B = score_F(b_seq)
    if F_B <= 0:
        fail("degenerate baseline (should not happen)")

    # ---- parse & validate candidate output ----
    otok = read_tokens(outf)
    if len(otok) < 1:
        fail("empty output")
    try:
        L = int(otok[0])
    except ValueError:
        fail("first token is not an integer length")
    if L < 1 or L > L_max:
        fail("L=%d out of range [1,%d]" % (L, L_max))
    if len(otok) != 1 + L:
        fail("expected 1+%d tokens, got %d" % (L, len(otok)))

    seq = []
    for i in range(L):
        tok = otok[1 + i]
        try:
            v = int(tok)
        except ValueError:
            fail("non-integer fragment id token %r" % tok)
        if v < 0 or v >= M:
            fail("fragment id %d out of range [0,%d]" % (v, M - 1))
        seq.append(v)

    S = cost_of(seq)
    if S > BUDGET:
        fail("total cost %d exceeds BUDGET %d" % (S, BUDGET))

    F = score_F(seq)
    sc = min(1000.0, 100.0 * F / max(1e-9, F_B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, F_B, sc / 1000.0))


if __name__ == "__main__":
    main()
