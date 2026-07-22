#!/usr/bin/env python3
"""
verify.py <in> <out> <ans> -- deterministic checker for fsx_A_0685.

Instance (in <in>):
    n k
    a_0 .. a_{n-1}      (0/1 quarry-approved mask over Z_n)
    c_0 .. c_{n-1}      (nonneg integer quarry fee, meaningful where a_i=1)

Participant artifact (in <out>), stdout of the solution:
    line 1: k integers  -- the tile B (positions in Z_n, must be distinct, all approved)
    line 2: n/k integers -- the translation offsets T (positions in Z_n)

Feasibility: |B|=k, all distinct, all in [0,n-1], all approved (a_b=1);
|T|=n/k, all in [0,n-1]. Any violation, wrong token count, or non-finite/non-integer
token -> Ratio: 0.0.

Objective (minimize): let count[x] = #{(b,t) in B x T : (b+t) mod n == x}.
  D = sum_x |count[x] - 1|                       (paving defect)
  C = sum_{b in B} cost[b]                        (total quarry fee of the tile)
  W = 1 + sum_{i : a_i=1} cost[i]                 (fixed normalizer from the input)
  F = ALPHA*(D/n) + C/W                           (ALPHA = 0.35)
F is minimized; D dominates by construction (any nonzero defect costs far more than any
achievable cost saving) while C breaks ties among equal-defect tilings.

Score: an internal reference tiling REF (T=multiples of k, one arbitrary -- smallest
index -- approved cell per residue class mod k; always defect 0 given the guaranteed
>=1 approved cell per class) gives F_ref. Then
  sc = min(1000, 100*F_ref / max(1e-9, F_you)) ;  Ratio = sc/1000.
"""
import sys

ALPHA = 0.35


def fail(msg):
    print(f"INFEASIBLE: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints_strict(path, need):
    """Tokenize whitespace-separated tokens, parse as STRICT base-10 ints (rejects
    nan/inf/floats/garbage), require exactly `need` tokens."""
    txt = open(path).read()
    toks = txt.split()
    if len(toks) != need:
        return None, f"expected {need} tokens, got {len(toks)}"
    vals = []
    for t in toks:
        try:
            if not (t.lstrip("+-").isdigit()):
                return None, f"non-integer token {t!r}"
            vals.append(int(t))
        except ValueError:
            return None, f"non-integer token {t!r}"
    return vals, None


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as f:
        head = f.readline().split()
        n, k = int(head[0]), int(head[1])
        allowed = list(map(int, f.readline().split()))
        cost = list(map(int, f.readline().split()))
    if len(allowed) != n or len(cost) != n:
        fail("corrupt instance file")
    M = n // k

    out_txt = open(outf).read()
    out_toks = out_txt.split()
    if len(out_toks) != k + M:
        fail(f"expected {k + M} output tokens (k={k} for B, n/k={M} for T), got {len(out_toks)}")

    def parse_int_tok(t):
        s = t.lstrip("+-")
        if not s.isdigit():
            return None
        return int(t)

    B_raw = [parse_int_tok(t) for t in out_toks[:k]]
    T_raw = [parse_int_tok(t) for t in out_toks[k:k + M]]
    if any(v is None for v in B_raw) or any(v is None for v in T_raw):
        fail("non-integer / non-finite token in output")

    if any(b < 0 or b >= n for b in B_raw):
        fail("B index out of range [0,n-1]")
    if len(set(B_raw)) != k:
        fail("B contains duplicate positions")
    if any(allowed[b] == 0 for b in B_raw):
        fail("B contains a non-quarry-approved position")
    if any(t < 0 or t >= n for t in T_raw):
        fail("T index out of range [0,n-1]")

    B, T = B_raw, T_raw

    def defect(Bset, Tset):
        cnt = [0] * n
        for b in Bset:
            for t in Tset:
                cnt[(b + t) % n] += 1
        return sum(abs(x - 1) for x in cnt)

    D_you = defect(B, T)
    C_you = sum(cost[b] for b in B)
    W = 1 + sum(cost[i] for i in range(n) if allowed[i])
    F_you = ALPHA * (D_you / n) + C_you / W

    # internal reference: T = H (multiples of k), one arbitrary (smallest-index)
    # approved cell per residue class mod k.
    classes = [[] for _ in range(k)]
    for i in range(n):
        if allowed[i]:
            classes[i % k].append(i)
    if any(len(c) == 0 for c in classes):
        fail("instance has an empty residue class (generator invariant violated)")
    B_ref = [min(c) for c in classes]
    H = [t * k for t in range(M)]
    D_ref = defect(B_ref, H)
    C_ref = sum(cost[b] for b in B_ref)
    F_ref = ALPHA * (D_ref / n) + C_ref / W

    sc = min(1000.0, 100.0 * F_ref / max(1e-9, F_you))
    print(f"D={D_you} C={C_you} F={F_you:.6f} F_ref={F_ref:.6f}")
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)


if __name__ == "__main__":
    main()
