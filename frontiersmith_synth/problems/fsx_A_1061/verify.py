#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  ->  prints 'Ratio: <x in [0,1]>' (last line authoritative).

Deterministic exact scorer for the two-scale additive-energy channel-set problem.

Objective (minimize): TOTAL(A) = E_M(A) + W * E_D(A), where
    E_n(A) = #{(a,b,c,e) in A^4 : a+b == c+e (mod n)}          (exact additive energy)
and E_D(A) is computed from the residues (a mod D) of A (i.e. the energy of A's image
under the quotient map Z_M -> Z_D).

Checker baseline B: the TOTAL score of the checker's own trivial construction, the first k
multiples of D taken mod M (A = {0, D, 2D, ..., (k-1)D}) -- a positive, feasible, cheaply
built reference that is maximally concentrated in one coarse band (worst case for E_D) and
carries no fine-grained structure either. A submission matching this baseline scores ~0.1;
the theoretical floor (2k^2-k) + W*Ed_floor(D,k) sits comfortably below B on every generated
instance (checked empirically at authoring time across the full test ladder, with margin),
so the true optimum lands well under the ratio-1.0 saturation point -- headroom is preserved
without needing to fabricate an artificial cap.

Minimization normalization: sc = min(1000, 100 * B / max(1e-9, TOTAL)); Ratio = sc / 1000.

Any feasibility violation (wrong token count, duplicate, out of range, non-integer,
non-finite) prints 'Ratio: 0.0' and exits 0.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    M, D, k, W = int(toks[0]), int(toks[1]), int(toks[2]), int(toks[3])
    return M, D, k, W


def energy_of_multiset(vals, n):
    """Exact additive energy sum_x r(x)^2 of a list of residues mod n. O(len(vals)^2 + n)."""
    r = [0] * n
    for i in range(len(vals)):
        vi = vals[i]
        for j in range(len(vals)):
            r[(vi + vals[j]) % n] += 1
    return sum(x * x for x in r)


def baseline_total(M, D, k, W):
    """Checker's own trivial construction: first k multiples of D mod M."""
    triv = [(i * D) % M for i in range(k)]
    Em = energy_of_multiset(triv, M)
    Ed = energy_of_multiset([a % D for a in triv], D)
    return Em + W * Ed


def fail(reason):
    sys.stdout.write("reason: %s\nRatio: 0.0\n" % reason)
    sys.exit(0)


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    M, D, k, W = read_instance(inp)

    try:
        with open(outp) as f:
            raw = f.read().split()
    except Exception:
        fail("cannot read output")

    if len(raw) != k:
        fail("expected exactly %d tokens, got %d" % (k, len(raw)))

    A = []
    for tok in raw:
        # int() on a well-formed string never yields non-finite; anything like
        # "nan"/"inf"/"1.5"/"1e9" fails to parse as int and is rejected here.
        try:
            v = int(tok)
        except ValueError:
            fail("non-integer / non-finite token: %r" % tok)
        A.append(v)

    seen = set()
    for v in A:
        if v < 0 or v >= M:
            fail("channel %d out of range [0,%d)" % (v, M))
        if v in seen:
            fail("duplicate channel %d" % v)
        seen.add(v)

    Em = energy_of_multiset(A, M)
    Ed = energy_of_multiset([a % D for a in A], D)
    total = Em + W * Ed

    B = float(baseline_total(M, D, k, W))

    sc = min(1000.0, 100.0 * B / max(1e-9, float(total)))
    ratio = sc / 1000.0
    sys.stdout.write(
        "Em=%d Ed=%d W=%d TOTAL=%d  B(baseline)=%.6f\nRatio: %.6f\n"
        % (Em, Ed, W, total, B, ratio)
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
