#!/usr/bin/env python3
# verify.py <in> <out> <ans>
#
# Deterministic scorer for the binary covering-code problem (format C, objective = MIN size).
#
# Instance (in): "n r".
# Artifact  (out): one codeword per line, each a length-n string over {0,1}.
#   The set C of DISTINCT codewords must be a covering code of radius r:
#   every x in {0,1}^n must satisfy Hamming(x, c) <= r for some c in C.
#
# Score (minimization):  sc = min(1000, 100 * B / F),  ratio = sc/1000,
#   B = size of the checker's own first-fit covering code (internal baseline),
#   F = number of DISTINCT valid codewords the participant submitted.
#   Reproducing the baseline -> ~0.1 ; a 10x-smaller code caps at 1.0.
#
# ANY feasibility violation -> "Ratio: 0.0".  Bit-for-bit deterministic on reruns.
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    n = int(toks[0]); r = int(toks[1])
    return n, r


def masks_upto(n, r):
    """All bit-masks over n bits with popcount <= r (offsets of the radius-r ball)."""
    N = 1 << n
    out = []
    for m in range(N):
        if bin(m).count("1") <= r:
            out.append(m)
    return out


def firstfit_baseline(n, r, masks):
    """Natural-order first-fit covering code.  Deterministic; positive size."""
    N = 1 << n
    covered = bytearray(N)
    count = 0
    for p in range(N):
        if not covered[p]:
            count += 1
            for m in masks:
                covered[p ^ m] = 1
    return count


def fail(reason):
    print("reason: " + reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    n, r = read_instance(in_path)
    N = 1 << n
    masks = masks_upto(n, r)

    # ---- parse participant artifact strictly ----
    cap = 4 * N + 10  # more codewords than this is never useful and signals garbage
    seen = set()
    nlines = 0
    try:
        with open(out_path) as f:
            for line in f:
                s = line.strip()
                if s == "":
                    continue
                nlines += 1
                if nlines > cap:
                    fail("too many lines")
                if len(s) != n:
                    fail("codeword wrong length")
                # strict binary check (also rejects nan/inf/any non-0/1 token)
                v = 0
                for ch in s:
                    if ch == "0":
                        v = v << 1
                    elif ch == "1":
                        v = (v << 1) | 1
                    else:
                        fail("non-binary character")
                seen.add(v)
    except OSError:
        fail("cannot read output")

    F = len(seen)
    if F < 1:
        fail("empty covering code")

    # ---- feasibility: full coverage ----
    covered = bytearray(N)
    for c in seen:
        for m in masks:
            covered[c ^ m] = 1
    if len(covered) - sum(covered) > 0:
        # count uncovered without materializing; sum(covered) counts 1s
        pass
    ncov = sum(covered)
    if ncov != N:
        fail("not a covering code: %d of %d words uncovered" % (N - ncov, N))

    # ---- baseline + score ----
    B = firstfit_baseline(n, r, masks)
    if B <= 0:
        B = 1
    sc = min(1000.0, 100.0 * float(B) / max(1e-9, float(F)))
    print("n=%d r=%d F=%d baseline=%d" % (n, r, F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
