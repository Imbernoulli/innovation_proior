#!/usr/bin/env python3
"""Deterministic checker for the deep-space LDPC batch-decoding problem (format C).

Usage: python3 verify.py <in> <out> <ans>

<in>  : the instance
          line 1        : n r m T
          next r lines  : parity-check matrix H (length-n binary strings)
          next m lines  : received frames (length-n binary strings)
<out> : participant artifact -- EXACTLY m whitespace-separated tokens, the i-th being a
        length-n binary string = the codeword the solver decoded frame i to.
<ans> : ignored placeholder.

Scoring (maximization). A frame i counts as CORRECTED iff the submitted word is a true
codeword of the LDPC code (H * word = 0) AND lies within Hamming distance T of the
received frame. F = number of corrected frames. The checker builds its own trivial
baseline B = number of frames that are ALREADY codewords (i.e. decodable by doing
nothing). Reproducing B scores Ratio ~ 0.1; a genuinely good decoder scores higher.

Prints exactly one line ending in "Ratio: <x>" (x in [0,1]) and exits 0. On ANY
feasibility violation prints "Ratio: 0.0" and exits 0.
"""
import sys


def fail(reason):
    print("INVALID (%s) Ratio: 0.0" % reason)
    sys.exit(0)


def parity(x):
    return bin(x).count("1") & 1


def bits_to_int(s, n):
    """Strict: s must be exactly n chars, all '0'/'1'. Returns int or None."""
    if len(s) != n:
        return None
    v = 0
    for j, ch in enumerate(s):
        if ch == "1":
            v |= (1 << j)
        elif ch != "0":
            return None
    return v


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    in_path, out_path = sys.argv[1], sys.argv[2]

    # ---- read instance ----
    with open(in_path) as f:
        lines = f.read().split("\n")
    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    header = lines[idx].split()
    idx += 1
    if len(header) < 4:
        fail("bad instance header")
    n, r, m, T = (int(header[0]), int(header[1]), int(header[2]), int(header[3]))

    H = []
    for _ in range(r):
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        v = bits_to_int(lines[idx].strip(), n)
        if v is None:
            fail("bad H row")
        H.append(v)
        idx += 1

    Y = []
    for _ in range(m):
        while idx < len(lines) and lines[idx].strip() == "":
            idx += 1
        v = bits_to_int(lines[idx].strip(), n)
        if v is None:
            fail("bad received frame")
        Y.append(v)
        idx += 1

    # ---- checker-internal trivial baseline: frames that are already codewords ----
    def is_codeword(x):
        for row in H:
            if parity(row & x):
                return False
        return True

    B = 0
    for y in Y:
        if is_codeword(y):
            B += 1
    if B <= 0:
        B = 1

    # ---- read participant artifact (strict) ----
    with open(out_path) as f:
        raw = f.read()
    toks = raw.split()
    if len(toks) != m:
        fail("expected %d frame decodings, got %d tokens" % (m, len(toks)))

    words = []
    for tk in toks:
        v = bits_to_int(tk, n)   # rejects wrong length, non-binary, nan/inf, floats
        if v is None:
            fail("frame decoding is not a length-%d binary string" % n)
        words.append(v)

    # ---- objective ----
    F = 0
    for i in range(m):
        w = words[i]
        if not is_codeword(w):
            continue
        # Hamming distance to the received frame
        if bin(w ^ Y[i]).count("1") <= T:
            F += 1

    sc = 100.0 * float(F) / max(1e-9, float(B))
    if sc > 1000.0:
        sc = 1000.0
    ratio = sc / 1000.0
    print("valid. F=%d B=%d Ratio: %.6f" % (F, B, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
