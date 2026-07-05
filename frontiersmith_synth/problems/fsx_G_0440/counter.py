import sys

# Format D checker -- minimum-XOR straight-line program for a fixed GF(2) linear map.
#
#   INPUT  <in>:   m n
#                  m rows, each n entries in {0,1}  (matrix M, y = M x over GF(2))
#
#   OUTPUT <out>:  G                       (number of 2-input XOR gates)
#                  G lines "a b"           gate t (t=1..G) has signal index n+t-1,
#                                          and equals signal[a] XOR signal[b];
#                                          both a,b must reference EARLIER signals
#                                          (inputs are indices 0..n-1, gate t is n+t-1,
#                                           so a,b < n+t-1).
#                  one line of m indices   o_0 .. o_{m-1}: the signal index whose
#                                          GF(2) value equals output row j.
#
#   1) Parse + feasibility gate: every gate references only earlier signals; every
#      output index is a valid signal; non-integer / nan / inf / out-of-range /
#      wrong token count  ->  Ratio: 0.0.
#   2) EXACT equivalence: recompute each signal as a bitmask over the n inputs and
#      verify output signal j == row j of M exactly.  Mismatch -> Ratio: 0.0.
#   3) Objective (minimize) = G.  Baseline B = naive per-row cost
#      = sum_j (popcount(row_j) - 1).  Ratio = min(1, 0.1 * B / G).

MAXG = 500000

def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)

def main():
    try:
        inp = open(sys.argv[1]).read().split()
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("io error")

    it = iter(inp)
    try:
        m = int(next(it)); n = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= m <= 100000 and 1 <= n <= 100000):
        fail("bad dims")

    rowmask = [0] * m
    B = 0
    try:
        for j in range(m):
            pc = 0
            for i in range(n):
                v = int(next(it))
                if v not in (0, 1):
                    fail("matrix entry not 0/1")
                if v:
                    rowmask[j] |= (1 << i)
                    pc += 1
            B += max(0, pc - 1)
    except SystemExit:
        raise
    except Exception:
        fail("bad matrix")
    if B <= 0:
        fail("degenerate matrix (baseline 0)")

    # ---- parse participant output (strict integer tokens only) ----
    if not out:
        fail("empty output")

    def as_int(tok):
        # rejects nan/inf/floats/garbage -> non-finite guard
        t = tok[1:] if tok[:1] in "+-" else tok
        if not t.isdigit():
            fail("non-integer / non-finite token: %r" % tok)
        return int(tok)

    pos = 0
    def nxt():
        nonlocal pos
        if pos >= len(out):
            fail("truncated output")
        tok = out[pos]; pos += 1
        return as_int(tok)

    G = nxt()
    if G < 0:
        fail("G < 0")
    if G > MAXG:
        fail("G too large")

    need = 1 + 2 * G + m
    if len(out) != need:
        fail("wrong token count (got %d, need %d)" % (len(out), need))

    # signal[k] = bitmask over n inputs.  inputs 0..n-1 are unit vectors.
    total = n + G
    sig = [0] * total
    for i in range(n):
        sig[i] = 1 << i

    for t in range(G):
        idx = n + t                 # index of this new signal
        a = nxt(); b = nxt()
        if not (0 <= a < idx and 0 <= b < idx):
            fail("gate %d references non-earlier signal (a=%d b=%d, must be < %d)" % (t, a, b, idx))
        sig[idx] = sig[a] ^ sig[b]

    outs = []
    for j in range(m):
        o = nxt()
        if not (0 <= o < total):
            fail("output %d index out of range: %d" % (j, o))
        outs.append(o)

    # ---- exact equivalence check ----
    for j in range(m):
        if sig[outs[j]] != rowmask[j]:
            fail("output %d does not equal row %d of M" % (j, j))

    if G < 1:
        fail("no gates but nonzero linear map")

    ratio = min(1.0, 0.1 * B / max(1e-9, float(G)))
    print("G=%d B=%d Ratio: %.6f" % (G, B, ratio))

if __name__ == "__main__":
    main()
