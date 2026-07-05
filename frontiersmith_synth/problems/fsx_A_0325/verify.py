import sys

# Deterministic scorer for the "peak-resonance" (first autocorrelation
# inequality) construction problem.  MINIMIZATION.
#
#   c1(f) = 2*n * max_k (f * f)[k] / (sum f)^2
#
# where (f * f) is the acyclic self-convolution of the non-negative integer
# intensity vector f of length n.  Smaller c1 = flatter hype echo = better.
#
# All arithmetic below is exact integer arithmetic (f is integer), so the
# comparison used for the peak is bit-for-bit deterministic.  The final ratio
# is a rational rendered as a float.


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def peak_autoconv(f):
    """max_k sum_i f[i]*f[k-i], acyclic, exact integer.  O(n^2)."""
    n = len(f)
    best = 0
    # lag k ranges over 0 .. 2n-2
    for k in range(2 * n - 1):
        lo = 0 if k < n else k - n + 1
        hi = k if k < n else n - 1
        s = 0
        fi = f
        for i in range(lo, hi + 1):
            s += fi[i] * fi[k - i]
        if s > best:
            best = s
    return best


def c1_value(f, n):
    """Returns (numerator, denominator) of c1 as an exact rational, or None
    if the vector carries no mass."""
    S = 0
    for v in f:
        S += v
    if S <= 0:
        return None
    P = peak_autoconv(f)
    # c1 = 2*n*P / S^2
    return (2 * n * P, S * S)


def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    # ---- parse instance ----
    try:
        n = int(inp[0])
        M = int(inp[1])
    except Exception:
        fail("bad instance")
    if n <= 0 or M <= 0:
        fail("bad instance params")

    # ---- parse participant artifact: EXACTLY n integers in [0, M] ----
    if len(out) != n:
        fail("expected %d integers, got %d tokens" % (n, len(out)))
    f = []
    for tok in out:
        # strict integer parse: rejects floats, 'nan', 'inf', '1e3', etc.
        try:
            v = int(tok)
        except Exception:
            fail("non-integer token %r" % tok)
        if v < 0 or v > M:
            fail("intensity %d out of range [0,%d]" % (v, M))
        f.append(v)
    if sum(f) <= 0:
        fail("all-zero schedule")

    # ---- objective of participant ----
    fr = c1_value(f, n)
    if fr is None:
        fail("zero mass")
    Fnum, Fden = fr           # F = Fnum/Fden
    if Fnum <= 0:
        fail("degenerate objective")

    # ---- internal baseline B: a naive "fill the front half of the arena
    #      at uniform intensity" schedule.  c1 = 2n / ceil(n/2) ~= 4. ----
    L = (n + 1) // 2
    base = [1] * L + [0] * (n - L)
    br = c1_value(base, n)
    Bnum, Bden = br           # B = Bnum/Bden

    # ---- minimization score: sc = min(1000, 100 * B / F) ----
    #   B/F = (Bnum/Bden)/(Fnum/Fden) = (Bnum*Fden)/(Bden*Fnum)
    ratio_num = 100.0 * (Bnum * Fden)
    ratio_den = (Bden * Fnum)
    sc = ratio_num / ratio_den if ratio_den > 0 else 0.0
    if sc > 1000.0:
        sc = 1000.0
    Fval = Fnum / Fden
    Bval = Bnum / Bden
    print("n=%d F=%.6f B=%.6f Ratio: %.6f" % (n, Fval, Bval, sc / 1000.0))


if __name__ == "__main__":
    main()
