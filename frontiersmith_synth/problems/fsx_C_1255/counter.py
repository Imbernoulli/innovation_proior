import sys
from fractions import Fraction as Fr

# Format D checker -- ADC resolution/rate/filter-order/gain allocation.
#
# Input <in>:  NBINS BUDGET WFILT PFS NFLOOR FLO FHI
#              f_1 p_1 ... f_NBINS p_NBINS
# Output <out>: B R K G   (four integers, the artifact)
#
# Feasible iff 1<=B<=16, R>=1, 0<=K<=8, 0<=G<=12, B*R + WFILT*K <= BUDGET.
#
# atten(f) = 1 if K==0 else 1/(1+(2f/R)^(2K))   (order-K Butterworth, cutoff fc=R/2)
# usable(f) = min(p*atten(f), FS), clip(f) = max(0, p*atten(f)-FS), FS=PFS/2^G
# signal  = sum usable(f_i) over in-band tones
# aliased = sum usable(f_i) over out-of-band tones with f_i > R/2 (beyond Nyquist)
# noise   = FS/4^B + aliased + sum(clip) + NFLOOR
# SNR = signal / noise   (maximize)
#
# Baseline B_ref: fixed reference choice B=6,K=2,G=0, rest of budget on R.
#   Ratio = min(1000, 100*SNR/(10*B_ref)) / 1000

BMAX = 16
KMAX = 8
GMAX = 12
MAX_TOKEN_DIGITS = 15

B0_REF, K0_REF, G0_REF = 6, 2, 0


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def parse_int(tok, lo=None, hi=None):
    if not tok:
        raise ValueError("empty token")
    t = tok[1:] if tok and tok[0] in "+-" else tok
    if not t.isdigit() or len(t) > MAX_TOKEN_DIGITS:
        raise ValueError("not a small plain integer: %r" % tok)
    v = int(tok)
    if lo is not None and v < lo:
        raise ValueError("below range")
    if hi is not None and v > hi:
        raise ValueError("above range")
    return v


def atten(f, R, K):
    if K == 0:
        return Fr(1)
    ratio = Fr(2 * f, R)
    return Fr(1) / (Fr(1) + ratio ** (2 * K))


def evaluate(bins, FLO, FHI, PFS, NFLOOR, B, R, K, G):
    FS = Fr(PFS, 2 ** G)
    quant = FS / Fr(4) ** B
    sig = Fr(0)
    ali = Fr(0)
    clip_total = Fr(0)
    for f, p in bins:
        a = atten(f, R, K)
        fp = Fr(p) * a
        c = fp - FS if fp > FS else Fr(0)
        usable = fp - c
        clip_total += c
        if FLO <= f <= FHI:
            sig += usable
        elif 2 * f > R:
            ali += usable
    noise = quant + ali + clip_total + Fr(NFLOOR)
    return sig, noise


def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    it = iter(inp)
    try:
        NBINS = int(next(it)); BUDGET = int(next(it)); WFILT = int(next(it))
        PFS = int(next(it)); NFLOOR = int(next(it))
        FLO = int(next(it)); FHI = int(next(it))
        bins = []
        for _ in range(NBINS):
            f = int(next(it)); p = int(next(it))
            bins.append((f, p))
    except Exception:
        fail("bad instance (should never happen)")

    # ---- baseline B_ref: fixed reference choice, rest of budget on R ----
    R0 = (BUDGET - WFILT * K0_REF) // B0_REF
    if R0 < 1:
        fail("degenerate instance (baseline infeasible, should never happen)")
    sig0, noise0 = evaluate(bins, FLO, FHI, PFS, NFLOOR, B0_REF, R0, K0_REF, G0_REF)
    if noise0 <= 0 or sig0 <= 0:
        fail("degenerate baseline (should never happen)")
    Bref = sig0 / noise0

    # ---- parse participant output ----
    if len(out) != 4:
        fail("wrong token count (got %d, need 4)" % len(out))
    try:
        B = parse_int(out[0], lo=1, hi=BMAX)
        R = parse_int(out[1], lo=1, hi=10 ** 9)
        K = parse_int(out[2], lo=0, hi=KMAX)
        G = parse_int(out[3], lo=0, hi=GMAX)
    except Exception as e:
        fail("bad token (%s)" % e)

    cost = B * R + WFILT * K
    if cost > BUDGET:
        fail("cost %d exceeds budget %d" % (cost, BUDGET))

    sig, noise = evaluate(bins, FLO, FHI, PFS, NFLOOR, B, R, K, G)
    if noise <= 0:
        fail("degenerate noise (should never happen)")
    SNR = sig / noise

    ratio = min(1000.0, 100.0 * float(SNR) / max(1e-9, float(Bref))) / 1000.0
    print("SNR=%.6f Bref=%.6f B=%d R=%d K=%d G=%d Ratio: %.6f" % (
        float(SNR), float(Bref), B, R, K, G, ratio))


if __name__ == "__main__":
    main()
