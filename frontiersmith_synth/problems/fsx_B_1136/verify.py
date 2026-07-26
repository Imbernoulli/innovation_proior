import sys, math
from fractions import Fraction


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def parse_int_token(tok):
    # strict integer parse; reject floats, nan, inf, junk
    try:
        if tok.lower() in ("nan", "inf", "-inf", "+inf", "infinity", "-infinity"):
            return None
        f = float(tok)
        if not math.isfinite(f):
            return None
    except ValueError:
        return None
    try:
        v = int(tok)
    except ValueError:
        return None
    return v


def read_input(path):
    toks = open(path).read().split()
    it = iter(toks)
    Q = int(next(it)); VolumeBudget = int(next(it)); M = int(next(it))
    Dmax = int(next(it)); Lmax = int(next(it))
    PenNum = int(next(it)); PenDen = int(next(it))
    thr, mass, val, scour = {}, {}, {}, {}
    for c in range(1, 8):
        thr[c] = int(next(it)); mass[c] = int(next(it))
        val[c] = int(next(it)); scour[c] = int(next(it))
    return dict(Q=Q, VolumeBudget=VolumeBudget, M=M, Dmax=Dmax, Lmax=Lmax,
                PenNum=PenNum, PenDen=PenDen, thr=thr, mass=mass, val=val, scour=scour)


def simulate(inst, basins):
    """basins: list of (d, l, b) already validated feasible (bounds + budget).
    Returns total objective F (a Fraction, >= 0)."""
    Q = inst["Q"]; thr = inst["thr"]; mass = inst["mass"]; val = inst["val"]; scour = inst["scour"]
    PenNum, PenDen = inst["PenNum"], inst["PenDen"]

    remaining = dict(mass)  # class -> mass still in flow
    correct = {c: 0 for c in range(1, 8)}   # mass routed to its OWN bin
    wrong = {c: 0 for c in range(1, 8)}     # mass of foreign classes routed to bin c

    for (d, l, b) in basins:
        vol = d * l
        captured_now = []
        for c in range(1, 8):
            if remaining[c] <= 0:
                continue
            # R_i >= thr_c  <=>  d*l >= Q*thr_c   (pure integer comparison, no division)
            r_ok = vol >= Q * thr[c]
            # v_i <= scour_c  <=>  Q/d <= scour_c  <=>  Q <= scour_c * d
            scour_ok = Q <= scour[c] * d
            if r_ok and scour_ok:
                captured_now.append(c)
        for c in captured_now:
            m = remaining[c]
            remaining[c] = 0
            if b == 0:
                continue  # drained to waste: no value, no penalty
            if b == c:
                correct[b] += m
            else:
                wrong[b] += m

    F = Fraction(0)
    for t in range(1, 8):
        raw = Fraction(val[t] * correct[t] * PenDen - PenNum * val[t] * wrong[t], PenDen)
        if raw > 0:
            F += raw
    return F


def internal_baseline(inst, k_classes):
    """Trivial feasible construction: resolve just the `k_classes` SMALLEST
    residence-time thresholds (a prefix of the correctly-ordered cascade,
    scour-safe depth), ignore the rest entirely. Since these are the strict-
    smallest thresholds, this prefix captures exactly those classes with zero
    contamination by construction -> always positive, robust reference."""
    Q = inst["Q"]; thr = inst["thr"]; scour = inst["scour"]
    Dmax = inst["Dmax"]; Lmax = inst["Lmax"]
    order = sorted(range(1, 8), key=lambda c: (thr[c], c))
    basins = []
    for k in range(k_classes):
        c = order[k]
        thr_hi_excl = thr[order[k + 1]] if k + 1 < 7 else None
        d, l = find_basin(Q, thr[c], thr_hi_excl, scour[c], Dmax, Lmax)
        if d is None:
            continue
        basins.append((d, l, c))
    return basins, simulate(inst, basins)


def find_basin(Q, thr_lo, thr_hi_excl, scour_c, Dmax, Lmax):
    """Find integer (d,l) with 1<=d<=Dmax, 1<=l<=Lmax such that:
       d*l >= Q*thr_lo                (meets the residence-time threshold)
       d*l <  Q*thr_hi_excl  (if given)  (does NOT also cross the next boundary)
       Q <= scour_c * d               (scour-safe: no resuspension)
    Returns (d, l) or (None, None)."""
    d_min = -(-Q // scour_c) if scour_c > 0 else Dmax
    d_min = max(1, d_min)
    for d in range(d_min, Dmax + 1):
        need = Q * thr_lo
        l = -(-need // d)  # ceil
        if l < 1:
            l = 1
        if l > Lmax:
            continue
        prod = d * l
        if thr_hi_excl is not None and prod >= Q * thr_hi_excl:
            continue
        return d, l
    return None, None


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]
    inst = read_input(in_path)
    Q, VolumeBudget, M, Dmax, Lmax = (inst[k] for k in ("Q", "VolumeBudget", "M", "Dmax", "Lmax"))

    raw = open(out_path).read()
    toks = raw.split()
    if not toks:
        fail("empty output")

    mprime_tok = toks[0]
    mprime = parse_int_token(mprime_tok)
    if mprime is None:
        fail("bad basin count")
    if mprime < 0 or mprime > M:
        fail("basin count out of range")

    expected_tokens = 1 + 3 * mprime
    if len(toks) != expected_tokens:
        fail("token count mismatch: got %d expected %d" % (len(toks), expected_tokens))

    basins = []
    total_vol = 0
    for i in range(mprime):
        d_tok, l_tok, b_tok = toks[1 + 3 * i], toks[2 + 3 * i], toks[3 + 3 * i]
        d = parse_int_token(d_tok); l = parse_int_token(l_tok); b = parse_int_token(b_tok)
        if d is None or l is None or b is None:
            fail("non-finite/non-integer token in basin %d" % (i + 1))
        if not (1 <= d <= Dmax):
            fail("depth out of range in basin %d" % (i + 1))
        if not (1 <= l <= Lmax):
            fail("length out of range in basin %d" % (i + 1))
        if not (0 <= b <= 7):
            fail("bin id out of range in basin %d" % (i + 1))
        total_vol += d * l
        if total_vol > VolumeBudget:
            fail("volume budget exceeded")
        basins.append((d, l, b))

    F = simulate(inst, basins)
    if F < 0:
        fail("negative objective (should not happen)")

    # CAP: headroom multiplier so a near-perfect (all-7-classes-clean) solution does
    # not saturate the score against the (single easiest class) reference.
    CAP = 12
    _, B = internal_baseline(inst, 1)
    B = B if B > 0 else Fraction(1, 1)

    ratio_frac = F / (Fraction(CAP) * B)
    if ratio_frac > 1:
        ratio_frac = Fraction(1)
    ratio = float(ratio_frac)
    if ratio < 0:
        ratio = 0.0
    print("F=%s B=%s Ratio: %.6f" % (str(F), str(B), ratio))


if __name__ == "__main__":
    main()
