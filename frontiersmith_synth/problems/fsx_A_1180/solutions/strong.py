# TIER: strong
import sys, math


def allowed(h, k, l, cent):
    if h == 0 and k == 0 and l == 0:
        return False
    if cent == "P":
        return True
    if cent == "I":
        return (h + k + l) % 2 == 0
    if cent == "F":
        return (h % 2 == k % 2) and (k % 2 == l % 2)
    return False


def gen_candidates(cent, need):
    box = 3
    while True:
        cands = []
        for h in range(0, box + 1):
            for k in range(0, box + 1):
                for l in range(0, box + 1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    if not allowed(h, k, l, cent):
                        continue
                    cands.append((h * h + k * k + l * l, h, k, l))
        if len(cands) >= need + 3:
            cands.sort()
            return [(h, k, l) for _, h, k, l in cands]
        box += 1


def enumerate_cent(a, b, c, cent, cutoff_deg, lam):
    q_max = (2.0 * math.sin(math.radians(cutoff_deg / 2.0)) / lam) ** 2
    out = []
    hb = int(a * math.sqrt(q_max)) + 2
    for h in range(0, hb + 1):
        qh = (h / a) ** 2
        if qh > q_max + 1e-12:
            break
        kb = int(b * math.sqrt(max(0.0, q_max - qh))) + 2
        for k in range(0, kb + 1):
            qhk = qh + (k / b) ** 2
            if qhk > q_max + 1e-12:
                break
            lb = int(c * math.sqrt(max(0.0, q_max - qhk))) + 2
            for l in range(0, lb + 1):
                q = qhk + (l / c) ** 2
                if q > q_max + 1e-12:
                    break
                if h == 0 and k == 0 and l == 0:
                    continue
                if not allowed(h, k, l, cent):
                    continue
                sin_t = lam * math.sqrt(q) / 2.0
                if sin_t > 1.0:
                    continue
                theta2 = 2.0 * math.degrees(math.asin(sin_t))
                out.append((theta2, h, k, l))
    return out


INT_LO, INT_HI = 4, 13   # the disclosed range the hidden lattice constants live in
EXACT_TOL_DEG = 1e-3


def fit_hypothesis(cent, qs, gmax, lam):
    """Exhaustive search over the disclosed integer grid (exact, no interpolation gap):
    score each candidate cell by how many given peaks it explains to near machine
    precision, then tie-break by residual and by using the FEWEST possible reflections
    to do it (an oversized cell can also fit everything, just less specifically)."""
    best = None
    best_key = None
    for a in range(INT_LO, INT_HI + 1):
        for b in range(INT_LO, INT_HI + 1):
            for c in range(INT_LO, INT_HI + 1):
                af, bf, cf = float(a), float(b), float(c)
                cands = enumerate_cent(af, bf, cf, cent, gmax, lam)
                if not cands:
                    continue
                positions = [t_[0] for t_ in cands]
                resid = 0.0
                exact = 0
                for q in qs:
                    d = min(abs(p - q) for p in positions)
                    resid += d * d
                    if d < EXACT_TOL_DEG:
                        exact += 1
                key = (-exact, resid, len(cands))
                if best_key is None or key < best_key:
                    best_key = key
                    best = (af, bf, cf)
    a, b, c = best
    cands = enumerate_cent(a, b, c, cent, gmax, lam)
    if not cands:
        fallback = gen_candidates(cent, 1)[0]
        cands = [(0.0,) + fallback]
    assigned = []
    for q in qs:
        best_c = min(cands, key=lambda t_: abs(t_[0] - q))
        assigned.append((best_c[1], best_c[2], best_c[3]))
    return a, b, c, assigned


TOL_CHECK_DEG = 0.08   # must exceed the generator's peak-merge tolerance (0.05deg) so a
                        # merged group's representative position doesn't look like a miss
MISMATCH_TOL = 8       # true hypotheses land at <=5 mismatches (merge-drift / accidental
                        # near-coincidences); wrong hypotheses land at 14+ -- wide margin


def mismatch_count(cent, a, b, c, gmax, qs, lam):
    """THE INSIGHT: score a hypothesis by whether the reflections it forbids are
    genuinely silent and the ones it allows are genuinely present, not by how well
    it merely fits the peaks it was handed (any hypothesis can be made to fit those --
    see fit_hypothesis above, which finds a clean fit under EVERY centering)."""
    mism = 0
    for (theta2, h, k, l) in enumerate_cent(a, b, c, "P", gmax, lam):
        would_allow = allowed(h, k, l, cent)
        is_obs = any(abs(theta2 - q) <= TOL_CHECK_DEG for q in qs)
        if would_allow != is_obs:
            mism += 1
            if mism > 50:
                return mism
    return mism


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    _t = int(next(it))
    lam = float(next(it))
    gmax = float(next(it))
    _fmax = float(next(it))
    M = int(next(it))
    qs = [float(next(it)) for _ in range(M)]

    fits = {}
    for cent in ("P", "I", "F"):
        fits[cent] = fit_hypothesis(cent, qs, gmax, lam)

    chosen = None
    for cent in ("F", "I", "P"):
        a, b, c, _assigned = fits[cent]
        mism = mismatch_count(cent, a, b, c, gmax, qs, lam)
        if mism <= MISMATCH_TOL:
            chosen = cent
            break
    if chosen is None:
        chosen = "P"

    a, b, c, assigned = fits[chosen]
    out = ["%.6f %.6f %.6f" % (a, b, c), chosen]
    for (h, k, l) in assigned:
        out.append("%d %d %d" % (h, k, l))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
