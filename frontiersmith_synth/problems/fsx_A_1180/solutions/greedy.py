# TIER: greedy
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
    """The lattice constants are disclosed to be integers in [4,13]: an EXHAUSTIVE
    search over that finite grid is exact (no interpolation gap), so the true cell is
    always reachable. Score each candidate cell by (a) how many given peaks it explains
    to near machine precision, then tie-break by (b) total residual and (c) how many
    reflections it would place in range at all -- preferring the cell that explains the
    data with the FEWEST possible reflections, not just the cell with the most of them."""
    best = None
    best_key = None
    best_assigned = None
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
                    best_assigned = None  # computed lazily below only for the winner
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


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    _t = int(next(it))
    lam = float(next(it))
    gmax = float(next(it))
    _fmax = float(next(it))
    M = int(next(it))
    qs = [float(next(it)) for _ in range(M)]

    a, b, c, assigned = fit_hypothesis("P", qs, gmax, lam)

    out = ["%.6f %.6f %.6f" % (a, b, c), "P"]
    for (h, k, l) in assigned:
        out.append("%d %d %d" % (h, k, l))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
