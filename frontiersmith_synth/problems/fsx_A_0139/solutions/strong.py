# TIER: strong
# Fixed-aware portfolio completion. Build several candidate fills of the m new
# cameras (Hammersley, a handful of Fibonacci/Korobov rank-1 lattices, a jittered
# grid), UNION each with the pre-installed fixed cameras, and keep the candidate
# whose EXACT full-set star discrepancy is smallest. Because it (a) evaluates the
# true union objective and (b) tries constructions the greedy Hammersley fill does
# not, it reliably matches-or-beats greedy and blows past the diagonal baseline.
import sys

def star_discrepancy(pts, n):
    xs = sorted(set([p[0] for p in pts] + [1.0]))
    ys = sorted(set([p[1] for p in pts] + [1.0]))
    best = 0.0
    for qx in xs:
        for qy in ys:
            V = qx * qy
            nc = 0; no = 0
            for (x, y) in pts:
                if x <= qx and y <= qy:
                    nc += 1
                    if x < qx and y < qy:
                        no += 1
            dplus = nc / n - V
            dminus = V - no / n
            mm = dplus if dplus > dminus else dminus
            if mm > best:
                best = mm
    return best

def vdc(i, base=2):
    f = 1.0; r = 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r

def hammersley(m):
    return [((i + 0.5) / m, vdc(i, 2)) for i in range(m)]

def lattice(m, g):
    return [((i + 0.5) / m, (((i * g) % m) + 0.5) / m) for i in range(m)]

def jittered_grid(m):
    # near-square grid, cell-centered (deterministic, no randomness)
    import math
    cols = int(math.ceil(math.sqrt(m)))
    rows = int(math.ceil(m / cols))
    pts = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= m:
                break
            pts.append(((c + 0.5) / cols, (r + 0.5) / rows))
            idx += 1
    return pts

def candidate_generators(m):
    # generators near the golden ratio give strong rank-1 lattices
    g0 = int(round(m * 0.6180339887498949))
    cands = set()
    for g in (g0 - 1, g0, g0 + 1):
        if 1 <= g < m:
            cands.add(g)
    # a couple of extra coprime-ish spreads
    for g in (int(round(m * 0.381966)), int(round(m * 0.7236068))):
        if 1 <= g < m:
            cands.add(g)
    return sorted(cands)

def main():
    toks = sys.stdin.read().split()
    M = int(toks[0]); k = int(toks[1])
    fixed = []
    for j in range(k):
        fixed.append((float(toks[2 + 2 * j]), float(toks[2 + 2 * j + 1])))
    m = M - k

    candidates = [hammersley(m), jittered_grid(m)]
    for g in candidate_generators(m):
        candidates.append(lattice(m, g))

    best_pts = None
    best_d = None
    for cand in candidates:
        d = star_discrepancy(fixed + cand, M)
        if best_d is None or d < best_d:
            best_d = d
            best_pts = cand

    out = ["%.10f %.10f" % (x, y) for (x, y) in best_pts]
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
