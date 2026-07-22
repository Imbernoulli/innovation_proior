import sys
import itertools
from fractions import Fraction

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

# ---------- shared font construction (baseline normalizer) ----------
# Glyph model used only by the checker's internal baseline:
#   spine = column 0 inked on every row (guarantees 4-connectivity);
#   an "arm" at row r inks the entire row r.
# A glyph is thus determined by its set of arm rows (a codeword in {0,1}^H).
def build_glyph(arm_rows, H, W):
    g = [[0] * W for _ in range(H)]
    for r in range(H):
        g[r][0] = 1                     # spine
    for r in arm_rows:
        for c in range(W):
            g[r][c] = 1                 # arm (full row)
    return g

def subsets(H, w):
    return [set(c) for c in itertools.combinations(range(H), w)]

def baseline_codewords(N, H):
    # Deliberately LOW-distance code: the first two words differ in a single
    # arm row, so the min pairwise distance is only (W-1) -> a weak font.
    order, seen = [], set()
    def add(s):
        k = frozenset(s)
        if k not in seen:
            seen.add(k); order.append(set(s))
    add({0, 1}); add({0, 1, 2})
    for s in subsets(H, 2): add(s)
    for s in subsets(H, 3): add(s)
    return order[:N]

def baseline_font(N, H, W):
    return [build_glyph(cw, H, W) for cw in baseline_codewords(N, H)]

# ---------- objective on an arbitrary font ----------
def connected(g, H, W):
    cells = [(r, c) for r in range(H) for c in range(W) if g[r][c]]
    if not cells:
        return False
    start = cells[0]
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and g[nr][nc] and (nr, nc) not in seen:
                seen.add((nr, nc)); stack.append((nr, nc))
    return len(seen) == len(cells)

def hamming(a, b, H, W):
    d = 0
    for r in range(H):
        ar, br = a[r], b[r]
        for c in range(W):
            if ar[c] != br[c]:
                d += 1
    return d

def min_distance(fonts, H, W):
    n = len(fonts)
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming(fonts[i], fonts[j], H, W)
            if best is None or d < best:
                best = d
    return best if best is not None else 0

def vocab(fonts, H, W, mh, mw):
    seen = set()
    for g in fonts:
        for r in range(H - mh + 1):
            for c in range(W - mw + 1):
                key = 0
                bit = 1
                for dr in range(mh):
                    for dc in range(mw):
                        if g[r + dr][c + dc]:
                            key |= bit
                        bit <<= 1
                seen.add(key)
    return len(seen)

def objective(fonts, H, W, mh, mw, gnum, gden):
    D = min_distance(fonts, H, W)
    V = vocab(fonts, H, W, mh, mw)
    # D^2 / (1 + gamma V), gamma = gnum/gden  ==  D^2 * gden / (gden + gnum*V)
    return Fraction(D * D * gden, gden + gnum * V)

def main():
    try:
        inp = open(sys.argv[1]).read().split()
        it = iter(inp)
        N = int(next(it)); H = int(next(it)); W = int(next(it))
        inkLo = int(next(it)); inkHi = int(next(it))
        mh = int(next(it)); mw = int(next(it))
        gnum = int(next(it)); gden = int(next(it))
    except Exception:
        fail("bad input")

    # ---- parse participant font: exactly N*H*W tokens, each '0' or '1' ----
    raw = open(sys.argv[2]).read()
    if len(raw) > 5_000_000:
        fail("output too large")
    toks = raw.split()
    need = N * H * W
    if len(toks) != need:
        fail("expected %d values, got %d" % (need, len(toks)))
    for tk in toks:
        if tk != "0" and tk != "1":
            fail("non-binary token %r" % tk)

    fonts = []
    k = 0
    for _ in range(N):
        g = [[0] * W for _ in range(H)]
        for r in range(H):
            for c in range(W):
                g[r][c] = 1 if toks[k] == "1" else 0
                k += 1
        fonts.append(g)

    # ---- feasibility: ink budget + 4-connectivity per glyph ----
    for gi, g in enumerate(fonts):
        ink = sum(sum(row) for row in g)
        if ink < inkLo or ink > inkHi:
            fail("glyph %d ink %d out of [%d,%d]" % (gi, ink, inkLo, inkHi))
        if not connected(g, H, W):
            fail("glyph %d not 4-connected" % gi)

    F = objective(fonts, H, W, mh, mw, gnum, gden)

    # ---- internal baseline B (weak feasible font) ----
    B = objective(baseline_font(N, H, W), H, W, mh, mw, gnum, gden)
    if B <= 0:
        B = Fraction(1)

    sc = min(1000.0, 100.0 * float(F) / max(1e-9, float(B)))
    print("D=%d V=%d F=%.6f B=%.6f" % (
        min_distance(fonts, H, W), vocab(fonts, H, W, mh, mw), float(F), float(B)))
    print("Ratio: %.6f" % (sc / 1000.0))

if __name__ == "__main__":
    main()
