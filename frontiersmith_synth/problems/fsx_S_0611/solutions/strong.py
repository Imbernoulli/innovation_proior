# TIER: strong
# INSIGHT (error-correcting-code reformulation of typographic style):
# fix a tiny motif basis first -- here every glyph is built from ONE stroke type,
# the full-row arm on a shared spine -- then place glyphs as far-apart codewords
# in that basis. Restricting to full-row arms pins the 2x3 motif vocabulary to a
# handful of windows (style coherence), while choosing an EVEN-weight code over the
# arm rows forces every pair of glyphs to differ in >=2 rows, i.e. min Hamming
# distance 2*(W-1). Coherence and distinctness are obtained together instead of
# being traded against each other.
import sys, itertools

def build_glyph(arm_rows, H, W):
    g = [[0] * W for _ in range(H)]
    for r in range(H):
        g[r][0] = 1
    for r in arm_rows:
        for c in range(W):
            g[r][c] = 1
    return g

def subsets(H, w):
    return [set(c) for c in itertools.combinations(range(H), w)]

def main():
    d = sys.stdin.read().split()
    N, H, W = int(d[0]), int(d[1]), int(d[2])
    inkLo, inkHi = int(d[3]), int(d[4])
    # Even-weight codewords (weights 2 then 4) whose row-ink stays in budget.
    order = []
    for w in (2, 4, 6, 8):
        ink = H + w * (W - 1)          # spine H + w full-row arms
        if ink < inkLo or ink > inkHi:
            continue
        order.extend(subsets(H, w))
    order = order[:N]
    out = []
    for cw in order:
        g = build_glyph(cw, H, W)
        for r in range(H):
            out.extend(str(x) for x in g[r])
    sys.stdout.write(" ".join(out) + "\n")

main()
