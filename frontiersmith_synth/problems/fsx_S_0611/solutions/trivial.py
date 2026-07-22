# TIER: trivial
# Reproduces the checker's internal weak baseline font (low min-distance code).
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

def baseline_codewords(N, H):
    order, seen = [], set()
    def add(s):
        k = frozenset(s)
        if k not in seen:
            seen.add(k); order.append(set(s))
    add({0, 1}); add({0, 1, 2})
    for s in subsets(H, 2): add(s)
    for s in subsets(H, 3): add(s)
    return order[:N]

def main():
    d = sys.stdin.read().split()
    N, H, W = int(d[0]), int(d[1]), int(d[2])
    out = []
    for cw in baseline_codewords(N, H):
        g = build_glyph(cw, H, W)
        for r in range(H):
            out.extend(str(x) for x in g[r])
    sys.stdout.write(" ".join(out) + "\n")

main()
