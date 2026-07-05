# TIER: greedy
# Concentric-ring filler: stack L equal-thickness rings across the annulus (trying L=1..8 and
# keeping the best), spreading the N zones over the rings. Beats the single-ring baseline by
# using more than one radial layer. Deterministic.
import sys, math

def build(N, R, r_in, L):
    tol = 1e-6
    placed = []
    remaining = N
    th = (R - r_in) / L            # radial thickness of each ring layer
    rr = th / 2.0 * (1.0 - 1e-7)   # zone radius (radial half-width of a layer)
    for layer in range(L):
        if remaining <= 0:
            break
        rc = r_in + (layer + 0.5) * th        # centre radius of this ring
        # how many zones of radius rr fit on the circle of radius rc without overlap
        s = rr / rc
        if s >= 1.0:
            cnt = 1
        else:
            cnt = int(math.pi / math.asin(min(1.0, s)))
        cnt = max(0, min(cnt, remaining))
        for i in range(cnt):
            a = 2.0 * math.pi * i / cnt + layer * 0.3   # stagger layers
            placed.append((rc * math.cos(a), rc * math.sin(a), rr))
        remaining -= cnt
    return placed

def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); R = float(toks[1]); r_in = float(toks[2])
    best = []
    best_sum = -1.0
    for L in range(1, 9):
        p = build(N, R, r_in, L)
        s = sum(r for (_, _, r) in p)
        if s > best_sum:
            best_sum = s
            best = p
    out = [str(len(best))]
    for (x, y, r) in best:
        out.append("%.10f %.10f %.10f" % (x, y, r))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
