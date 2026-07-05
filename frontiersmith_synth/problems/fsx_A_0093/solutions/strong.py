# TIER: strong
# Best-first greedy insertion over a fine candidate grid: at each step place the single
# feasible zone with the largest achievable radius, naturally filling several concentric
# layers of the annulus. Deterministic (fixed grid, fixed tie-breaking).
import sys, math

def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); R = float(toks[1]); r_in = float(toks[2])
    tol = 1e-6

    G = 46
    cand = []
    for gy in range(G):
        for gx in range(G):
            x = -R + (gx + 0.5) * (2.0 * R / G)
            y = -R + (gy + 0.5) * (2.0 * R / G)
            d = math.hypot(x, y)
            if d < r_in or d > R:
                continue
            cand.append((x, y, d))

    placed = []
    for _ in range(N):
        best = None
        best_r = 1e-4
        for (x, y, d) in cand:
            r = min(R - d, d - r_in)
            if r <= best_r:
                continue
            ok = True
            for (px, py, pr) in placed:
                lim = math.hypot(x - px, y - py) - pr
                if lim < r:
                    r = lim
                    if r <= best_r:
                        ok = False
                        break
            if ok and r > best_r:
                best_r = r
                best = (x, y, r - tol)
        if best is None:
            break
        placed.append(best)

    out = [str(len(placed))]
    for (x, y, r) in placed:
        out.append("%.10f %.10f %.10f" % (x, y, r))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
