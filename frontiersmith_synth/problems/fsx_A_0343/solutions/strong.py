# TIER: strong
# Two-phase refinement: (1) spread centers with a deterministic physical relaxation
# (disk-repulsion) over several seeded restarts, then (2) tighten radii to the largest
# feasible values by iterated coordinate ascent. Keeps the best feasible total.
import sys, math, random

TOL = 1e-6

def tighten_radii(centers, W, H, rmax, sweeps=40):
    # Iterated coordinate ascent: r_i = min(walls, rmax, min_j dist_ij - r_j).
    # Gauss-Seidel order guarantees the layout is feasible after each full sweep.
    n = len(centers)
    r = [0.0] * n
    for _ in range(sweeps):
        for i in range(n):
            xi, yi = centers[i]
            ri = min(xi, W - xi, yi, H - yi, rmax)
            for j in range(n):
                if j == i:
                    continue
                xj, yj = centers[j]
                d = math.hypot(xi - xj, yi - yj) - r[j]
                if d < ri:
                    ri = d
            if ri < 0.0:
                ri = 0.0
            r[i] = ri
    return r

def relax_centers(centers, W, H, rho, iters=60):
    # Push overlapping disks (assumed radius rho) apart; clamp into the box.
    n = len(centers)
    c = [[x, y] for (x, y) in centers]
    for _ in range(iters):
        for i in range(n):
            for j in range(i + 1, n):
                dx = c[i][0] - c[j][0]
                dy = c[i][1] - c[j][1]
                d = math.hypot(dx, dy)
                mind = 2.0 * rho
                if d < mind and d > 1e-12:
                    push = 0.5 * (mind - d) / d
                    c[i][0] += dx * push; c[i][1] += dy * push
                    c[j][0] -= dx * push; c[j][1] -= dy * push
        lo = rho * 0.05
        for i in range(n):
            if c[i][0] < lo: c[i][0] = lo
            if c[i][0] > W - lo: c[i][0] = W - lo
            if c[i][1] < lo: c[i][1] = lo
            if c[i][1] > H - lo: c[i][1] = H - lo
    return [(p[0], p[1]) for p in c]

def grid_centers(N, W, H):
    G = int(math.ceil(math.sqrt(N)))
    R = int(math.ceil(N / float(G)))
    cw = W / G; ch = H / R
    cs = []
    k = 0
    for row in range(R):
        for col in range(G):
            if k >= N:
                break
            cs.append(((col + 0.5) * cw, (row + 0.5) * ch))
            k += 1
    return cs

def total(centers, W, H, rmax):
    r = tighten_radii(centers, W, H, rmax)
    return sum(r), centers, r

def main():
    toks = sys.stdin.read().split()
    N = int(toks[0]); W = float(toks[1]); H = float(toks[2]); rmax = float(toks[3])
    rng = random.Random(1234567 + N)

    base = grid_centers(N, W, H)
    best_sum, best_c, best_r = total(base, W, H, rmax)

    # target equal radius for the repulsion phase (area heuristic)
    rho0 = min(rmax, 0.5 * math.sqrt((W * H) / (N * math.pi)) * 1.6)

    for trial in range(6):
        rho = rho0 * (0.85 + 0.06 * trial)
        cs = []
        for (x, y) in base:
            jx = (rng.random() - 0.5) * rho * 0.8
            jy = (rng.random() - 0.5) * rho * 0.8
            cs.append((x + jx, y + jy))
        cs = relax_centers(cs, W, H, rho, iters=70)
        s, c, r = total(cs, W, H, rmax)
        if s > best_sum:
            best_sum, best_c, best_r = s, c, r

    out = ["%.9f %.9f %.9f" % (best_c[i][0], best_c[i][1], best_r[i]) for i in range(N)]
    sys.stdout.write("\n".join(out) + "\n")

main()
