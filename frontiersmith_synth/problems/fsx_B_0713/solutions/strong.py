# TIER: strong
# Insight: the checker's own scoring formula is cheap to simulate locally (a handful of
# leaves, a 1-D sweep per sun ray), so instead of hard-coding a recipe we treat parameter
# choice as a tiny 4-D black-box optimization and run a seeded coordinate/random local
# search AGAINST THE ACTUAL INSTANCE (its depth, its biomass cost coefficients, its own
# multi-ray sun schedule). That is what lets it dodge the two failure modes a fixed
# recipe cannot see coming: (a) a narrow branching angle that self-shades once the crown
# gets deep, and (b) a large length/taper that looks good on light alone but is not worth
# its structural cost once cost_len/cost_leaf are read from the input.
import sys, math, random


def build_tips(D, L0, r, theta_deg, taper, skew, A0):
    theta = math.radians(theta_deg)
    tips = []
    total_length = [0.0]

    def rec(x, y, phi, length, depth):
        nx = x + length * math.cos(phi)
        ny = y + length * math.sin(phi)
        total_length[0] += length
        if depth < D:
            dl = theta * (1.0 + skew)
            dr = theta * (1.0 - skew)
            rec(nx, ny, phi + dl, length * r, depth + 1)
            rec(nx, ny, phi - dr, length * r, depth + 1)
        else:
            tips.append((nx, ny))

    rec(0.0, 0.0, math.pi / 2.0, L0, 1)
    return tips, A0 * taper, total_length[0]


def sweep_harvest(tips, leaf_area, alpha_deg):
    a = math.radians(alpha_deg)
    dx, dy = math.sin(a), -math.cos(a)
    ex, ey = math.cos(a), math.sin(a)
    rho = math.sqrt(leaf_area / math.pi) if leaf_area > 0 else 0.0
    items = []
    for (x, y) in tips:
        t = x * dx + y * dy
        s = x * ex + y * ey
        items.append((t, s - rho, s + rho))
    items.sort(key=lambda z: z[0])
    union = []
    harvested = 0.0
    for _, lo, hi in items:
        covered = 0.0
        for (ulo, uhi) in union:
            if uhi <= lo or ulo >= hi:
                continue
            covered += min(hi, uhi) - max(lo, ulo)
        span = hi - lo
        if span <= 1e-12:
            pt_covered = any(ulo <= lo <= uhi for (ulo, uhi) in union)
            frac = 0.0 if pt_covered else 1.0
        else:
            frac = max(0.0, span - covered) / span
        harvested += leaf_area * frac
        newu, nlo, nhi = [], lo, hi
        for (ulo, uhi) in union:
            if uhi < nlo or ulo > nhi:
                newu.append((ulo, uhi))
            else:
                nlo = min(nlo, ulo); nhi = max(nhi, uhi)
        newu.append((nlo, nhi))
        newu.sort()
        union = newu
    return harvested


def objective(D, L0, A0, cost_len, cost_leaf, suns, params):
    r, theta, taper, skew = params
    tips, leaf_area, total_length = build_tips(D, L0, r, theta, taper, skew, A0)
    wsum = sum(w for _, w in suns) or 1.0
    harvested = sum((w / wsum) * sweep_harvest(tips, leaf_area, ang) for ang, w in suns)
    cost = cost_len * total_length + cost_leaf * (len(tips) * leaf_area)
    return harvested - cost


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    D = int(next(it))
    L0 = float(next(it)); A0 = float(next(it))
    cost_len = float(next(it)); cost_leaf = float(next(it))
    K = int(next(it))
    suns = [(float(next(it)), float(next(it))) for _ in range(K)]

    bounds = [(0.3, 0.95), (1.0, 80.0), (0.3, 1.0), (-0.5, 0.5)]
    # init: shrink the branching angle a bit for deeper crowns (avoids angle wrap),
    # keep a fair amount of biomass to start from.
    best = [0.6, min(55.0, 380.0 / max(D, 1)), 0.75, 0.0]
    for k in range(4):
        lo, hi = bounds[k]
        best[k] = min(hi, max(lo, best[k]))
    best_F = objective(D, L0, A0, cost_len, cost_leaf, suns, best)

    rng = random.Random(4242 + D)  # deterministic given the instance's own depth
    step = [0.15, 15.0, 0.2, 0.15]
    iters = 320
    for it_i in range(iters):
        cand = best[:]
        k = rng.randrange(4)
        cand[k] += rng.uniform(-1.0, 1.0) * step[k]
        lo, hi = bounds[k]
        cand[k] = min(hi, max(lo, cand[k]))
        F = objective(D, L0, A0, cost_len, cost_leaf, suns, cand)
        if F > best_F:
            best_F, best = F, cand
        if it_i % 60 == 59:
            step = [s * 0.6 for s in step]

    print("%.6f %.6f %.6f %.6f" % tuple(best))


if __name__ == "__main__":
    main()
