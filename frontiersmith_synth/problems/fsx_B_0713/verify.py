import sys, math

# ---- shared canopy model (checker ground truth) ----------------------------------
# Full binary L-system skeleton of depth D. Every node (1..D) contributes wood length
# to the structural cost. Only the 2^(D-1) TERMINAL tips (depth == D) bear a leaf disk
# of area A0*taper. Rays from the sun schedule cast deterministic parallel shadows:
# leaves are processed nearest-sun-first, and a leaf's harvested area is the fraction
# of its projected footprint not already covered by nearer leaves (nearer leaves cover
# their FULL footprint regardless of their own exposure -- they still block light).

R_LO, R_HI = 0.3, 0.95
THETA_LO, THETA_HI = 1.0, 80.0
TAPER_LO, TAPER_HI = 0.3, 1.0
SKEW_LO, SKEW_HI = -0.5, 0.5

BASE_PARAMS = (0.45, 38.0, 0.6, 0.0)  # checker's own trivial construction (r, theta, taper, skew)


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
    leaf_area = A0 * taper
    return tips, leaf_area, total_length[0]


def sweep_harvest(tips, leaf_area, alpha_deg):
    a = math.radians(alpha_deg)
    dx, dy = math.sin(a), -math.cos(a)   # ray travel direction (downward-ish)
    ex, ey = math.cos(a), math.sin(a)    # projection axis, perpendicular to the ray
    rho = math.sqrt(leaf_area / math.pi) if leaf_area > 0 else 0.0
    items = []
    for (x, y) in tips:
        t = x * dx + y * dy               # depth along ray: smaller = closer to sun
        s = x * ex + y * ey                # projected (shadow) position
        items.append((t, s - rho, s + rho))
    items.sort(key=lambda z: z[0])

    union = []   # sorted, disjoint covered intervals (from nearer-sun leaves so far)
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
    total_leafarea = len(tips) * leaf_area
    cost = cost_len * total_length + cost_leaf * total_leafarea
    return harvested - cost


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    D = int(next(it))
    L0 = float(next(it)); A0 = float(next(it))
    cost_len = float(next(it)); cost_leaf = float(next(it))
    K = int(next(it))
    suns = []
    for _ in range(K):
        ang = float(next(it)); w = float(next(it))
        suns.append((ang, w))
    return D, L0, A0, cost_len, cost_leaf, suns


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    try:
        D, L0, A0, cost_len, cost_leaf, suns = read_instance(in_path)
    except Exception as e:
        print("Ratio: 0.0  # bad instance: %s" % e)
        return 0

    try:
        with open(out_path) as f:
            toks = f.read().split()
    except Exception:
        print("Ratio: 0.0  # cannot read output")
        return 0

    if len(toks) != 4:
        print("Ratio: 0.0  # expected exactly 4 numbers (r theta taper skew), got %d" % len(toks))
        return 0

    try:
        vals = [float(t) for t in toks]
    except Exception:
        print("Ratio: 0.0  # non-numeric token")
        return 0

    if not all(math.isfinite(v) for v in vals):
        print("Ratio: 0.0  # non-finite value")
        return 0

    r, theta, taper, skew = vals
    if not (R_LO - 1e-9 <= r <= R_HI + 1e-9):
        print("Ratio: 0.0  # r out of [%.2f,%.2f]" % (R_LO, R_HI))
        return 0
    if not (THETA_LO - 1e-9 <= theta <= THETA_HI + 1e-9):
        print("Ratio: 0.0  # theta out of [%.1f,%.1f]" % (THETA_LO, THETA_HI))
        return 0
    if not (TAPER_LO - 1e-9 <= taper <= TAPER_HI + 1e-9):
        print("Ratio: 0.0  # taper out of [%.2f,%.2f]" % (TAPER_LO, TAPER_HI))
        return 0
    if not (SKEW_LO - 1e-9 <= skew <= SKEW_HI + 1e-9):
        print("Ratio: 0.0  # skew out of [%.2f,%.2f]" % (SKEW_LO, SKEW_HI))
        return 0

    F = objective(D, L0, A0, cost_len, cost_leaf, suns, (r, theta, taper, skew))
    B = objective(D, L0, A0, cost_len, cost_leaf, suns, BASE_PARAMS)

    sc = 100.0 * F / max(1e-9, B)
    sc = max(0.0, min(1000.0, sc))
    ratio = sc / 1000.0
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, ratio))
    return 0


if __name__ == "__main__":
    sys.exit(main())
