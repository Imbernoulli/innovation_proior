# TIER: greedy
# The "obvious first attempt": pre-plan a regular, edge-to-edge 4x4 grid across the whole
# panel BEFORE looking at any history, spend the entire budget laying it down exactly as
# planned (ignores the running history completely when choosing WHERE to probe), then
# honestly compute the tightest bound the standard Lipschitz-envelope formula allows over the
# official inspection grid from whatever the layout happened to land on. This "textbook
# coverage" recipe is a genuinely decent, near-optimal blind layout for GENERIC coverage --
# but it is systematically loose wherever a defect happens to sit far from every point of
# EVERY plausible fixed layout, since it can never revisit a still-loose spot once its 16
# probes are spent.
import sys, json, math


def z_nom(coeffs, x_lo, y_lo, S, x, y):
    a0, a1, a2, a3, a4, a5 = (coeffs["a0"], coeffs["a1"], coeffs["a2"],
                               coeffs["a3"], coeffs["a4"], coeffs["a5"])
    cx = x_lo + S / 2.0
    cy = y_lo + S / 2.0
    return (a0 + a1 * x + a2 * y + a3 * (x * y) / S
            + a4 * ((x - cx) ** 2) / S + a5 * ((y - cy) ** 2) / S)


def main():
    inst = json.load(sys.stdin)
    dom = inst["domain"]
    x_lo, x_hi, y_lo, y_hi = dom["x_lo"], dom["x_hi"], dom["y_lo"], dom["y_hi"]
    S = x_hi - x_lo
    coeffs = inst["z_nom_coeffs"]
    Lc = inst["lipschitz_L"]
    m_cap = inst["m_cap"]
    n_inspect = inst["n_inspect"]

    GX = GY = 4
    grid_lines = [x_lo + S * i / (GX - 1) for i in range(GX)]
    all_grid = [(gx, gy) for gx in grid_lines for gy in grid_lines]   # 16 fixed points

    phase = inst.get("phase")
    if phase == "probe":
        rnd = inst["round"]
        max_this = inst["max_probes_this_round"]
        start = rnd * max_this
        pts = all_grid[start:start + max_this]
        print(json.dumps({"probes": [{"x": px, "y": py} for (px, py) in pts]}))
        return

    # ---- certify: honestly compute the tightest sound bound over the OFFICIAL inspection grid ----
    history = inst["history"]
    hist_dev = []
    for h in history:
        dev = abs(h["z"] - z_nom(coeffs, x_lo, y_lo, S, h["x"], h["y"]))
        hist_dev.append((h["x"], h["y"], dev))

    N = n_inspect
    best = 0.0
    for i in range(N):
        qx = x_lo + S * (i + 0.5) / N
        for j in range(N):
            qy = y_lo + S * (j + 0.5) / N
            v = m_cap
            for (px, py, dev) in hist_dev:
                d = math.hypot(qx - px, qy - py)
                val = dev + Lc * d
                if val < v:
                    v = val
            if v > best:
                best = v
    print(json.dumps({"bound": best}))


main()
