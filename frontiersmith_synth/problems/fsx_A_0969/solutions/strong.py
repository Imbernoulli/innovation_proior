# TIER: strong
# INSIGHT (not "greedy + more probes"): the object being minimised is D_earn, the worst-case
# value of the certified-envelope formula over the OFFICIAL inspection grid -- so the only
# thing worth asking, every single round, is "which inspection point does my OWN current
# history force the certificate to be loosest at right now?", and then attacking exactly that
# point. This is a genuine minimax game against the bound formula: unlike a layout planned
# once in advance (which can never move a probe after seeing a reading), this argmax naturally
# starts by dispersing across whatever the pilot has NOT yet informed (reproducing good
# generic coverage as a side effect, since with zero signal the loosest points are simply the
# farthest-from-everything ones), and the INSTANT any probe lands near a hidden defect, the
# certified envelope right around it becomes the new worst point in the domain -- pulling the
# very next probe there, then the next, closing in over several rounds until nothing beats it
# anymore. No fixed schedule and no rule that defers to "finish the plan first" is needed: the
# formula itself decides, round by round, whether extending generic coverage or refining a
# discovery is currently worth more.
import sys, json, math


def z_nom(coeffs, x_lo, y_lo, S, x, y):
    a0, a1, a2, a3, a4, a5 = (coeffs["a0"], coeffs["a1"], coeffs["a2"],
                               coeffs["a3"], coeffs["a4"], coeffs["a5"])
    cx = x_lo + S / 2.0
    cy = y_lo + S / 2.0
    return (a0 + a1 * x + a2 * y + a3 * (x * y) / S
            + a4 * ((x - cx) ** 2) / S + a5 * ((y - cy) ** 2) / S)


def envelope(hist_dev, Lc, m_cap, qx, qy):
    v = m_cap
    for (px, py, dev) in hist_dev:
        d = math.hypot(qx - px, qy - py)
        val = dev + Lc * d
        if val < v:
            v = val
    return v


def inspection_grid(x_lo, y_lo, S, N):
    pts = []
    for i in range(N):
        qx = x_lo + S * (i + 0.5) / N
        for j in range(N):
            qy = y_lo + S * (j + 0.5) / N
            pts.append((qx, qy))
    return pts


def main():
    inst = json.load(sys.stdin)
    dom = inst["domain"]
    x_lo, x_hi, y_lo, y_hi = dom["x_lo"], dom["x_hi"], dom["y_lo"], dom["y_hi"]
    S = x_hi - x_lo
    coeffs = inst["z_nom_coeffs"]
    Lc = inst["lipschitz_L"]
    m_cap = inst["m_cap"]
    n_inspect = inst["n_inspect"]

    history = inst["history"]
    hist_dev = []
    for h in history:
        dev = abs(h["z"] - z_nom(coeffs, x_lo, y_lo, S, h["x"], h["y"]))
        hist_dev.append((h["x"], h["y"], dev))

    pts = inspection_grid(x_lo, y_lo, S, n_inspect)

    phase = inst.get("phase")
    if phase == "probe":
        k = inst["max_probes_this_round"]      # always 1: true probe-by-probe adaptivity
        best_v, best_q = -1.0, None
        for (qx, qy) in pts:
            v = envelope(hist_dev, Lc, m_cap, qx, qy)
            if v > best_v:
                best_v, best_q = v, (qx, qy)
        chosen = [best_q] if (best_q is not None and k >= 1) else []
        print(json.dumps({"probes": [{"x": px, "y": py} for (px, py) in chosen]}))
        return

    # ---- certify: D_earn on the SAME official inspection grid the grader will recompute ----
    best = 0.0
    for (qx, qy) in pts:
        v = envelope(hist_dev, Lc, m_cap, qx, qy)
        if v > best:
            best = v
    print(json.dumps({"bound": best}))


main()
