import sys, json, math, isorun
import numpy as np

# ==========================================================================
# fsx_B_0379 -- seeded-numerical-sim (Format B, isolated candidate)
# Theme: "asteroid mining" -- design a phase-only diffractive optical element
# (DOE / phase mask) that splits a single high-power ablation laser into a
# prescribed array of K surface spots. The evaluator simulates Fraunhofer
# propagation (a 2-D FFT) with a fixed, deterministic grid and returns a
# composite diffraction-efficiency x uniformity metric.
# Objective: MAXIMIZE M = eta * U. Ill-formed / non-finite masks score 0.
# Normalization: superposition-of-gratings reference DOE == 0.1.
# ==========================================================================


def _far_metric(phi, targets, N):
    """Simulate propagation for phase array phi (N x N). Return composite M."""
    A = np.exp(1j * phi)
    G = np.fft.fft2(A)
    I = np.abs(G) ** 2
    P = float(I.sum())
    if not np.isfinite(P) or P <= 0.0:
        return 0.0
    spot = np.array([float(I[ky % N, kx % N]) for (ky, kx) in targets], dtype=float)
    if not np.all(np.isfinite(spot)):
        return 0.0
    eta = float(spot.sum()) / P
    mx = float(spot.max())
    mn = float(spot.min())
    U = 1.0 - (mx - mn) / (mx + mn + 1e-30)
    M = eta * U
    if not (M == M) or M < 0.0:
        return 0.0
    return M


def _superposition_phase(targets, N):
    """Classic superposition-of-gratings reference DOE (deterministic)."""
    y, x = np.mgrid[0:N, 0:N]
    acc = np.zeros((N, N), dtype=complex)
    for (ky, kx) in targets:
        acc += np.exp(1j * 2.0 * np.pi * (ky * y + kx * x) / N)
    return np.angle(acc)


def _pattern(name, N):
    c = N // 2
    if name == "grid3x3_s4":
        pts = [[c - 4 + 4 * i, c - 4 + 4 * j] for i in range(3) for j in range(3)]
    elif name == "grid3x3_s5":
        pts = [[c - 5 + 5 * i, c - 5 + 5 * j] for i in range(3) for j in range(3)]
    elif name == "Lshape":
        pts = [[c + d, c - 6] for d in (-6, -3, 0, 3, 6)] + \
              [[c + 6, c + d] for d in (-3, 0, 3, 6)]
    elif name == "grid2x3":
        pts = [[c - 4 + 8 * i, c - 3 + 3 * j] for i in range(2) for j in range(3)]
    elif name == "ring8_r7":
        pts = [[int(round(c + 7 * math.sin(2 * math.pi * t / 8))),
                int(round(c + 7 * math.cos(2 * math.pi * t / 8)))] for t in range(8)]
    elif name == "cross9":
        pts = [[c, c]] + [[c + d, c] for d in (-6, -3, 3, 6)] + \
              [[c, c + d] for d in (-6, -3, 3, 6)]
    elif name == "crossX":
        pts = [[c + d, c + d] for d in (-6, -3, 3, 6)] + \
              [[c + d, c - d] for d in (-6, -3, 3, 6)]
    elif name == "grid2x2_s6":
        pts = [[c - 3 + 6 * i, c - 3 + 6 * j] for i in range(2) for j in range(2)]
    else:
        raise ValueError(name)
    # dedup preserving order, drop the DC pixel if it ever appears
    seen = []
    for p in pts:
        t = (p[0] % N, p[1] % N)
        if t == (0, 0):
            continue
        if t not in [tuple(q) for q in seen]:
            seen.append([t[0], t[1]])
    return seen


def make_instances():
    N = 24
    specs = [
        ("grid3x3_s4", 3701),
        ("grid3x3_s5", 3702),
        ("Lshape",     3703),
        ("grid2x3",    3704),
        ("ring8_r7",   3705),
        ("cross9",     3706),
        ("crossX",     3707),
        ("grid2x2_s6", 3708),
    ]
    out = []
    for name, seed in specs:
        targets = _pattern(name, N)
        pub = {"N": N, "targets": targets, "seed": seed, "pattern": name}
        out.append({"public": pub, "hidden": {}})
    return out


def baseline(inst):
    pub = inst["public"]
    N = pub["N"]
    targets = pub["targets"]
    phi = _superposition_phase(targets, N)
    return _far_metric(phi, targets, N)


def score(inst, ans):
    pub = inst["public"]
    N = pub["N"]
    targets = pub["targets"]
    if not isinstance(ans, dict) or "phase" not in ans:
        return False, 0.0
    ph = ans["phase"]
    if not isinstance(ph, list) or len(ph) != N:
        return False, 0.0
    grid = []
    for row in ph:
        if not isinstance(row, list) or len(row) != N:
            return False, 0.0
        clean = []
        for v in row:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return False, 0.0
            v = float(v)
            if v != v or v in (float("inf"), float("-inf")):
                return False, 0.0
            clean.append(v)
        grid.append(clean)
    phi = np.asarray(grid, dtype=float)
    if phi.shape != (N, N) or not np.all(np.isfinite(phi)):
        return False, 0.0
    M = _far_metric(phi, targets, N)
    if not (M == M) or M < 0.0 or not math.isfinite(M):
        return False, 0.0
    return True, M


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, stt = isorun.run_candidate(cand, inst["public"], timeout=20)
        if stt != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        # maximization F/B analog: reproducing the reference DOE (obj==b) -> 0.1
        r = min(1.0, 0.1 * obj / max(b, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
