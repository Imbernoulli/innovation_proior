import sys, json, numpy as np, isorun

# ==========================================================================
# fsx_B_0169 -- seeded-numerical-sim (Format B, isolated candidate)
# Theme: "festival stage layout" -- a single laser projector illuminates a
# binary phase-plate ("gobo" / DMD) that must fan the beam out into a chosen
# constellation of light spots on the stage backdrop. The plate has only
# L=2 physical phase states (0 or pi), so its far field is conjugate-symmetric
# and roughly half the light is unavoidably lost to a mirror-image order.
#
# The designer emits an N x N phase mask. The evaluator SNAPS every value to
# the nearest of the L device levels (untrusted output cannot dodge the
# hardware constraint), Fraunhofer-propagates it with a fixed FFT, and grades
#   Q = efficiency * uniformity**unif_power
#   efficiency = (light landing on the target spots) / (total light)
#   uniformity = min_spot_intensity / max_spot_intensity   (equal brightness)
# Objective: MAXIMIZE Q. The reference (baseline) construction is a
# random-phase superposition of blazed gratings; a good design beats it by
# iteratively balancing the spots (Gerchberg-Saxton / IFTA style).
# Everything is deterministic and seeded; scoring is pure numpy.
# ==========================================================================


def _grating(N, r, c):
    cy = N // 2
    cx = N // 2
    ky = r - cy
    kx = c - cx
    y, x = np.mgrid[0:N, 0:N]
    return 2.0 * np.pi * (ky * y + kx * x) / N


def _rand_super_phase(N, spots, seed):
    """Reference construction: superpose one blazed grating per target spot,
    each with an independent random piston phase, then keep the argument."""
    rng = np.random.default_rng(seed)
    field = np.zeros((N, N), dtype=complex)
    for (r, c) in spots:
        field += np.exp(1j * (_grating(N, r, c) + rng.uniform(0.0, 2.0 * np.pi)))
    return np.angle(field)


def _quantize(phase, L):
    step = 2.0 * np.pi / L
    return (np.round(phase / step) * step) % (2.0 * np.pi)


def _Q(phase, spots, L, p):
    """Composite figure of merit for a (continuous) phase mask."""
    q = _quantize(phase, L)
    fld = np.fft.fftshift(np.fft.fft2(np.exp(1j * q)))
    I = np.abs(fld) ** 2
    total = I.sum()
    if total <= 0:
        return 0.0
    Ik = np.array([I[r, c] for (r, c) in spots], dtype=float)
    mx = Ik.max()
    if mx <= 0:
        return 0.0
    eta = float(Ik.sum() / total)
    U = float(Ik.min() / mx)
    val = eta * (U ** p)
    if val != val or val < 0:
        return 0.0
    return val


def make_instances():
    # (N aperture pixels, M target spots, seed). A spread of sizes/densities;
    # the denser / larger boards are the harder, generalization-flavoured cases.
    specs = [
        (48, 10, 1),
        (64, 14, 2),
        (64, 20, 3),
        (48, 16, 4),
        (56, 12, 5),
        (64, 24, 6),
        (48, 8, 7),
        (56, 18, 8),
    ]
    L = 2
    p = 3.0
    out = []
    for (N, M, seed) in specs:
        rng = np.random.default_rng(1000 + seed)
        cy = N // 2
        cx = N // 2
        R = N // 4
        coords = set()
        # spots confined to the right half-plane (dc >= 1) so the target set is
        # asymmetric -- the binary plate's forced twin order is genuinely wasted.
        while len(coords) < M:
            dr = int(rng.integers(-R, R + 1))
            dc = int(rng.integers(1, R + 1))
            coords.add((cy + dr, cx + dc))
        spots = sorted(coords)
        base_seed = 100 + seed  # seed of the reference random-superposition
        pub = {
            "N": N,
            "L": L,
            "unif_power": p,
            "spots": [[int(r), int(c)] for (r, c) in spots],
            "seed": base_seed,
        }
        out.append({"public": pub, "hidden": {}})
    return out


def baseline(inst):
    pub = inst["public"]
    N = pub["N"]
    L = pub["L"]
    p = pub["unif_power"]
    spots = [tuple(s) for s in pub["spots"]]
    ph = _rand_super_phase(N, spots, pub["seed"])
    return _Q(ph, spots, L, p)


def score(inst, ans):
    pub = inst["public"]
    N = pub["N"]
    L = pub["L"]
    p = pub["unif_power"]
    spots = [tuple(s) for s in pub["spots"]]
    if not isinstance(ans, dict) or "phase" not in ans:
        return False, 0.0
    ph = ans["phase"]
    if not isinstance(ph, list) or len(ph) != N:
        return False, 0.0
    arr = np.empty((N, N), dtype=float)
    for i, row in enumerate(ph):
        if not isinstance(row, list) or len(row) != N:
            return False, 0.0
        for j, v in enumerate(row):
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return False, 0.0
            fv = float(v)
            if fv != fv or fv in (float("inf"), float("-inf")):
                return False, 0.0
            arr[i, j] = fv
    q = _Q(arr, spots, L, p)
    if q != q or q < 0:
        return False, 0.0
    return True, q


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
            obj = 0.0
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        r = min(1.0, 0.1 * obj / max(b, 1e-12))   # maximization: F/B analog
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
