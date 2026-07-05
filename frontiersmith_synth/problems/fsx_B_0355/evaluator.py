import sys, json, math, random, isorun
import numpy as np

# ------------------------------------------------------------------ instances
# Skin: forest-fire watchtower "beacon panel". A watchtower carries an N x N
# array of individually tiltable mirror facets (a phase-only spatial light
# modulator). Collimated light hits the facets; the panel's Fraunhofer (far-
# field) pattern -- computed by a fixed, seeded FFT propagation -- lights up a
# ridgeline of K high-risk fire-lookout cells. We want as much beam energy as
# possible ON the lookout cells (efficiency), split EVENLY across them so no
# lookout is left dim (uniformity). This is the diffraction-efficiency +
# uniformity spot-array holography objective, deterministically simulated.

N_GRID = 24

def _targets(seed, K, N=N_GRID):
    rng = random.Random(1000 + seed)
    win = N // 2 - 1
    seen = set()
    while len(seen) < K:
        r = N // 2 + rng.randint(-win, win)
        c = N // 2 + rng.randint(-win, win)
        if (r, c) == (N // 2, N // 2):      # never the undiffracted (DC) order
            continue
        seen.add((r, c))
    return sorted(seen)

def make_instances():
    specs = [(0, 10), (1, 14), (2, 18), (3, 22), (4, 12), (5, 16), (6, 20), (7, 24)]
    out = []
    for s, K in specs:
        out.append({"public": {"N": N_GRID, "targets": [list(t) for t in _targets(s, K)]},
                    "hidden": {}})
    return out

# ------------------------------------------------------------------ physics
def _far_intensity(phase):
    U = np.exp(1j * phase)
    G = np.fft.fftshift(np.fft.fft2(U))
    return np.abs(G) ** 2

def _metric(phase, targets):
    I = _far_intensity(phase)
    tot = float(I.sum())
    if not (tot == tot and tot > 0):
        return 0.0
    tv = np.array([float(I[r, c]) for (r, c) in targets])
    if tv.size == 0 or not np.all(np.isfinite(tv)):
        return 0.0
    eff = float(tv.sum()) / tot                     # fraction of energy on lookouts
    mx = float(tv.max())
    u = float(tv.min()) / mx if mx > 0 else 0.0      # dimmest / brightest lookout
    m = eff * u
    return m if (m == m and 0.0 <= m <= 1.0) else 0.0

def _superposition_phase(N, targets):
    """Trivial construction: sum a steering grating per lookout, keep the phase."""
    yy, xx = np.mgrid[0:N, 0:N]
    f = np.zeros((N, N), dtype=complex)
    for (r, c) in targets:
        ky = r - N // 2
        kx = c - N // 2
        f += np.exp(1j * (2 * np.pi * (ky * yy + kx * xx) / N))
    return np.angle(f)

def baseline(inst):
    N = inst["public"]["N"]
    targets = [tuple(t) for t in inst["public"]["targets"]]
    return _metric(_superposition_phase(N, targets), targets)

# ------------------------------------------------------------------ scoring
def score(inst, ans):
    N = inst["public"]["N"]
    targets = [tuple(t) for t in inst["public"]["targets"]]
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
            if not math.isfinite(v):
                return False, 0.0
            arr[i, j] = float(v)
    obj = _metric(arr, targets)
    if not (obj == obj and 0.0 <= obj <= 1.0):
        return False, 0.0
    return True, obj

# ------------------------------------------------------------------ main
def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * obj / max(b, 1e-12))
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    print("Ratio: %.6f" % (sum(vec) / len(vec)))
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))

main()
