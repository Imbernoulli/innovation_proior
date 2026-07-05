import sys, json, isorun
import numpy as np

# ==========================================================================
# fsx_B_0363 -- seeded-numerical-sim (Format B, isolated candidate)
# Theme: "telescope array" -- a laser-guide-star / metrology transmitter behind
# a segmented telescope carries a phase-only Spatial Light Modulator (SLM).
# A monochromatic plane wave (unit amplitude) illuminates the SLM; the far
# field is the 2D discrete Fourier transform of exp(i*phase). We must sculpt
# the pupil PHASE so the far field lights up a prescribed ARRAY OF SPOTS
# (a fan-out of guide beacons) that is BOTH power-efficient and UNIFORM.
#
#   field  = exp(i*phi)            (phase-only: |field| == 1 everywhere)
#   F      = FFT2(field);  P = |F|^2
#   eta    = (sum of P over the target spots) / (sum of P over the whole plane)
#   u      = 1 - (Pmax - Pmin)/(Pmax + Pmin)  over the target spots   (1 == flat)
#   M      = eta * u**gamma                    (the composite figure of merit)
#
# Objective: MAXIMIZE M. This is phase retrieval / iterative Fourier-transform
# design (Gerchberg-Saxton / Dammann-grating fan-out) -- non-convex, no closed
# form, many viable heuristics, and a phase-only element can never reach M==1
# (light always scatters into non-target orders) so there is permanent headroom.
# The evaluator re-simulates propagation itself; the candidate's claimed metric
# is never trusted.
# ==========================================================================

GAMMA = 6

# (N, K, R): SLM grid, number of target spots, target-frequency half-window
SPECS = [
    (64, 16, 10),
    (64, 25, 12),
    (56, 12,  8),
    (72,  9,  6),
    (80, 36, 16),
    (64, 20, 11),
    (72, 30, 14),
    (56, 14,  9),
    (96, 49, 20),
    (80, 24, 13),
]


def _make_targets(seed, N, K, R):
    rng = np.random.default_rng(1000 + seed)
    seen = set()
    tg = []
    while len(tg) < K:
        fy = int(rng.integers(-R, R + 1))
        fx = int(rng.integers(-R, R + 1))
        if (fy, fx) == (0, 0) or (fy, fx) in seen:
            continue
        seen.add((fy, fx))
        tg.append([fy % N, fx % N])       # unshifted FFT index coordinates
    return tg


def make_instances():
    out = []
    for si, (N, K, R) in enumerate(SPECS):
        tg = _make_targets(si, N, K, R)
        pub = {"N": int(N), "gamma": GAMMA, "targets": tg}
        out.append({"public": pub, "hidden": {}})
    return out


def _objective(phi, targets, N, gamma):
    P = np.abs(np.fft.fft2(np.exp(1j * phi))) ** 2
    tot = float(P.sum())
    idx = np.asarray(targets, dtype=int)
    I = P[idx[:, 0], idx[:, 1]]
    eta = float(I.sum()) / tot
    imax = float(I.max())
    imin = float(I.min())
    u = 1.0 - (imax - imin) / (imax + imin + 1e-30)
    return eta * (u ** gamma)


def _superpose_zero(targets, N):
    """Deterministic in-phase superposition of single-spot gratings, then keep
    only the argument (phase-only projection). The reference construction."""
    y = np.arange(N).reshape(-1, 1)
    x = np.arange(N).reshape(1, -1)
    C = np.zeros((N, N), dtype=complex)
    for fy, fx in targets:
        C += np.exp(2j * np.pi * (fy * y + fx * x) / N)
    return np.angle(C)


def baseline(inst):
    pub = inst["public"]
    phi = _superpose_zero(pub["targets"], pub["N"])
    return _objective(phi, pub["targets"], pub["N"], pub["gamma"])


def score(inst, ans):
    pub = inst["public"]
    N = pub["N"]
    targets = pub["targets"]
    gamma = pub["gamma"]
    if not isinstance(ans, dict) or "phase" not in ans:
        return False, 0.0
    ph = ans["phase"]
    if not isinstance(ph, list) or len(ph) != N:
        return False, 0.0
    for row in ph:
        if not isinstance(row, list) or len(row) != N:
            return False, 0.0
    arr = np.asarray(ph, dtype=float)
    if arr.shape != (N, N):
        return False, 0.0
    if not np.all(np.isfinite(arr)):
        return False, 0.0
    M = _objective(arr, targets, N, gamma)
    if not np.isfinite(M) or M < 0.0:
        return False, 0.0
    return True, float(M)


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
        r = min(1.0, 0.1 * obj / max(b, 1e-12))   # maximization normalization
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
