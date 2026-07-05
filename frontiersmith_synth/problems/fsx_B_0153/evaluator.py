#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0153 -- "Interstellar Relay Beam Fan-Out Phase Mask".

Family: seeded-numerical-sim (Frontier-Eng Optics/Communications), skinned as an
interstellar relay station that must split ONE transmit laser into a fan of downlink
spots aimed at a set of remote relay nodes. The transmitter is a phase-only spatial
light modulator: an N x N grid of pixels, each imposing a phase shift on a uniformly
illuminated aperture. The far field (Fraunhofer diffraction) is the 2-D FFT of the
aperture field, and each far-field pixel is one diffraction "order" (one possible
downlink direction).

The candidate must choose the N x N phase mask so that, in the far field:
  * as much of the transmitted power as possible lands on the M requested spots
    (diffraction EFFICIENCY), AND
  * the power is spread EVENLY across those spots (UNIFORMITY) -- an unevenly lit
    fan starves the dimmest relay node.
These two goals fight each other (a single blazed grating is perfectly efficient into
ONE spot but ignores the rest), so the objective is a composite that both must satisfy.

Forward model (deterministic, pure numpy, run in the PARENT process):
    E      = exp(1j * phase)                      # uniform unit illumination, phase-only
    G      = fftshift(fft2(E))                    # far field, DC at (N//2, N//2)
    I      = |G|**2                               # far-field intensity
    eff    = sum(I at the M target pixels) / sum(I over all pixels)      in [0, 1]
    unif   = min(spot_I) / max(spot_I)                                   in [0, 1]
    obj    = eff * unif                                                  in [0, 1]
Higher obj is better (objective = MAX).

Baseline (the evaluator computes it ITSELF): the cheapest fabricable diffractive mask
-- a BINARY (two-level, 0/pi) phase mask made by thresholding the grating
superposition. Binary etching is the cheapest DOE to make, but it wastes half the
control (conjugate ghost orders, uneven fan), so it is a weak baseline. Its composite
is b.

Scoring (deterministic; no wall-time):
    r = min(1, 0.1 * obj / max(b, 1e-12))         # F/B analog for a MAX objective
  -> the cheap binary mask (obj == b) maps to exactly 0.1; a mask whose
     composite is k times the baseline maps to min(1, 0.1*k). A malformed answer,
     wrong shape, a non-finite value, or obj <= 0 scores 0 on that instance.

The candidate is UNTRUSTED: it is run as an isolated stdin->stdout subprocess via
`isorun`, sees only the public instance, and can never reach the evaluator's frames,
FFT, target list, or scorer.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import numpy as np
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)
    return nxt


# ----------------------------- forward optical model -----------------------
def _farfield_intensity(phase):
    """phase: (N,N) real -> far-field intensity I = |fftshift(fft2(exp(i*phase)))|^2."""
    E = np.exp(1j * phase)
    G = np.fft.fftshift(np.fft.fft2(E))
    return (G.real ** 2 + G.imag ** 2)


def _objective(phase, targets):
    """Composite efficiency*uniformity for a phase mask given target pixel list.
    efficiency = fraction of far-field power on the M target pixels; uniformity =
    (dimmest spot)/(brightest spot) -- the harsh min/max ratio, so one starved
    downlink node tanks the whole design."""
    I = _farfield_intensity(phase)
    total = float(np.sum(I))
    if total <= 0.0:
        return 0.0
    rows = np.array([t[0] for t in targets], dtype=int)
    cols = np.array([t[1] for t in targets], dtype=int)
    spot = I[rows, cols]
    eff = float(np.sum(spot)) / total
    smax = float(np.max(spot))
    if smax <= 0.0:
        return 0.0
    unif = float(np.min(spot)) / smax
    if unif < 0.0:
        unif = 0.0
    return eff * unif


def _super_acc(N, targets):
    """Complex superposition of one unit blazed grating per target. A grating steering
    to far-field (shifted) pixel (r,c) has spatial frequency (r-N//2, c-N//2)."""
    y = np.arange(N).reshape(N, 1)
    x = np.arange(N).reshape(1, N)
    acc = np.zeros((N, N), dtype=complex)
    for (r, c) in targets:
        acc += np.exp(2j * np.pi * ((r - N // 2) * y + (c - N // 2) * x) / N)
    return acc


def _binary_phase(N, targets):
    """Naive baseline: the cheapest fabricable diffractive mask -- a BINARY (two-level,
    0 / pi) phase mask obtained by thresholding the grating superposition. Binary
    etching is the cheapest DOE to make, but it throws away half the control (it
    creates conjugate ghost orders and lights the fan unevenly), so it is a weak
    baseline that leaves plenty of room for analog / iterative designs."""
    acc = _super_acc(N, targets)
    return np.where(acc.real >= 0.0, 0.0, np.pi)


# ----------------------------- instance family -----------------------------
def make_instances():
    """Deterministic, seeded. Returns list of {'public':..., 'hidden':{}}.
    Public view fully specifies the forward model: grid size N, uniform unit
    illumination, and the list of target far-field pixels (diffraction orders) to
    illuminate. The target set is a seeded subset of low-order pixels inside a
    central window, never including the DC (central) pixel."""
    specs = [
        # (seed, N, M) -- small instances
        (201, 20, 6), (202, 22, 7), (203, 24, 8),
        (204, 20, 9), (205, 22, 6), (206, 24, 10),
        # larger / held-out instances (generalization)
        (207, 28, 8), (208, 30, 9), (209, 32, 10),
        (210, 28, 12), (211, 30, 7), (212, 32, 11),
    ]
    out = []
    for seed, N, M in specs:
        r = _rng(seed)
        w = N // 4                      # target window half-width around DC
        dc = N // 2
        chosen = set()
        guard = 0
        while len(chosen) < M and guard < 10000:
            guard += 1
            rr = dc + r(-w, w)
            cc = dc + r(-w, w)
            if (rr, cc) == (dc, dc):
                continue                # never the DC pixel
            if 0 <= rr < N and 0 <= cc < N:
                chosen.add((rr, cc))
        targets = sorted(chosen)
        public = {"N": N, "targets": [[int(a), int(b)] for a, b in targets]}
        out.append({"public": public, "hidden": {}})
    return out


# ----------------------------- scoring -------------------------------------
def baseline(inst):
    p = inst["public"]
    N = p["N"]
    targets = [(t[0], t[1]) for t in p["targets"]]
    return float(_objective(_binary_phase(N, targets), targets))


def score(inst, answer):
    """Strictly validate the answer against the instance; return (ok, obj)."""
    p = inst["public"]
    N = p["N"]
    targets = [(t[0], t[1]) for t in p["targets"]]
    if not isinstance(answer, dict):
        return False, None
    ph = answer.get("phase", None)
    if not isinstance(ph, list) or len(ph) != N:
        return False, None
    for row in ph:
        if not isinstance(row, list) or len(row) != N:
            return False, None
    try:
        a = np.asarray(ph, dtype=float)
    except (TypeError, ValueError):
        return False, None
    if a.shape != (N, N):
        return False, None
    if not np.all(np.isfinite(a)):
        return False, None
    obj = _objective(a, targets)
    if not np.isfinite(obj) or obj <= 0.0:
        return False, None
    return True, float(obj)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
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
            ok, obj = False, None
        if not ok or obj is None or obj <= 0.0:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * obj / max(b, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
