#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0177 -- "Wildlife Corridor Guide-Light Phase Mask".

Family: seeded-numerical-sim (Frontier-Eng Optics/Communications), skinned as a
wildlife-corridor conservation project. A single infrared guide-light projector must
steer one beam into a fan of dim guide spots strung ALONG a corridor, one spot per
habitat patch, so that nocturnal animals are gently drawn along the safe crossing.
The projector is a phase-only spatial light modulator: an N x N grid of pixels, each
imposing a phase shift on a uniformly illuminated aperture. The far field (Fraunhofer
diffraction) is the 2-D FFT of the aperture field, and each far-field pixel is one
diffraction "order" -- one aim-able guide-spot direction.

WHAT MAKES THIS VARIANT DISTINCT: the M target patches lie (roughly) along a line --
the corridor -- and each patch carries a PRESCRIBED illumination WEIGHT w_i (a large
clearing needs more light than a narrow underpass). So the goal is NOT an equal spot
array; it is to reproduce a SPECIFIED non-uniform intensity PROFILE along the corridor
while dumping as little power as possible outside it. Efficiency and profile-fidelity
fight each other (a single blazed grating is perfectly efficient into ONE patch but
ignores the profile), so the objective is a composite both must satisfy.

Forward model (deterministic, pure numpy, run in the PARENT process):
    E       = exp(1j * phase)                          # uniform unit illumination
    G       = fftshift(fft2(E))                        # far field, DC at (N//2, N//2)
    I       = |G|**2                                   # far-field intensity
    eff     = sum(I at the M patch pixels) / sum(I)                       in [0, 1]
    p_i     = I_i / sum_j I_j          (achieved fraction on patch i)     in [0, 1]
    q_i     = w_i / sum_j w_j          (prescribed fraction on patch i)   in [0, 1]
    fidel   = 1 - 0.5 * sum_i |p_i - q_i|   (1 - total-variation distance) in [0, 1]
    obj     = eff * fidel                                                 in [0, 1]
Higher obj is better (objective = MAX).

Baseline (the evaluator computes it ITSELF): the classic strawman -- a SINGLE blazed
grating that dumps the whole beam onto the highest-weight patch. It is perfectly
efficient into that ONE patch but ignores the rest of the corridor, so its profile
fidelity is only the top patch's prescribed share. Its composite is b.

Scoring (deterministic; no wall-time):
    r = min(1, 0.1 * obj / max(b, 1e-12))     # F/B analog for a MAX objective
  -> the naive weighted-superposition mask (obj == b) maps to exactly 0.1; a mask
     whose composite is k times the baseline maps to min(1, 0.1*k). A malformed
     answer, wrong shape, a non-finite value, or obj <= 0 scores 0 on that instance.

The candidate is UNTRUSTED: it is run as an isolated stdin->stdout subprocess via
`isorun`, sees only the public instance, and can never reach the evaluator's frames,
FFT, patch list, weights, or scorer.

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
    E = np.exp(1j * phase)
    G = np.fft.fftshift(np.fft.fft2(E))
    return (G.real ** 2 + G.imag ** 2)


def _objective(phase, targets, weights):
    """Composite efficiency * profile-fidelity for a phase mask."""
    I = _farfield_intensity(phase)
    total = float(np.sum(I))
    if total <= 0.0:
        return 0.0
    rows = np.array([t[0] for t in targets], dtype=int)
    cols = np.array([t[1] for t in targets], dtype=int)
    spot = I[rows, cols].astype(float)
    ssum = float(np.sum(spot))
    if ssum <= 0.0:
        return 0.0
    eff = ssum / total
    p = spot / ssum
    w = np.array(weights, dtype=float)
    q = w / float(np.sum(w))
    tv = 0.5 * float(np.sum(np.abs(p - q)))
    fidel = 1.0 - tv
    if fidel < 0.0:
        fidel = 0.0
    return eff * fidel


def _naive_phase(N, targets, weights):
    """Naive baseline: a SINGLE blazed grating that steers the whole beam to the
    highest-weight patch. It is perfectly efficient into that ONE patch but ignores
    the rest of the corridor, so its profile fidelity is just the top patch's
    prescribed share. A grating steering to far-field (shifted) pixel (r,c) has spatial
    frequency (r-N//2, c-N//2)."""
    k = int(np.argmax(np.array(weights, dtype=float)))   # first max index (deterministic)
    r, c = targets[k]
    y = np.arange(N).reshape(N, 1)
    x = np.arange(N).reshape(1, N)
    fr = r - N // 2
    fc = c - N // 2
    return np.angle(np.exp(2j * np.pi * (fr * y + fc * x) / N))


# ----------------------------- instance family -----------------------------
def make_instances():
    """Deterministic, seeded. Returns list of {'public':..., 'hidden':{}}.
    Public view fully specifies the forward model: grid size N, uniform unit
    illumination, the list of patch far-field pixels along the corridor, and the
    per-patch prescribed weights. Patches are a seeded set of low-order pixels
    strung along a line through the field, never including the DC (central) pixel."""
    specs = [
        # (seed, N, M) -- small instances
        (2201, 16, 5), (2202, 18, 6), (2203, 20, 6),
        (2204, 16, 7), (2205, 18, 5), (2206, 20, 8),
        # larger / held-out instances (generalization)
        (2207, 22, 7), (2208, 24, 8), (2209, 22, 9),
        (2210, 24, 6), (2211, 26, 9), (2212, 26, 7),
    ]
    out = []
    for seed, N, M in specs:
        r = _rng(seed)
        w = N // 3                       # corridor half-length around DC
        dc = N // 2
        slope_choice = r(0, 2) - 1       # -1, 0, or +1  (corridor orientation)
        # pick M distinct non-zero offsets along the corridor axis
        offs = []
        seen = set()
        guard = 0
        while len(offs) < M and guard < 10000:
            guard += 1
            o = r(-w, w)
            if o == 0 or o in seen:
                continue
            seen.add(o)
            offs.append(o)
        offs.sort()
        targets = []
        weights = []
        tset = set()
        for o in offs:
            cc = dc + o
            rr = dc + slope_choice * o
            # keep inside the grid; skip DC and duplicates
            if not (0 <= rr < N and 0 <= cc < N):
                continue
            if (rr, cc) == (dc, dc) or (rr, cc) in tset:
                continue
            tset.add((rr, cc))
            targets.append([int(rr), int(cc)])
            weights.append(int(r(1, 5)))   # prescribed illumination weight 1..5
        public = {"N": N,
                  "targets": [[int(a), int(b)] for a, b in targets],
                  "weights": [int(x) for x in weights]}
        out.append({"public": public, "hidden": {}})
    return out


# ----------------------------- scoring -------------------------------------
def baseline(inst):
    p = inst["public"]
    N = p["N"]
    targets = [(t[0], t[1]) for t in p["targets"]]
    weights = list(p["weights"])
    return float(_objective(_naive_phase(N, targets, weights), targets, weights))


def score(inst, answer):
    """Strictly validate the answer against the instance; return (ok, obj)."""
    p = inst["public"]
    N = p["N"]
    targets = [(t[0], t[1]) for t in p["targets"]]
    weights = list(p["weights"])
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
    obj = _objective(a, targets, weights)
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
