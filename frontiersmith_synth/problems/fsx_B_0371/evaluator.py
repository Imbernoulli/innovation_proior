#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0371 -- "Tweezer-Array Wiring: Phase-Mask Hologram for a
Qubit Register" (family: seeded-numerical-sim; format B, quality-metric).

THEME.  A neutral-atom quantum lab "wires up" its qubit register by carving a single
laser beam into an ARRAY of optical tweezers (bright focal spots), one trap per qubit.
The beam passes through a phase-only spatial light modulator (SLM): the modulator
imprints a phase mask phi[y,x] on the (fixed) illumination field, and a lens Fourier-
transforms the modulated field to the trapping plane.  The engineer must design phi so
that the far-field intensity has bright, *equally-bright* spots exactly at the target
qubit sites -- high diffraction EFFICIENCY (light lands in the traps, not wasted) and
high UNIFORMITY (every qubit sees the same trap depth).

PHYSICS (deterministic, seeded per instance; pure numpy, no hardware).
  Illumination amplitude A[y,x] (>=0, fixed, given in the public instance) is a
  truncated Gaussian times a fixed seeded ripple -- it models the real, non-ideal beam.
  The SLM applies phase only:   U_in = A * exp(1j * phi).
  A lens produces the far field: U_out = fftshift( fft2( U_in ) )   (numpy convention).
  Intensity:                     I = |U_out|^2.
  Target qubit sites are pixel coordinates (r,c) into the M x M U_out grid.

METRICS (higher is better).
  efficiency  eta = ( sum of I over the target pixels ) / ( sum of I over ALL pixels )
  uniformity  u   = 1 - (Imax - Imin) / (Imax + Imin)   over the target-pixel intensities
  composite   Q   = eta * u                              (both must be good)

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str,
             "M": int,                       # grid side length
             "seed": int,
             "spots": [[r,c], ...],          # K target pixel sites, 0 <= r,c < M
             "amp":  [[...],...]}            # M x M illumination amplitude A (>=0)
  stdout: ONE JSON object:
            {"phase": [[...],...]}           # M x M real phase mask phi (radians; any finite)

  A mask is VALID iff `phase` is an M x M array of finite real numbers.  Wrong shape,
  a non-finite entry (nan/inf), a crash, a timeout, or non-JSON -> that instance scores 0.

SCORING (deterministic; no wall-time).  Per instance the parent computes:
    Q_base = composite of the internal PLAIN-SUPERPOSITION baseline (a weak analytic
             hologram: superpose one grating per spot, all in phase, keep the phase).
    Q_cand = composite of the candidate's mask.
  and normalizes with an affine anchor (weak baseline -> 0.1, physically-perfect -> 1.0):
    r = clamp( 0.1 + 0.9 * (Q_cand - Q_base) / (1.0 - Q_base), 0, 1 )
  A candidate reproducing plain superposition scores ~0.1; the perfect Q=1 hologram
  (unreachable for a phase-only element under a truncated, rippled Gaussian beam) would
  score 1.0, so even strong Gerchberg-Saxton style optimizers keep headroom below 1.0.

ISOLATION.  The candidate is untrusted and runs OS-sandboxed in a fresh subprocess via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The baseline reference is
computed by THIS parent process, so an introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import numpy as np
import isorun


# ----------------------------- instance family -----------------------------
def _build_amp(M, seed):
    """Truncated-Gaussian illumination times a fixed seeded ripple (the real beam)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:M, 0:M]
    c = (M - 1) / 2.0
    w = M * 0.42
    amp = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2.0 * w * w))
    ripple = 1.0 + 0.15 * rng.standard_normal((M, M))
    amp = amp * np.clip(ripple, 0.4, 1.6)
    return amp.astype(np.float64)


def _spot_grid(M, K, step):
    cen = M // 2
    off = (K - 1) * step // 2
    return [[cen - off + step * i, cen - off + step * j]
            for i in range(K) for j in range(K)]


def _build_instances():
    """Deterministic instance family. (name, seed, M, K, step)."""
    specs = [
        ("reg_a", 101, 48, 3, 8),
        ("reg_b", 102, 48, 4, 6),
        ("reg_c", 103, 48, 4, 5),
        ("reg_d", 104, 48, 5, 5),
        ("reg_e", 105, 48, 4, 7),
        ("reg_f", 106, 48, 3, 10),
        ("reg_g", 107, 48, 5, 4),
        # harder / larger held-out registers
        ("hold_h", 211, 64, 4, 7),
        ("hold_i", 212, 64, 5, 6),
        ("hold_j", 213, 64, 6, 5),
        ("hold_k", 214, 48, 4, 4),
        ("hold_l", 215, 64, 4, 9),
    ]
    out = []
    for name, seed, M, K, step in specs:
        amp = _build_amp(M, seed)
        spots = _spot_grid(M, K, step)
        out.append({"name": name, "M": M, "seed": seed,
                    "spots": spots, "amp": amp})
    return out


# ----------------------------- physics + metrics ---------------------------
def _farfield_I(phi, amp):
    U = np.fft.fftshift(np.fft.fft2(amp * np.exp(1j * phi)))
    return np.abs(U) ** 2


def _composite(phi, amp, spots):
    I = _farfield_I(phi, amp)
    tot = float(I.sum())
    if tot <= 0.0:
        return 0.0
    tv = np.array([I[r, c] for (r, c) in spots], dtype=np.float64)
    eff = float(tv.sum()) / tot
    imax = float(tv.max())
    imin = float(tv.min())
    u = 1.0 - (imax - imin) / (imax + imin + 1e-18)
    q = eff * u
    if not np.isfinite(q):
        return 0.0
    return max(0.0, min(1.0, q))


def _baseline_phi(M, spots):
    """Plain superposition of gratings, all spots in phase; keep the phase (weak)."""
    tgt = np.zeros((M, M), dtype=np.complex128)
    for (r, c) in spots:
        tgt[r, c] = 1.0
    field = np.fft.ifft2(np.fft.ifftshift(tgt))
    return np.angle(field)


# ----------------------------- validation ----------------------------------
def _valid_phase(inst, answer):
    """Return an M x M float64 phase array, or None if the answer is invalid."""
    if not isinstance(answer, dict):
        return None
    ph = answer.get("phase")
    if not isinstance(ph, list):
        return None
    M = inst["M"]
    if len(ph) != M:
        return None
    out = np.empty((M, M), dtype=np.float64)
    for i, row in enumerate(ph):
        if not isinstance(row, list) or len(row) != M:
            return None
        for j, v in enumerate(row):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            fv = float(v)
            if not np.isfinite(fv):
                return None
            out[i, j] = fv
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        M = inst["M"]
        amp = inst["amp"]
        spots = inst["spots"]
        q_base = _composite(_baseline_phi(M, spots), amp, spots)
        denom = 1.0 - q_base
        if denom < 1e-9:
            denom = 1e-9

        public = {"name": inst["name"], "M": M, "seed": inst["seed"],
                  "spots": [[int(r), int(c)] for (r, c) in spots],
                  "amp": amp.tolist()}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            phi = _valid_phase(inst, ans)
        except Exception:
            phi = None
        if phi is None:
            vec.append(0.0)
            continue
        try:
            q_cand = _composite(phi, amp, spots)
        except Exception:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (q_cand - q_base) / denom
        if not np.isfinite(r):
            vec.append(0.0)
            continue
        r = max(0.0, min(1.0, float(r)))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
