# TIER: strong
# Gerchberg-Saxton / iterative-Fourier-transform with adaptive amplitude weighting.
#   Start from a seeded random phase.  Each iteration:
#     1) simulate the flow spectrum F = fftshift(fft2(exp(i*phase)));
#     2) in the spectrum, KEEP the current phase of F but REPLACE the amplitude by the
#        target: the corridors get a weight (all others 0), and the weight of each
#        corridor is nudged up when it is dimmer than the corridor average -- this
#        actively equalizes them (raises uniformity), not just efficiency;
#     3) inverse-transform and keep only the phase (unit amplitude) as the new offsets.
#   The best design seen (by the same composite the evaluator uses) is returned.  This
#   beats plain grating superposition on both efficiency and uniformity, but a phase-only
#   grid cannot reach the ideal, so the normalized score stays below 1.0.
import sys, json
import numpy as np

inst = json.load(sys.stdin)
n = inst["n"]
spots = inst["spots"]

spot_idx = np.array(spots)
su = spot_idx[:, 0]
sv = spot_idx[:, 1]


def composite(phase):
    U = np.exp(1j * phase)
    F = np.fft.fftshift(np.fft.fft2(U))
    I = np.abs(F) ** 2
    E = I.sum()
    if E <= 1e-12:
        return -1.0, None
    vals = I[su, sv]
    eff = vals.sum() / E
    mx = vals.max()
    unif = (vals.min() / mx) if mx > 1e-12 else 0.0
    return eff * unif, vals


rng = np.random.default_rng(20240617)
phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, n))

# target amplitude mask on the corridors (adaptively re-weighted for uniformity)
weight = np.ones(len(spots), dtype=float)

best_M = -1.0
best_phase = phase.copy()

for it in range(120):
    U = np.exp(1j * phase)
    F = np.fft.fftshift(np.fft.fft2(U))
    ang = np.angle(F)

    # score current design; adapt corridor weights to boost the dim ones
    M, vals = composite(phase)
    if M > best_M and vals is not None:
        best_M = M
        best_phase = phase.copy()
    if vals is not None:
        amp = np.sqrt(np.maximum(vals, 1e-12))
        mean_amp = amp.mean()
        # push weight toward the corridors that are currently below the mean amplitude
        weight *= (mean_amp / amp) ** 0.5
        weight = np.clip(weight, 0.2, 5.0)
        weight /= weight.mean()

    target = np.zeros((n, n), dtype=float)
    target[su, sv] = weight

    Fnew = target * np.exp(1j * ang)
    Uback = np.fft.ifft2(np.fft.ifftshift(Fnew))
    phase = np.angle(Uback)

# final check of the last phase too
M, vals = composite(phase)
if M > best_M:
    best_M = M
    best_phase = phase.copy()

print(json.dumps({"phases": best_phase.tolist()}))
