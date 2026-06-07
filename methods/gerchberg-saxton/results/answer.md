# Gerchberg–Saxton (GS) and Weighted Gerchberg–Saxton (WGS)

## Problem

A detector measures intensity, not phase. Given the amplitude of a wave in two Fourier-conjugate planes — `|f|` in the object/image plane and `|F|` in the diffraction/Fourier plane, with `F = FT(f)` — recover the lost phase (phase retrieval). The synthesis form: given a *fixed* illumination amplitude `|A|` on a phase-only device (kinoform, diffractive optic, or SLM) and a desired Fourier-plane intensity (an image, or an array of focal spots), find the per-pixel phase `φ_j` whose Fourier transform has the target amplitude. The binding constraint is that only phase is free — amplitude is dictated by the measurement or by the hardware — so one cannot simply inverse-transform the target.

## Key idea

Cast it as a feasibility problem: find a field lying in two sets, `M_o = {g : |g| = |f|}` and `M_F = {g : |FT g| = |F|}`. Both are products of fixed-modulus circles (non-convex). The nearest point of a fixed-modulus set to a field `x` keeps `x`'s phase and resets its modulus:

  `P(x) = a · x/|x| = a · e^{i·arg x}` (minimizes `|x − a e^{iθ}|²`; if `x = 0`, choose any phase convention).

Alternating projection onto the two sets, with an FFT coupling them, is the **Gerchberg–Saxton algorithm**. Because each step is either a Fourier-domain hop with the matching Parseval scaling or a nearest-point projection, the modulus-mismatch error is monotone non-increasing:

  `E_{k+1} ≤ e_k ≤ E_k`,

with `e_k² = Σ_n (|y_k[n]| − |f[n]|)²` (object plane) and `E_k² = N Σ_n (|X_k[n]| − |F[n]|)²` (Fourier plane), using the convention `F_m = (1/N)Σ_n f_n exp(-2πinm/N)`. Non-convexity means this monotone error decrease does not guarantee a global solution — the iteration can plateau at a local fixed point — so one seeds with a random phase (which also breaks the centrosymmetric-input degeneracy) and may restart.

For an equal-amplitude spot array, plain GS gives *non-uniform* spot intensities and ghost spots, because a phase-only mask cannot split power evenly among structured equal-amplitude targets. **Weighted GS (WGS/GSW)** fixes this by treating the per-spot target amplitudes as adaptive weights and driving the *delivered* intensities to uniformity:

  `w_m^{(k)} = w_m^{(k-1)} · ⟨|V|⟩ / |V_m|`  (equivalently `w_m ← w_m · √(I_target/I_m)` with `I_target = ⟨|V|⟩²`),

where `V_m` is the delivered complex field at spot `m`, `I_m = |V_m|²`, and `⟨·⟩` averages over spots. The uniform state `|V_m| = ⟨|V|⟩` is the fixed point of this update (weights stop changing). Multiplicative keeps weights positive; ratio-to-the-mean corrects the spread directly while efficiency is tracked separately.

Metrics: efficiency `e = Σ_m I_m` (power in wanted spots), uniformity `u = 1 − (max I_m − min I_m)/(max I_m + min I_m)` (→ 1 when equal).

## Algorithm (one GS iteration)

1. `x_k = |f| · e^{iφ_{k-1}}` — impose object-plane amplitude, keep phase.
2. `X_k = FFT(x_k) = |X_k| e^{iψ_k}` — propagate to Fourier plane.
3. `Y_k = |F| · e^{iψ_k}` — keep computed phase, reset modulus to the target/measured `|F|`.
4. `y_k = IFFT(Y_k) = |y_k| e^{iφ_k}` — propagate back; keep `φ_k`, discard `|y_k|`. Loop.

WGS inserts, between forward propagation and re-imposing the modulus, the weight update above and uses the weighted amplitudes `w_m` in place of the flat target.

## Code

Image-target GS and WGS with numpy FFTs (phase-only SLM, flat illumination):

```python
import numpy as np

def synthesize_phase(target_amplitude, illumination=1.0, n_iter=30):
    """target_amplitude: desired Fourier-plane amplitude, normalized to [0,1].
    Returns (phase mask to write on the SLM, final delivered intensity)."""
    h, w = target_amplitude.shape
    phase = np.random.rand(h, w)                       # random seed
    I = None
    for _ in range(n_iter):
        u = illumination * np.exp(1j * phase)          # |A| fixed, keep phase  (project onto M_o)
        U = np.fft.fftshift(np.fft.fft2(u))            # lens: SLM -> target plane
        I = np.abs(U) ** 2
        psi = np.angle(U)
        U = target_amplitude * np.exp(1j * psi)        # keep phase, reset modulus (project onto M_F)
        u = np.fft.ifft2(np.fft.ifftshift(U))          # lens: target -> SLM plane
        phase = np.angle(u)                            # discard amplitude -> phase-only hologram
    return phase, I


def synthesize_balanced_phase(target_amplitude, n_iter=30):
    """Weighted GS: drive the spot intensities to uniformity."""
    def normalization(a):
        a_min = a.min()
        a_max = a.max()
        return (a - a_min) / (a_max - a_min + 1e-12)

    h, w = target_amplitude.shape
    mask = (target_amplitude == 1)                     # locations we care about
    phase = np.random.rand(h, w)
    weights = target_amplitude.astype(float).copy()
    prev_w = weights.copy()
    I = None
    for _ in range(n_iter):
        u = np.exp(1j * phase)
        U = np.fft.fftshift(np.fft.fft2(u))
        I = np.abs(U) ** 2
        Inorm = normalization(I)
        psi = np.angle(U)
        # feedback: w <- w * sqrt(I_target / normalized I)
        weights[mask] = np.sqrt(target_amplitude[mask] / np.maximum(Inorm[mask], 1e-12)) * prev_w[mask]
        weights = normalization(weights)                # bound level, correct relative spread
        prev_w = weights.copy()
        U = weights * np.exp(1j * psi)                 # impose WEIGHTED target amplitude, keep GS phase
        u = np.fft.ifft2(np.fft.ifftshift(U))
        phase = np.angle(u)
    return phase, I
```

Spot-array WGS by direct Fourier-optics propagation (no full-grid FFT; `delta[j,m]` is the per-pixel ramp `Δ_j^m = (2π/λf)(x_j x_m + y_j y_m) + (z_m π/λf²)(x_j²+y_j²)` that focuses light to spot `m`):

```python
import numpy as np

def synthesize_spot_phase(delta, illumination, n_iter=30):
    """delta: (N_pixels, M) propagation phases; illumination: (N_pixels,) fixed |A|.
    Returns (hologram phase phi_j, efficiency e, uniformity u)."""
    N, M = delta.shape
    theta = np.random.rand(M) * 2 * np.pi              # random per-spot phase (random-superposition seed)
    w = np.ones(M)                                     # equal weights to start
    phi = np.zeros(N)
    I = np.zeros(M)
    e = u = None
    for _ in range(n_iter):
        # backward propagation: weighted, phased superposition of single-spot ramps -> SLM phase
        field = (w * np.exp(1j * theta))[None, :] * np.exp(1j * delta)    # (N, M)
        phi = np.angle(field.sum(axis=1))                                 # phi_j = arg sum_m w_m e^{i(delta_jm + theta_m)}
        # forward propagation: hologram back to each spot
        A = illumination * np.exp(1j * phi)
        V = (A[:, None] * np.exp(-1j * delta)).sum(axis=0) / N            # V_m
        I = np.abs(V) ** 2
        theta = np.angle(V)                                              # GS: keep computed per-spot phase
        w = w * (np.mean(np.abs(V)) / np.abs(V))                         # WGS weight update
        e = np.sum(I)
        u = 1 - (I.max() - I.min()) / (I.max() + I.min())
    return phi, e, u
```

The FFT code follows the numpy hologram loop: build `cos/sin` from the current phase, FFT/shift, take the output phase, impose either the target or the normalized weight image, inverse FFT, and keep only `angle(u)`. The spot-array code follows the Fourier-optics trap loop: backward propagation uses `exp(i(Delta + theta))`, forward propagation reads `V_m` with `exp(i(phi - Delta))`, `update_weights` applies `w ← w · ⟨|V|⟩/|V|`, and the SLM field is normalized to unit modulus to enforce the phase-only constraint.
