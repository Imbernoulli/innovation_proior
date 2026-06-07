# Fienup's Hybrid Input–Output (HIO) algorithm

## Problem

Recover a real, non-negative, finitely supported object `f(x)` from a single Fourier modulus
`|F(u)|` (a far-field diffraction pattern, a speckle-interferometry power spectrum, or
crystallographic structure-factor magnitudes), where `F = FT(f)`. Only `|F|` is measured; the
object plane supplies no measured amplitude, only the a-priori constraints `f` real, `f(x) ≥ 0`,
and `f = 0` outside a support `D`. The DFT loses the Fourier phase, and `f(x)`, its translates,
and its inverted conjugate `f*(−x)` all share the same `|F|` (the twin-image degeneracy), so the
constraints must do the disambiguating work.

## Key idea

Cast it as finding a field in the intersection of two sets coupled by an FFT: the **Fourier set**
`M = {y : |FT y| = |F|}` (a product of circles — non-convex) and the **object set**
`S = {y : y = 0 outside D, y ≥ 0}` (convex). The Fourier projection keeps the computed phase and
resets the modulus, `P_M(x) = IFT(|F| · FT(x)/|FT x|)`; the object projection zeros points outside
`D` and clips negatives inside.

Alternating these projections is the **error-reduction (ER)** algorithm — equivalently a
double-length-step steepest descent on `B = Σ_u (|G(u)| − |F(u)|)²`, whose gradient
`∂B/∂g(x) = 2[g(x) − g'(x)]` is free from the same two transforms. Each step is a Parseval-
preserving hop or a nearest-point projection, so the modulus error is monotone non-increasing,
`E_{F,k+1} ≤ E_{O,k} ≤ E_{F,k}`. But monotone decrease on a *non-convex* pair is not convergence:
ER plateaus at striped local minima and is painfully slow for the single-intensity problem.

The fix reframes the three transform steps `g' = P_M(g)` as a nonlinear box whose **output always
lies in `M`**. The input therefore need not be the current object estimate — it is a *driving
function*. If a change `Δg'` of the output is wanted, drive the input by `β Δg'`. To make the
output satisfy the object constraints, the desired change at the violation set `γ` (negative inside
`D`, or outside `D`) is `−g'_k`, giving the input–output family. The **hybrid input–output (HIO)**
algorithm accepts the good output where constraints hold and applies accumulating input feedback
where they are violated:

  `g_{k+1}(x) = g'_k(x)` for `x ∉ γ`,  `g_{k+1}(x) = g_k(x) − β g'_k(x)` for `x ∈ γ`.

At satisfied points it keeps `g'_k` (correct modulus, feasible); at violators it grows the input by
`−β g'_k` until the output is forced non-negative — escaping the output–output stagnation. For the
support-only case with `β = 1` this is the averaged double reflection (reflect across `M`, reflect
across `S`, average): `x_{k+1} = ½(R_S R_M + I)(x_k)` with `R = 2P − I`; general `β` gives
`x_{k+1} = ½[R_S(R_M + (β−1)P_M) + I + (1−β)P_M](x_k)`. The reflections overshoot each constraint
set, which is exactly what lets HIO leave a basin that the contractive ER projection is pinned in.
HIO trades ER's monotonicity for escape, so `E_O` may rise early; the working strategy alternates
blocks of HIO (escape plateaus) with a few ER iterations (consolidate the overshoot). `β` is the
feedback gain of order one — larger is faster but unstable, `≈ 0.9` is a safe default.

## Algorithm

1. Estimate support `D` by thresholding the autocorrelation `IFT(|F|²)` (object diameter ≈ half
   the autocorrelation diameter); keep it loose (tighter early, looser later).
2. Seed with a random Fourier phase (breaks centro-symmetry and the twin-image degeneracy).
3. Repeat: `g'_k = P_M(g_k)`; form `γ = {g'_k < 0 inside D} ∪ {outside D}`; update
   `g_{k+1} = g'_k` on `γ^c`, `g_k − β g'_k` on `γ` (HIO).
4. Alternate ~10–30 HIO iterations with ~5–10 ER iterations (`g_{k+1} = g'_k` on `γ^c`, `0` on `γ`).
5. Monitor the object-domain error `E_O = √(Σ_{x∈γ}[g'_k]² / Σ_x [g'_k]²)`; restart from a new
   random seed on stagnation; reconstruct 2–3 times to confirm the solution is not the twin.

The variant family (object-domain update at `x ∈ γ`, with `g_{k+1} = g'_k` at `x ∉ γ` except basic
input–output): error-reduction `0`; output–output `(1−β)g'_k` (`= ER` if `β=1`); basic
input–output keeps `g_k` at `x ∉ γ` and sets `g_k − β g'_k` at `x ∈ γ`; **HIO** keeps `g'_k` at
`x ∉ γ` and `g_k − β g'_k` at `x ∈ γ`.

## Code

```python
import numpy as np

def project_fourier(x, mag):
    """The nonlinear box: keep computed phase, reset |FT| to the measured modulus.
    Its output always satisfies the Fourier-modulus constraint (lies in M)."""
    X = np.fft.fft2(x)
    X = mag * np.exp(1j * np.angle(X))
    return np.real(np.fft.ifft2(X))           # object is real -> drop tiny imaginary part

def support_from_autocorrelation(mag, frac=0.04):
    """Object diameter ~ half the autocorrelation diameter; autocorr = IFT(|F|^2)."""
    autoc = np.abs(np.fft.fftshift(np.fft.ifft2(mag ** 2)))
    return autoc > frac * autoc.max()         # loose support mask


def fienup(mag, support=None, beta=0.9, n_iter=200, mode='hybrid', init=None):
    """ER / input-output / output-output / hybrid (HIO) phase retrieval from |F|.
    mode='output-output' with beta=1 is the error-reduction (Gerchberg-Saxton) step.
    Pass init to continue from a running estimate (so blocks chain)."""
    if support is None:
        support = support_from_autocorrelation(mag)
    if init is None:
        # random-phase seed: breaks centrosymmetry and the twin-image degeneracy
        init = np.real(np.fft.ifft2(mag * np.exp(1j * 2*np.pi*np.random.rand(*mag.shape))))
    g_in = project_fourier(init, mag)
    g = np.zeros_like(mag, dtype=float)
    for _ in range(n_iter):
        g_out = project_fourier(g_in, mag)                # output g'_k (correct Fourier modulus)
        gamma = (g_out < 0) | (~support)                  # object-constraint violations
        if mode in ('hybrid', 'output-output'):
            g = g_out.copy()                              # satisfied points: keep good output
        if mode == 'error-reduction':
            g = np.where(gamma, 0.0, g_out)               # project violations to zero
        elif mode == 'hybrid':
            g[gamma] = g_in[gamma] - beta * g_out[gamma]  # g_k - beta g'_k   (HIO)
        elif mode == 'input-output':
            g = g_in.copy()
            g[gamma] = g_in[gamma] - beta * g_out[gamma]  # g_k - beta g'_k   (basic IO)
        elif mode == 'output-output':
            g[gamma] = g_out[gamma] - beta * g_out[gamma] # (1-beta) g'_k     (=> ER if beta=1)
        g_in = g                                          # this iterate drives the next output
    return project_fourier(g_in, mag)


def object_error(g_out, support):
    """Meaningful metric for input-output: energy-normalized constraint violation."""
    gamma = (g_out < 0) | (~support)
    return np.sqrt(np.sum(g_out[gamma] ** 2) / np.sum(g_out ** 2))


def reconstruct(mag, support=None, beta=0.9, hio_block=20, er_block=5, rounds=10):
    """The working strategy: HIO blocks to escape plateaus, ER blocks to consolidate."""
    if support is None:
        support = support_from_autocorrelation(mag)
    g = np.real(np.fft.ifft2(mag * np.exp(1j * 2*np.pi*np.random.rand(*mag.shape))))
    for _ in range(rounds):
        g = fienup(mag, support, beta, hio_block, mode='hybrid', init=g)
        g = fienup(mag, support, beta, er_block, mode='error-reduction', init=g)
    return g
```
