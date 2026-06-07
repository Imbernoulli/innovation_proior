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
double-length-step steepest descent on `B = N^{-2}Σ_u (|G(u)| − |F(u)|)²`, whose gradient
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
feedback gain of order one — larger is faster but unstable, and `0.8` is a conservative
implementation default.

## Algorithm

1. Estimate support `D` by thresholding the autocorrelation `IFT(|F|²)` (object diameter ≈ half
   the autocorrelation diameter); keep it loose (tighter early, looser later).
2. Seed with a random Fourier phase (breaks the centro-symmetry that stalls a constant-phase
   start; the twin/inversion ambiguity remains, so it is checked by restarting — see step 5).
3. Repeat: `g'_k = P_M(g_k)`; form `γ = {g'_k < 0 inside D} ∪ {outside D}`; update
   `g_{k+1} = g'_k` on `γ^c`, `g_k − β g'_k` on `γ` (HIO).
4. To consolidate, run a few ER iterations: the same output-output rule with `β = 1`.
5. Monitor the object-domain error `E_O = √(Σ_{x∈γ}[g'_k]² / Σ_x [g'_k]²)`; restart from a new
   random seed on stagnation; reconstruct 2–3 times to gain confidence the solution is unique (up
   to the translation/inversion/twin symmetries `|F|` cannot break).

The variant family (object-domain update at `x ∈ γ`, with `g_{k+1} = g'_k` at `x ∉ γ` except basic
input–output): error-reduction `0`; output–output `(1−β)g'_k` (`= ER` if `β=1`); basic
input–output keeps `g_k` at `x ∉ γ` and sets `g_k − β g'_k` at `x ∈ γ`; **HIO** keeps `g'_k` at
`x ∉ γ` and `g_k − β g'_k` at `x ∈ γ`.

## Code

```python
import numpy as np

def support_from_autocorrelation(mag, frac=0.04):
    """Threshold IFT(|F|^2) for a loose support mask."""
    autoc = np.abs(np.fft.fftshift(np.fft.ifft2(mag ** 2)))
    return autoc > frac * autoc.max()


def object_domain_step(candidate, previous, mask, beta=0.8, mode=None):
    """Input-output family update at the object-domain constraint step."""
    if mode is None:
        mode = "hybrid"
    if mode not in {"input-output", "output-output", "hybrid"}:
        raise ValueError("mode must be 'input-output', 'output-output', or 'hybrid'")

    mask = mask.astype(bool)
    gamma = ((candidate < 0) & mask) | (~mask)

    if mode in {"output-output", "hybrid"}:
        updated = candidate.copy()
    else:
        updated = previous.copy()

    if mode in {"input-output", "hybrid"}:
        updated[gamma] = previous[gamma] - beta * candidate[gamma]
    else:
        updated[gamma] = candidate[gamma] - beta * candidate[gamma]
    return updated


def phase_retrieval(mag, mask=None, beta=0.8, steps=200, mode=None, init=None):
    """Phase retrieval from a measured Fourier magnitude.

    mode=None runs HIO. Use mode="output-output", beta=1.0 for an ER cleanup pass.
    """
    if mask is None:
        mask = np.ones(mag.shape, dtype=bool)
    else:
        mask = mask.astype(bool)

    if init is None:
        spectrum = mag * np.exp(1j * 2 * np.pi * np.random.rand(*mag.shape))
    else:
        spectrum = mag * np.exp(1j * np.angle(np.fft.fft2(init)))

    previous = None
    for _ in range(steps):
        candidate = np.real(np.fft.ifft2(spectrum))
        if previous is None:
            previous = candidate.copy()
        image = object_domain_step(candidate, previous, mask, beta, mode)
        spectrum = mag * np.exp(1j * np.angle(np.fft.fft2(image)))
        previous = image
    return previous


def object_error(output, mask):
    """Energy-normalized violation of non-negativity and support."""
    mask = mask.astype(bool)
    gamma = ((output < 0) & mask) | (~mask)
    energy = np.sum(output ** 2)
    if energy == 0:
        return 0.0
    return np.sqrt(np.sum(output[gamma] ** 2) / energy)
```
