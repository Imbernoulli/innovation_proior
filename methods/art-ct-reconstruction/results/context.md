## Research question

A specimen — a biological macromolecule under an electron beam, a slice of a human head under X-rays — is opaque, and one cannot see inside it directly. What a detector records is a *projection*: for each ray that passes through the object, the detector measures how much was absorbed, which (after taking a logarithm of the attenuation) is the line integral of the object's density along that ray. Take many such rays from many directions and the question is to recover the internal density distribution `f(x, y)` of a cross-section from its line integrals. The measurement conditions vary by application. In electron microscopy the tilt range is mechanically limited (often no more than ±60°) and the dose is kept low, giving few views and noisy data. In X-ray body-section work one aims to keep patient exposure low, again with a limited number of measurements. Discretized, the problem involves a number of unknowns (pixels) and equations (rays) both in the tens of thousands; physical prior knowledge is available — density is nonnegative, the object is zero outside a known region.

## Background

**The Radon transform and the line-integral model.** The mathematical object underneath all of this is the Radon transform: the map from a 2-D function `f(x, y)` to the set of its line integrals `p(θ, s) = ∫ f along the line at angle θ, offset s`. The full set of projections over all angles is a *sinogram*. Reconstruction is inverting the Radon transform. There is a clean analytic inverse via the **projection-slice (central-slice) theorem**: the 1-D Fourier transform of a projection at angle θ equals a radial slice, at angle θ, of the 2-D Fourier transform of `f`. Assembling all slices fills Fourier space, and an inverse 2-D transform recovers `f`. This is the basis of the analytic reconstruction methods (see Baselines).

**Discretization into a linear system.** Superimpose a square grid of `N` cells on the unknown image and assume `f` is constant `f_j` within cell `j`. A "ray" is a fat line of finite width; its measured ray-sum `p_i` is then a weighted sum of the cells it crosses, `Σ_j w_ij f_j = p_i`, where `w_ij` is the contribution of cell `j` to ray `i` — naturally, the length (or area) of the intersection of ray `i` with cell `j`. Stacking `M` rays gives a linear system `A f = p` with `A = [w_ij]`. Two facts about this matrix characterize it: it is **enormous** (for a 256×256 image, `N ≈ 65000`, and `M` of comparable size, so `A` is ~65000×65000) and it is **extremely sparse** (each ray crosses only `O(√N)` cells, so almost every `w_ij` is zero). For a 100×100 grid with 100 views of 150 rays the dense coefficient table has roughly `1.5e8` entries. When measurement noise is present, or `M < N`, the system is inconsistent or underdetermined: no exact inverse exists, or many solutions do.

**The geometry of one equation: a hyperplane.** A grid image is a single point `f = (f_1, …, f_N)` in `N`-dimensional space. Each equation `Σ_j w_ij f_j = p_i` constrains that point to lie on a *hyperplane*: the set `{f : a_i · f = p_i}`, where `a_i = (w_i1, …, w_iN)` is the i-th row. The hyperplane is perpendicular to the vector `a_i`, and its signed distance from the origin is `p_i / ‖a_i‖`. If a unique solution exists, it is the single point where all `M` hyperplanes meet — the common intersection of the `M` hyperplanes.

**The method of projections (Kaczmarz, 1937).** Stefan Kaczmarz gave an iterative scheme for `A x = b` built entirely on orthogonal projection: starting from any guess, project the current point onto the hyperplane of the first equation, then onto the second, and so on, cycling through the equations repeatedly. The orthogonal projection of a point `x` onto the hyperplane `a_i · z = b_i` is the closest point on it; it moves `x` only along `a_i`. For a consistent, nonsingular system the cyclic projections converge to the solution. Two further facts about this convergence were established. Tanabe (1971) showed that for an **underdetermined consistent** system (`M < N`, infinitely many solutions) the iteration started from `x_0` converges to the solution *closest to* `x_0` — the minimum-norm solution when `x_0 = 0`. And the *rate* depends on the angles between successive hyperplanes: if they are mutually orthogonal, one sweep suffices; if they are nearly parallel, convergence is slower. For an **overdetermined, inconsistent** system (`M > N`, no common intersection) the cyclic projections settle into a limit cycle in the neighborhood of the near-intersections of the hyperplanes.

**The discrete forward model.** The discrete forward model represents the continuous line integrals only approximately: the pixel-basis assumption (`f` constant within a cell) makes the computed ray-sums approximate, so the equation system is mildly **inconsistent** even with noiseless data. The fidelity of the model depends on how the weights `w_ij` are formed — finer weighting (intersection length or area) gives more accurate computed ray-sums than a crude binary `0/1` occupancy. The forward model can be evaluated cell-by-cell on the fly without storing the full operator.

## Baselines

**Direct matrix inversion / least squares.** With `A f = p` in hand, the textbook answer is `f = A⁻¹ p`, or for the rectangular/noisy case the least-squares normal-equations solution `f = (AᵀA)⁻¹ Aᵀ p`. Core idea: solve the linear system exactly. Algorithm: form and invert an `N×N` (or solve the normal equations) system, operating on the dense operator at scale `N ≈ 10⁴–10⁵`.

**Analytic inversion: backprojection and filtered backprojection.** Using the projection-slice theorem one can invert the Radon transform analytically. Plain **backprojection** smears each projection back uniformly along its rays and sums over angles; it reconstructs a blurred image (a `1/r` blur, because uniform angular sampling oversamples low spatial frequencies). **Filtered backprojection (FBP)** first applies a 1-D **ramp filter** `H(ω) = |ω|` to each projection (high-pass, to undo the `1/r` weighting), then backprojects. Core idea: filter-then-smear, an exact inversion in the continuous limit. Algorithm: per view, FFT the projection, multiply by `|ω|` (optionally windowed — Shepp-Logan, Hamming — to attenuate high frequencies), inverse-FFT, and backproject; `O(M · N log N)` overall, non-iterative, very fast. It is a fixed linear operator derived from dense, uniformly-sampled, straight-line projections over a full angular range. (For diffracting sources — ultrasound, microwave — rays *bend*, and the straight-line Fourier-slice relationship does not hold.)

**SIRT (Simultaneous Iterative Reconstruction Technique).** An iterative scheme that treats all equations together: compute the correction each equation would make to each pixel, but do not apply any of them immediately; instead, at the end of a full pass average all the corrections accumulated at each pixel and apply that average, then repeat. Core idea: a simultaneous-correction scheme in which all corrections within a pass are applied against the same image. Algorithm: one full forward-and-correct pass per iteration, corrections applied only at iteration's end. It reconsiders the entire equation set each iteration and is order-agnostic.

**Cyclic projection (Kaczmarz).** Applying Kaczmarz directly to `A f = p` — project onto each ray's hyperplane in turn, exactly satisfying it. Core idea: cyclic orthogonal projection, `O(nnz)` per sweep, tiny memory, computes weights on the fly. Algorithm: for each ray `i`, replace `f` by its orthogonal projection onto `{a_i · f = p_i}`, cycling through all rays repeatedly.

## Evaluation settings

The natural test object is the **Shepp–Logan head phantom**: a configuration of overlapping ellipses of prescribed densities chosen so that the line integral through it can be written in closed form (an ellipse's projection is analytically known), so exact, noiseless "real" projection data can be generated for it by summing per-ellipse line integrals — and a variant adds a small subdural-hematoma ellipse to probe low-contrast detail. The standard discretization in this setting is a **128×128 sampling lattice**, reconstructed from on the order of **100 projections of ~127 rays each** (so `N = 16384` pixels and `M ≈ 12700` rays, an ~25%-underdetermined system once the circular reconstruction region is accounted for). The reconstruction region is taken as the inscribed **circle**, with the object vanishing outside it. Metrics are the **root-mean-squared error** against the known phantom and, because RMSE hides texture, **line plots through fine structures** (e.g. through three small low-contrast tumors along a fixed scan line) to see whether the detail rises above the reconstruction noise; image displays are window-thresholded (e.g. to the density range 0.95–1.05) to reveal fine contrast. For the electron-microscopy motivation the relevant protocol is a **limited tilt range** (angles spanning no more than ~±60°) with few views and low dose; for the diffracting-source motivation, **curved rays** from ray tracing rather than straight lines.

## Code framework

The available pieces are a sparse forward operator (a ray tracer that, given a parallel-beam geometry, returns for each ray the list of cells it crosses and the intersection lengths) plus the standard array and sparse-matrix primitives. The inverse routine is the empty slot.

```python
import numpy as np
from scipy.sparse import csr_matrix

def build_system_matrix(n, angles, n_rays=None, det_spacing=1.0):
    """Parallel-beam forward model: A[i, j] = intersection length of ray i with pixel j.
    Returns A (sparse, M x n*n) and, per ray, which projection/view it belongs to.
    The reconstruction routine below inverts it."""
    # ... trace each ray of each view across the n x n grid, accumulate lengths ...
    pass

def forward_project(A, x):
    """Computed ray-sums q = A x for the current image estimate."""
    return A.dot(x)

# --- inverse routine ---
def reconstruct(A, b, ray_view, x0=None, nonneg=True, **params):
    """Recover the image vector x from measured ray-sums b given the forward
    operator A.
    """
    # TODO: implement the reconstruction.
    pass

def apply_constraints(x, nonneg=True):
    """Fold in cheap physical priors such as density >= 0."""
    if nonneg:
        np.clip(x, 0.0, None, out=x)
    return x

if __name__ == "__main__":
    n = 32
    img = ...                      # phantom
    angles = np.linspace(0, np.pi, 30, endpoint=False)
    A, ray_view = build_system_matrix(n, angles, n_rays=n)
    b = forward_project(A, img.ravel())     # simulated projection data
    x = reconstruct(A, b, ray_view)         # iterative inversion step
```
