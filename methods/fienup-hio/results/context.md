## Research question

A coherent wave's far-field diffraction pattern records only an intensity, so what reaches the
detector is the *Fourier modulus* `|F(u)|` of the object `f(x)`, never the Fourier phase `ψ(u)`.
In stellar speckle interferometry the power spectrum yields `|F(u)|` for a self-luminous object
that is, physically, real and non-negative and of finite extent; in X-ray crystallography the
structure-factor magnitudes are `|F|` with the phases lost. The task is to recover the missing
Fourier phase — equivalently `f(x)` itself — from **one** modulus `|F(u)|` plus whatever is known
a priori about the object: that it is real, that it is non-negative `f(x) ≥ 0`, and that it lives
inside a bounded region (its support). This is harder than the two-modulus problem of electron
microscopy or wavefront sensing, where a focused image supplies a *second* measured modulus
`|f(x)|`: here the object plane has no measured amplitude at all, only inequality/support
constraints standing in for it. A 128×128 field carries N² unknown phases, so brute search is
hopeless; the problem needs an iteration that uses only `|F|`, an FFT, and the object constraints,
and it must have a way to escape the long stalls produced by plain alternating projection.

## Background

**The phase problem and its conjugate-symmetry degeneracies.** That detectors keep intensity and
discard phase is the oldest obstacle in diffractive imaging, dating to the Braggs in
crystallography. Two facts about the Fourier-modulus map make the single-intensity version
delicate. First, `|F(u)|² = |FT f|²` is the Fourier transform of the object's **autocorrelation**
`f ⋆ f`, so the data fix the autocorrelation but not `f` directly. Second, `f(x)`, its translate,
and its **inverted conjugate** `f*(−x)` all share the same `|F(u)|` — the "twin image" degeneracy.
A reconstruction can therefore lock onto a blend of the object and its centro-symmetric twin, a
genuine ambiguity the constraints (real, non-negative, support) are there to break.

**Support from the autocorrelation.** Because the autocorrelation `IFT(|F|²)` is computable from
the data, and the support of an autocorrelation is the difference set of the object's support, a
bound on the object's diameter is available — it is half the diameter of the autocorrelation. In
two dimensions the support of the autocorrelation does not in general fix the object's support
uniquely (Fienup, Crimmins & Holsztynski, *J. Opt. Soc. Am.* 72, 1982), so the support mask can be
bounded but only loosely; locator sets contain all candidate supports. A finite-support plus
non-negativity constraint is exactly the side information that, with enough sampling of `|F|`,
makes 2-D recovery essentially unique in practice for complicated objects.

**Constraint sets and projections.** The set of fields with a *prescribed Fourier modulus* and
free Fourier phase is a product of circles in the complex plane — a non-convex set. The nearest
member of such a set to a given field keeps the field's Fourier phase and resets its Fourier
modulus to `|F|`. By contrast the object-domain constraints here — vanish outside the support,
be real and non-negative — define a *convex* set (a subspace intersected with a non-negative
cone), whose nearest point is found by zeroing the field outside the support and clipping negative
values to zero inside. Alternating between the closest point of each set is the natural way to
seek a field lying in both; for two convex sets this alternating-projection idea converges
(von Neumann; Bregman 1965; Gubin–Polyak–Raik 1967), but here one of the two sets is non-convex,
which is the source of the trouble below.

**Diagnostic facts that frame the problem.** Two observed phenomena set the stage. (i) An
iteration that only transforms back and forth, resetting the Fourier modulus and then projecting
onto the object constraints, drives the modulus-mismatch error *down sharply for the first ~30
iterations and then onto long plateaus*: in a representative single-intensity run the normalized
RMS error falls fast, sticks near 0.16, much later eases to 0.02, then to 0.003, with thousands of
iterations needed — unsatisfactory for this application. (ii) The same iteration frequently
**stagnates at a local minimum marked by a low-contrast pattern of stripes** across the
reconstruction (Fienup, "Fourier Modulus Image Construction," RADC-TR-81-63, 1981); this is a
convergence problem — a local, not global, minimum — not a uniqueness problem. Plateaus and
stripes are precisely what a better single-intensity algorithm must overcome, and neither is a
defect of the *data*; they are pathologies of the *iteration* on a non-convex constraint set.

## Baselines

**Gerchberg–Saxton, two intensities (Gerchberg & Saxton, *Optik* 35, 1972).** The original
iterative scheme for the *two-modulus* problem (a focused image gives `|f(x)|`, a diffraction
pattern gives `|F(u)|`). Four steps: Fourier transform the current object estimate; replace the
computed Fourier modulus by the measured `|F|`, keeping the computed phase; inverse transform;
replace the computed object modulus by the measured `|f|`, keeping that phase. Core idea:
alternately enforce each domain's measured modulus. Gerchberg and Saxton prove a defined error decreases
monotonically each iteration. **Gap:** it presumes a *measured* object modulus `|f|`. In the
single-intensity (astronomy/crystallography) problem there is no `|f|` — only the inequality
constraints — so the fourth step has nothing to reset the modulus *to*.

**Error-reduction (generalized Gerchberg–Saxton).** The four-step scheme above, generalized so
that step four makes the *minimum change* needed to satisfy whatever object-domain constraints
hold — measured modulus, or support/non-negativity. For a single intensity the first three steps
are unchanged and the fourth becomes a projection onto the object constraints: with `g'_k` the
inverse transform of the modulus-corrected spectrum and `γ` the set of points where `g'_k`
violates the constraints (negative, or outside the support),

  `g_{k+1}(x) = g'_k(x)` for `x ∉ γ`,  `g_{k+1}(x) = 0` for `x ∈ γ`.

Core idea: each step is either a Fourier hop or a nearest-point projection, so by Parseval the
modulus error chain `E_{F,k+1} ≤ E_{O,k} ≤ E_{F,k}` is monotone non-increasing — the algorithm
provably never makes the Fourier-modulus mismatch worse. It is moreover equivalent to a
double-length-step steepest descent on the Fourier squared error
`B = N^{-2}Σ_u (|G(u)| − |F(u)|)²`, whose gradient
`∂B/∂g(x) = 2[g(x) − g'(x)]` is computed for free by the same two transforms.
**Gap:** monotone decrease on a *non-convex* landscape is not convergence to the global solution;
the iteration plateaus at local minima (the stripe pattern), and for the single-intensity problem
it is *painfully slow* — thousands of iterations. As a fixed step on a near-flat gradient it
crawls, and it "constantly runs into the object-domain constraints," wasting its step against
them.

**Gradient-search variants (steepest descent, conjugate gradient).** Since the gradient is so
cheap, one can do honest line searches or conjugate-gradient directions instead of the fixed
double-length step. Core idea: choose `h_k` to better minimize `B` along `−∇B`, or build conjugate
directions `D_k(x) = g_k(x) − g'_k(x) + (B_k/B_{k-1}) D_{k-1}(x)`. **Gap:** every step must still be
followed by a projection onto the object constraints, so the search keeps colliding with the
constraint set; a fixed step size is suboptimal (the gradient is tiny on plateaus, so one needs a
large `h` there but a small `h` elsewhere to stay stable), and adaptively tuning `h` is itself
unsolved. What is wanted is a scheme that satisfies the object constraints *inherently* while
reducing the Fourier error, rather than alternately fighting them.

## Evaluation settings

The natural single-intensity test object is a digitized photograph of a satellite, ~60×40 pixels
embedded in a 128×128 field on a dark background, with `|F(u)|` either taken noise-free or
synthesized realistically: the undegraded object is convolved with ~150 different atmospheric
point-spread functions, each blurred image is corrupted with Poisson photon noise (of order 100
photons/pixel over the object, ~3×10⁵ photons/image, realistic for a 1.2-m telescope), and a
speckle-interferometry power-spectrum estimate (Labeyrie's method as modified by Goodman &
Belsher) produces the noisy `|F|`. Array sizes are 128×128, FFT-based, running on an array
processor of the era. The yardsticks: the normalized RMS Fourier-modulus error
`sqrt(N^{-2}Σ_u (|G_k| − |F|)²) / sqrt(N^{-2}Σ_u |F|²)` plotted versus iteration on log–log axes, and
the normalized object-domain error `E_O` = square root of `Σ_{x∈γ} [g'_k(x)]²` divided by the
square root of the total output-image energy `Σ_x [g'_k(x)]²`. Visual image quality is tracked alongside, since an error scalar alone
does not reveal stripe artifacts or truncation. Candidate iterations should be compared from the
same starting input and mask, with fixed iteration budgets, and one restarts from a different
random seed on stagnation and reconstructs two or three times to confirm uniqueness.

## Code framework

The primitives that already exist: a 2-D FFT and its inverse (`numpy.fft.fft2/ifft2`), complex-
array construction from amplitude and phase, support-mask handling, the autocorrelation
`IFT(|F|²)` from which a support estimate is thresholded, and the error metrics. The slot to be
filled is the object-domain update after the inverse FFT; everything else is the generic
Fourier-magnitude loop.

```python
import numpy as np

def support_from_autocorrelation(mag, frac=0.04):
    """Threshold the autocorrelation IFT(|F|^2) to bound the object's support."""
    autoc = np.abs(np.fft.fftshift(np.fft.ifft2(mag ** 2)))
    return autoc > frac * autoc.max()

def object_domain_step(candidate, previous, mask, beta=0.8, mode=None):
    """Update the object-domain image after the inverse Fourier step.
    TODO: define the violation set and the per-region rule that produces the next image."""
    pass

def phase_retrieval(mag, mask=None, beta=0.8, steps=200, mode=None, init=None):
    """Run the FFT loop from a measured Fourier magnitude and an object-domain rule.
    TODO: initialize the Fourier phase, iterate inverse FFT -> object_domain_step -> FFT,
          and return the recovered image."""
    pass

def object_error(output, mask):
    """Measure violation of non-negativity and the support mask in an output image."""
    pass
```
