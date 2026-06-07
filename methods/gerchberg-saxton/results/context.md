## Research question

A detector — a photographic plate, a CCD, the human eye — records the *intensity* of a wave, never its phase. Yet for a coherent optical or electron wave the phase carries most of the structural information: the wave that forms an image and the wave in its diffraction (Fourier) plane are an exact Fourier-transform pair, and knowing the modulus in both planes still leaves the phase undetermined. The first problem is **phase retrieval**: given the image-plane amplitude `|f|` and the diffraction-plane amplitude `|F|` of one and the same wave, with `F = FT(f)`, recover the missing phase so the full complex wave — and through it the specimen that produced it — can be reconstructed. Brute search over the phase is hopeless: an N×N field has N² independent phases.

The mirror-image problem is **synthesis**. Suppose the amplitude in one plane is fixed by hardware: a beam of fixed profile illuminates a device that can only retard the phase of the wave pixel-by-pixel, not attenuate it — a kinoform, a diffractive optical element, or a liquid-crystal spatial light modulator (SLM). One wants the wave a focal length away (its Fourier transform) to land on a chosen intensity pattern: a picture, or an array of bright focal spots used to trap and move micron-sized particles. The task is to choose the per-pixel phase `φ_j` so that the Fourier transform of the fixed-amplitude, phase-`φ` field has the desired modulus. The binding difficulty in both problems is the same: **only phase is free**; the amplitude in at least one plane is dictated (by the measurement, or by the modulator hardware), so one cannot simply inverse-transform the target and read off the answer.

For a useful solution one needs an iteration that (i) uses nothing but the two known moduli and an FFT, (ii) is cheap enough to run on the data sizes of the day (128×128 and up), (iii) provably does not make things worse as it runs, and — for the spot-array synthesis case — (iv) delivers *uniform* intensity across the wanted spots and wastes little power on unwanted orders.

## Background

**The phase problem.** That intensity detectors discard phase is the oldest obstacle in diffractive imaging. In X-ray crystallography the diffraction pattern gives the moduli of the Fourier coefficients of the electron density, but the phases are lost; this "phase problem" dates to the Braggs (1912–13) and has its own large literature (e.g. G. Taylor, *Acta Cryst.* D59, 2003). In transmission electron microscopy one can record both a focused image and a diffraction pattern of the same specimen, giving `|f|` and `|F|` directly; D. L. Misell (*J. Phys. D* 6, 1973, "A method for the solution of the phase problem in electron microscopy") framed exactly the two-intensity recovery that motivated the iterative approach. The shared structure: two measured moduli, one Fourier transform connecting them, phase to be filled in.

**Fourier optics of a lens.** A thin lens performs an optical Fourier transform: the complex field in the front focal plane and the field in the back focal plane are a Fourier pair. So "image plane / diffraction plane" and "SLM plane / target plane" are the same mathematical relationship, `F = FT(f)`, realized in glass. A phase mask written on the SLM at the back focal plane therefore controls, through a single FFT, the field delivered to the target plane. For a single off-axis focal spot at `(x_m,y_m,z_m)` the required pixel phase is a known linear-plus-quadratic ramp `Δ_j^m = (2π/λf)(x_j x_m + y_j y_m) + (z_m π/λf²)(x_j²+y_j²)`; a lateral shift is a blazed grating, an axial shift is a Fresnel lens term.

**Constraint sets and projections.** A set of complex fields with a *prescribed modulus* and free phase is a product of circles in the complex plane — geometrically a non-convex set. The nearest member of such a set to a given field is found by keeping the field's phase and resetting its modulus to the prescribed value. Two such moduli, in two Fourier-conjugate planes, define two non-convex sets whose intersection is the sought solution.

**Diagnostic facts.** Two empirical observations frame the problem. First, an iteration that only resets moduli and transforms back and forth is observed to drive the modulus-mismatch error *down rapidly for a few iterations and then onto long plateaus*, sometimes stagnating with a characteristic low-contrast stripe pattern across the reconstruction (J. R. Fienup, *Phase retrieval algorithms: a comparison*, Appl. Opt. 21, 1982); the plateau is a convergence problem (a local, not global, minimum), not a uniqueness problem. Second, in the spot-array synthesis case, the simplest recipes — taking the complex superposition of the single-spot phase ramps with random offsets ("random superposition") — produce holograms with good speed but *poor uniformity and spurious "ghost" spots*, because a phase-only mask cannot split power evenly among a structured set of equal-amplitude targets (a phenomenon long noted in holographic-tweezers work, e.g. Curtis, Koss & Grier, *Opt. Commun.* 207, 2002). These two facts — slow stagnation, and non-uniform spot intensities — are exactly what a better algorithm must overcome.

## Baselines

**Direct inverse transform (the thing you cannot do).** If both amplitude *and* phase were free in the SLM plane one would simply set the SLM field to `IFT(target)` and be done. Because the SLM modulates phase only (fixed illumination amplitude), this is unavailable; it is the baseline that defines the difficulty rather than a usable method.

**Random superposition (SR / "random mask").** For an array of `M` target spots, each spot `m` has a known single-spot phase ramp `Δ_j^m`. Superpose them with random per-spot phase offsets `θ_m` and keep only the phase: `φ_j = arg Σ_m exp(i(Δ_j^m + θ_m))` (Lesem–Hirsch–Jordan kinoform lineage; used for tweezers by Curtis–Koss–Grier 2002). Core idea: each ramp focuses one spot; the random offsets decorrelate them so interference doesn't pile all the energy into one place. Algorithm: one pass, no iteration, `O(MN)`. **Gap:** the random offsets give no control over how power actually divides — efficiency is mediocre and uniformity is poor, with ghost spots; acceptable for crude real-time steering, inadequate when even intensities are required.

**Error-reduction / iterative Fourier-transform without weighting.** The generic iterate: transform the current SLM field to the target plane, replace its modulus by the target modulus (keeping phase), transform back, replace its modulus by the fixed illumination modulus (keeping phase), repeat (Fienup 1982 calls the generalized form "error-reduction"). Core idea: alternately satisfy each plane's modulus constraint. Algorithm: two FFTs per iteration, `O(N² log N)`. It provably never increases the modulus-mismatch error and is closely related to steepest-descent error minimization (Fienup 1982, §III and Appendix). **Gap:** convergence stalls on plateaus at non-global minima (the non-convex sets have spurious fixed points); and when applied to an equal-amplitude spot array it inherits the non-uniformity problem above — equal *target* amplitudes do not yield equal *delivered* spot intensities.

**Feedback variants (input–output family).** Replacing the object-plane reset with a feedback rule that drives constraint-violating points toward zero — the input–output and hybrid input–output (HIO) algorithms (Fienup 1978–1982) — converges much faster on the single-intensity, support-constrained astronomy problem and escapes many plateaus. Core idea: the three transform-and-reset steps are a nonlinear system whose input need not be the current best estimate but a *driving function*; pushing the input by `β·(violation)` steers the output. **Gap (for this setting):** HIO targets the single-modulus-plus-support problem and does not by itself address per-spot intensity *uniformity* in a multi-spot array; it is a different cure for the stagnation symptom, not for the unevenness symptom.

## Evaluation settings

The natural test objects are: (a) a two-plane pair `(|f|, |F|)` synthesized from a known complex function (often a smooth profile times a Gaussian) so the true phase is available for comparison; (b) a grayscale target image to be reproduced in the Fourier plane by a phase-only mask; (c) a 2-D or 3-D array of focal spots — a square grid, or a 3×3×3 cubic lattice — for optical trapping. Hardware-side settings: a liquid-crystal SLM with ~512–768 pixels across, 8-bit grayscale mapped linearly to a 0–2π phase retardation, illuminated by a uniform or Gaussian beam; a focusing objective of focal length `f` and wavelength `λ` (e.g. 532 or 800 nm).

The standard yardsticks are modulus/phase agreement and three power metrics. With the phase-retrieval DFT convention `F_m = (1/N)Σ_n f_n exp(-2πinm/N)`, Parseval gives `Σ|f|² = NΣ|F|²`, so the summed-squared modulus error is scaled as object plane `e_k = (Σ_n (|y_k[n]| − |f[n]|)²)^{1/2}` and diffraction plane `E_k = (N Σ_n (|X_k[n]| − |F[n]|)²)^{1/2}`. For spot arrays, with `I_m = |V_m|²` the fraction of power in spot `m`: the **efficiency** `e = Σ_m I_m` (power landing in wanted spots) and the **uniformity** `u = 1 − (max I_m − min I_m)/(max I_m + min I_m)` (which is 1 when all spots are equal). A percent-RMS spread of the spot intensities about their mean is the companion measure. Iteration count to reach a tolerance, and wall-clock time, complete the protocol.

## Code framework

The primitives that already exist: a 2-D FFT and its inverse with centered spectra (`numpy.fft.fft2/ifft2`, `fftshift/ifftshift`), complex-array construction from amplitude and phase, and the metric helpers. The slots to be filled are the iterations that turn a fixed illumination amplitude and a target modulus, or a list of focal-spot propagation phases, into a phase mask.

```python
import numpy as np

def to_field(amplitude, phase):
    return amplitude * np.exp(1j * phase)

def forward(u):          # SLM/object plane -> target/Fourier plane (a lens)
    return np.fft.fftshift(np.fft.fft2(u))

def backward(U):         # target/Fourier plane -> SLM/object plane
    return np.fft.ifft2(np.fft.ifftshift(U))

def normalization(a):
    a_min = a.min()
    a_max = a.max()
    return (a - a_min) / (a_max - a_min + 1e-12)

def uniformity(intensity, mask):
    I = intensity[mask == 1] / np.max(intensity)
    return 1.0 - (I.max() - I.min()) / (I.max() + I.min())

def image_efficiency(norm_intensity, target, mask):
    return np.sum(norm_intensity[mask == 1]) / np.sum(target[mask == 1])

def spot_efficiency(spot_intensity):
    return np.sum(spot_intensity)

def synthesize_phase(target_amplitude, illumination=1.0, n_iter=30):
    """Return a phase-only mask whose Fourier transform approximates
    target_amplitude, under the fixed `illumination` amplitude."""
    phase = np.random.rand(*target_amplitude.shape)   # initial guess
    # TODO: choose the transform-domain replacement rule and the object-plane
    #       phase-only replacement rule.
    pass

def synthesize_balanced_phase(target_amplitude, n_iter=30):
    """Return a phase-only mask for target points whose relative intensities
    should be made even."""
    phase = np.random.rand(*target_amplitude.shape)
    weights = target_amplitude.astype(float).copy()
    prev_w = weights.copy()
    # TODO: choose how observed intensity should revise the target amplitudes
    #       before the transform-domain replacement rule is applied.
    pass

def synthesize_spot_phase(delta, illumination, n_iter=30):
    """Return a phase-only mask for a set of focal spots.
    `delta[j, m]` is the known propagation phase from pixel j to spot m."""
    theta = np.random.rand(delta.shape[1]) * 2 * np.pi
    weights = np.ones(delta.shape[1])
    # TODO: choose how the single-spot phase ramps are combined, how the
    #       resulting mask is projected back to phase-only form, and how the
    #       spot fields are re-read.
    pass
```
