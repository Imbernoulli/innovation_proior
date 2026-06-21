The problem is to recover a real, non-negative object of finite extent from a single measured Fourier modulus, such as a far-field diffraction pattern or a speckle-interferometry power spectrum. Detectors record only intensity, so we know |F(u)| but not the Fourier phase; the object-domain information is reduced to the a-priori constraints that f(x) is real, non-negative, and zero outside a bounded support. This single-intensity setting is harder than the two-modulus Gerchberg-Saxton problem because there is no measured object-plane amplitude to project onto, only an inequality/support constraint. A straightforward alternating projection between the Fourier-modulus set and the object-constraint set—error reduction—does decrease the modulus mismatch monotonically, but because the Fourier-modulus set is non-convex it plateaus for thousands of iterations and often stagnates in a striped local minimum.

The right fix is to stop treating the iterate as an object estimate and instead treat the Fourier-projection step as a nonlinear box whose output always has the correct modulus. The input to that box can then be used as a driving function. Where the output already satisfies the object constraints we should keep it, since it has both the right modulus and the right support/sign. Where the output violates the constraints, we should feed back an accumulated correction that can overshoot the constraint set and escape a basin. This leads to Fienup's hybrid input-output (HIO) algorithm.

The hybrid input-output algorithm updates the object-domain image after each inverse Fourier step as follows. Let g'_k be the output of the Fourier projection—i.e., the inverse transform of the measured modulus multiplied by the current computed phase. Let γ be the set of points where g'_k violates the constraints: negative values inside the support, or any value outside the support. Then the next input g_{k+1} is set to g'_k at the satisfying points, and to g_k − β g'_k at the violating points, where β is a feedback gain of order one. At the good points we accept the correct-modulus output; at the bad points we accumulate a correction until the output is forced non-negative. This sacrifices the monotone error decrease of error reduction in exchange for the ability to leave spurious local minima. In practice one alternates blocks of HIO iterations with a few error-reduction iterations to consolidate the progress.

A few implementation details matter. The support mask can be estimated by thresholding the autocorrelation, since the inverse transform of |F|² is the object's autocorrelation and its diameter is about twice the object's diameter; the mask should be kept loose so an off-center partial reconstruction is not truncated. The initial Fourier phase should be random rather than constant, to break the centro-symmetry that otherwise stalls the iteration. Because |F| is invariant under translation and under inversion/conjugation, the reconstruction may land on a twin image; restarting from a few random seeds and checking agreement resolves this up to the inherent symmetries.

The feedback gain β controls how aggressively HIO pushes against constraint violations. A value near one gives rapid motion but can oscillate or diverge, while a smaller value is more stable but slower; β = 0.8 is a common conservative default. Because HIO is no longer monotone, the useful diagnostic is the object-domain error of the Fourier-projection output—the energy-normalized size of the violation set—rather than the input's Fourier error. When that output error stalls, a short burst of error-reduction iterations projects the current output back onto the object constraints and often locks in the progress made during the HIO escape phase.

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
