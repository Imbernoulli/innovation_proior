# Context — wavefront correction from slope measurements in a closed loop

## Research question

Light from a star arrives at the top of the atmosphere as a flat wavefront. By the time it reaches a ground telescope it has passed through kilometres of turbulent air whose refractive index fluctuates, so the wavefront is crumpled: the phase φ(x, y, t) over the telescope pupil is a randomly varying aberration. A crumpled wavefront does not focus to a point — the image blurs to the *seeing* limit, much larger than the diffraction limit the aperture could in principle deliver. The question is how to **measure this time-varying phase aberration and cancel it in real time** by reshaping a corrector so that the wavefront leaving the system is flat again, restoring near-diffraction-limited imaging.

The setting has three features. The phase cannot be measured directly: optical detectors record intensity, and there is no device that reads φ at optical frequencies in real time, so any measurement is *indirect* and must be inverted to recover the phase. The aberration is not static — turbulence blows across the aperture on a millisecond timescale, so the correction runs as a feedback loop that tracks the evolving wavefront. And the corrector and the sensor are both finite and discrete — a finite number of actuators, a finite number of sub-measurements — so the inversion is a finite linear-algebra problem, and the loop closes every millisecond.

## Background

**Atmospheric turbulence.** The Kolmogorov model describes turbulence as energy injected at large scales and cascading to small scales, producing refractive-index fluctuations with a power-law spectrum. The phase imprinted on a wavefront has structure function D_φ(r) = ⟨[φ(x+r) − φ(x)]²⟩ = 6.88 (r/r₀)^(5/3), where **r₀** (the Fried parameter) is the coherence length of the atmosphere: r₀ ∝ [λ⁻² (cos γ)⁻¹ ∫ Cₙ²(z) dz]⁻³/⁵, with γ the zenith angle and Cₙ² the index structure constant along the path. Over an aperture of diameter D, the number of independent turbulent cells is ~(D/r₀)², which sets how many degrees of freedom a corrector needs. The spectrum is *red* — most of the phase variance is at low spatial and temporal frequencies. The temporal scale comes from the wind blowing the frozen turbulence pattern across the pupil (Taylor's frozen-flow hypothesis): the **Greenwood frequency** f_G = 0.426 v/r₀ (v the layer wind speed) sets the bandwidth a correction loop must reach, and the coherence time is τ₀ ≈ 0.314 r₀/v. For a loop with finite latency Δt that feeds back the latest measured phase with opposite sign ("zero-order prediction"), the residual variance from being one step behind the turbulence is σ²_servo = 28.4 (f_G Δt)^(5/3) for Kolmogorov turbulence.

**Sensing slopes, not phase — the Shack–Hartmann wavefront sensor.** A Shack–Hartmann sensor places an array of small lenslets across an image of the pupil. Each lenslet forms a focal spot from the light in its sub-aperture; if the wavefront over that sub-aperture is tilted, the spot shifts. The spot displacement in x and y is proportional to the **average wavefront slope (gradient)** over the sub-aperture. So the sensor returns, per sub-aperture, two numbers (sₓ, s_y) — local gradients — and never the phase itself. A whole sensor frame is a vector **s** of all sub-aperture slopes. Recovering φ from s is a *gradient-integration* problem: a discrete derivative has been measured and the antiderivative is wanted. Because only gradients are seen, any spatially constant offset (piston) is invisible to the sensor — an ambiguity baked into slope sensing.

**Sampling geometry and unsensed modes.** When the sub-aperture corners coincide with the corrector's actuator locations (the **Fried geometry**), a single sub-aperture's slopes relate to the four corner displacements d₁…d₄ by sₓ = −d₁ + d₂ − d₃ + d₄ and s_y = −d₁ − d₂ + d₃ + d₄. Stacking all sub-apertures gives a linear forward map s = A d. This map has a non-trivial null space. Besides global piston, the Fried geometry has the **waffle** mode: pushing the two diagonal corners c₁, c₄ up and the other diagonal c₂, c₃ down by equal amounts gives a checkerboard surface with w = d₁ − d₂ − d₃ + d₄ ≠ 0 but sₓ = s_y = 0 in every sub-aperture — the sensor reads zero for it in the ideal local stencil. In finite and misregistered systems the waffle-like component can be mixed across near-null directions rather than appearing as one clean singular vector.

**The deformable mirror.** The corrector is a thin reflective surface backed by an array of actuators; pushing an actuator deforms the surface locally (its *influence function*), and a command vector c of actuator amplitudes produces a mirror shape that is the linear superposition of influence functions. Its key parameters — actuator spacing and number, response time, stroke — match what the turbulence demands (spacing set by r₀, response time by τ₀, stroke by the peak-to-valley aberration). Because the actuator count is finite, the mirror reproduces spatial frequencies up to its Nyquist; the residual high-frequency wavefront beyond that is the **fitting error**.

**Putting a number on residual quality.** For small residual phase variance σ² (in rad²) over the pupil, the Strehl ratio — peak intensity relative to the diffraction-limited peak — is well approximated by the (extended) Maréchal relation S ≈ exp(−σ²). The residual variance decomposes, assuming the contributors are uncorrelated, into σ²_WFE = σ²_fit + σ²_rec + σ²_bw + …: the fitting error of the finite mirror, the reconstruction error (measurement noise, aliasing, calibration, sampling propagated through the inversion), and the temporal bandwidth (servo-lag) error of the finite-speed loop. Each depends on sampling density, loop gain, and loop speed.

## Baselines

**Modal estimation (Zernike fit).** One way to invert slopes is to expand the phase in a small set of orthogonal aperture modes — Zernike polynomials Z_j(x,y) (piston, tip, tilt, focus, astigmatism, coma, …), which diagonalise the Kolmogorov statistics and are the natural basis for circular pupils. One fits φ ≈ Σ_j a_j Z_j by least squares to the measured slopes (since ∇Z_j is known analytically), reconstructing a *smooth* low-order estimate. It uses few parameters, is naturally regularized (high orders simply dropped), and is robust to noise. A separate modal-to-actuator projection then maps the fitted modes to mirror commands.

**Direct zonal integration of gradients.** Treating the slopes as finite differences of phase values on a grid, one can integrate them — path integration, or solving the discrete Poisson/least-squares gradient equations on the Hudgin/Southwell/Fried grids. This gives a phase map at grid points, which is then mapped to actuator commands.

**Single-shot inversion without feedback.** Given a forward map s = A d (or s = D c with D the measured interaction matrix between actuators and slopes), the most direct estimate is one matrix multiply by an inverse of A. Because A is rectangular and degenerate, the literal inverse does not exist; the Moore–Penrose pseudo-inverse minimizes ‖A d − s‖². This produces a correction in a single open-loop multiply from one sensor frame.

## Evaluation settings

The natural yardsticks are simulation and on-sky AO testbeds. A simulator generates Kolmogorov/von Kármán phase screens with a chosen r₀, L₀ (outer scale) and wind profile, propagates them through a modelled Shack–Hartmann sensor (lenslet array, finite photon flux and read noise, centroiding the spots) and a modelled deformable mirror with a given actuator grid, and runs the control loop at a fixed frame rate (e.g. ~1 kHz) with a fixed latency. Configuration knobs: telescope diameter D, sub-aperture / actuator counts (D/r₀ sampling), loop frame rate and delay, inversion conditioning, and loop gain. Performance is read out as the residual wavefront error variance (and its fit/recon/bandwidth split) and the resulting Strehl ratio or point-spread function, evaluated across r₀ values, guide-star magnitudes (photon noise), and wind speeds. On the bench, the same loop is closed on an optical setup with a known aberration source.

## Code framework

A pre-existing numerical stack: `numpy` for arrays and linear algebra (`numpy.linalg.svd`, `numpy.linalg.pinv`, least-squares solves), `scipy` for interpolation, plus turbulence and Zernike utilities that already exist (phase-screen generators for a given r₀, Zernike-mode generators, a circular-pupil mask, a centre-of-gravity centroider for sub-aperture spots). The deformable-mirror and sensor abstractions provide influence functions and a routine that maps a mirror shape to the slopes a Shack–Hartmann sensor would report.

What does not yet exist is the bridge from slopes to commands and the loop that closes around it:

```python
import numpy as np

# --- already available ---
def make_phase_screen(r0, size): ...          # Kolmogorov/von Karman turbulence
def dm_influence_functions(n_act, grid): ...   # mirror shape per actuator poke
def shack_hartmann_slopes(phase): ...          # returns slope vector s from a phase map
def centre_of_gravity(spots): ...              # sub-aperture centroids -> slopes

class Reconstructor:
    """Turn a wavefront-sensor slope vector into corrector commands."""
    def __init__(self):
        self.interaction_matrix = None
        self.control_matrix = None
        self.actuator_values = None

    def build_interaction_matrix(self, dm, wfs, poke=1.0, imat_noise=False):
        # TODO
        pass

    def build_command_matrix(self, conditioning=1e-3, alpha=None):
        # TODO
        pass

    def reconstruct(self, slopes):
        # TODO
        pass

    def apply_gain(self, command_increment, gain, leak=0.0):
        # TODO
        pass

def close_loop(reconstructor, dm, wfs, atmos, gain, n_iter, leak=0.0):
    commands = np.zeros(dm.n_act)
    slopes = np.zeros(wfs.n_measurements)
    for k in range(n_iter):
        phase = atmos.step()                      # turbulence evolves
        # TODO
        pass
    return commands
```

The empty methods are the missing bridge from sensor slopes to corrector commands and the loop that closes around them.
