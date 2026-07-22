Let me start from what's actually broken on the sky. A star is, for all practical purposes, a point at infinity — a flat wavefront arriving at the top of the atmosphere. If my telescope of diameter D got that flat wavefront, it would focus it to a diffraction-limited spot of width ~λ/D. It doesn't. The light spends its last few kilometres crossing turbulent air whose refractive index fluctuates, and by the time the wavefront reaches my pupil it's crumpled: there's a spatially varying phase φ(x, y) across the aperture, and it's *changing* — the wind blows the turbulence pattern across the pupil on a millisecond timescale. A crumpled wavefront doesn't focus to a point; it smears into the seeing disk, which for a big telescope is enormously larger than λ/D. So the whole game is: figure out φ(x, y, t), put an equal-and-opposite shape on a corrector, and do it fast enough that φ hasn't moved on by the time I correct it.

How bad is the crumpling, and how fast does it move? The turbulence statistics tell me. Kolmogorov says the phase has a structure function D_φ(r) = 6.88 (r/r₀)^(5/3), where r₀ — the Fried parameter — is the patch size over which the wavefront stays roughly flat (to about a radian RMS). Over my aperture there are roughly (D/r₀)² of these independent patches, and that number is basically how many independent things I'll have to measure and correct — it sets the size of the problem. The time axis comes from the wind: if a layer moves at speed v, the pattern sweeps across r₀ in about τ₀ ≈ 0.314 r₀/v, and the relevant correction bandwidth is the Greenwood frequency f_G = 0.426 v/r₀. These aren't free parameters I get to choose; they're what the atmosphere hands me, and they say "you have (D/r₀)² degrees of freedom to fix, and you have a millisecond to do it." Good — that frames the size and the speed of everything that follows.

Now the immediate wall: I want φ, but I can't measure φ. Detectors count photons — they record intensity, |field|², which throws the phase away. There's no device that reads optical phase directly in real time. So whatever I build has to measure something *intensity-like* that depends on φ, and then invert. The cleanest trick I know: chop the pupil into little sub-apertures with a lenslet array — a Shack–Hartmann sensor. Each lenslet takes the light over its little patch and focuses it to a spot on a detector. If the wavefront over that patch were flat, the spot sits dead centre. If the wavefront over that patch is *tilted*, the spot shifts — and the shift is proportional to the average tilt, the average wavefront slope, over the sub-aperture. So per sub-aperture I read off two numbers: the x-displacement and y-displacement of the spot, which are the local gradients sₓ = ⟨∂φ/∂x⟩ and s_y = ⟨∂φ/∂y⟩ over that patch.

Stare at that for a second, because it changes the shape of the problem. I asked for φ and the instrument hands me ∇φ — and only its local average, sampled on a grid, with photon noise on each spot centroid. I've measured the *gradient* of the thing I want. So the central problem isn't sensing, it's *inversion*: reconstruct φ (or, more usefully, the corrector commands) from a vector of noisy, discretely-sampled slopes. It's a discrete antiderivative problem.

And right away there's an ambiguity I can't wish away. Gradients are blind to a constant. If I add the same number to φ everywhere — global piston — every slope is unchanged. So piston is fundamentally unmeasurable by this sensor. Fine, I don't actually care about piston for imaging (a constant phase offset across the whole pupil doesn't move the focal spot), but I have to remember that my forward map will have piston in its null space, so I can't naively invert it.

Let me set up the linear algebra honestly. I have a corrector — a deformable mirror — that's a thin surface pushed by a grid of actuators. Command actuator j by amount c_j and the surface takes the shape Σ_j c_j f_j(x, y), where f_j is actuator j's influence function. I want to drive the residual wavefront φ − (mirror shape) flat, which means flattening the slopes the sensor reports. So really I want a linear map from *commands* to *slopes*, and then to invert it.

I could write that map analytically. In the natural geometry where I put the actuators at the corners of the sub-apertures — actuator corners c₁, c₂, c₃, c₄ around one sub-aperture — the average slope across that little square in terms of the four corner heights d₁…d₄ is, to a good approximation, sₓ = −d₁ + d₂ − d₃ + d₄ and s_y = −d₁ − d₂ + d₃ + d₄ (difference of the right edge minus left edge for x, top minus bottom for y). Adjacent sub-apertures share corners, so when I stack every sub-aperture's two slope equations I get a big sparse linear system

  s = A d,

where d is the vector of all actuator heights and s is the vector of all slopes. That's my forward model. But I'm slightly uneasy writing A from this idealized stencil — a real mirror's influence functions aren't perfect tent functions, and the real sensor's response to a given mirror shape isn't exactly this averaging. Let me come back to whether I should *build* A from the analytic stencil or *measure* it; for now, A exists and s = A d.

Inversion. A is rectangular — typically more slope measurements than actuators, because each sub-aperture gives two slopes and I want the system overdetermined so noise averages down — and I suspect it's degenerate. There's no literal inverse for a rectangular matrix anyway. The honest thing for an overdetermined noisy system is least squares: find the command d that makes the predicted slopes A d match the measured s as closely as possible,

  minimize ‖A d − s‖².

Differentiate ‖A d − s‖² = (A d − s)ᵀ(A d − s) with respect to d and set it to zero: ∂/∂d = 2Aᵀ(A d − s) = 0, so AᵀA d = Aᵀ s — the normal equations — and d̂ = (AᵀA)⁻¹ Aᵀ s when AᵀA is invertible. The reconstructor would then be the matrix R = (AᵀA)⁻¹ Aᵀ, and I'd get commands by one matrix–vector multiply, d̂ = R s — a single matrix computed once offline, applied every frame. That's the object I'm reaching for, provided AᵀA is actually invertible.

It isn't, and I should find out *how* badly before I lean on it. Piston is already in the null space of A, so AᵀA is singular and (AᵀA)⁻¹ doesn't exist. But I want to know whether piston is the *only* problem, because the corner geometry has a reputation for one more pathology. Let me just push a checkerboard through the stencil and see. Take one sub-aperture and push the two diagonal corners c₁ and c₄ up by ε and pull c₂, c₃ down by ε — so (d₁,d₂,d₃,d₄) = (ε, −ε, −ε, ε). Plug in:

  sₓ = −d₁ + d₂ − d₃ + d₄ = −ε + (−ε) − (−ε) + ε = 0,
  s_y = −d₁ − d₂ + d₃ + d₄ = −ε − (−ε) + (−ε) + ε = 0.

So a full checkerboard — every actuator pushed opposite to its four neighbours — produces zero slope in this sub-aperture, and by the same arithmetic in every sub-aperture. Let me sanity-check that I haven't accidentally made *everything* vanish: with a pure x-tilt (d₁,d₂,d₃,d₄) = (−1, 1, −1, 1) the same stencil gives sₓ = 1 + 1 + 1 + 1 = 4 and s_y = 1 − 1 − 1 + 1 = 0, and a y-tilt (−1,−1,1,1) gives sₓ = 0, s_y = 4. (I ran exactly these four vectors through the stencil to be sure: piston → (0,0), checkerboard → (0,0), x-tilt → (4,0), y-tilt → (0,4).) Good — tilts are seen strongly, so the sensor isn't blind across the board; it is specifically blind to the checkerboard. That's the waffle mode, w = d₁ − d₂ − d₃ + d₄, and it's a second vector in the null space of A, separate from piston and much nastier: it's a high-spatial-frequency corrugation that, if it ever gets onto the mirror, ruins the correction, and the sensor will never tell me it's there.

So AᵀA is genuinely rank-deficient, and I expect the modes that *are* sensed to span a huge range of how-well-sensed: a mode that barely tilts any spot shows up as a tiny singular value of A, and inverting it means dividing by that tiny number, which multiplies the photon noise on the slopes enormously. Plain (AᵀA)⁻¹ would (a) blow up on the exact null modes and (b) catastrophically amplify noise on the near-null modes. I need to regularize.

The clean way to see and control all of this at once is the singular value decomposition. Write A = U Σ Vᵀ, with U, V orthogonal and Σ diagonal holding the singular values σ₁ ≥ σ₂ ≥ … ≥ 0. The columns of V are the *command modes* (combinations of actuator pokes), the columns of U are the *slope modes*, and σ_i says how strongly command-mode v_i shows up in slope-mode u_i. The least-squares solution is

  d̂ = Σ_i (u_iᵀ s / σ_i) v_i = V Σ⁺ Uᵀ s,

where Σ⁺ inverts the nonzero σ_i. Now I can read off most of the danger. Piston has no slope signature, so it belongs in the null space and the pseudo-inverse leaves it out. The checkerboard waffle pattern is zero in the ideal local stencil too — I just confirmed that — but I have to be more careful with a finite real pupil: edge effects, invalid actuators, influence-function overlap, and misregistration can mix the waffle-like shape across several small-singular-value directions instead of giving me one clean singular vector to throw away. So SVD conditioning should reduce the damage from unsensed and barely sensed modes, but I don't think it by itself guarantees "no waffle," and I'll want to keep that worry alive. The barely-sensed modes have small σ_i, and the factor 1/σ_i is the noise gain on that mode: a slope-noise component along u_i lands on the command as (noise/σ_i) v_i. So the whole numerical danger lives in the small singular values. The cure at this stage is to truncate or soften them: drop every mode with σ_i below a threshold (set by a conditioning parameter rcond, relative to σ_max), i.e. set its 1/σ_i to 0. That's the pseudo-inverse with conditioning — keep the well-sensed modes, discard directions whose slope signal is mostly noise or absent.

I can make the trade smoother instead of a hard cut. Add a penalty on the command size to the least-squares cost — minimize ‖A d − s‖² + α‖d‖², which says "fit the slopes, but don't reach for a huge command to do it". Differentiate: 2Aᵀ(A d − s) + 2α d = 0 → (AᵀA + αI) d = Aᵀ s → d̂ = (AᵀA + αI)⁻¹ Aᵀ s. In SVD terms I'd expect each singular value's inverse 1/σ_i to be replaced by some filtered factor; let me derive what it is rather than guess. Substituting A = UΣVᵀ, AᵀA + αI = V(Σ² + αI)Vᵀ and Aᵀ = VΣUᵀ, so d̂ = V(Σ² + αI)⁻¹ΣUᵀ s, and the i-th diagonal factor is σ_i/(σ_i² + α). I don't fully trust an algebra step like that until I've checked it on numbers, so let me take a random 8×5 A and a random s, compute d̂ both ways — the direct solve of (AᵀA + αI)d = Aᵀs and the SVD form with factor σ/(σ²+α) — at α = 0.3. The two vectors agree to ~1e-15 (machine precision). So the filter factor is σ_i/(σ_i² + α). Read its limits off the formula and check them on the same run: for σ_i = 10 ≫ √α the factor is 0.0997 ≈ 1/σ = 0.10 (well-sensed modes pass through essentially as 1/σ); for σ_i = 0.01 ≪ √α the factor is 0.0333 ≈ σ/α = 0.0333 → 0 (poorly-sensed modes are smoothly suppressed instead of exploding). That's Tikhonov regularization, and α is the knob trading reconstruction noise against fidelity. The αI also makes the matrix strictly invertible, so even the exact null modes are handled — they just get reconstructed as zero. Truncated-SVD and Tikhonov are two faces of the same thing: refuse to divide by singular values that are mostly noise.

So now the offline recipe is: form A, take its regularized pseudo-inverse R, done. But I left a question open — do I write A from the analytic corner stencil or measure it? The analytic stencil assumes ideal tent influence functions and an ideal averaging sensor; the real mirror's f_j leak into neighbours, the real sensor has its own response, and there are alignment errors. If I instead *poke* the system — command actuator j alone to a small amplitude, read the actual slope vector it produces, and make that the j-th response of the forward map — then the map captures the true end-to-end response of *this* mirror and *this* sensor, calibration and all. That's clearly the more honest A. But there's a sign trap, and I want to nail it down concretely because the whole loop's stability hangs on getting the feedback polarity right. The corrector is applied by subtracting its phase from the incoming wavefront. If I poke the mirror with no atmospheric phase in the path, the WFS is effectively seeing `−mirror`, so the raw measured slopes are the negative of the correction slope the mirror can provide. So I should store the *negative* of that raw WFS response. Then a positive stored row says "this actuator can supply this slope correction." Let me verify this actually inverts to the right thing rather than just talk myself into it: in a small synthetic model where the true command→slope map is some matrix G and the WFS returns G·phase − G·correction, I poked each actuator, stored `−(raw response)`, and checked the stored interaction matrix against G — transposed for the actuator-by-slope layout, it came out equal to G to machine precision. So storing the negated poke response really does recover the true forward map with the correction sign baked in; the polarity is right. In the common simulator layout this interaction matrix is stored actuator-by-slope rather than slope-by-actuator: I has shape `(n_actuators, n_slopes)`, `pinv(I)` has shape `(n_slopes, n_actuators)`, and the command increment is `pinv(I).T @ slopes`. That's the same pseudo-inverse as R = A⁺, just transposed and with the correction sign made explicit.

I now have a one-shot corrector: measure s, command d̂ = R s. Is that enough? I don't think so, and the reasons all push the same way. R inherits every imperfection in the single calibration — a slow drift in alignment, a temperature change in the mirror — and my open-loop command is then wrong with no way to notice. The WFS slope-vs-tilt relationship is only linear for small spot displacements, so a big aberration drives spots out of their linear range and a single multiply mis-estimates it. And, worst of all, the aberration is *moving* — by the time I've applied d̂, the turbulence has evolved, and a static one-shot is always correcting a stale wavefront.

The fix for all three is the same: don't try to nail it in one shot, drive the *residual* to zero iteratively. Put the corrector in front of the sensor so the sensor sees the wavefront *after* correction — closed loop. Now s is the residual slope, the error signal, and I want a control law that nudges the commands each frame to push that error toward zero and keeps pushing as the turbulence wanders. With the stored sign above, the reconstructed increment R s is the mirror command that should be added to cancel the current residual:

  c_{k+1} = c_k + g · R s_k.

Each frame, measure the residual slopes s_k, reconstruct the residual command R s_k, and add a fraction g of it to the running command. I should confirm the plus sign is actually negative feedback and not a sign-flipped recipe for divergence, so let me close this loop in the same synthetic model: a static command-space aberration a, residual = G·a − G·c, with c updated by c += g·R·s at g = 0.4. The residual slope norm went 4.30 → 0.026 at frame 10 → 3.5e-13 at frame 59 — it collapses to zero. So the plus sign is genuinely negative feedback (the mirror phase enters the residual as atmosphere minus mirror, so adding mirror command reduces the residual WFS signal), and the loop nulls the error as intended. If I had stored the raw `−mirror` response instead, the same physical feedback law would show up with a minus sign.

Why *accumulate* (an integrator) rather than just set c = g R s (proportional)? The intuition is that the integrator has infinite gain at DC, but let me actually watch both on a constant disturbance instead of asserting it. Scalar model, one well-sensed mode, disturbance a = 1, g = 0.4. Proportional, m ← g(a − m): it settles at m = 0.286, leaving a residual error of 0.714 — and 0.714 is exactly a/(1+g) = 1/1.4, the offset a proportional law is stuck with because it needs a nonzero error to hold a nonzero command. Integrator, m ← m + g(a − m): it settles at m = 1.000, residual 0.000 — a type-1 system with zero steady-state error on a constant input. So the integrator's DC advantage is real, not just a slogan: it drives a steady aberration to zero where the proportional law leaves a permanent fraction of it uncorrected. And turbulence is *red* — most of its power is at low temporal frequency — so a controller with huge low-frequency gain is exactly what rejects the bulk of the disturbance. This is also the cheapest possible predictor: feed back the latest measured phase with the opposite sign and hope the next frame looks like this one ("zero-order prediction"). It's not optimal, but it's the natural first thing, and it's what an integrator does.

Now the loop can go unstable, so I have to find the bound on g rather than pick one by feel. There's delay around the loop: the sensor integrates for a frame, the centroids and the matrix multiply take time, the mirror takes time to move — at least one full frame, realistically closer to two, between sensing an error and correcting it. Model one well-sensed mode after reconstruction. With no turbulence input, the residual slope is just the negative of the delayed mirror command, so the homogeneous update is c_{k+1} = c_k − g c_{k-d}. In transfer-function language the integrator from error to command is g/(1 − z⁻¹), the plant contributes the negative residual sign and a pure delay z⁻ᵈ, and the characteristic equation is 1 + g z⁻ᵈ/(1 − z⁻¹) = 0. For a single frame of delay (d = 1): 1 + g z⁻¹/(1 − z⁻¹) = 0 ⇒ 1 − z⁻¹ + g z⁻¹ = 0 ⇒ z = 1 − g, stable for |1 − g| < 1, i.e. 0 < g < 2. For the realistic two-frame delay (d = 2): 1 + g z⁻²/(1 − z⁻¹) = 0 ⇒ z²(1 − z⁻¹) + g = 0 ⇒ z² − z + g = 0, whose roots multiply to g and sum to 1. Rather than hand-wave the boundary, let me sweep g and read off the largest root magnitude. For d = 2: g = 0.3 → 0.548, g = 0.5 → 0.707, g = 0.7 → 0.837, g = 0.9 → 0.949, g = 1.0 → 1.000, g = 1.1 → 1.049. So the two-frame loop is stable exactly up to g = 1 and unstable past it — the boundary really is g = 1, not the g = 2 of the single-delay case (which the same sweep confirms: d = 1 only reaches |z| = 1 at g = 2). That g = 1 edge is too sharp for real hardware: the loop rings and any model error tips it over. So where should I actually sit? At g = 0.5 the d = 2 roots are 0.5 ± 0.5j, magnitude 0.707, and the discrete-pole damping ratio works out to ζ ≈ 0.40 — a genuinely well-damped loop with real phase margin, comfortably inside the unit circle. That matches the standard rule of thumb of keeping the usable gain around g ≲ 0.5 for this two-step-delay loop. So with a couple of frames of latency I keep the loop gain below about a half — not because it's the bare stability boundary (that's g = 1), but because that's where the loop is actually well-damped and robust.

That ceiling collides with what I *want* from the gain, which is the next thing to reason through. A higher g closes the loop faster — more closed-loop bandwidth, so the correction keeps up with the moving turbulence and the servo-lag residual σ²_servo = 28.4 (f_G Δt)^(5/3) for a zero-order delay Δt shrinks as the loop reduces the effective lag. But a higher g also pumps more of the slope *noise* straight onto the commands every frame (the reconstruction error), and pushes the poles toward the unit circle — exactly what the sweep above shows. A lower g is quiet and stable but sluggish, and a sluggish loop sits behind the turbulence and pays a big servo-lag. So g is the knob balancing bandwidth/servo-lag against noise propagation, and the atmosphere's f_G tells me where the balance has to sit: the faster the turbulence (higher f_G), the more bandwidth I need, the harder I'm pressing against that g ≲ 0.5 stability wall. On a bright star (low noise) I push g up toward the stability limit; on a faint star (noisy slopes) I back g off to keep the noise term down and accept more servo-lag. There's no universally right g — it's tuned to r₀, v, and the guide-star brightness.

One more thing nags at me, and it's the waffle problem coming back. Piston can be handled directly by mean-subtracting the mirror shape, but the high-frequency checkerboard is more slippery. The reconstructor is built to avoid small singular values, yet waffle-like content can still sneak into the running command through numerical round-off, actuator coupling, calibration mismatch, or noise projected through the imperfectly conditioned directions. The sensor is blind or nearly blind to that content — I checked at the top that the checkerboard reads (0, 0) in every sub-aperture — so the error signal does not reliably report it, and the pure integrator leaves whatever unsensed component is already on c exactly where it is. I need to actively bleed off anything the loop can't see. The simplest way: make the integrator slightly *leaky* —

  c_{k+1} = (1 − ℓ) c_k + g · R s_k,

with a small ℓ (say 0.01). The (1 − ℓ) factor pulls every component of c gently toward zero each frame; for a sensed, actively-driven mode the loop just re-commands it and the leak is negligible, but for an unsensed mode — waffle, piston — which nothing ever re-commands, the (1 − ℓ)^k decay is the *only* force acting on it, and it drains away. Look at what the leak does to the pole, too: the integrator's transfer function becomes g/(1 − (1−ℓ)z⁻¹), moving its pole from z = 1 (marginally stable) to z = 1 − ℓ, strictly inside the unit circle — so the leak also pulls the loop off the stability edge and buys robustness to model error, which is exactly what protects the modes the gain bound alone doesn't. The cost is that I've broken the perfect DC rejection, and I'd better quantify how much I'm giving up rather than wave it away. Run the leaky integrator on the same constant disturbance a = 1, g = 0.4, ℓ = 0.01: it settles at m = 0.9756, residual 0.0244. The fixed point of m = (1−ℓ)m + g(a − m) is m = ga/(g+ℓ), so the residual is ℓa/(g+ℓ) = 0.01/0.41 = 0.0244 — matches. So a truly constant aberration now settles at ~2.4% residual instead of exactly zero. With ℓ tiny that residual is negligible, and the protection against waffle buildup is worth far more than the slight loss of DC gain. So: small leak, real gain, and the loop is both stable and self-cleaning.

That's the whole method, and it falls out as: turbulence crumples the wavefront → the only thing I can measure is local slopes on a grid (Shack–Hartmann), not phase → so I have to invert a gradient, which means a forward map from commands to slopes (built by poking the real hardware and storing the correction sign in the interaction matrix) and its least-squares inverse → which is degenerate and noise-sensitive, so the inverse is a regularized SVD pseudo-inverse R that suppresses piston, near-null, and noise-dominated directions without pretending to solve waffle by itself → one shot isn't enough because the aberration moves and the loop has errors, so close the loop with an integrator c_{k+1} = (1−ℓ)c_k + g R s_k that drives the residual slopes to zero → with gain g kept under ~0.5 for stability against the loop delay and tuned for the bandwidth-vs-noise (servo-lag vs reconstruction) trade, and with either explicit mean removal or a small leak ℓ to bleed off the modes the sensor can't see.

```python
import numpy as np

class Reconstructor:
    """Turn wavefront-sensor slopes into corrector commands."""

    def __init__(self):
        self.interaction_matrix = None
        self.control_matrix = None
        self.actuator_values = None

    def build_interaction_matrix(self, dm, wfs, poke=1.0, imat_noise=False):
        n_act = dm.n_act
        n_slope = wfs.n_measurements          # 2 per sub-aperture (sx, sy)
        interaction = np.zeros((n_act, n_slope))
        for j in range(n_act):
            cmd = np.zeros(n_act)
            cmd[j] = poke                     # push one actuator
            mirror = dm.shape(cmd)            # influence function of actuator j
            # The corrector is subtracted from the incoming phase, so store
            # the correction slope, not the raw residual slope from the poke.
            slopes = wfs.measure(
                phase_correction=mirror,
                noise=imat_noise,
                calibration=True,
            )
            interaction[j] = -slopes / poke
        self.interaction_matrix = interaction
        return interaction

    def build_command_matrix(self, conditioning=1e-3, alpha=None):
        if alpha is None:
            # Stored as actuator-by-slope, so pinv has shape slope-by-actuator.
            self.control_matrix = np.linalg.pinv(
                self.interaction_matrix, rcond=conditioning
            )
        else:
            U, sig, Vt = np.linalg.svd(self.interaction_matrix, full_matrices=False)
            filt = sig / (sig**2 + alpha)     # soft filter: sigma/(sigma^2+alpha)
            self.control_matrix = (Vt.T * filt) @ U.T
        return self.control_matrix

    def reconstruct(self, slopes):
        return self.control_matrix.T @ slopes

    def apply_gain(self, command_increment, gain=0.4, leak=0.0):
        if self.actuator_values is None:
            self.actuator_values = np.zeros(command_increment.size)
        self.actuator_values = (
            (1.0 - leak) * self.actuator_values + gain * command_increment
        )
        return self.actuator_values


def close_loop(reconstructor, dm, wfs, atmos, gain=0.4, leak=0.0, n_iter=1000):
    commands = np.zeros(dm.n_act)
    slopes = np.zeros(wfs.n_measurements)
    for k in range(n_iter):
        phase = atmos.step()                          # turbulence evolves
        delta = reconstructor.reconstruct(slopes)     # command from previous residual
        commands = reconstructor.apply_gain(delta, gain=gain, leak=leak)
        correction = dm.shape(commands)
        slopes = wfs.measure(phase, phase_correction=correction)
    return commands

reconstructor = Reconstructor()
reconstructor.build_interaction_matrix(dm, wfs)
reconstructor.build_command_matrix(conditioning=1e-3)
final = close_loop(reconstructor, dm, wfs, atmos, gain=0.4)
```
