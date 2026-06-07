# Synthesis — AO wavefront reconstruction + closed-loop integrator control

## Three-source bottom line
1. **Primary**: Fried 1977 "Least-square fitting a wave-front distortion estimate to an array of phase-difference measurements" (JOSA 67(3):370). Original is paywalled on Optica (opg.optica.org/josa-67-3-370) and OCR-hostile; **flagged gap**. Faithful secondary that reproduces Fried-geometry equations and the least-squares s=Ad → d=A#s formulation, plus the waffle null mode: Praus (MZA) "Development and Analysis of a Waffle Constrained Reconstructor (WCR) for Fried Geometry AO" (amostech.com/.../PRAUS.pdf). Standard integrator-control treatment: Davies & Kasper review + Doelman time-delay preprint.
2. **Background**: Davies & Kasper, "Adaptive Optics for Astronomy" (arXiv:1201.5741) — Kolmogorov, r0, Greenwood, SH WFS measures slopes, s=Dv interaction matrix, R=pinv, error budget σ²=σ²_fit+σ²_rec+σ²_bw, Maréchal Strehl. Doelman, "The minimum of the time-delay wavefront error in AO" (arXiv:1906.03128) — Greenwood f_G=0.426 v/r0, servo-lag σ²_ref=28.4(f_G Δt)^(5/3), zero-order prediction = feed back measured phase with opposite sign (the integrator).
3. **Third-party explainer**: HCIPy tutorial (docs.hcipy.org pyramid WFS) — poke loop builds interaction matrix, forward S=Aφ, normal eq φ=(AᵀA)⁻¹AᵀS, Tikhonov inverse_tikhonov(rcond), leaky integrator actuators=(1-leak)*actuators - gain*R.dot(diff). AOtools paper (arXiv:1910.04414).

## Code (1.4): canonical implementation
- **soapy** `reconstruction.py`: `make_dm_iMat` pokes each actuator (actCommands[i]=1), records WFS response → interaction matrix; `MVM.calcCMat`: `control_matrix = numpy.linalg.pinv(interaction_matrix, svdConditioning)`; `reconstruct`: `new = control_matrix.T.dot(slopes)`; `apply_gain` closed loop: `actuator_values += gain * new` (closed) i.e. integrator; open: leaky `gain*new + (1-gain)*old`.
- **HCIPy**: `inverse_tikhonov`, leaky integrator.
- **soapy** `DM.Piezo.makeIMatShapes`: influence function = grid with poked actuator=1, interpolated, mean-subtracted.

## Key facts (sourced, pre-method)
- Kolmogorov: phase structure function D_φ(r)=6.88(r/r0)^(5/3). r0 ∝ [λ⁻²(cosγ)⁻¹∫Cn²dz]⁻³/⁵ = coherence length; aperture seeing ~ D/r0.
- Greenwood frequency f_G=0.426 v/r0; coherence time τ0≈0.314 r0/v. Servo-lag variance σ²_ref=28.4(f_G Δt)^(5/3) (Kolmogorov, zero-order prediction).
- SH WFS: lenslet array, each subaperture spot displacement ∝ average wavefront slope (gradient) over the subaperture, in x and y. Measures slopes, NOT phase.
- Fried geometry: actuators at subaperture corners. Slope from corner displacements: sx=-d1+d2-d3+d4, sy=-d1-d2+d3+d4 (Praus eq 1,2).
- Waffle null mode: w=d1-d2-d3+d4 (push c1,c4 up, c2,c3 down) → sx=sy=0, unsensed. SVD doesn't suppress it because it isn't a natural singular mode of A.
- Forward model s=Ad (or s=Dv). Reconstructor R=A# = pinv(A) via SVD, regularized.
- Maréchal: Strehl S≈exp(-σ²) for small residual phase variance σ² (rad²).
- Error budget: σ²_WFE = σ²_fit + σ²_rec + σ²_bw + ... (uncorrelated).
- Fitting error: DM finite actuator count can't fit high spatial freqs.

## Derivation arc for reasoning.md (discovery order)
1. Pain: turbulence → time-varying phase aberration φ(x,t) over pupil; want flat wavefront for diffraction-limited imaging. r0 sets how bad; τ0 sets how fast.
2. Need to measure φ. Can't measure phase directly (no phase-sensitive detector at optical freqs in real time). SH WFS: lenslets, each gives a focal spot whose displacement = local average gradient (sx,sy) over that subaperture. So I only get SLOPES, not φ.
3. Reconstruct φ (or DM commands) from slopes. Slope = gradient of phase → discrete derivative. Stack all subaperture slopes into s; phase values at grid points φ. Linear: s = Γφ (gradient operator). Or in actuator space directly: poke actuator j by unit, measure slope response = column j of interaction matrix D → s = D c.
4. Invert. D is tall/degenerate, not square. Least squares: minimize ||Dc - s||² → normal equations DᵀD c = Dᵀ s → c = (DᵀD)⁻¹ Dᵀ s. Derive normal equations from setting gradient of ||Dc-s||² to 0.
5. Walls: (a) DᵀD singular/ill-conditioned — piston null (global offset unsensed since only gradients measured), waffle null (Fried geometry checkerboard unsensed). Pseudo-inverse via SVD: D=UΣVᵀ, D#=VΣ#Uᵀ, drop/regularize small singular values. (b) Noise amplification: tiny σ_i blow up 1/σ_i → noise. Tikhonov: (DᵀD+αI)⁻¹Dᵀ, or truncate SVD (rcond). Derive Tikhonov as MAP/ridge.
6. Why regularization picks the trade: 1/σ vs σ/(σ²+α). Filter factor.
7. Closed loop. One-shot R·s isn't enough: DM/WFS nonlinear away from null; calibration drift; sensor is now measuring RESIDUAL in closed loop (null-seeking). So iterate: integrator. Want steady state where residual slopes → 0. Discrete integrator c_{k+1}=c_k - g R s_k. Why integrator: zero-order prediction, rejects DC/low-freq turbulence with infinite DC gain (type-1). 
8. Stability: loop has delay (≥1 frame WFS integrate + compute + DM). Open-loop TF g·z⁻ᵈ/(1-z⁻¹). Integrator pole at z=1 marginally stable; with delay, gain g must be < ~0.5 (2-step delay) for stability. Derive: for pure integrator with one-frame delay, characteristic 1 + g z⁻¹/(1-z⁻¹)=0 → stability bound on g.
9. Leak: replace c_{k+1}=(1-l)c_k - g R s_k. Why: (1-l) pulls unsensed/poorly-sensed modes (piston, waffle) back to zero — they're in the null space of D so R s never corrects them, but noise/numerical error injects them; leak bleeds them off. Moves integrator pole from z=1 to z=1-l (strictly stable). Cost: imperfect DC rejection (small steady-state error). Trade l small (0.01) vs robustness.
10. Gain vs bandwidth: high g → fast, more closed-loop bandwidth, less servo-lag σ²_bw, but more noise propagation and instability; low g → stable but slow, large servo-lag. The g balances σ²_bw (∝ servo-lag, wants high g/bandwidth f_3dB vs f_G) against noise (σ²_rec, wants low g). Greenwood frequency sets the lag scale.
11. Land on code: build interaction matrix by poking, pinv with conditioning, integrator with gain+leak.

## Design decisions → why
- Least squares (not exact solve): D overdetermined (more slope measurements than actuators typically) and noisy → minimize residual norm.
- SVD pseudo-inverse (not direct inverse): D degenerate (piston/waffle nulls), rectangular.
- Regularization/conditioning (truncate small σ or Tikhonov): noise on slopes amplified by 1/σ_i for near-null modes → discard them. rcond threshold.
- Negative sign / feedback: drive residual to zero (null-seeking), correction opposes measured aberration.
- Integrator (not proportional): type-1 system, zero steady-state error on constant aberration, rejects low-freq turbulence (where most power is, since Kolmogorov is red).
- Gain g<~0.5: stability margin given the 1–2 frame loop delay.
- Leak (1-l): suppress null-space/unsensed mode buildup (waffle, piston), strict stability, robustness — at cost of small DC error.
- Poke to build D (not analytic): captures real DM influence functions + WFS response; numerically well-conditioned; calibration of the actual system.
