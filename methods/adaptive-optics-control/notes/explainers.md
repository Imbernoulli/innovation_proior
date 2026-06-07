# Explainer / source capture

## HCIPy tutorial (docs.hcipy.org pyramid WFS) — closed-loop AO
- Interaction matrix built by poking each DM mode +/- probe_amp; slope += s*(image-image_ref)/(2*probe_amp).
- Forward model S = A phi. Inverse: phi = (A^T A)^-1 A^T S (normal equations).
- Implemented as regularized pseudo-inverse: inverse_tikhonov(transformation_matrix, rcond=1e-3).
- Leaky integrator closed loop: actuators = (1-leakage)*actuators - gain*R.dot(diff_image). leakage=0.01, gain=0.5, ~1 kHz.

## soapy reconstruction.py — canonical AO reconstructor
- make_dm_iMat: actCommands[i]=1 (poke), record -1*wfs.frame -> interaction matrix column.
- MVM.calcCMat: control_matrix = numpy.linalg.pinv(interaction_matrix, svdConditioning).
- reconstruct: new_actuator_values = control_matrix.T.dot(wfs_measurements).
- apply_gain (closed loop): actuator_values += gain * new_actuator_values  (integrator).
  open loop: gain*new + (1-gain)*old (leaky form).
- DM.Piezo.makeIMatShapes: influence function = grid, poked actuator=1, interpolated up, mean-subtracted.

## Praus / MZA "Waffle Constrained Reconstructor for Fried Geometry" — reproduces Fried equations
- Fried geometry: actuators at sub-aperture corners c1..c4.
- Slope influence: sx = -d1+d2-d3+d4 ; sy = -d1-d2+d3+d4.
- Forward map s = A d (over-constrained MIMO). Reconstructor d_est = A# s (pseudo-inverse via SVD).
- Waffle null mode w = d1-d2-d3+d4: pushing c1,c4 up and c2,c3 down -> sx=sy=0, unsensed; not a natural SVD singular mode so plain SVD conditioning doesn't suppress it.

## Davies & Kasper "Adaptive Optics for Astronomy" (arXiv:1201.5741)
- Kolmogorov; Fried r0 ∝ [λ^-2 (cosγ)^-1 ∫Cn^2 dz]^-3/5; (D/r0)^2 degrees of freedom.
- SH WFS: lenslet array, spot displacement ∝ average wavefront slope over sub-aperture.
- Reconstruction: v = R s, R = pseudo-inverse of interaction matrix D (s = D v), regularized to limit noise amplification.
- Error budget σ²_WFE = σ²_fit + σ²_rec + σ²_bw (uncorrelated). Maréchal: S ≈ exp(-σ²).

## Doelman "Minimum of the time-delay wavefront error in AO" (arXiv:1906.03128)
- Zero-order prediction (the integrator): feed back latest measured phase with opposite sign; ε_ref(t)=φ(t+Δt)-φ(t).
- σ²_ref = 28.4 (f_G Δt)^(5/3) for Kolmogorov.
- Greenwood frequency f_G = 0.426 v/r0 (eq context); single-layer f_G expression given.
- Many closed-loop AO systems implement zero-order prediction as a discrete-time integrator; integrator may not reach the zero-order minimum due to closed-loop stability constraint.

## Source-gap note
- Fried 1977 original (JOSA 67:370) is paywalled on Optica and OCR-hostile. The exact Fried-geometry slope-influence equations and the least-squares s=Ad -> d=A#s formulation are taken from the Praus/MZA reproduction (faithful secondary), cross-checked against the Davies & Kasper review's s=Dv, R=pinv treatment. Gap flagged.
