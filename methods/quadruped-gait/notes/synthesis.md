# Synthesis — quadruped-gait CPG (Righetti & Ijspeert 2008 + Ijspeert review machinery)

## Pain point
A quadruped gait = the four legs executing the SAME periodic stride but with fixed inter-leg
PHASE offsets (trot = diagonal pairs in phase, ½ out of phase with the other diagonal; bound =
front pair in phase, hind pair in phase, ½ apart; pace = ipsilateral in phase; walk = a 4-beat
sequence ~¼ apart). Speed needs different gaits (slow→walk statically stable; fast→trot/bound
dynamically stable). Hand-scripting joint trajectories (open-loop clocked splines) is brittle:
(a) you must hand-author and re-tune a full trajectory per gait per speed; (b) any
discrete event (re-tuning ω, perturbation, terrain bump) injects a discontinuity / loses
phase, and there is no restoring mechanism — a position-replay has no notion of its own phase,
so a kicked leg just resumes its clock wrong. Want: smooth, retunable rhythm with a built-in
restoring force on both shape and inter-leg timing, reducible to a few interpretable knobs.

## The chain to re-derive (discovery order)
1. A gait is periodic per-leg + phase locking across legs → need a *clock* per leg that is a
   stable oscillator (forgets transients) and a coupling that pins relative phases.
2. Limit cycle from a dynamical system: Andronov–Hopf normal form. In polar form
   ṙ = α(μ − r²)r, θ̇ = ω. r → √μ (stable radius, structurally stable harmonic cycle); θ
   advances at ω. Amplitude μ and frequency ω are explicit, independent knobs. The cycle's
   SHAPE is invariant to ω → retune speed without re-authoring the waveform.
3. Output mapping: one oscillator per leg drives a foot trajectory. x_i = −d·r_i·cos θ_i sets
   fore/aft (step length d); z_i lifts the foot only in swing (sin θ_i > 0 → clearance) and
   keeps/penetrates ground in stance (sin θ_i < 0). So θ ∈ (0,π) = swing, (π,2π) = stance.
4. Gait = relative phases. Couple the phases Kuramoto-style:
   θ̇_i = ω + Σ_j r_j K sin(θ_j − θ_i − Φ_ij). The sin(Δ−Φ) term has stable fixed point at
   θ_j − θ_i = Φ_ij → the offset matrix Φ *is* the gait. Trot Φ: legs FR,FL,RR,RL with
   {0,π,π,0} pattern (diagonals in phase). Change Φ → change gait; nothing else changes.
   This is exactly the Bellegarda–Ijspeert CPG-RL polar HopfNetwork (canonical code).
5. (Righetti's specific oscillator, primary #1) Biology: stance duration controls speed
   (linear inverse-stance↔speed in mammals, Frigon & Rossignol), swing ≈ constant. So you want
   to set swing and stance durations INDEPENDENTLY. A plain Hopf has one ω. Fix: make ω
   phase-dependent — fast ω in one half, slow ω in the other:
   ω = ωstance/(e^{−b y}+1) + ωswing/(e^{b y}+1), a smooth sigmoid switch on the sign of y.
   Now swing freq and stance freq are separate knobs; speed ← ωstance, stability ← ωswing.
   Cartesian-polar form: ẋ = α(μ−r²)x − ωy, ẏ = β(μ−r²)y + ωx; using different α,β (e.g.
   α=5, β=50) gives faster convergence on y so x (the control policy) stays smooth.
6. Network via symmetric coupled-cell theory (Golubitsky–Stewart H/K theorem): pick the
   coupling matrix k_ij so the desired symmetric periodic solution (the gait) is the stable
   one; add k_ij y_j to the ẏ equation (eqs 4-5). Coupling matrices in Fig 2 give trot, pace,
   bound, walk; type of gait set by matrix, independent of swing/stance durations.
7. Feedback (eqs 6-9): add control u_i to ẏ_i. "Stop before transition": u = −ω x_i − Σ k_ij y_j
   makes ẋ=ẏ=0 at y=0 (fixed point at x=±√μ), so the oscillator waits at top/bottom until the
   foot sensor allows the phase switch. "Fast transition": u = −sign(y)·F drives y→0 quickly
   (delay ≈ y(t)/F). Couples CPG ↔ mechanics; robust to bad params/terrain.
8. The few interpretable params (μ, ωswing, ωstance, Φ/coupling, ground_clearance, step_len,
   duty via swing/stance ratio) → low-dim search → black-box/evolutionary tuning in sim to
   maximize forward speed. (Adaptive frequency oscillators, Buchli–Iida–Ijspeert, as the
   feedback route to auto-tune ω.)

## Design-decision → why
- **Oscillator not scripted spline** → limit cycle = built-in restoring force; transient
  perturbations forgotten; phase is a state, so events/retuning don't break continuity.
- **Hopf normal form specifically** → harmonic, structurally stable cycle whose shape is
  *independent of ω*; lets you retune frequency/amplitude without re-authoring the waveform.
  (Van der Pol works too but its cycle shape changes with parameters.)
- **Polar (r,θ) state** → amplitude and phase are *directly* the meaningful quantities (step
  length and where-in-stride); coupling acts on θ cleanly.
- **Kuramoto coupling sin(θ_j−θ_i−Φ_ij)** → stable fixed point at exactly the prescribed
  offset Φ_ij; r_j weight so a dead/small oscillator doesn't drag others; robust phase-locking
  that *re-locks* after a kick. Gait ⇔ Φ matrix → one knob switches gaits.
- **Phase-dependent ω (sigmoid)** → independent swing/stance durations, matching the biology
  (stance↔speed, swing≈const); sigmoid keeps ω(y) smooth so no jerk at the switch.
- **α≠β (faster y convergence)** → x is the actuated policy; want its derivative bounded/smooth,
  so converge hard on y, gently on x.
- **Feedback on y not x** → y's sign = swing/stance indicator; modulating y leaves x (the joint
  command) smooth. Stop-control = exact fixed point at y=0; fast-control = bounded-delay reset.
- **Coupling via symmetric-cell theory** → design network from symmetry/algebra independent of
  cell dynamics; guarantee only the desired gait is the stable symmetric solution; scalable.
- **Output mapping x=−d r cosθ, z swing-lift only** → minimal foot trajectory; step length d
  and clearance are explicit; knee/elbow are functions of the hip oscillator → 1 oscillator/leg.
- **Few params → black-box tuning** → the whole point of the dimensionality reduction; speed is
  a smooth function of (ωstance, ωswing, μ, step_len), tunable by evolutionary/grid search in sim.

## Code to land on (canonical = EPFL/CPG-RL HopfNetwork, PyBullet)
polar Hopf: r_dot = α(μ−r²)r; θ_dot = ω(swing/stance by sign of sinθ) + Σ_j r_j K sin(θ_j−θ_i−Φ_ij);
Euler integrate; mod 2π; map to foot x=−d r cosθ, z = −h + clearance·sinθ (swing) /
penetration·sinθ (stance); IK → joint targets → joint PD (+ Cartesian PD) → torques → step sim.
PHI matrices per gait. run loop measures base velocity (forward speed objective). I will also
show the Righetti Cartesian-form oscillator (ẋ,ẏ with phase-dependent ω) as the variant that
gives independent swing/stance, since that is primary #1's specific contribution.

## In-frame reminders
- Never name the target papers / "review" / authors-as-citation. May name "Hopf oscillator",
  "Kuramoto model", Golubitsky–Stewart, Frigon & Rossignol (prior-art ancestors — keep).
- No method speed numbers. Biological stance↔speed fact is fine (context).
