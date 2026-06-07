# Sources retrieved this run (quadruped-gait CPG)

## PRIMARY
1. **Righetti & Ijspeert (2008), "Pattern generators with sensory feedback for the control of
   quadruped locomotion", ICRA 2008, pp. 819-824.** — FULL TEXT obtained.
   `refs/righetti08-quadruped.pdf` (6 pp, PDF v1.4) from EPFL infoscience record 130740.
   Extracted to `notes/righetti08-text.txt`. This is the concrete CPG-quadruped paper:
   modified Hopf oscillator with phase-dependent frequency (eqs 1-3), symmetric coupled-cell
   network → walk/trot/pace/bound (eqs 4-5, Fig 2 matrices), touch-sensor feedback (eqs 6-9).
   Load-bearing pre-method biological fact it cites: stance-duration ↔ speed linear relation in
   mammals (Frigon & Rossignol 2006, ref [6]); swing ~constant.

2. **Ijspeert (2008), "Central pattern generators for locomotion control in animals and robots:
   a review", Neural Networks 21(4):642-653.** — FULL TEXT *not obtained*. EPFL infoscience
   record 125126 has the PDF bundle ACCESS-RESTRICTED (only LICENSE bundle public);
   Wayback never captured the bytes (all snapshots are 302/301 redirects); CMU/UW course
   mirrors are dead (Google-Sites redirect / 404); ResearchGate/SemanticScholar are
   abstract-only walls. GAP FLAGGED. Its load-bearing robotics content — the
   amplitude-controlled phase oscillator (ACPO, Kuramoto-derived) and the salamander
   limb/body CPG hierarchy — is reproduced faithfully in source #3 and #5 below and matches
   the equations I use; the quadruped half (gaits, oscillator design, feedback) is fully covered
   by primary #1. I verified the ACPO equation form against the Kuramoto definition and the
   canonical code (#4).

## BACKGROUND (lineage)
- Kuramoto coupled-phase-oscillator model of synchrony (φ̇_i = ω_i + Σ_j K_ij sin(φ_j−φ_i)).
- Hopf bifurcation / Andronov–Hopf normal form → structurally stable harmonic limit cycle.
- Golubitsky & Stewart symmetric coupled-cell theory; Golubitsky, Stewart, Buono & Collins
  (1998) "A modular network for legged locomotion", Physica D 115:56-72 (Righetti ref [10]).
- Biological CPGs in the spinal cord; half-centre (Brown); flexor/extensor anti-phase;
  stance-duration↔speed (Frigon & Rossignol 2006).
- Quadruped gait phase relationships: walk (4-beat, ~0,1/2,1/4,3/4), trot (diagonal in phase,
  ±1/2), pace (ipsilateral in phase), bound (fore in phase, hind in phase, ±1/2), gallop.
  (sourced: eLife 29495; arXiv gait surveys.)

## THIRD-PARTY EXPLAINER
3. **"Central Pattern Generators for the control of robotic systems" (arXiv 1509.02417), 12 pp.**
   `refs/cpg-review-1509.02417.pdf`, extracted to `notes/cpg-review-text.txt`,
   `notes/cpg-review-p79.txt`. Covers half-centres, leaky-integrator, Kuramoto coupled
   oscillators, salamander hierarchy, sensory feedback, the parameter-tuning pain point, and
   reproduces Ijspeert/Crespi ACPO equations (1)-(4) verbatim with the Euler-integration code.

## CANONICAL CODE
4. **EPFL MICRO-507 "Legged Robots" / CPG-RL (Bellegarda & Ijspeert) HopfNetwork** — the
   modern runnable quadruped descendant of Righetti08, polar-coordinate Hopf network in
   PyBullet. `code/hopf_network_epfl.py` (completed; all gait matrices + foot mapping + full
   sim loop with IK, joint PD, Cartesian PD), `code/hopf_network_amirrazmjoo.py` (skeleton
   header crediting CPG-RL Bellegarda & Ijspeert), `code/run_cpg_epfl.py`. Header of the file
   itself cites Righetti & Ijspeert 2008. r is amplitude, θ phase, ω switches swing/stance,
   Kuramoto coupling with phase-offset matrix Φ, foot xz mapping (eqs 8-9).

## NOTE on empirical discipline
- Righetti08's Fig-5 speed-vs-(ωswing,ωstance) maps and "speed correlates with stance
  duration" are the PROPOSED METHOD's own results → EXCLUDED from context/reasoning.
- The *biological* stance↔speed fact (mammals, Frigon & Rossignol) IS a pre-method fact → context.
- No locomotion-speed numbers of the method will be fabricated or quoted.
