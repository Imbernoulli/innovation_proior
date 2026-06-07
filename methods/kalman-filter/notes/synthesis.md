# Synthesis — Kalman filter

## SELF-ACCOUNT (backbone of reasoning.md)

**The train story (Grewal & Andrews, *Kalman Filtering: Theory and Practice*, §1.2, "Discovery of the Kalman Filter"):**
> "In late November of 1958, not long after coming to RIAS, Kalman was returning by train to Baltimore from a visit to Princeton. At around 11 PM, the train was halted for about an hour just outside Baltimore. It was late, he was tired, and he had a headache. While he was trapped there on the train for that hour, an idea occurred to him: **Why not apply the notion of state variables to the Wiener filtering problem?** He was too tired to think much more about it that evening, but it marked the beginning of a great exercise to do just that. **He read through Loève's book on probability theory and equated expectation with projection. That proved to be pivotal in the derivation of the Kalman filter.** With the additional assumption of **finite dimensionality**, he was able to derive the Wiener filter as what we now call the Kalman filter. With the change to state-space form, the mathematical background needed for the derivation became much simpler, and the proofs were within the mathematical reach of many undergraduates."

**Kalman's own words (1991 UCLA talk "System Theory: Past and Present"):**
- Why he published in a *mechanical* engineering journal (ASME J. Basic Eng.) not an EE journal: *"When you fear stepping on hallowed ground with entrenched interests, it is best to go sideways."*
- The continuous-time (second) paper was rejected because a referee said one step "cannot possibly be true." (It was true.)

**Kalman's framing (his early-research note, Grewal-Andrews):** From reading Ragazzini (1952) on sampled-data systems, "the idea occurred to him that there is no fundamental difference between continuous and discrete linear systems" — drove his state-space/algebraic program. Discrete-time first because "we can keep the mathematics rigorous and yet elementary" (paper §Notation).

**Bucy's role:** Richard S. Bucy (also at RIAS) "suggested to Kalman that the Wiener–Hopf equation is equivalent to the matrix Riccati equation — if one assumes a finite-dimensional state-space model." (continuous-time, later paper; the discrete recursion (32) is already the Riccati equation in disguise.)

**Schmidt / NASA Ames (Schmidt & McGee, NASA TM-86847, "Discovery of the Kalman Filter as a Practical Tool," 1985; first-hand):**
- Fall 1960: Kalman, unaware of the Apollo work, called and arranged to visit Schmidt at Ames; presented his 1960 paper to the Dynamics Analysis Branch. "the presentation hit a responsive chord" — they had been thinking of filter theory for circumlunar midcourse navigation, and the *sequential* solution features relieved IBM 704 / on-board computer load.
- Wiener filter could not be applied: nonlinearity + irregular discrete measurements; approximations "would either severely restrict the observation system or destroy the inherent accuracy."
- Schmidt's two breakthroughs: (1) **decompose** Kalman's formulation "into a discrete-time update portion and a discrete-time optical measurement update portion" = predict/update split ("Looking back, the decomposition seems almost trivial; at the time, however, it was a major and critical step forward"); (2) **relinearize about the current estimate** rather than the nominal trajectory → the **extended Kalman filter** ("on the average, the estimated state would be closer to the actual... than to the reference").

## PRIMARY SOURCE: Kalman 1960 (refs/kalman-1960.txt)

Four stated limitations of Wiener methods (motivation, §Introduction):
1. optimal filter specified by its **impulse response** — hard to synthesize the filter from that.
2. numerical determination of the impulse response "often quite involved and poorly suited to machine computation; the situation gets rapidly worse with increasing complexity."
3. generalizations (growing-memory, nonstationary) "require new derivations, frequently of considerable difficulty."
4. "The mathematics of the derivations are not transparent. Fundamental assumptions and their consequences tend to be obscured."

Highlights/ingredients:
- (5) **Optimal estimate = conditional expectation** (Theorem 1/1-a); under Gaussian OR (linear estimate + quadratic loss), it is the **orthogonal projection** of x(t1) onto the linear manifold Y(t) = span of observations (Theorem 2). All results use only **first and second order averages**.
- (6) **Bode–Shannon** models for random processes: arbitrary random signal = output of a linear dynamic system driven by **white noise** ("shaping filter"); described in **state / state-transition** form — first-order difference (or differential) equations. x(t+1)=Φx(t)+u(t), y=Mx.
- (7) **State-transition solution** of the Wiener problem: one derivation covers all cases (growing/infinite memory, stationary/nonstationary). Riccati-type nonlinear difference equation for the error covariance P*(t).
- (8) **Duality theorem (Theorem 4):** the Wiener (estimation) problem is the dual of the noise-free optimal regulator (LQR) problem, which Kalman had already solved. Both reduce to (32). Observation↔control, M↔M̂, etc. (Table 1).

Core derivation (Solution of the Wiener Problem, Theorem 3):
- Problem I: x(t+1)=Φ(t+1;t)x(t)+u(t); y(t)=M(t)x(t); u white, zero mean, cov Q(t).
- Solution = Ê[x(t1)|Y(t)] (orthogonal projection). Recurse: Y(t) = Y(t-1) ⊕ Z(t), where Z(t) is generated by the **new information** ỹ(t|t-1) = y(t) − M(t)x*(t|t-1) — the component of y(t) orthogonal to Y(t-1). (Eq. 20; the "innovation".)
- Ê[x(t+1)|Y(t)] = Ê[x(t+1)|Y(t-1)] + Ê[x(t+1)|Z(t)] (orthogonal increments). First term = Φ x*(t|t-1) (the prediction); second term = Δ*(t) ỹ(t|t-1) (the correction). (Eq. 18-19.)
- Gain: Δ*(t) is determined by requiring the residual x(t+1) − Δ*(t)ỹ(t|t-1) to be orthogonal to ỹ(t|t-1):
  **Δ*(t) = Φ(t+1;t) P*(t) M'(t) [M(t) P*(t) M'(t)]⁻¹** (Eq. 25/28).
- Error covariance recursion (Eq. 30, 32):
  P*(t+1) = Φ(t+1;t){P*(t) − P*(t)M'[M P* M']⁻¹ M P*}Φ'(t+1;t) + Q(t). "Plays a role analogous to that of the Wiener-Hopf equation." Initialized with P*(t0)=Ex(t0)x'(t0).
- Φ*(t+1;t) = Φ(t+1;t) − Δ*(t)M(t) is the transition matrix of both the estimator and the error.
- Loss = trace P*(t).

## CANONICAL IMPLEMENTATION: FilterPy (code/filterpy_kalman_filter.py)
predict:  x = Fx + Bu ;  P = F P F' + Q
update:   y = z − Hx ;  S = H P H' + R ;  K = P H' S⁻¹ ;  x = x + Ky ;  P = (I−KH)P(I−KH)' + K R K'  (Joseph form)
Mapping to paper: F=Φ, H=M, K = predict-step gain folded into the contemporary "filtered" form x(t|t) (the modern textbook splits Kalman's one-step predictor x*(t+1|t) into a measurement-update producing x(t|t) then a time-update Φ x(t|t) — exactly Schmidt's decomposition). K_modern = P⁻ H'(H P⁻ H' + R)⁻¹; Kalman's Δ* = Φ K_modern.

## ANTECEDENTS / BASELINES (context.md)
- **Wiener filter (1949; Wiener-Hopf integral eq.; spectral factorization):** optimal LTI filter for *stationary* signal+noise with known spectra; minimizes MSE; defined in continuous time, frequency domain. Output specified as an impulse response via spectral factorization of rational spectra. Limits: stationary only; infinite/whole-past memory; recompute from scratch per new setting; impulse response hard to realize and hard to compute on a machine (Kalman's four points).
- **Kolmogorov (1941):** discrete-time stationary prediction, same theory.
- **Bode–Shannon (1950):** "A simplified derivation of linear least-square smoothing and prediction theory" — the whitening / shaping-filter idea: represent a signal as white noise through a linear system; whiten, then the optimal predictor is trivial. Gives Kalman the random-process *model* but still stationary/frequency-flavored.
- **Zadeh–Ragazzini (1950):** finite-memory Wiener case; Kalman's discrete sampled-data roots are Ragazzini's.
- **Recursive least squares / Gauss (1795/1809):** sequential update of a least-squares estimate as data arrive, with a gain on the residual and a recursively updated information/covariance matrix — the deterministic ancestor of predict/update. (Gauss explicitly: update old estimate when a new observation arrives without redoing all the arithmetic.)
- **State-space / state-transition (Kalman's own LQR, 1960 "Contributions to the theory of optimal control"; Laning–Battin 1956 used state space for time-varying systems):** dx/dt=Fx+Du, y=Mx; the regulator problem solved via Riccati. This is the machinery Kalman drags into estimation.

## DESIGN-DECISION → WHY
- Discrete time first → keeps the math elementary/rigorous; matches sampled measurements; matches the digital computer.
- State variables / time domain (not spectra) → handles nonstationary & time-varying automatically; finite-dimensional carry-state; one derivation for all cases (kills Wiener limitations 1-4).
- Conditional expectation = projection → optimal estimate is *linear* projection onto observation span; needs only 1st/2nd moments; Gaussian makes projection = full conditional mean.
- Innovation ỹ = y − Hx⁻ as the new coordinate → makes Y(t) = Y(t-1) ⊕ Z(t) orthogonal, so the estimate updates by *adding* a term instead of re-projecting → recursion (no growing history). This is THE move.
- Gain by orthogonality (residual ⟂ innovation) → minimizes error covariance; balances model-trust (P⁻) vs measurement-trust (R): K = P⁻H'(HP⁻H'+R)⁻¹.
- Carry P (covariance) and propagate it (Riccati) → the filter is self-tuning; gain computed offline from model alone; trace P = expected loss = built-in accuracy gauge.
- Joseph form for P update → numerically stable, stays symmetric PSD, valid for non-optimal K (the spaceborne-computer fix lineage).
- Duality with regulator → reuse the already-solved LQR Riccati solver; theoretical leverage (stability via observability/controllability).
