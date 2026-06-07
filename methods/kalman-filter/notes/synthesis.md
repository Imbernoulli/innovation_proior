# Synthesis — Kalman filter

## Pain point (research question)
Estimate the state x of a linear dynamic system from a stream of noisy measurements, *recursively*
(online, one measurement at a time), with no requirement of stationarity or a frequency-domain detour.

## The prior art and exactly where each falls short
- **Wiener filter (Wiener 1949) / Wiener–Hopf integral equation.** Optimal linear MMSE filter for
  stationary signal+noise, specified by its impulse response, found via spectral factorization in the
  frequency domain. Limitations (stated in Kalman's own Introduction): (1) optimal filter specified by
  impulse response — hard to synthesize a physical filter from it; (2) numerical determination of the
  optimal impulse response is involved, poorly suited to machine computation, worse as complexity grows;
  (3) generalizations (growing-memory / nonstationary) require new, hard derivations each time;
  (4) the math obscures the fundamental assumptions. Also: it is a *batch*, frequency-domain method —
  needs the whole record and stationary, rational spectra.
- **Bode–Shannon (1950) "A Simplified Derivation of Linear Least-Squares Smoothing and Prediction."**
  Represents a random signal as the output of a linear shaping filter driven by white noise — the
  "whitening" idea. This is the seed of the *state-space / shaping-filter* model: arbitrary second-order
  statistics ⇒ linear system excited by white noise.
- **Zadeh–Ragazzini (1950), Booton (1952), Blum (1958).** Finite-memory / nonstationary / growing-memory
  extensions of Wiener. Blum gave recursion formulas for growing-memory filters — the only prior explicit
  recursion Kalman cites — but "much more complicated than ours."
- **Gauss least squares / recursive least squares.** Minimize squared residuals; the recursive form
  updates the estimate as each new observation arrives without re-inverting the whole normal-equations
  system. The ancestor of the recursive, online flavor.
- **Orthogonality principle (Doob, Loève, Pugachev).** The MMSE estimate is the orthogonal projection of
  the unknown onto the span of the observations; the error is orthogonal to (uncorrelated with) every
  observation. Well known in probability theory (Doob pp. 75–78, 148–155; Loève pp. 455–464) but
  "not yet used extensively in engineering."
- **Gaussian conditioning.** For jointly Gaussian variables, the conditional expectation is linear in the
  conditioning variables and equals the orthogonal projection; conditional distributions stay Gaussian
  (Theorem 5 in the appendix: linear functions of a Gaussian process are Gaussian; orthogonal Gaussian
  variables are independent; any second-order process is matched by a unique Gaussian process).

## The state-space model (the reframing that unlocks everything)
x(t+1) = Φ(t+1;t) x(t) + u(t),  y(t) = M(t) x(t),  with {u(t)} independent Gaussian, zero mean,
E u(t)u'(s)=0 (t≠s), E u(t)u'(t)=Q(t). (Kalman's notation: Φ=transition, M=measurement, Q=excitation cov.
Modern notation: F, H, Q, plus explicit measurement noise R and v.) Carry a Gaussian belief on x:
mean x̂, covariance P. "Guess the state correctly" ⇒ a single recursion covers filtering/prediction/
smoothing, stationary or not, growing or infinite memory. Difficulty (3),(4) vanish.

## The derivation (Kalman's orthogonal-projection route + the modern predict/update split)
Loss L(ε)=ε² ⇒ optimal estimate = conditional expectation (Theorem 1-a) = orthogonal projection onto
Y(t)=span of observations (Theorem 2). For Gaussian, projection = conditional expectation exactly; for
non-Gaussian, projection = best *linear* estimator. So x*(t|t) = Ê[x(t)|Y(t)].

Innovation: decompose Y(t) = Y(t-1) ⊕ Z(t), where Z(t) is spanned by ỹ(t|t-1) = y(t) − M(t)x*(t|t-1),
the part of the new measurement orthogonal to the past — the *innovation*. Projection is additive over
orthogonal subspaces:
  x*(t+1|t) = Φ x*(t|t-1) + Ê[x(t+1)|Z(t)],  and the Z-term = ∆*(t) ỹ(t|t-1) for some matrix ∆*.
Find ∆* by orthogonality: the error x(t+1) − ∆* ỹ must be ⊥ ỹ, i.e.
  E[x(t+1) − ∆* ỹ] ỹ' = 0 ⇒ ∆* M P M' = Φ P M' ⇒ ∆*(t) = Φ P*(t) M' [M P*(t) M']⁻¹.   (eq 25/28)
Error transition matrix Φ* = Φ − ∆* M (eq 22/29). Covariance recursion (eq 24/30):
  P*(t+1) = Φ* P*(t) Φ' + Q(t),  and eliminating ∆*,Φ* gives the nonlinear (Riccati) difference eq 32:
  P*(t+1) = Φ{ P* − P* M'[M P* M']⁻¹ M P* } Φ' + Q   (note: transcriber flags eq 32 has a P*M vs MP*
  ordering typo in the original; correct order is M P*). trace P*(t) = expected quadratic loss (eq 27).
Initialize P*(t0)=E x(t0)x'(t0); needs P*(t0) and Q(t).

**Modern predict/update split (Welch & Bishop), with explicit measurement noise R:**
  z_k = H x_k + v_k, v ~ N(0,R), w ~ N(0,Q).
  PREDICT (time update):  x̂⁻ = F x̂,  P⁻ = F P F' + Q.
  UPDATE (measurement update):
    innovation/residual  y = z − H x̂⁻
    innovation cov       S = H P⁻ H' + R
    gain                 K = P⁻ H' S⁻¹ = P⁻ H' (H P⁻ H' + R)⁻¹
    state                x̂ = x̂⁻ + K y
    cov                  P = (I − K H) P⁻   [Joseph form: (I−KH)P⁻(I−KH)' + K R K' for stability]
Limits: R→0 ⇒ K→H⁻¹ (trust measurement); P⁻→0 ⇒ K→0 (trust prediction). Gain weights prior vs
measurement by inverse covariance.

## Gain derivation by minimizing trace P (the "trust" route, Welch & Bishop)
e_k = x_k − x̂_k = (I−KH)e⁻_k − K v_k. With e⁻ ⊥ v:
  P = (I−KH)P⁻(I−KH)' + K R K'.
  trace P = tr P⁻ − 2 tr(K H P⁻) + tr(K(H P⁻ H'+R)K').   (use tr(KHP⁻)=tr((KHP⁻)') since P⁻ symmetric)
  d/dK tr P = −2 (H P⁻)' + 2 K (H P⁻ H' + R) = 0  ⇒  K = P⁻ H'(H P⁻ H'+R)⁻¹.   ✓ same as orthogonality.
Substituting K back, (I−KH)P⁻(I−KH)'+KRK' collapses to P=(I−KH)P⁻.

## Scalar sanity check (product of Gaussians — Labbe)
1-D, H=1: prior N(x̂⁻,P⁻), likelihood N(z,R). Posterior mean (P⁻ z + R x̂⁻)/(P⁻+R) =
x̂⁻ + [P⁻/(P⁻+R)](z−x̂⁻), so K=P⁻/(P⁻+R); posterior var P⁻R/(P⁻+R) = (1−K)P⁻. Inverse-variance
(precision) weighting. Matrix gain is the multivariate version of exactly this.

## Why optimal
Linear-Gaussian: projection = conditional expectation, so the Kalman estimate is THE MMSE estimate
(no nonlinear estimator beats it). Non-Gaussian, same first/second moments: it is the best *linear*
(linear-MMSE) estimator, and by Theorem 2(B) optimal among linear estimators under squared loss.
Nonlinear estimation can only beat it if the process is nongaussian AND you use ≥3rd-order statistics.

## Duality (Theorem 4) — note, do NOT re-derive LQR
Problem I (estimation) is the dual of the noise-free optimal regulator (Problem II / LQR): swap
X(t0+τ) ↔ X̂'(T−τ). Both reduce to the same Riccati equation. Mention as the structural twin; LQR lives
in a separate method.

## Design decisions → why
- Carry only mean+cov (first two moments): Gaussian stays Gaussian under linear map + additive Gaussian
  noise (Thm 5A), so two moments are a *sufficient* statistic — no need to propagate a full density.
- Project P by FPF': covariance of a linear image F x is F P F'. Add Q: independent process noise adds
  its covariance; uncertainty must *grow* under prediction.
- Innovation y=z−Hx̂⁻ not z: the predictable part Hx̂⁻ carries no new info; only the orthogonal residual
  updates the estimate. Whitening / Bode–Shannon.
- Gain via inverse innovation covariance S⁻¹: precision weighting of the two information sources.
- P=(I−KH)P⁻: uncertainty must *shrink* after a measurement; Joseph form for numerical stability and
  validity at non-optimal K.
- R measured offline (can sample the sensor); Q hard (can't observe the process) — tune, or inject
  uncertainty.

## Canonical code (filterpy, rlabbe)
predict: x=Fx(+Bu); P = FPF' + Q. update: y=z−Hx; S=HPH'+R; K=PH'S⁻¹; x=x+Ky;
P=(I−KH)P(I−KH)'+KRK' (Joseph). Matches synthesis exactly.

## Unsourced facts
None — every equation traced to Kalman 1960 (primary) or Welch&Bishop / Labbe (explainers).
