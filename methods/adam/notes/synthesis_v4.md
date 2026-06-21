# Synthesis (V4) — composing the deliverables from notes, NOTES-FIRST

This is the working notes file the V4 deliverables are composed FROM (per the skill's 1.5).
It reuses the verified derivations/code from notes/research.md and results_v3, and adds the
two V4-specific things: (a) a complete design-decision -> why table with rejected
alternatives + failure modes, and (b) the load-bearing-ancestor write-ups in one place, so
reasoning.md becomes transcription-with-voice rather than figuring-it-out-while-writing.

Method = Adam (and the L-infinity sibling, AdaMax). In-frame ban: context.md must not name
"Adam"/"AdaMax"; no file may treat the source paper as a published artifact. Prior-art
ancestor citations (AdaGrad/RMSProp/AdaDelta/Nesterov/Zinkevich/Amari) stay.

---

## The questions the trace must answer (filled in)

- **Pain point.** Train large models (millions of params) on objectives seen only through
  noise — minibatch subsampling AND deliberately injected noise (dropout). Need first-order +
  ~linear memory (no Hessian / theta x theta matrix). Want per-parameter step sizes, robust to
  noisy/sparse/non-stationary gradients, with a step-size knob that is interpretable and needs
  little tuning. SGD+momentum gives none of the per-parameter scaling; adaptive methods each
  have one fatal wart.
- **Tools on the table & where each falls short.** Momentum (first-moment EMA; no
  per-parameter scale). AdaGrad (per-parameter scale via growing SUM of g^2; LR decays to zero
  on non-stationary problems). RMSProp (EMA of g^2 fixes the decay-to-zero; but no bias
  correction => blows up for beta2 near 1, and momentum sits on the rescaled gradient, no clean
  moment story). AdaDelta (unit-matching heuristic, same no-debias gap). SFO/quasi-Newton
  (memory grows with #minibatch partitions; assumes deterministic subfunctions, breaks under
  dropout).
- **The first-principles object.** Out of a stream of noisy gradient vectors with ~linear
  memory, manufacture a good per-coordinate step. Central difficulty: each g_t is one noisy
  sample of the true gradient; we must estimate BOTH the mean direction and the per-coordinate
  spread, cheaply, and combine them — under non-stationarity.
- **Chain of approximations theory -> practice.** Want first AND second moment -> keep two EMAs
  -> zero-init biases both toward zero -> unroll + take expectation -> bias is exactly the
  multiplicative (1 - beta^t) -> divide it out (bias correction) -> step along m_hat/sqrt(v_hat)
  -> ratio is scale-invariant => alpha is a trust region & an SNR auto-annealer -> regret proof
  (OCO) -> L-infinity limit gives AdaMax.
- **Which prior method falls out as a special case.** RMSProp = beta1=0 (still with the bias
  correction RMSProp lacks). AdaGrad = beta1=0 + infinitesimal (1-beta2) + annealed alpha*t^-1/2
  (then v_hat_t -> t^-1 sum g^2). Correspondences only hold WITH bias correction.

---

## Load-bearing ancestor write-ups (verified vs primary; from research.md)

1. **SGD with momentum / Nesterov (Sutskever, Martens, Dahl & Hinton, ICML 2013).**
   v_t = mu*v_{t-1} + g_t ; theta -= alpha*v_t. v_t is an (un-normalized) first-moment EMA:
   averages minibatch noise, builds speed downhill, damps ravine oscillation. NAG uses a
   look-ahead gradient at theta+mu*v, more stable at high momentum. Their result: well-init nets
   with momentum *scheduled* (ramp up, then reduce near the end) rival Hessian-free. Two
   takeaways: a smoothed first moment is genuinely part of the answer; the momentum coefficient
   wants to DECAY late. **Gap:** one global learning rate, no per-parameter scaling; average is
   over the raw gradient so coordinates with wildly different magnitudes step at the same scale.

2. **AdaGrad (Duchi, Hazan & Singer, JMLR 2011).**
   theta_{t+1,i} = theta_{t,i} - alpha*g_{t,i}/sqrt(sum_{s<=t} g_{s,i}^2). Per-parameter rate
   shrunk by accumulated squared gradient: rare features keep a small denominator and take big
   steps when they fire — ideal for sparse high-dim NLP/BoW. Regret O(sqrt(T)); sparse adaptive
   bound ~O(log d * sqrt(T)) vs O(sqrt(dT)). Brings the OCO/regret frame. **Gap:** denominator is
   a monotonically growing SUM that never forgets; sqrt(sum g^2) climbs, effective LR -> 0,
   learning stalls — fatal on non-stationary deep objectives.

3. **RMSProp (Tieleman & Hinton, Coursera 6.5, 2012; momentum variant Graves 2013).**
   v_t = beta2*v_{t-1} + (1-beta2)*g_t^2 ; theta -= alpha*g_t/(sqrt(v_t)+eps). EMA forgets old
   gradients -> windowed (not cumulative) second-moment estimate -> works under non-stationarity.
   Graves adds momentum on the already-rescaled gradient. **Gaps:** (a) no bias correction — with
   v_0=0 and beta2 near 1 (which sparse second-moment estimation demands), early v_t is far too
   small -> huge early steps / divergence; (b) momentum on the rescaled gradient, not a clean
   separately-maintained first moment — no "estimate both moments and combine" story.

4. **AdaDelta (Zeiler, 2012).** Same EMA-of-g^2 denominator plus an EMA of squared parameter
   updates in the numerator for unit-consistency (no global LR). **Gap:** still no
   moment-estimation/bias-correction view; unit-matching is a heuristic and shares the no-debias
   gap.

5. **Online convex optimization / regret (Zinkevich, ICML 2003).**
   R(T) = sum_t [f_t(theta_t) - f_t(theta*)]. OGD with eta_t = t^-1/2 gets O(sqrt(T)), so
   R(T)/T -> 0. Entry point: convexity hyperplane bound f_t(theta_t)-f_t(theta*) <=
   g_t^T(theta_t-theta*). This is the lever the new method's regret proof pulls.

6. **Natural gradient / Fisher (Amari 1998; Pascanu & Bengio 2013).** Second-moment estimate ~
   diagonal of Fisher; dividing by sqrt of it is a cheap, conservative, diagonal cousin of NGD
   (sqrt of inverse diagonal Fisher vs full inverse Fisher). SFO (Roux & Fitzgibbon 2010;
   Sohl-Dickstein 2014): quasi-Newton on minibatches, memory grows with #partitions, assumes
   deterministic subfunctions => infeasible on GPU + breaks under dropout. Motivates a cheap
   first-order method.

---

## DESIGN-DECISION -> WHY table (with rejected alternatives + their failure mode)

| # | Design decision | Why this | Rejected alternative -> failure mode |
|---|---|---|---|
| 1 | Stay strictly first-order, ~linear memory | Models have millions of params; can't store/invert theta x theta or a Hessian on a GPU | Second-order / quasi-Newton (SFO): memory grows with #minibatch partitions; assumes deterministic subfunctions -> breaks under dropout |
| 2 | Keep an EMA of the gradient (first moment m) | Smoothed mean direction: averages minibatch noise, builds downhill speed, damps ravine oscillation; this is momentum's win | No smoothing (raw g): noisy single-sample direction yanks the step around |
| 3 | Keep an EMA of the squared gradient (second raw moment v) for per-parameter scaling | Per-coordinate magnitude estimate -> per-parameter step; division by sqrt(v) is a cheap diagonal sqrt-Fisher preconditioner | Single global LR (SGD/momentum): structurally wrong — conv vs dense layers have very different gradient scales |
| 4 | Use an EMA (forgetting) for v, NOT a growing sum | EMA tracks the RECENT gradient scale, stays O(1), adapts to non-stationarity | AdaGrad's cumulative sum: sqrt(sum g^2) climbs -> effective LR -> 0 -> learning stalls on non-stationary problems |
| 5 | Write both EMAs in normalized (1-beta) form, m=beta*m+(1-beta)*g | Weights sum to ~1 -> m is a true weighted average with the units of a gradient; keeps the ratio m/sqrt(v) clean and not contaminated by a stray 1/(1-beta) | Un-normalized momentum form v=mu*v+g: carries an accumulation factor that pollutes the ratio |
| 6 | Step along the ratio m/sqrt(v) (combine both moments) | Get per-parameter scale AND smoothed direction at once; moment view shows they don't compete (1st vs 2nd moment) | Momentum-on-rescaled-gradient (RMSProp+Graves): no clean separable moment story; couples the two estimates |
| 7 | BIAS-CORRECT both moments by /(1-beta^t) | Unrolling the zero-init EMA + expectation gives E[v_t]=E[g^2](1-beta_2^t)+zeta exactly; the (1-beta^t) factor is a structural early-training bias toward zero | No correction (RMSProp/AdaDelta): with beta2 near 1, v_1 ~= (1-beta2)g^2 is ~1000x too small -> denominator tiny -> enormous divergent first step |
| 8 | Correct BOTH moments, not just the denominator | At t=1 correcting both restores m_hat/sqrt(v_hat) ~= sign(g) (honest unit step) | Correct only v: ratio at t=1 becomes 0.1*g/|g| = 0.1*sign(g) — now 10x too SMALL; trades one mismatch for another |
| 9 | Denominator form sqrt(v_hat) + eps (additive eps OUTSIDE the sqrt) | Floors divide-by-zero on dead coords; caps the blow-up when v_hat is tiny (step -> alpha*m/eps, a bounded momentum step) | eps=0: 0/0 or unbounded step on zero-gradient coordinates |
| 10 | eps = 1e-8 | Below any healthy gradient RMS (typically >>1e-8), so invariance holds in the normal regime; nonzero so degenerate coords are tamed | eps too large: erodes scale-invariance, denominator no longer tracks magnitude |
| 11 | beta_1 = 0.9 (window ~1/(1-beta)=~10) | Short, responsive window: kills minibatch noise yet stays responsive as the landscape drifts; same regime as well-tuned momentum | beta_1 too high: direction lags the moving landscape |
| 12 | beta_2 = 0.999 (window ~1000), i.e. beta_2 >> beta_1 | A variance estimate is intrinsically noisier (squaring amplifies spread) and it's the DENOMINATOR (jittery denom jitters every step) -> estimate it more carefully | beta_2 = beta_1: denominator too jittery; beta_2 lower fails sparse coords (too few samples of a rare gradient) |
| 13 | alpha = 0.001 default | Trust-region cap: |Delta_t| <~ alpha, so alpha caps per-step parameter movement; conservative, SNR-annealing handles the fine end | alpha as a raw gradient-scaled rate: coupled to arbitrary loss magnitude, hard to set |
| 14 | Fold both corrections into one scalar step_size = alpha*sqrt(1-beta2^t)/(1-beta1^t) | Avoids materializing m_hat,v_hat each step; one scalar, two in-place EMA updates, one fused divide-add. With nonzero eps this is the reference fixed-floor convention, not exactly Algorithm 1's time-scaled eps floor | Materialize m_hat,v_hat: extra tensors/ops; exact Algorithm 1 eps placement but less like shipped code |
| 15 | Decaying beta_{1,t} = beta_1*lambda^{t-1} (lambda just below 1) in the regret proof | The momentum-penalty regret term sums beta_{1,t}/(1-beta_{1,t})*sqrt(t); lambda<1 collapses sum_t t*lambda^t to a constant 1/(1-lambda)^2 | Constant beta_1: that term is sum_t sqrt(t) = Theta(T^3/2) -> bound breaks. (Matches the "reduce momentum near the end" folklore as a hard requirement.) |
| 16 | gamma = beta_1^2/sqrt(beta_2) < 1 condition | "Momentum doesn't outrun the denominator's forgetting": makes old effective steps decay geometrically -> the m_hat^2/sqrt(v_hat) sum stays O(sqrt(T)) | gamma >= 1: effective steps pile up, lemma fails |
| 17 | L-infinity sibling: u_t = max(beta_2*u_{t-1}, |g_t|) (AdaMax) | Limit p->inf of an Lp power-EMA: prefactor (1-beta2^p)^1/p ->1, Lp norm of a finite seq -> max; collapses to one max, nothing to overflow | Generic finite-p Lp: |g|^p overflows / underflows numerically |
| 18 | AdaMax needs NO bias correction on u | A max doesn't average in the zero init — once any |g|>0 arrives the u_0=0 loses every comparison; no shrink-toward-zero artifact | Apply (1-beta2^t) to u: unnecessary, and there's no zero-bias to undo |
| 19 | AdaMax default alpha = 0.002 (a touch larger) | u_t is a max (upper envelope), tends larger than an RMS, so the safe step is a touch bigger; the envelope is alpha-scale and avoids Adam's sparse-case two-way bound | alpha=0.001: needlessly conservative given the larger denominator |
| 20 | (Forward-looking) Polyak-Ruppert iterate averaging via theta_bar EMA, de-biased | Last iterate of a stochastic method rattles at the noise floor; averaging improves stochastic-approximation convergence; EMA weights recent iterates more | Uniform Polyak average: weights stale early iterates equally |

Every method constant (alpha, beta1, beta2, eps, alpha_adamax=0.002, lambda) and structural
choice (EMA-not-sum, both moments, debias both, eps placement, scalar folding, gamma<1,
L-inf limit) has a why + a rejected alternative above; proof caveats are handled separately in
`notes/source_matrix.md`.

---

## STATE-THEN-JUSTIFY paragraphs to keep FLIPPED in reasoning.md (insight-before-method)

These are the spots where it is tempting to state the formula then justify; reasoning.md must
walk the motivation first and let each piece DROP OUT:
- bias correction: simulate the t=1 blow-up FIRST (pain), then unroll+expectation reveals the
  (1-beta^t) factor, then "divide it out" is forced — never "v_hat = v/(1-beta^t), because...".
- the ratio m/sqrt(v): want both per-param scale and smoothed direction -> moment view shows
  they're orthogonal -> ratio appears; THEN notice scale-invariance, trust region, SNR.
- eps: hit the 0/0 / tiny-v blow-up FIRST, then add eps as the floor.
- defaults: derive from EMA window need (noise-kill vs responsiveness; noisier denominator) ->
  the numbers fall out; never "0.9/0.999 because".
- AdaMax: ask "why L2? what if Lp, p->inf?" -> the max drops out of the limit; THEN note no
  bias correction needed.
- decaying beta_1: in the proof, find the Theta(T^3/2) wall FIRST, then the lambda^{t-1}
  schedule is the patch (and it retro-explains the momentum folklore).

## CODE-FRAMEWORK note (V4 critical change vs V3)

V3's context.md code-framework section PRESUPPOSED the method: it named first/second moment
EMAs, exp_avg/exp_avg_sq, RMSProp-core code, bias correction, AdaMax, exp_inf. That violates
the pre-method scaffold rule. For V4 the scaffold must presuppose NOTHING about the update
rule: just a generic first-order stochastic-optimization harness —

  class Optimizer: def step(self, params, grads): # TODO: the per-parameter update rule we design
  + a training loop that draws a minibatch, computes grads, calls step()
  + plugged into an existing model + loss.

NO moment-estimates / bias-correction / RMSProp-core / method names. The final PyTorch code
fills in step() and corresponds piece-for-piece to this scaffold. Derive the scaffold by
hollowing out the final code: remove the two EMA buffers, the bias-correction scalar, the
addcdiv — leaving the bare Optimizer + loop. The known-before primitives are only: an
Optimizer base with per-parameter state, in-place tensor ops, autograd .backward(), a
minibatch loop, a model + loss.

## Final code is faithful to (grounded)
code/pytorch_v0.3.1_adam.py (PyTorch torch/optim/adam.py): exp_avg.mul_(beta1).add_(1-beta1,grad);
exp_avg_sq.mul_(beta2).addcmul_(1-beta2,grad,grad); bc1=1-beta1**t; bc2=1-beta2**t;
step_size=lr*sqrt(bc2)/bc1; p.addcdiv_(-step_size, exp_avg, exp_avg_sq.sqrt().add_(eps)).
This v0.3.1 reference is pre-AMSGrad. Exclude AMSGrad, AdamW, Reddi/Yogi, and other later
posterior fixes from context/reasoning. AdaMax faithful to code/pytorch_v0.3.1_adamax.py.
