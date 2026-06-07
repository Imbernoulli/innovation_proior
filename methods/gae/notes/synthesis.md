# GAE — synthesis notes (Phase 1.5)

## The pain point (problem being solved)
Policy gradient (PG) methods optimize expected return directly and compose
cleanly with neural-net function approximators. Two chronic failures:
1. **Variance.** The score-function gradient estimator's variance scales badly
   with horizon: the credit for one action gets confounded with the rewards
   caused by all *other* actions (past and future). Long-horizon → noisy
   gradient → tiny stable step size → huge sample complexity.
2. **Stability/nonstationarity.** Even with a good gradient direction, naive SGD
   steps overshoot because the data distribution shifts with the policy.

GAE attacks #1 (variance of the gradient via a better advantage estimator).
TRPO + value-function trust region attacks #2. The reasoning trace's heart is #1.

## First-principles object
The score-function (likelihood-ratio) policy gradient:
  g = E[ Σ_t Ψ_t ∇θ log π(a_t|s_t) ].
Ψ_t can be many things (full return, reward-to-go, Q, A, TD residual). The
**advantage** A^π(s,a) = Q^π − V^π gives (almost) minimal variance: it asks
"is this action better than average from here?" — a baseline V(s_t) that's a
function of s_t only is subtracted, which is allowed because
E_{a~π}[∇log π · b(s)] = 0 (baseline doesn't bias the gradient).

## The two knobs and what they each do
- **γ (discount-as-variance-reduction).** Treat the problem as undiscounted
  (maximize Σ r_t) but introduce γ<1 in the *estimator*. γ downweights rewards
  far in the future, dropping terms beyond ≈1/(1−γ) of the response function.
  This is biased w.r.t. the true (undiscounted) gradient but cuts variance.
  Analyzed by Marbach&Tsitsiklis 2003, Kakade 2001b, Thomas 2014.
  Key fact: **γ biases the gradient even with a perfect value function** — it
  changes the objective (the scale of V^{π,γ}).
- **λ (the GAE knob).** Exponentially-weighted average of k-step advantage
  estimators. λ controls bias↔variance *given* γ. Key fact: **λ<1 biases only
  when V is inaccurate**; with V=V^{π,γ}, every λ gives a γ-just estimator.

## The derivation chain (the spine of reasoning.md)
1. TD residual δ_t^V = r_t + γV(s_{t+1}) − V(s_t). If V=V^{π,γ}, then
   E_{s_{t+1}}[δ_t] = Q^{π,γ}(s_t,a_t) − V^{π,γ}(s_t) = A^{π,γ}(s_t,a_t).
   So δ is an unbiased one-step advantage estimate *when V is exact*.
2. k-step sum telescopes:
   Â^(k)_t = Σ_{l=0}^{k-1} γ^l δ_{t+l}
           = −V(s_t) + r_t + γr_{t+1} + … + γ^{k-1}r_{t+k-1} + γ^k V(s_{t+k}).
   = k-step return minus baseline V(s_t). Bias→0 as k→∞ because the γ^k V(s_{t+k})
   term is ever more discounted; −V(s_t) never affects bias (it's a baseline).
   - k=1: δ_t (low variance, high bias unless V exact).
   - k=∞: Σ γ^l r_{t+l} − V(s_t) (Monte Carlo return minus baseline; unbiased
     w.r.t. γ-discounted advantage but high variance).
3. **GAE = exponentially-weighted average of the Â^(k):**
   Â^GAE(γ,λ)_t := (1−λ)( Â^(1) + λÂ^(2) + λ²Â^(3) + … ).
   Substitute Â^(k)=Σ_{l=0}^{k-1} γ^l δ_{t+l}, regroup by δ_{t+l}:
   coefficient of γ^l δ_{t+l} is (1−λ)(λ^l + λ^{l+1} + …) = (1−λ)·λ^l/(1−λ) = λ^l.
   ⇒ **Â^GAE_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}.**  (the clean closed form)
   The (1−λ) is exactly the normalizer that makes the weights (1−λ)λ^k sum to 1.
4. Limits:
   - λ=0: Â_t = δ_t = r_t + γV(s_{t+1}) − V(s_t).   (one-step TD; high bias/low var)
   - λ=1: Â_t = Σ γ^l δ_{t+l} = Σ γ^l r_{t+l} − V(s_t). (MC minus baseline; low bias/high var)
5. **γ-just property** (Prop. 1 + proof, appendix). Definition: Â_t is γ-just if
   E[Â_t ∇log π(a_t|s_t)] = E[A^{π,γ}(s_t,a_t) ∇log π(a_t|s_t)]. Sufficient
   condition: Â_t = Q_t(s_{t:∞},a_{t:∞}) − b_t(s_{0:t},a_{0:t−1}) where Q_t is an
   unbiased estimate of Q^{π,γ}. Proof splits into Q-term and b-term:
   - Q-term: condition on s_{0:t},a_{0:t}, pull ∇log π out, inner expectation of
     Q_t gives Q^{π,γ}; baseline V(s_t) inside ⇒ A^{π,γ}. (Uses tower property.)
   - b-term: b depends only on past (s_{0:t},a_{0:t−1}); condition so a_t is the
     last sampled variable, then E_{a_t}[∇log π(a_t|s_t)] = ∫π ∇log π = ∇∫π = ∇1 = 0.
   ⇒ baseline contributes nothing. GAE(γ,1) is γ-just for ANY V; GAE(γ,0) is
   γ-just only for V=V^{π,γ}.
6. **Reward shaping connection** (Ng et al. 1999). Potential shaping
   r̃ = r + γΦ(s') − Φ(s) leaves A^{π,γ} invariant. With Φ=V, the shaped reward
   r̃ **equals** the TD residual δ_t^V. The γλ-discounted sum of shaped rewards
   = Σ(γλ)^l δ_{t+l} = Â^GAE. So GAE = "shape the reward with V to compress the
   temporal spread of credit, then apply a steeper discount γλ to cut the
   long-delay noise." With Φ=V^{π,γ} the response function χ(l;s,a) collapses to
   l=0 only (all credit becomes immediate); an approximate V partially does this.
7. **Response function** χ(l;s,a) = E[r_{t+l}|s_t,a_t] − E[r_{t+l}|s_t]. Then
   A^{π,γ}(s,a) = Σ γ^l χ(l;s,a) — decomposes advantage across time delays.
   γ<1 drops terms with l≫1/(1−γ); γλ cuts δ-terms with l≫1/(1−γλ). Justifies
   why two knobs and why best λ < best γ empirically (λ adds less bias).

## Value function estimation
- Targets: Monte-Carlo / TD(1) returns V̂_t = Σ γ^l r_{t+l}. Regress V_φ to V̂.
- Trust-region VF fit: minimize Σ‖V_φ(s_n)−V̂_n‖² s.t.
  (1/N)Σ ‖V_φ(s_n)−V_old(s_n)‖²/(2σ²) ≤ ε (≈ avg KL of a Gaussian with var σ²).
  Solve via CG with Gauss-Newton / Fisher matrix H = (1/N)Σ j_n j_nᵀ,
  j_n = ∇_φ V_φ(s_n); step s≈−H⁻¹g via matrix-vector products, rescale to hit ε.
- TD(λ) target footnote: V̂^λ_t = V_old(s_n) + Σ(γλ)^l δ_{t+l}; tried, no
  improvement over λ=1 target.

## Policy update (TRPO)
maximize L(θ)=(1/N)Σ [π_θ(a|s)/π_old(a|s)] Â_n  s.t. mean-KL ≤ ε.
Linearize objective, quadraticize KL ⇒ step θ−θ_old ∝ −F⁻¹g (natural gradient
direction; same direction as Kakade natural PG / Peters natural actor-critic),
solved by CG with Fisher-vector products, line search to satisfy constraint.

## Algorithm ordering subtlety (important, design choice)
Update policy with V_{φ_i} (the OLD value function), THEN update value function.
If you fit V first, you'd overfit Bellman residuals toward zero ⇒ δ_t≈0 ⇒
gradient≈0. So fitting VF first introduces extra bias. Order matters.

## Appendix FAQ
- **Compatible features** (Konda&Tsitsiklis): PG only depends on the projection
  of A onto span{∇log π}. Projecting a γ-just Â onto compatible features by least
  squares = the natural policy gradient. Any Â (incl. GAE) plugs in. Orthogonal
  to GAE's contribution (which is about temporal structure, not the subspace).
- **Why state-value V, not Q?** (1) V has lower-dim input, easier to learn;
  (2) GAE smoothly interpolates λ∈[0,1] between high-bias and low-bias; a
  parameterized Q only gives the high-bias (one-step) estimator, and one-step
  bias was empirically prohibitive.

## DESIGN-DECISION → WHY table
| Decision | Why this / why not the alternative |
|---|---|
| Use advantage A, not Q or raw return, as Ψ_t | A=Q−V centers on "better than average"; subtracting baseline V(s) is variance-optimal among baselines and unbiased (E[∇log π·b(s)]=0). Raw return / reward-to-go has far higher variance. |
| Subtract V(s_t) as baseline (not a const) | State-dependent baseline removes the part of variance explained by the state; const baseline can't. Must be a fn of s_t (not a_t) to stay unbiased. |
| Introduce γ<1 although problem is undiscounted | γ downweights far-future rewards → drops high-variance long-delay terms (l≫1/(1−γ)). Bias accepted as variance-reduction (Marbach, Kakade, Thomas). |
| Build estimator from TD residual δ_t, not raw rewards | δ has expectation A when V exact; using V as a learned baseline soaks up variance; residuals telescope into k-step returns cleanly. |
| Exponential weighting (1−λ)λ^{k−1} over k-step estimators (not pick one k) | A single k is a hard bias/var choice; geometric weighting gives ONE smooth knob λ and collapses to the closed form Σ(γλ)^l δ. Mirrors TD(λ)'s λ-return but for the advantage, not the value. |
| The (1−λ) prefactor | Normalizer: Σ_k (1−λ)λ^{k−1} = 1, so weights are a probability distribution over horizons; yields coefficient λ^l on γ^l δ_{t+l}. |
| Two separate knobs γ and λ (not one) | They bias differently: γ biases even with perfect V (changes objective scale); λ biases only when V is wrong. So best λ ≪ best γ; collapsing them loses control. |
| γλ as the effective discount on δ's | Reward-shaping view: shape with Φ=V (r̃=δ), then apply steeper discount γλ to cut long-delay noise after V has compressed the response function. |
| TRPO for the policy update | Need a stable step under nonstationary data; KL trust region gives monotone-ish improvement and natural-gradient direction; plain SGD overshoots. |
| Trust-region (KL/Fisher) for the VALUE function too | Plain LSQ regression overfits the latest batch; KL constraint (Gaussian interpretation) limits per-iteration change → robust NN value fitting. |
| CG + Fisher/Gauss-Newton-vector products | Forming/inverting F or H explicitly is O(d²)–O(d³); matrix-vector products keep it O(d) per CG iter for 10^4-param nets. |
| Update policy before value function | Fitting VF first drives δ→0 (overfit Bellman residual) → zero gradient; using old V_{φ_i} for advantages avoids that extra bias. |
| State-value V over Q-function | Lower-dim input (easier), and enables the full λ∈[0,1] interpolation; Q only affords the high-bias one-step estimator, which was empirically too biased. |
| MC / TD(1) targets for VF (not TD(λ) target) | Tried TD(λ) target V_old+Σ(γλ)^l δ; no measurable gain over the λ=1 (MC) target, so keep the simpler one. |

## Canonical code (grounded)
Two equivalent forms in the wild:
- Backward recursion (OpenAI baselines ppo2 runner):
  δ_t = r_t + γ·V(s_{t+1})·nonterminal − V(s_t);
  A_t = δ_t + γλ·nonterminal·A_{t+1};  returns = A + V.
- Forward discounted-cumsum (SpinningUp PPO buffer):
  deltas = rews[:-1] + γ·vals[1:] − vals[:-1];
  adv = discount_cumsum(deltas, γλ);  ret = discount_cumsum(rews, γ)[:-1].
Both compute Â_t = Σ(γλ)^l δ_{t+l}. The backward recursion is the unrolled form
of the closed sum: A_t = δ_t + (γλ)A_{t+1}.
