# TD3 — synthesis notes (written before Phase 2)

Target method: Twin Delayed Deep Deterministic policy gradient (TD3). arXiv:1802.09477, Fujimoto/van Hoof/Meger, ICML 2018.
Builds on DDPG (continuous-control actor-critic). Three contributions: (1) Clipped Double Q-learning, (2) Delayed policy updates + target networks for variance, (3) Target policy smoothing regularization.

## Pain point / research question
In continuous-control actor-critic (DDPG is SOTA), the learned critic Q_θ overestimates value, the actor exploits the critic's errors, a feedback loop builds, and learning is brittle/divergent and seed-sensitive. Overestimation from function-approximation noise is well-studied for *discrete* Q-learning but had been "largely left untouched" for actor-critic continuous control. Goal: characterize and fix overestimation + error accumulation (variance) in the actor-critic deterministic-policy-gradient setting, without giving up sample efficiency.

## Load-bearing ancestors (verified against primary text)

### Q-learning + function approximation (Watkins 1989; Sutton 1988 TD; Bellman)
- Q*(s,a)=E[r+γ max_a' Q*(s',a')]. Tabular Q-learning converges (Robbins-Monro). With FA, minimize L(θ)=E[(y−Q_θ(s,a))²], y=r+γ max_a' Q_θ⁻(s',a') (semi-gradient; target's θ-dependence dropped).
- The TD target is built from an estimate of the next state → error accumulates across bootstraps.

### Overestimation bias from the max of noisy estimates (Thrun & Schwartz 1993)
- If target subject to zero-mean error ε: E_ε[max_a'(Q(s',a')+ε)] ≥ max_a' Q(s',a'). The max operator over a noisy estimate is a biased estimator of the max of the true values → consistent positive bias. With FA, this noise (approximation error) is unavoidable. Bias is then propagated through the Bellman backup and compounds.

### Double Q-learning (van Hasselt 2010)
- Two estimators Q^A, Q^B. Decouple SELECTION from EVALUATION: a*=argmax_a Q^A(s',a); update Q^A toward r+γ Q^B(s',a*). Because the action is selected by one estimator and valued by the *other* (independent), the value of the selected action is not systematically the max of its own noise → unbiased (E[Q^B(s',a*)] ≤ max_a Q^B). Removes maximization bias.
- Limitation TD3 reacts to: requires the two estimators to be (near-)independent.

### Double DQN (van Hasselt 2016)
- Deep version: reuse the existing target network as the second estimator. Selection by online net Q_θ, evaluation by target net Q_θ⁻: y=r+γ Q_θ⁻(s', argmax_a' Q_θ(s',a')).
- Limitation in actor-critic: the policy (=the argmax surrogate) changes slowly, so the action selected by current policy and the action the target critic would prefer are basically the same → the target and current value estimates are too similar to be independent → bias not removed. (TD3 measures this: "DDQN-AC" still overestimates like DDPG.)

### DPG (Silver 2014) / DDPG (Lillicrap 2015)
- DPG: for a deterministic actor π_φ, ∇_φ J(φ)=E_s[∇_a Q^π(s,a)|_{a=π(s)} ∇_φ π_φ(s)]. The critic supplies the gradient direction; the actor ascends Q. This is the continuous-action replacement for the discrete max_a (you can't enumerate actions).
- DDPG = DPG + DQN machinery: deep nets, replay buffer (off-policy, decorrelate), soft target networks θ'←τθ+(1−τ)θ' for actor and critic, OU exploration noise. Critic loss MSBE with target y=r+γ Q_θ'(s',π_φ'(s')). One actor, one critic.
- Gap: single critic → overestimation; no analysis of the actor/critic coupling; sensitive to hyperparameters and seeds; high target variance from deterministic peaks.

### Two-timescale stochastic approximation (Konda & Tsitsiklis 2003)
- Actor-critic convergence (linear setting) often needs the critic to update on a faster timescale than the actor — the critic should be near-converged relative to the (slow) policy. Justifies delaying policy updates.

### SARSA / Expected SARSA (Sutton & Barto; van Seijen 2009)
- SARSA target uses the action actually taken (on-policy), y=r+γ Q(s',a'), a'~π. Yields "safer" values robust to action perturbations. Expected SARSA averages over the policy's action distribution. Motivates target policy smoothing as a SARSA-style bootstrap over a noisy policy.

## The derivation chain (insight → method)

1. **Does overestimation even exist in actor-critic?** Discrete bias comes from analytic max; here the actor ascends Q via gradient, no max. Prove it still happens (Sec 4.1):
   - φ_approx = φ + (α/Z1) E[∇_φπ ∇_a Q_θ], φ_true = φ + (α/Z2) E[∇_φπ ∇_a Q^π] (normalized gradients).
   - Local maximizer of the *approximate* objective ⇒ ∃ε1, α≤ε1 ⇒ E[Q_θ(s,π_approx)] ≥ E[Q_θ(s,π_true)]. (approx policy is better under the approx critic)
   - Local maximizer of *true* objective ⇒ ∃ε2, α≤ε2 ⇒ E[Q^π(s,π_true)] ≥ E[Q^π(s,π_approx)]. (true policy better under true critic)
   - Assume at π_true the estimate is at least the true value: E[Q_θ(s,π_true)] ≥ E[Q^π(s,π_true)]. Chain with α<min(ε1,ε2): E[Q_θ(s,π_approx)] ≥ E[Q_θ(s,π_true)] ≥ E[Q^π(s,π_true)] ≥ E[Q^π(s,π_approx)] ⇒ overestimation at π_approx.
   - Appendix B (unnormalized gradients): φ_true maximizes the true rate of change Δ^π_true = Q^π(s,π_true)−Q^π(s,π_φ) so Δ^π_true ≥ Δ^π_approx; condition (matching values along the segment toward φ_true) gives Δ^θ_approx ≥ Δ^π_true; with Q_θ(s,π_φ)=Q^π(s,π_φ) ⇒ Q_θ(s,π_approx) ≥ Q^π(s,π_true) ≥ Q^π(s,π_approx).
   - Verified empirically (motivating/diagnostic): plotting DDPG's value estimate vs true (discounted return averaged over 1000 episodes from buffer states) over 1M steps shows clear systematic overestimation. (CDQ curve, shown for contrast, is the proposed thing — that's the method, fine to mention as what we'll build toward, but don't claim its win as a result.)

2. **Try the discrete fix: Double DQN in actor-critic.** y=r+γ Q_θ'(s',π_φ(s')) (current policy, target critic). FAILS — slow policy ⇒ current and target too similar ⇒ still overestimates (measured: DDQN-AC like DDPG). WALL.

3. **Try original Double Q-learning with a pair of actor-critics.** (π_φ1,Q_θ1),(π_φ2,Q_θ2): y1=r+γ Q_θ2'(s',π_φ1(s')), y2=r+γ Q_θ1'(s',π_φ2(s')). Better (each critic valued by the other), but not independent (shared replay buffer + each target uses the opposite critic) ⇒ for some states Q_θ2(s,π_φ1(s)) > Q_θ1(s,π_φ1(s)). Since Q_θ1 already overestimates, those states get *further* inflated, propagated. Reduction insufficient. WALL.

4. **Clipped Double Q-learning (the fix).** Use the biased estimate as an upper bound on the less-biased one ⇒ take the minimum:
   y1 = r + γ min_{i=1,2} Q_θi'(s', π_φ1(s')).
   - "Cannot introduce any additional overestimation over the standard Q-learning target." May induce *under*estimation — acceptable, because underestimated actions are NOT propagated by the policy (the actor avoids low-value actions), whereas overestimated actions ARE chased.
   - Implementation simplification: single actor π_φ optimized wrt Q_θ1, same target y for both critics. If Q_θ2>Q_θ1 update = standard (no extra bias); if Q_θ2<Q_θ1, value reduced (Double-Q-like). 
   - Secondary benefit (variance): E[min of RVs] decreases as variance increases ⇒ min prefers low-variance states ⇒ safer targets.
   - Convergence proof (Appendix A): finite MDP, two tabular Q^A,Q^B, a*=argmax Q^A(s',·), y=r+γ min(Q^A,Q^B)(s',a*), both updated. Apply the SARSA-convergence lemma (Singh 2000) with Δ_t=Q^A−Q*, ζ_t=α_t. Write F_t = F^Q_t + c_t where F^Q_t=r+γ Q^A(s',a*)−Q* is the standard-Q term (E[F^Q_t|P_t] ≤ γ||Δ_t||, known) and c_t = γ min(Q^A,Q^B)(s',a*) − γ Q^A(s',a*). Need c_t→0, i.e. Δ^BA=Q^B−Q^A→0. With both updated to the same target y: Δ^BA_{t+1} = Δ^BA_t + α_t((y−Q^B)−(y−Q^A)) = Δ^BA_t + α_t(Q^A−Q^B) = (1−α_t)Δ^BA_t → 0. So c_t→0, lemma condition 3 holds, Q^A→Q*. Symmetric for Q^B. □

5. **Variance / accumulating error (Sec 5).** With residual TD-error δ(s,a): Q_θ(s,a)=r+γ E[Q_θ(s',a')]−δ(s,a). Unrolling: Q_θ(s_t,a_t)=E[Σ_{i=t}^T γ^{i−t}(r_i − δ_i)] — value estimate = expected return minus expected discounted future TD-errors. Var(estimate) ∝ Var(future reward + estimation error); large γ ⇒ variance grows fast if per-update error not tamed. Each minibatch step only controls error inside the batch.

6. **Target networks ⇄ variance (Sec 5.2).** Diagnostic: with fixed policy, all τ converge similarly (volatility higher at τ=1). With a *learned* policy, fast targets (τ=1) diverge. ⇒ divergence = policy updates against a high-variance value estimate; the actor/critic feedback loop (bad value ⇒ bad policy ⇒ bad value). Fix the target to slow error growth.

7. **Delayed policy updates (the fix).** If targets reduce error over many critic updates, and policy updates on high-error states diverge, then update actor + targets only every d critic updates (d=2). Less likely to repeat updates against an unchanged critic; the policy update that does happen uses a lower-variance critic. Two-timescale (Konda & Tsitsiklis 2003): critic fast, actor slow. Empirically improves performance with fewer actor updates.

8. **Target policy smoothing (the fix, Sec 5.3).** Deterministic policies overfit narrow Q peaks; the deterministic-action target has high variance from FA error. SARSA intuition: similar actions should have similar value. Fit the value of a small region around the target action: y=r+E_ε[Q_θ'(s',π_φ'(s')+ε)] ≈ r+γ Q_θ'(s',π_φ'(s')+ε), ε~clip(N(0,σ),−c,c). Clip ε to keep target near the real action; also clip the smoothed action to the valid action range. Reminiscent of Expected SARSA but off-policy and noise chosen independently of exploration.

9. **Assemble TD3** = DDPG + clipped double-Q (twin critics, min target) + smoothed target action + delayed actor/target updates:
   ỹ = π_φ'(s') + clip(N(0,σ̃),−c,c); y = r + γ min_i Q_θi'(s',ỹ); critics → MSE to y each step; every d steps: actor by DPG wrt Q_θ1, then soft-update θ'_i, φ'.

## Design-decision → why (with rejected alternatives)

| Decision | Why | Rejected alternative / failure |
|---|---|---|
| Two critics, take min | Upper-bound the less-biased estimate by the biased one; cannot add overestimation over standard Q-learning | Single critic (DDPG) → overestimates; pair w/o min (DQ-AC) → not independent, still inflates some states; DDQN-AC → policy too slow, no independence |
| Accept resulting underestimation | Under-valued actions aren't chased by the actor; over-valued ones are | Optimism / over-estimating critic → propagated, suboptimal policy |
| Min also reduces variance | E[min] favors low-variance states → safer targets | — |
| Single actor wrt Q_θ1, shared target | Cheaper; if Q_θ2>Q_θ1 it's the standard update (no extra bias) | Two full actor-critics → 2× cost, marginal gain |
| Delayed policy update d=2 | Critic must be near-converged before actor moves (two-timescale); avoids policy updates on high-variance value | d=1 (DDPG) diverges in the no-target / fast-target regime; large d cripples actor learning (critic trained 1×/step) |
| Soft target τ=0.005 | Stable bootstrap target slows error accumulation | τ=1 (no/fast target) → divergence with a learned policy |
| Target policy smoothing σ=0.2, clip c=0.5 | Smooths Q over similar actions (SARSA-style), kills overfit to narrow peaks, lowers target variance; clip keeps target near real action | Deterministic target → high-variance peaks, brittle |
| Clip smoothed action to action range | Avoid valuing impossible actions | — |
| Uncorrelated Gaussian exploration N(0,0.1) | Simpler than OU, no measured benefit from OU | OU process (DDPG) → no gain |
| 400/300 (paper) hidden, Adam lr 1e-3, batch 100, γ0.99 | Re-tuned DDPG base ("AHE") | original DDPG: actor lr 1e-4, batch 64, τ1e-3, obs-norm, critic L2 |
| Terminal handling: y=r if terminal (not horizon-cap) | Don't bootstrap past true episode end; horizon truncation isn't a real terminal | — |

NOTE on code grounding: canonical repo github.com/sfujim/TD3. Current master uses 256/256 hidden, Adam lr 3e-4, batch 256, start_timesteps random warmup — these are the maintained defaults; the original write-up used 400/300, lr 1e-3, batch 100. Ground final code on the repo structure (Actor/Critic with twin Q, Q1 helper, TD3 train loop) and note the original hyperparameters in comments. policy_noise/noise_clip scaled by max_action.

## In-frame reminders
- Never say "this paper"/"the authors"/arXiv id. May name TD3, DDPG, DPG, Double Q-learning, Thrun&Schwartz, Konda&Tsitsiklis, SARSA as prior art / the thing being built.
- context.md: 5 sections, pre-method, no TD3 name, no proposed-method results; the overestimation-in-DDPG diagnostic and target-network diagnostic are pre-method facts about an existing system → allowed in Background.
- reasoning.md: continuous first-person, no headers, discovery order, all proofs worked inline (Sec 4.1 chain, Appendix A convergence, Appendix B unnormalized, Sec 5 unrolled TD-error).
