# SAC — Synthesis (Phase 1.5)

## The pain point (state of field, ~2017)
Model-free deep RL on continuous control suffers two chronic problems:
1. **Sample inefficiency.** On-policy methods (TRPO, PPO, A3C) throw away every batch after one
   gradient step — millions of env steps for simple tasks.
2. **Brittleness / instability.** Off-policy actor-critic (DDPG) reuses data via a replay buffer
   and is sample-efficient, but is notoriously hard to stabilize and hyperparameter-sensitive;
   it fails outright on high-dim tasks (Ant, Humanoid). On-policy methods still beat it there.

Goal: an algorithm that is BOTH off-policy (sample-efficient) AND stable on high-dim continuous
control, with little per-task tuning.

## The two tools on the table, and exactly where each falls short

### DDPG (Lillicrap 2015) = deep DPG (Silver 2014)
- Continuous Q-learning: target y = r + γ Q_targ(s', μ_targ(s')). Q trained by MSBE.
- The max_a Q in continuous spaces is intractable, so a **deterministic actor** μ_θ(s) is trained
  to approximate argmax_a Q via the chain rule: ∇_θ Q(s,μ(s)) = ∇_a Q · ∇_θ μ.
- LIMITATIONS:
  - deterministic policy → zero intrinsic exploration → must bolt on external OU/Gaussian noise.
  - deterministic actor + Q interplay is brittle, hyperparameter-sensitive, overestimates Q.
  - collapses to a single action; no notion of multimodality; fragile on high-dim.

### Soft Q-learning / energy-based policies (Haarnoja 2017, arXiv 1702.08165)
- Maximum-entropy RL: augment reward with policy entropy. Soft Bellman:
  V(s) = α log ∫ exp(Q(s,a)/α) da ("soft max"), optimal policy π ∝ exp(Q(s,·)/α) — energy-based.
- Solves exploration (multimodal, robust) but **sampling from exp(Q) in continuous actions has no
  closed form**: needs an amortized SVGD sampling network to approximate samples.
- LIMITATIONS:
  - it's a Q-learning method, not actor-critic: Q estimates the OPTIMAL Q*, the "actor" is only an
    approximate sampler; convergence hinges on how well the sampler matches the true posterior ∝exp Q.
  - the SVGD inference is complex and a source of instability.

### Max-ent RL framework (Ziebart 2008/2010, Todorov, Toussaint, Rawlik, Fox/Tishby)
- Objective J(π) = Σ_t E[r(s,a) + α H(π(·|s))]. α → 0 recovers standard RL.
- Why entropy: better exploration (acquire diverse near-optimal behaviors), robustness to model/
  estimation error (Ziebart 2010: max-ent policies robust to perturbations), captures multiple modes.

### Reparameterization trick (Kingma 2013 VAE) / SVG(0) (Heess 2015)
- To get low-variance gradients of E_{a~π_θ}[f(a)] w.r.t. θ when f is differentiable (here f = Q,
  a NN we can backprop through), write a = f_θ(ε; s), ε ~ fixed noise; gradient flows through a.
- SVG(0) is a zero-step special case but optimizes standard return, no entropy, no separate V net.
- Alternative = likelihood-ratio / REINFORCE: ∇ E[f] = E[f ∇ log π], high variance, doesn't use
  that Q is differentiable. Reparam is strictly better when the target is a differentiable NN.

### Double-Q / clipped double-Q (van Hasselt 2010; TD3 Fujimoto 2018)
- Max in Q-learning → positive bias (overestimation). Use min of two independently-trained Qs as
  the value estimate to mitigate.

## The central object & difficulty
Optimize the max-ent objective J(π) = Σ E[r + αH(π)] off-policy, with a STOCHASTIC actor (for
exploration/stability) but WITHOUT the intractable energy-based sampling of soft Q-learning, and
WITHOUT DDPG's deterministic-actor brittleness. Need: (a) a tractable stochastic actor, (b) a way
to evaluate the CURRENT policy's soft value (true actor-critic, not Q*-learning), (c) a low-variance
off-policy gradient for the actor.

## The chain of approximations theory → practice
1. **Soft policy iteration** (tabular, exact): alternate
   - soft policy EVALUATION: iterate soft Bellman backup T^π Q = r + γ E_{s'}[V(s')], where
     V(s) = E_{a~π}[Q(s,a) − log π(a|s)]. Converges (contraction via entropy-augmented reward).
   - soft policy IMPROVEMENT: π_new = argmin_{π'∈Π} KL(π'(·|s) ‖ exp(Q^{π_old}(s,·))/Z(s)).
     Provably non-decreasing soft value (Lemma 2). Together → converges to optimal π in Π (Thm 1).
   - KEY: restrict π to a tractable family Π and **project** the energy-based target into Π via KL.
     This is what removes the need for SVGD sampling — we never sample from exp(Q); we fit a
     tractable π to it.
2. **Function approximation + SGD instead of to-convergence:**
   - parameterize Q_θ, π_φ, and (for stability) a separate V_ψ.
   - V loss: J_V = E[(V_ψ(s) − E_{a~π}[Q_θ(s,a) − log π_φ(a|s)])²]  (eq 5/6)
   - Q loss (soft Bellman residual): J_Q = E[(Q_θ(s,a) − (r + γ E_{s'}[V_ψ̄(s')]))²]  (eq 7/8/9),
     V_ψ̄ a target (EMA) network.
   - π loss: J_π = E_s[ KL(π_φ(·|s) ‖ exp(Q_θ(s,·))/Z_θ(s)) ]  (eq 10). Z drops (no φ-gradient).
3. **Reparameterize the actor**: a = f_φ(ε; s), Gaussian + tanh squash. Then
   J_π = E_{s,ε}[ log π_φ(f_φ(ε;s)|s) − Q_θ(s, f_φ(ε;s)) ]  (eq 11/12). Low-variance, DDPG-style
   gradient extended to a STOCHASTIC policy (eq 12 generalizes DPG).
4. **Two Q-functions** (min) to fight overestimation in V and π updates.

## Which prior method falls out as special case
- α → 0 recovers standard RL objective.
- The deterministic ablation (drop entropy, deterministic policy, two Qs, hard target, no target
  actor) ≈ DDPG → SAC is "DDPG done with a stochastic max-ent actor and a value net".
- SVG(0) = SAC's reparam policy gradient but with standard return and no V net.

## Design decisions → why (with rejected alternatives)
- **Stochastic actor (vs DDPG deterministic):** exploration is intrinsic (no OU noise to tune);
  multimodal; entropy in the value fn makes high-entropy regions valuable → stabilizes seeds.
- **Separate V network (vs deriving V from Q,π):** "in principle unnecessary" since
  V(s)=E_a[Q−logπ]; but a separate V_ψ stabilizes training and is convenient. (Modern variants like
  Spinning Up drop V and use target-Q directly — but the paper's own choice is to keep V.)
- **KL projection into Π (vs energy-based sampling of soft Q-learning):** sidesteps SVGD entirely;
  makes it a true actor-critic; gives a convergence proof "regardless of policy parameterization".
- **Information projection (KL with π' first arg) target exp(Q)/Z:** convenient — the resulting
  objective is exactly E_a[log π − Q] (+const), and Z is φ-independent so it vanishes.
- **Reparameterization trick (vs likelihood-ratio):** Q is a differentiable NN; reparam gives lower
  variance by backpropagating ∇_a Q through the action. LR estimator ignores Q's differentiability.
- **Gaussian + tanh squash:** Gaussian is tractable/reparameterizable but unbounded; actions must
  live in [−1,1]. tanh squash → change-of-variables log-prob correction
  log π(a|s) = log μ(u|s) − Σ_i log(1 − tanh²(u_i)). (eq 21)
- **Two Q-functions, take min:** clipped double-Q (TD3/van Hasselt) to reduce positive bias;
  empirically speeds up hard tasks though one Q can solve Humanoid.
- **Target value network (EMA, τ=0.005):** slowly-tracking target stabilizes the bootstrapped
  regression (Mnih 2015). τ=1 + periodic copy = hard update variant.
- **Reward scale = inverse temperature:** with α subsumed into reward (scale reward by 1/α), the
  reward magnitude controls stochasticity of optimal policy. Only hyperparameter that needs tuning.
  Too small → near-uniform policy, ignores reward; too large → near-deterministic, poor local optima.
- **Adam, lr 3e-4, 2×256 ReLU MLPs, γ=0.99, buffer 1e6, batch 256:** standard.
- **Discounted max-ent objective subtlety (App A):** with a discount you don't discount the state
  distribution, only rewards; the "true" discounted max-ent objective weights future entropy+reward
  from each (s,a) by ρ_π — deferred detail, justifies the practical backup.

## Numerical detail for tanh log-prob (from Spinning Up, numerically stable form)
log(1 − tanh²(u)) = 2(log 2 − u − softplus(−2u)); use this instead of log(1−tanh²+ε) to avoid NaNs.

## Canonical implementations
- Official: github.com/haarnoja/sac (TF1, rllab). algos/sac.py: V loss (min of two Q − logπ),
  Q loss (MSBE to scale_reward*r + γ(1−d)V_targ), policy KL loss (reparam: mean(logπ − Q1)),
  EMA target on V. policies/gaussian_policy.py: Normal + tanh squash, _squash_correction =
  Σ log(1 − tanh²(u) + 1e-6).
- Clean reference: OpenAI Spinning Up PyTorch SAC (later twin-target-Q variant, no V net) — used
  for the readable squashed-Gaussian actor and numerically-stable log-prob.
- Final answer code grounds in the PAPER's formulation (V net + twin Q), structured like Spinning Up
  for readability.

## In-frame scaffold ↔ final code correspondence
Pre-method skeleton: replay buffer, MLP builder, a generic stochastic policy stub
(`class Policy: sample()/log_prob()` TODO), value-net stubs, an off-policy actor-critic training
loop with TODO update step. Final code fills: SquashedGaussianActor, twin Q, V + target V, the
three losses, polyak update.
