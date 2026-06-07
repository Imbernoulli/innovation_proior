# NoisyNet synthesis (arXiv 1706.10295, ICLR 2018, DeepMind)

## Pain point / research question
- Exploration in deep RL is driven by crude "dithering": ε-greedy (value methods) or entropy bonus (policy gradient). These add **state-independent, temporally decorrelated** noise to the policy at every step. Cannot produce coherent, multi-step, state-dependent exploratory behaviour. Reference: osband2017deep "deep exploration".
- Optimism-in-face-of-uncertainty methods have guarantees but don't scale to neural nets. Intrinsic-motivation methods separate generalisation from exploration and require hand-tuning the intrinsic-reward weight, which can distort the optimal policy.
- Goal: a single, simple exploration mechanism that (a) injects state-dependent, temporally-consistent noise, (b) is learned (self-tuning), (c) is a drop-in for DQN/Dueling/A3C with negligible overhead.

## Core idea
Perturb the network **weights** with parametric noise instead of perturbing the action. A single weight perturbation induces a consistent state-dependent policy change across the whole episode. The noise *intensity* is a learnable parameter, trained by the same RL gradient.

## Noisy linear layer
- Standard linear: y = wx + b, w∈R^{q×p}, b∈R^q, x∈R^p (p inputs, q outputs).
- Noisy: replace w by μ^w + σ^w ⊙ ε^w, b by μ^b + σ^b ⊙ ε^b.
  y = (μ^w + σ^w ⊙ ε^w) x + μ^b + σ^b ⊙ ε^b.
- Learnable: μ^w∈R^{q×p}, μ^b∈R^q, σ^w∈R^{q×p}, σ^b∈R^q. Noise ε^w∈R^{q×p}, ε^b∈R^q fixed-statistics zero-mean.
- θ ≜ μ + Σ ⊙ ε, ζ ≜ (μ,Σ). Loss wrapped in expectation over noise: L̄(ζ) = E[L(θ)].
- Gradient: ∇L̄(ζ) = E[∇_{μ,Σ} L(μ+Σ⊙ε)], Monte-Carlo with a single sample ξ per step: ∇L̄ ≈ ∇_{μ,Σ}L(μ+Σ⊙ξ).

## Two noise schemes
(a) **Independent Gaussian**: each ε^w_{i,j} and ε^b_j i.i.d. N(0,1). pq+q noise variables per layer. Used for A3C (distributed, compute not a concern).
(b) **Factorised Gaussian**: p input-noise unit Gaussians ε_i and q output-noise unit Gaussians ε_j (p+q total). Then
   ε^w_{i,j} = f(ε_i) f(ε_j),  ε^b_j = f(ε_j),  with f(x) = sgn(x) sqrt(|x|).
   - Note: bias could use f(x)=x but they keep f(ε_j) to reuse output noise.
   - Reason for factorisation: cut RNG cost (pq → p+q random draws). Matters for single-thread DQN/Dueling. Used for DQN and Dueling.
   - Why f(x)=sgn(x)√|x|: makes ε^w_{i,j}=f(ε_i)f(ε_j) have the same per-entry second moment behaviour as a unit variable — Var(f(ε)) for f=sgn·√|·| applied to N(0,1): E[f^2]=E[|ε|]=√(2/π); product of two such has E=2/π≈0.637. The transform keeps the magnitude O(1) rather than the heavy product N(0,1)·N(0,1) which would have larger spread/tails. The sign·sqrt keeps the noise sub-Gaussian-ish and unit-ish scale per factor.

## Initialisation
- **Unfactorised**: μ_{i,j} ~ U[-√(3/p), +√(3/p)]; σ_{i,j} = 0.017 (constant). (From Bayesian-RNN work fortunato2017bayesian; not tuned.)
- **Factorised**: μ_{i,j} ~ U[-1/√p, +1/√p]; σ_{i,j} = σ0/√p with σ0 = 0.5.
  - μ range √(3/p) for independent vs 1/√p for factorised: matches the variance of the effective noise. With factorised noise the per-weight effective noise variance is larger (product structure scaled), so μ init range is reduced.

## RL integration
- **DQN / Dueling**: drop ε-greedy; act greedily w.r.t. the *randomised* Q. Replace fully-connected layers (value/advantage heads) by noisy layers. Resample noise after every replay/optimisation step; for a replay batch the noise sample is held fixed across the batch. Because one optimisation step per action, noise re-sampled before every action.
  - NoisyNet-DQN loss: L̄(ζ) = E[ E_{(x,a,r,y)~D}[ r + γ max_b Q(y,b,ε';ζ⁻) − Q(x,a,ε;ζ) ]^2 ].
    Independent noise samples: ε (online), ε' (target), ε'' (action selection / greedy act). Independent ε,ε' avoid correlation bias between online and target.
  - NoisyNet-Dueling: double-DQN style target, b*(y)=argmax_b Q(y,b,ε'';ζ) (online net selects), evaluate with target.
- **A3C**: remove entropy bonus. Replace FC layers of policy network by noisy layers (independent Gaussian). Because on-policy & n-step returns, **noise must be fixed for the whole rollout** (∀i ε_i=ε) so Q̂_i is a consistent return estimate; resample after each optimisation step (every n steps).

## Why it works (in-frame rationale)
- State-dependent: weight noise propagates through the network so the induced action perturbation depends on the input — unlike ε-greedy's state-independent uniform action.
- Temporally consistent: noise held fixed between optimisation steps → coherent multi-step exploratory policy, not per-step jitter.
- Self-annealing: σ are learned; the network can drive σ→0 where it wants determinism (analysis shows last-layer Σ̄ decreases, but penultimate sometimes increases — problem-specific, not always toward deterministic). A deterministic optimiser of L(ζ) always exists since L positive continuous, so vanishing σ is feasible but not forced.
- Replaces a hand-tuned hyperparameter (ε schedule / entropy β) with a learned per-weight variance.

## Cost
- Doubles parameters in linear layers (μ and σ). But weights are an affine transform of noise; cost dominated by weight×activation matmul, so overhead marginal. Factorised noise keeps RNG cheap.

## Canonical implementation (Kaixhin/Rainbow model.py — widely used clean reimpl)
- std_init = 0.5 (=σ0).
- weight_mu, weight_sigma: shape (out, in). bias_mu, bias_sigma: (out,). epsilon buffers same shapes.
- reset_parameters: mu_range = 1/√in_features; weight_mu ~ U[-mu_range,mu_range]; weight_sigma = std_init/√in_features; bias_mu ~ U[-mu_range,mu_range]; bias_sigma = std_init/√out_features.
  - NOTE divergence from paper: paper sets σ_init = σ0/√p (p=inputs) for ALL params incl bias; Kaixhin uses √out_features for bias_sigma. Minor; flag.
- _scale_noise(size): x=randn(size); return x.sign() * x.abs().sqrt()  → this IS f(x)=sgn(x)√|x|.
- reset_noise: eps_in=_scale_noise(in), eps_out=_scale_noise(out); weight_epsilon = outer(eps_out, eps_in) [ger]; bias_epsilon = eps_out. → factorised.
- forward: training → F.linear(input, weight_mu + weight_sigma*weight_epsilon, bias_mu + bias_sigma*bias_epsilon); eval → use mu only (deterministic).
- Resample: call reset_noise() on all noisy layers each learn step.

## Design-decision table
| choice | why | rejected alt |
|---|---|---|
| perturb weights not actions | state-dependent + temporally consistent exploration | ε-greedy / entropy = decorrelated, state-independent |
| learnable σ | self-tuning, removes ε/β hyperparameter | fixed noise (Plappert parameter-space noise — constant Gaussian) |
| factorised noise for DQN | RNG cost pq→p+q, single-thread bottleneck | independent (used only where compute cheap: A3C) |
| f(x)=sgn(x)√|x| | keeps factor noise O(1)/unit-ish; product f(ε_i)f(ε_j) bounded magnitude | f=id (heavier-tailed product) |
| independent ε,ε' for online/target | avoid correlation bias in TD target | shared noise (biased) |
| fix noise per rollout (A3C) | on-policy consistency of n-step return estimate | resample per step (inconsistent Q̂) |
| eval uses μ only | greedy deterministic eval | sample at eval (noisy eval) |
