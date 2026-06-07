# TD3+BC synthesis

## Problem
Offline RL: learn a policy from a fixed dataset D, no environment interaction. Off-policy
algorithms (TD3/SAC) collapse offline because of extrapolation error: the critic queries Q on
out-of-distribution actions a=π(s) not in D, overestimates them, and the policy chases that
overestimation. Prior fixes (BCQ, BEAR, BRAC, CQL, Fisher-BRC) constrain the policy near the
behavior policy, but pile on machinery (generative models, logsumexp sampling, gradient penalties,
architecture changes, actor pre-training, entropy-term removal, reward bonuses) → many extra
hyperparameters and >2x runtime, hard to tune offline (can't validate by interacting).

## The method (two changes to TD3)
1. Add a BC regularizer to the deterministic-policy-gradient actor update:
   π = argmax_π E_{(s,a)~D}[ λ Q(s,π(s)) − (π(s)−a)^2 ]
2. Normalize each state feature over the dataset: s_i ← (s_i − μ_i)/(σ_i + ε), ε=1e-3.

λ is a normalizer (not a fixed weight): λ = α / ( (1/N) Σ_{(s_i,a_i)} |Q(s_i,a_i)| ), α=2.5.
- BC term max ≈ 4 for actions in [-1,1]; Q range scales with reward scale → must normalize Q.
- λ uses mean |Q| over the minibatch; Q not differentiated through (detached); used to scale loss.
- This also normalizes the effective LR across tasks since ∇_a Q scales with reward scale too.

## Underlying TD3 (the base, unchanged) — from sfujim/TD3
- Twin critics Q1,Q2 (clipped double-Q): target = r + γ(1-d) min(Q1',Q2')(s', ã)
- Target policy smoothing: ã = clip(π_target(s') + clip(N(0,σ),-c,c), -a_max, a_max), σ=0.2,c=0.5
- Delayed policy updates: update actor + soft-update targets every policy_freq=2 critic steps
- Soft target update τ=5e-3, γ=0.99, Adam lr=3e-4, batch=256
- Actor: 2 hidden layers 256, ReLU, tanh*max_action. Critic: 2x(256,256,ReLU) MLP.

## Code (TD3_BC.py) exact actor loss
  pi = actor(state); Q = critic.Q1(state, pi)
  lmbda = alpha / Q.abs().mean().detach()
  actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)   # minimize  ⇔  maximize λQ - (π-a)^2
Critic loss = MSE(Q1,target)+MSE(Q2,target). State normalization done at buffer level (mean/std).

## Design-decision → why
- Why MSE BC (not KL): no fundamental reason one divergence beats another; pick the simplest, an
  L2 to dataset action → one line, deterministic policy, no behavior-model to fit.
- Why normalize λ by mean|Q|: decouple the RL/BC balance from reward scale; one α works across tasks.
- Why detach Q in λ: λ is a scale, must not contribute gradient.
- Why state normalization: dataset is fixed → can precompute exact statistics; stabilizes the policy.
- Why minimal: in offline RL each extra knob is costly to validate (no env); want attribution.

## Baselines for context.md
- BC (supervised imitation), BCQ/extrapolation error (Fujimoto 2018), BEAR, BRAC (Wu 2019),
  CQL (Kumar 2020, conservative critic), Fisher-BRC (Kostrikov 2021, behavior model + offset),
  AWAC (advantage-weighted). DDPG/TD3 as the online base. D4RL benchmark (Gym MuJoCo), v0.
