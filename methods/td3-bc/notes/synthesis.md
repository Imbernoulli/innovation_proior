# TD3+BC synthesis (grounded this run)

## Method identity
TD3+BC = "A Minimalist Approach to Offline Reinforcement Learning" (Fujimoto & Gu, NeurIPS
2021, arXiv:2106.06860). Confirmed from td3_bc.edit.py (alpha=2.5, lmbda = alpha/Q.abs().mean(),
twin critics, target policy smoothing, delayed updates) == canonical sfujim/TD3_BC.

## Problem
Offline RL: learn from a fixed dataset D, no environment interaction. Off-policy algos
(TD3/SAC) collapse offline due to extrapolation error: the critic queries Q at OOD actions
a=pi(s) absent from D, overestimates them, error backs up through the Bellman recursion, and
the actor (which maximizes Q) walks straight toward the over-valued OOD actions. Self-correcting
online (fresh data refutes it); not offline. Prior fixes constrain pi near the behavior policy
but pile on machinery (generative models, logsumexp sampling, gradient penalties, architecture
changes, actor pre-training, entropy removal, reward bonuses) -> many extra hyperparameters,
>2x runtime, and -- critically -- hard to validate offline (can't interact). The cost of an
extra knob is much higher offline. Diagnostic findings (pre-method facts about existing systems):
removing the implementation adjustments from CQL/Fisher-BRC drops their scores sharply; all
offline-trained policies (even minimal ones) have high episodic variance absent online.

## The method (two changes to TD3)
1. Add a BC regularizer to the deterministic-policy-gradient actor objective:
   pi = argmax_pi E_{(s,a)~D}[ lambda Q(s,pi(s)) - (pi(s)-a)^2 ].
2. Normalize each state feature over the dataset: s_i <- (s_i - mu_i)/(sigma_i + eps), eps=1e-3,
   applied to state and next_state.
lambda is a normalizer, not a fixed weight: lambda = alpha / ((1/N) sum |Q(s_i,a_i)|), alpha=2.5.
   - BC term <= 4 for actions in [-1,1]; Q range scales with reward scale -> normalize Q so one
     alpha works across tasks.
   - mean |Q| over the minibatch; Q detached (it's a scale, must not contribute gradient).
   - Also normalizes effective LR across tasks since grad_a Q scales with reward scale.

## Underlying TD3 (base, unchanged) -- from sfujim/TD3
- Twin critics Q1,Q2 (clipped double-Q): y = r + gamma(1-d) min(Q1',Q2')(s', a_tilde).
- Target policy smoothing: a_tilde = clip(pi_target(s') + clip(N(0,sigma), -c, c), -a_max, a_max),
  sigma=0.2*a_max, c=0.5*a_max.
- Delayed policy updates: actor + soft target updates every policy_freq=2 critic steps.
- Soft target tau=5e-3, gamma=0.99, Adam lr=3e-4, batch=256, 256x256 ReLU MLPs.

## Design-decision -> why (no holes)
- BC term = L2 (pi(s)-a)^2 not KL: no fundamental reason one divergence beats another; pick the
  simplest. L2 to the dataset action is one line, works with a deterministic policy, needs NO
  behavior model to fit (unlike BCQ/BEAR/BRAC which fit pi_beta). This is the whole minimalist bet.
- Add BC as a +regularizer on the SAME states/actions already sampled for the critic: free,
  reuses the minibatch, no extra forward passes.
- Normalize lambda by mean|Q| (not a fixed lambda): the RL term magnitude ~ |Q| ~ reward scale,
  BC term <= 4 fixed; an un-normalized lambda would need re-tuning per task. Dividing by mean|Q|
  pins the RL term to O(alpha) regardless of reward scale -> one alpha for all D4RL tasks.
- Detach Q in lambda: lambda is a scalar SCALE on the loss; differentiating through it would
  add a spurious gradient term. Only the -lambda*Q.mean() and the BC MSE carry gradient.
- alpha=2.5: sets the RL:BC ratio; ablation shows sensitivity at alpha=1 (too much imitation)
  and alpha=4 (too much RL), 2.5 is the robust middle. (Ablation is the proposed method's own
  result -> excluded from context/reasoning; the value and its role are kept.)
- State normalization (precompute exact mu,sigma): dataset is FIXED, so exact per-feature stats
  are computable once; equalizes feature scales for the MLP, stabilizes the policy. eps=1e-3
  floors zero-variance features. Cheap; complements (not core to) the BC term.
- Why minimal at all: offline each extra knob is costly to validate (no env to check it);
  minimalism -> fewer hyperparameters, lower compute, clean attribution of gains.

## Code (exact actor loss, grounded in TD3_BC.py)
  pi = actor(state); Q = critic.Q1(state, pi)
  lmbda = alpha / Q.abs().mean().detach()
  actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)   # minimize <=> maximize lambda*Q - (pi-a)^2
Critic loss = MSE(Q1,target)+MSE(Q2,target). State norm at buffer level (mean/std, eps=1e-3).

## Baselines for context.md (all prior art, grounded)
- BC (supervised imitation, Pomerleau 1991): copy dataset actions; ceilinged by data quality,
  no value reasoning, can't beat the data.
- Online off-policy actor-critic: DDPG (Lillicrap 2015) / TD3 (Fujimoto 2018) / SAC -- the base
  that collapses offline due to extrapolation error.
- BCQ / extrapolation error (Fujimoto 2019): conditional VAE behavior model + perturbation net.
- BEAR/BRAC (Wu 2019): fit pi_beta, constrain via MMD/KL penalty.
- CQL (Kumar 2020): conservative critic regularizer (push down OOD Q); logsumexp sampling,
  actor pre-training, max over sampled actions, architecture/lr changes, remove SAC entropy.
- Fisher-BRC (Kostrikov 2021): behavior model + offset critic + gradient penalty + reward bonus.
- AWAC / AWR (advantage-weighted regression): weighted BC, exp(A/beta) weights.
Each gap stated as observed limitation, NOT as a prescription of the fix.

## Evaluation settings (pre-method facts, NO outcomes)
D4RL benchmark (Fu et al. 2021) on OpenAI Gym MuJoCo: HalfCheetah, Hopper, Walker2d (+Ant),
dataset types random/medium/medium-replay/medium-expert/expert. Metric: D4RL normalized score
(0 random, 100 expert), averaged over the final evaluations and several seeds, 10 episodes each,
1M gradient steps, eval every 5000 steps. Wall-clock runtime as a secondary axis.
