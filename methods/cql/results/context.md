# Context: learning control policies from a fixed dataset, without further interaction

## Research question

We have a large, previously collected dataset of transitions — tuples `(s, a, r, s')`
logged while some behavior policy `π_β(a|s)` acted in an environment — and we want to extract
the best possible control policy from it *without ever interacting with the environment again*.
This is the offline (batch) reinforcement-learning setting. It matters because in robotics,
healthcare, recommendation, and autonomous driving, online interaction is expensive, slow, or
dangerous, whereas logged data already exists in abundance, exactly the regime where supervised
learning thrives. A solution would have to take an arbitrary logged dataset — possibly from a
mediocre or mixed set of policies — and return a policy that is at least as good as the data,
and ideally substantially better, with high confidence and no online tuning.

The central obstacle is *distributional shift between the data-collecting policy and the policy
we are trying to learn*. Standard value-based RL evaluates and improves a policy by bootstrapping:
the target for `Q(s,a)` uses `Q(s', a')` at the next state, where `a'` is chosen by the policy
being improved. When that policy is pushed toward actions the dataset never contains, the
Q-values it queries there are unconstrained by data, and any errors are baked into the targets and
amplified through repeated backups. Online RL repairs such errors automatically by trying the
action and observing its real return; offline RL has no such corrective feedback. So the problem a
solution must solve is: how to assign values to actions outside the data distribution in a way
that does not let optimistic errors compound, while still allowing genuine improvement over the
behavior policy.

## Background

**MDPs and value-based RL.** An MDP is `(S, A, T, r, γ)` with transition kernel `T(s'|s,a)`,
reward `r(s,a)` bounded by `R_max`, and discount `γ ∈ (0,1)`. The Q-function `Q^π(s,a)` is the
expected discounted return of taking `a` in `s` and following `π` thereafter; it is the fixed
point of the Bellman operator `B^π Q = r + γ P^π Q`, where `P^π Q(s,a) = E_{s'~T, a'~π}[Q(s',a')]`.
Q-learning instead iterates the Bellman optimality operator `B* Q(s,a) = r(s,a) + γ
E_{s'}[max_{a'} Q(s',a')]`. Actor-critic methods alternate *policy evaluation* — fit `Q` toward
`B^π Q` — and *policy improvement* — move `π` toward actions that maximize `Q`. With a dataset
`D` sampled from `d^β(s) π_β(a|s)`, the backup can only be a single-sample empirical Bellman
operator `B̂^π`, which backs up the one observed `s'` per transition.

**The diagnostic phenomenon.** Run a standard off-policy actor-critic or Q-learning algorithm on
a fixed dataset and the learned Q-values blow up. The mechanism is specific. In the evaluation
target `r + γ E_{a'~π}[Q(s',a')]`, the actions `a'` are drawn from the *current learned policy*
`π`, but `Q` was only ever fit on actions from `π_β`. Because `π` is trained to maximize `Q`, it
gravitates to out-of-distribution (OOD) actions where `Q` happens to be erroneously high; those
inflated values enter the target and are bootstrapped into the next iterate, and the cycle
diverges. Crucially this is *action* distribution shift only — the backup never queries `Q` at
OOD *states*, since `s, s'` always come from `D`. A second, subtler observation: even when the
policy is held close to the data, *function-approximation coupling* can make a neural Q-network
report high values at OOD actions. Define, at iteration `k`, the empirical gap
`Δ̂^k = E_{s,a~D}[ max_{a'~Unif} Q̂^k(s,a') − Q̂^k(s,a) ]`, an estimate of how much an OOD action
looks better than the in-data action. On narrow datasets (e.g. data from a near-deterministic
expert) `Δ̂^k` is observed to be *positive and to grow during training* even for methods that
constrain the policy, and this drift precedes an "unlearning" collapse of policy performance.
On broader datasets the effect is milder but present. The low state-action-marginal entropy of
narrow datasets is what makes the coupling severe; with high-entropy (near-uniform) data it
largely disappears. This phenomenon — erroneously optimistic Q-values at OOD actions, uncorrected
because there is no online feedback — is the problem any offline value-learning method must defeat.

**Useful facts that will matter.** Constrained optimization of an expectation plus an entropy or
KL regularizer over a probability simplex has a closed-form Boltzmann solution. Empirical Bellman
backups differ from true ones by an amount controlled by concentration of the reward and dynamics
estimates, scaling as `1/√|D(s,a)|`. The matrix `(I − γP^π)^{-1}` has all non-negative entries,
so an additive negative correction inside it stays negative after discounted accumulation —
the standard device for turning a per-step underestimate into a value-level lower bound.

## Baselines

**Behavior-cloning** simply imitates `π_β` by supervised learning of `a` given `s`. It is the
trivial floor: it never exceeds the data and degrades badly when the data is suboptimal or mixed.

**Off-policy actor-critic / Q-learning applied directly offline** (DQN/SAC-style with a static
replay buffer). Core idea: the recipe above with `B̂^π` or `B̂*`. The gap: with no online
correction, OOD-action overestimation in the bootstrap target compounds and the value estimates
diverge; this is the phenomenon described above and the reason naive offline RL fails.

**BCQ — Batch-Constrained Q-learning (Fujimoto et al., 2018).** Trains a generative model of the
behavior policy `π_β` (a conditional VAE), samples candidate actions from it, perturbs them mildly,
and only backs up / acts with actions that stay near the data. By construction the backup never
queries far-OOD actions. Gap: it depends on an accurate generative model of `π_β`; it is tied to
that model's coverage and can be overly restrictive; it does not regularize the Q-function itself.

**BEAR — support-constrained Q-learning (Kumar et al., 2019).** Constrains the learned policy to
the *support* of `π_β` using a sampled MMD penalty between `π` and `π_β`, and uses an ensemble of
Q-functions with a conservative aggregation to discount uncertain values. Gap: still requires
sampling from / estimating the behavior distribution; uncertainty estimates good enough for online
exploration are not calibrated enough for offline pessimism; and empirically, because it leaves the
Q-function unregularized, function-approximation coupling can still inflate `Q` at OOD actions
(`Δ̂^k > 0` and growing), so the support constraint on the *policy* does not prevent the
*value* from being corrupted.

**Policy-constraint methods via explicit divergences (BRAC, AWR, ABM; Wu et al. 2019, Peng et al.
2019, Siegel et al. 2020; Jaques et al. 2019).** Penalize or constrain a divergence (KL,
Wasserstein) between `π` and `π_β`, then back up only constrained actions or add a value penalty.
Gap: all need a separately estimated behavior model — hard with multi-source data — and the
constraint acts on the policy, not the Q-function, so it shares BEAR's failure mode and trades off
between being too conservative and being ineffective.

**SPIBB — Safe Policy Improvement with Baseline Bootstrapping (Laroche et al., 2017).** Bootstraps
with the behavior policy on state-actions seen too few times, and proves safe-policy-improvement
guarantees `J(π, M) ≥ J(π_β, M) − ζ` with `ζ` decaying in the per-state counts. Gap: relies on
counts / a behavior model; the framing is the natural yardstick for a *safe-improvement* guarantee.

**Uncertainty / optimism methods (Osband et al. 2016/2017; Jaksch et al. 2010).** Built for online
*exploration*, they construct pointwise upper-confidence bounds `~ 1/√n(s,a)` and act optimistically.
Offline, the analogue would be a pointwise *lower* bound (pessimism). Gap: the uncertainty sets are
not high-fidelity enough for the demands of offline RL, where a single overestimated OOD action is
fatal.

**Robust MDP / high-confidence improvement (Iyengar 2005; Thomas 2015).** Optimize against
worst-case dynamics within an uncertainty set. Gap: tends to be uniformly over-conservative,
underrating value everywhere rather than only at OOD actions.

The recurring gap across baselines: they constrain the *policy* and lean on an explicit estimate
of `π_β`, while leaving the *Q-function* free to misvalue OOD actions.

## Evaluation settings

The natural offline benchmarks are continuous-control locomotion and manipulation from D4RL —
gym-MuJoCo (halfcheetah, hopper, walker2d) with dataset compositions of varying quality
(`random`, `medium`, `medium-expert`, `mixed`/`medium-replay`), the AntMaze navigation tasks
that require stitching sub-optimal trajectories, the Franka Kitchen multi-stage manipulation
tasks, and the Adroit dexterous-hand tasks with human-demonstration data — plus discrete-action
offline Atari built from logged DQN-replay data at reduced data fractions (e.g. 1% and 10%). The
metric is the normalized, smoothed undiscounted return of the learned policy, averaged over seeds.
Behavior cloning is the baseline floor; the prior offline methods above are the comparison points.
A defining protocol constraint is the absence of a reliable validation signal: there is no online
rollout to select hyperparameters or a stopping point, so methods are expected to work with
fixed settings, and any model-selection must rely only on quantities computable from the dataset
(e.g. the predicted Q-values on dataset state-action pairs).

## Code framework

The primitives that already exist: an actor-critic skeleton in the style of an entropy-regularized
off-policy algorithm (a stochastic Gaussian actor, twin Q-critics with target networks, an
automatically-tuned temperature), and a replay buffer that here just wraps the static dataset. The
only thing missing is whatever modifies the critic so that it does not overvalue out-of-distribution
actions. That single slot is left empty below.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):           # stochastic tanh-Gaussian policy, as in standard SAC
    def forward(self, s, repeat=None):
        pass                      # returns (action, log_prob)

class Critic(nn.Module):          # Q(s, a) -> scalar; supports a [B, N, .] batch of actions
    def forward(self, s, a):
        pass

class OfflineActorCritic:
    def __init__(self, actor, critic_1, critic_2, target_critic_1, target_critic_2, ...):
        pass

    def _critic_regularizer(self, observations, actions, next_observations, q1_data, q2_data):
        # TODO: design the critic correction that prevents unsupported actions from
        # receiving spuriously large values.
        pass

    def _q_loss(self, observations, actions, next_observations, rewards, dones, alpha):
        q1 = self.critic_1(observations, actions)
        q2 = self.critic_2(observations, actions)
        # standard TD target: twin-Q min at next state under the current policy (+ entropy)
        next_actions, next_log_pi = self.actor(next_observations)
        target_q = torch.min(self.target_critic_1(next_observations, next_actions),
                              self.target_critic_2(next_observations, next_actions))
        target_q = target_q - alpha * next_log_pi
        td_target = rewards + (1.0 - dones) * self.discount * target_q.detach()
        td_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        reg = self._critic_regularizer(observations, actions, next_observations, q1, q2)
        return td_loss + reg

    def _policy_loss(self, observations, alpha):
        new_actions, log_pi = self.actor(observations)
        q = torch.min(self.critic_1(observations, new_actions),
                      self.critic_2(observations, new_actions))
        return (alpha * log_pi - q).mean()       # maximize E[Q - alpha*log pi]

    def train(self, batch):
        # critic step, then policy step, then soft target update, on dataset minibatches
        pass
```
