# Context

## Research question

Offline-to-online reinforcement learning starts with a fixed dataset, trains a policy and value
function without new interaction, then spends a small online budget improving that initialization.
The central failure is not simply low offline return. A method can produce a strong offline policy and
still be a bad starting point once real interaction begins: early online updates can degrade the policy
before recovery. The question is what property the offline value estimate needs so that online
fine-tuning keeps the useful initialization and improves quickly.

## Background

**Conservative value learning.** In offline RL, a learned actor can exploit Q-values at actions that are
poorly covered by the dataset. Conservative Q-learning addresses this by adding a critic regularizer
that lowers values under policy-sampled actions and counterbalances this by raising values on dataset
actions:

`alpha * (E_{s~D,a~pi} Q(s,a) - E_{(s,a)~D} Q(s,a)) + Bellman error`.

In continuous-action SAC implementations, the first expectation is approximated by a
temperature-scaled log-sum-exp over sampled candidate actions, usually random actions, current-policy
actions, and next-state policy actions, with importance corrections. This gives a pessimistic value
estimate and helps prevent offline overestimation.

**SAC actor-critic machinery.** The common implementation uses a stochastic Tanh-Gaussian actor,
twin Q-functions, a clipped double-Q Bellman target, Polyak target updates, and automatic entropy
tuning with target entropy set to the negative action dimension. The critic may score multiple sampled
actions per state so that the same networks can support max-target backups and the conservative
log-sum-exp term.

**Return-to-go statistics.** A trajectory in a fixed dataset also carries direct Monte Carlo information:
for a transition at time `t`, the discounted future reward
`R_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...` is available by a reverse scan until the terminal or
episode boundary. This statistic is noisy but does not bootstrap through a learned critic.

## Baselines

**Conservative offline pretraining followed by online updates.** This starts from a pessimistic critic
and continues training with newly collected data. It can keep offline learning stable, but the learned
values may sit on a scale far below actual returns. When online data arrives, the critic must rapidly
change scale, and policy updates during that transient can move away from the useful initialization.

**Behavior-regularized and advantage-weighted methods.** Methods such as AWAC, TD3+BC, and IQL
avoid some severe value-extrapolation failures by keeping the actor near data-supported actions or by
using in-sample value regression. They can fine-tune stably, but they may improve slowly when the
online phase needs fast value-driven policy improvement.

**Online RL with offline data in replay.** SAC-style online training can mix old and new samples and
adapt quickly after enough interaction. Without a strong offline initialization, though, it spends online
samples discovering behavior that the dataset already demonstrated.

## Evaluation settings

The relevant benchmark pattern is fixed offline pretraining followed by a limited online budget. The
important curves are the score immediately after pretraining, the first part of online fine-tuning where
policy degradation can appear, final performance, and cumulative regret over the online phase. Typical
domains include sparse-reward AntMaze, dexterous Adroit tasks, Franka Kitchen, and diagnostic robotic
manipulation settings where the data distribution is narrow enough for value-scale errors to matter.

## Code framework

The available scaffold is a SAC+CQL training loop: replay buffers, actor and twin critics, multi-action
sampling, max-target backup, conservative log-sum-exp, automatic entropy tuning, and offline/online
batch mixing. The missing implementation slots are narrow:

```python
def discounted_returns_by_episode(rewards, terminals, discount):
    # reverse-scan each episode without bootstrapping through Q
    pass


def conservative_q_loss(critic1, critic2, actor, target1, target2, batch, hp):
    # Bellman loss plus CQL log-sum-exp over sampled actions
    pass


def offline_online_batch(offline_buffer, online_buffer, batch_size, mixing_ratio):
    # sample a fixed or data-size-dependent offline/online mixture
    pass
```
