# Cal-QL: Calibrated Q-Learning

## Method

Start with CQL, whose critic objective is

`alpha * (E_{s~D,a~pi} Q_theta(s,a) - E_{(s,a)~D} Q_theta(s,a)) + Bellman error`.

CQL gives useful offline pessimism, but the learned values can be far below any realistic return. For
offline-to-online fine-tuning, the desired value shape is

`V^mu(s) <= E_{a~pi} Q_theta^pi(s,a) <= V^pi(s)`.

The right inequality is the conservative lower-estimate direction. The left inequality keeps the
learned policy's estimated value above a worse reference policy `mu`, typically the behavior policy.

Cal-QL changes only the policy-action side of the conservative regularizer:

`E_{s~D,a~pi}[max(Q_theta(s,a), V^mu(s))] - E_{(s,a)~D}[Q_theta(s,a)]`.

In code, `V^mu(s)` is carried as a Monte Carlo return-to-go scalar from the sampled transition. The
floor is applied to current-policy and next-policy candidate Q-values before the CQL log-sum-exp. It
is not applied to random-action candidates, and the dataset-action subtraction remains unchanged.

## Guarantees

Calibration is an expected value condition over dataset states, not a claim that every neural-network
Q-value is pointwise correct. In the tabular case, with a sufficiently large CQL coefficient, the
modified regularizer prevents values below the reference from being pushed farther down and can
enforce the reference lower bound. With function approximation, it is the practical mechanism used to
keep conservative values on the return scale and avoid the initial online unlearning caused by
severely depressed Q-values.

## Canonical Implementation

The implementation keeps the standard SAC+CQL backbone:

- twin critics and clipped double-Q Bellman targets;
- optional max-target backup over `cql_n_actions` sampled next actions;
- CQL candidates from random actions, current-policy actions, and next-state policy actions evaluated
  at current observations;
- importance correction by subtracting `log(0.5 ** action_dim)` for uniform random actions and
  policy log probabilities for policy samples;
- online fine-tuning with an offline/online batch mixture.

When `enable_calql=True`, the implementation applies
`maximum(policy_candidate_q, mc_return)` to current and next policy-action candidates. The flag is
not canonically disabled at the online transition; online trajectories in the supported sparse-reward
environments are added with their own Monte Carlo returns. The CQL weight may be changed at the
transition, and CQL itself may be disabled by configuration, but that is separate from the Cal-QL
floor.

## Reference Code

```python
import numpy as np
import torch
import torch.nn.functional as F


def calc_return_to_go(rewards, terminals, gamma, reward_neg=None, sparse=False):
    rewards = np.asarray(rewards, dtype=np.float32)
    terminals = np.asarray(terminals, dtype=np.float32)
    if rewards.size == 0:
        return rewards
    if sparse and reward_neg is not None and np.all(rewards == reward_neg):
        return np.full_like(rewards, reward_neg / (1.0 - gamma), dtype=np.float32)
    out = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in range(rewards.size - 1, -1, -1):
        running = rewards[i] + gamma * running * (1.0 - terminals[i])
        out[i] = running
    return out


def calql_cql_loss(critic1, critic2, target1, target2, actor, batch, hp, enable_calql):
    obs, act, rew, next_obs, done, mc = batch
    q1_data, q2_data = critic1(obs, act), critic2(obs, act)

    next_act, next_logp = actor(next_obs, repeat=hp["n_actions"])
    target_q = torch.min(target1(next_obs, next_act), target2(next_obs, next_act))
    if hp["max_target_backup"]:
        idx = target_q.argmax(dim=-1, keepdim=True)
        target_q = target_q.gather(1, idx).squeeze(-1)
        next_logp = next_logp.gather(1, idx).squeeze(-1)
    else:
        target_q = target_q.squeeze(-1)
    if hp.get("backup_entropy", False):
        target_q = target_q - hp["entropy_alpha"] * next_logp
    td = (rew + (1.0 - done) * hp["discount"] * target_q).detach()
    bellman = F.mse_loss(q1_data, td) + F.mse_loss(q2_data, td)

    rand_act = act.new_empty((act.shape[0], hp["n_actions"], act.shape[-1])).uniform_(-1.0, 1.0)
    cur_act, cur_logp = actor(obs, repeat=hp["n_actions"])
    nxt_act, nxt_logp = actor(next_obs, repeat=hp["n_actions"])
    cur_act, cur_logp = cur_act.detach(), cur_logp.detach()
    nxt_act, nxt_logp = nxt_act.detach(), nxt_logp.detach()

    q1_rand, q2_rand = critic1(obs, rand_act), critic2(obs, rand_act)
    q1_cur, q2_cur = critic1(obs, cur_act), critic2(obs, cur_act)
    q1_nxt, q2_nxt = critic1(obs, nxt_act), critic2(obs, nxt_act)

    if enable_calql:
        lower = mc.reshape(-1, 1).repeat(1, q1_cur.shape[1])
        q1_cur, q2_cur = torch.maximum(q1_cur, lower), torch.maximum(q2_cur, lower)
        q1_nxt, q2_nxt = torch.maximum(q1_nxt, lower), torch.maximum(q2_nxt, lower)

    random_density = np.log(0.5 ** act.shape[-1])
    cat1 = torch.cat([q1_rand - random_density, q1_nxt - nxt_logp, q1_cur - cur_logp], dim=1)
    cat2 = torch.cat([q2_rand - random_density, q2_nxt - nxt_logp, q2_cur - cur_logp], dim=1)
    ood1 = torch.logsumexp(cat1 / hp["temp"], dim=1) * hp["temp"]
    ood2 = torch.logsumexp(cat2 / hp["temp"], dim=1) * hp["temp"]

    cql1 = torch.clamp(ood1 - q1_data, hp["clip_min"], hp["clip_max"]).mean()
    cql2 = torch.clamp(ood2 - q2_data, hp["clip_min"], hp["clip_max"]).mean()
    return bellman + hp["cql_alpha"] * (cql1 + cql2)
```
