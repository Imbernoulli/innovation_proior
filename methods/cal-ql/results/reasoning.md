I want the offline phase to hand online learning something more useful than a high offline score. The
online phase is short, so if the first few updates spend their budget repairing the critic, the whole
promise of pretraining is gone. The failure I need to explain is that the strongest offline critic can
make the policy worse right after interaction starts.

The conservative critic is the natural starting point. It is designed for exactly the offline danger:
policy improvement can query actions outside the dataset, and an ordinary critic can overestimate
those actions. So I add a CQL penalty to the Bellman loss,
`alpha * (E_{s~D,a~pi} Q(s,a) - E_{(s,a)~D} Q(s,a))`, and in continuous control I implement the first
expectation with a log-sum-exp over sampled actions. Minimizing that term pushes down the
policy-sampled candidates while the dataset-action term keeps the observed actions from collapsing.
That is why it works offline. It gives the actor a pessimistic surface instead of an optimistic fantasy.

But pessimism has a magnitude, not only a sign. A value can be a lower estimate and still be so far
below any plausible return that it becomes a bad coordinate system for online learning. If the offline
critic says that the current good policy has value far below the actual returns, then the first online
rollouts create Bellman targets on a much higher scale. The critic then jumps upward. During that jump,
some worse actions encountered online can look better than the depressed value of the pretrained
policy, so the actor can move away from the initialization before the critic has settled. The problem is
not that the critic is conservative; the problem is that its conservative estimate is allowed to be
arbitrarily low.

So the value estimate needs two inequalities at once. It should remain below the learned policy's true
value, because I still need the CQL safety direction. But it also should not fall below the value of a
reference policy that I can trust. If the reference policy is worse than the learned policy, then keeping
the learned policy's estimated value above that reference prevents a worse online behavior from
appearing attractive just because the old estimate was too depressed. The condition I want is
`V^mu(s) <= E_{a~pi} Q_theta^pi(s,a) <= V^pi(s)` on the dataset states. The right side is the conservative
lower-bound idea; the left side is the new scale condition.

The reference has to be computable from the offline data. The behavior policy is the obvious candidate:
it generated the trajectories, and I can estimate its value without bootstrapping by taking discounted
return-to-go along each trajectory. For terminal sparse-reward domains, a reverse scan gives me a
per-transition scalar. In all-failure sparse trajectories, I use the constant
`r_neg / (1 - gamma)` rather than a short-horizon partial sum, because the trajectory only shows the
same failure reward until timeout. This gives a cheap empirical reference value for each sampled state.

Now I need to put that scalar exactly where the excessive downward pressure enters. In CQL, the
push-down part is the policy-action side of the conservative regularizer. So I mask that pressure when
the sampled policy-action value is already below the reference: replace the candidate value inside the
regularizer by `max(Q_theta(s,a), reference(s))`. If the candidate is above the reference, the CQL
penalty is unchanged. If it is below, the log-sum-exp no longer receives a gradient that keeps dragging
that candidate downward. The dataset-action subtraction stays as ordinary CQL, and random-action
candidates can still be pushed down freely in the practical implementation. The policy-sampled current
and next action candidates are the ones that get the floor.

I should not overstate the theorem. With a tabular critic and a sufficiently strong conservative
coefficient, this masking can enforce the lower reference bound where the value would otherwise be
below the reference. With neural networks and sampled actions, it is an optimization mechanism, not a
pointwise guarantee. What I can defend is the intended inequality in expectation over dataset states and
the exact implementation: the loss stops applying additional conservative pressure to policy-action
samples that already sit below the Monte Carlo reference.

The resulting critic loss is still a SAC+CQL critic loss. The Bellman target uses twin target critics,
optionally with a max over several next actions. The CQL term samples random actions, current-policy
actions at the current observations, and policy actions sampled at next observations but evaluated at
the current observations. With importance sampling on, the
log-sum-exp concatenates `Q_rand - log uniform_density`, `Q_next_policy - log pi_next`, and
`Q_current_policy - log pi_current`; the dataset action appears as the subtraction outside the
log-sum-exp.

Online fine-tuning should not silently change the method into something else. I keep the same
reference-floor flag through both phases when it is enabled, mix offline and online samples according
to the chosen mixing ratio, and compute Monte Carlo returns for online rollouts before adding them to
the replay buffer in the supported sparse-reward environments. I can change the CQL weight at the
transition, and I can optionally turn off CQL itself, but I do not make "disable the floor online" a
separate step of the method.

Putting the pieces together, the method is calibrated Q-learning: start from CQL because I still need
pessimism against unsupported actions, but replace the policy-action values inside the conservative
regularizer by `max(Q, reference_return)` so pessimism cannot drive the policy estimate below a
behavior-reference scale. The insight is small in code but specific in placement. It is not a new actor,
not a new replay rule, and not a new Bellman target. It is a mask on the CQL push-down pressure that
keeps the offline critic conservative without letting its value scale become useless for online
fine-tuning.

```python
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
```

```python
def cql_q_loss(critic1, critic2, target1, target2, actor, batch, hp, enable_calql):
    s, a, r, s2, done, mc = batch
    q1, q2 = critic1(s, a), critic2(s, a)

    a2, logp2 = actor(s2, repeat=hp["n_actions"])
    tq = torch.min(target1(s2, a2), target2(s2, a2))
    if hp["max_target_backup"]:
        idx = tq.argmax(dim=-1, keepdim=True)
        tq = tq.gather(1, idx).squeeze(-1)
        logp2 = logp2.gather(1, idx).squeeze(-1)
    else:
        tq = tq.squeeze(-1)
    if hp.get("backup_entropy", False):
        tq = tq - hp["entropy_alpha"] * logp2
    target = (r + (1.0 - done) * hp["discount"] * tq).detach()
    bellman = F.mse_loss(q1, target) + F.mse_loss(q2, target)

    rand = a.new_empty((a.shape[0], hp["n_actions"], a.shape[-1])).uniform_(-1.0, 1.0)
    cur_a, cur_lp = actor(s, repeat=hp["n_actions"])
    nxt_a, nxt_lp = actor(s2, repeat=hp["n_actions"])
    cur_a, cur_lp = cur_a.detach(), cur_lp.detach()
    nxt_a, nxt_lp = nxt_a.detach(), nxt_lp.detach()
    q1_rand, q2_rand = critic1(s, rand), critic2(s, rand)
    q1_cur, q2_cur = critic1(s, cur_a), critic2(s, cur_a)
    q1_nxt, q2_nxt = critic1(s, nxt_a), critic2(s, nxt_a)

    if enable_calql:
        lb = mc.reshape(-1, 1).repeat(1, q1_cur.shape[1])
        q1_cur, q2_cur = torch.maximum(q1_cur, lb), torch.maximum(q2_cur, lb)
        q1_nxt, q2_nxt = torch.maximum(q1_nxt, lb), torch.maximum(q2_nxt, lb)

    random_density = np.log(0.5 ** a.shape[-1])
    cat1 = torch.cat([q1_rand - random_density, q1_nxt - nxt_lp, q1_cur - cur_lp], dim=1)
    cat2 = torch.cat([q2_rand - random_density, q2_nxt - nxt_lp, q2_cur - cur_lp], dim=1)
    ood1 = torch.logsumexp(cat1 / hp["temp"], dim=1) * hp["temp"]
    ood2 = torch.logsumexp(cat2 / hp["temp"], dim=1) * hp["temp"]
    diff1 = torch.clamp(ood1 - q1, hp["clip_min"], hp["clip_max"]).mean()
    diff2 = torch.clamp(ood2 - q2, hp["clip_min"], hp["clip_max"]).mean()
    return bellman + hp["cql_alpha"] * (diff1 + diff2)
```
