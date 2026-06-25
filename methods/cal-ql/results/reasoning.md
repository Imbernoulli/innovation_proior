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
Offline, that is the property I want: the actor sees a pessimistic surface rather than an optimistic
fantasy over actions it cannot check against data.

So why would such a critic hurt online? Pessimism has a magnitude, not only a sign. A value can be a
lower estimate and still be so far below any plausible return that it becomes a bad coordinate system
for online learning. Let me make this concrete before I trust the intuition. Take an AntMaze-style
sparse task, `gamma = 0.99`, where a success trajectory earns reward `1` only at the goal and `0`
elsewhere. The true return-to-go at a transition `k` steps before the goal is `gamma^k`, so the
discounted value of the good pretrained policy near the start of a five-step approach is about
`0.99^4 = 0.96`. Now suppose the offline CQL coefficient was strong enough to drag this state's
estimated policy value down to, say, `-120`. The moment online rollouts arrive, they generate Bellman
targets on the true scale near `+1`, and the critic must climb roughly 120 units to reach them. During
that climb the depressed value of the pretrained policy is below the value the critic temporarily
assigns to mediocre online actions, so the actor has every incentive to walk away from a good
initialization before the critic has caught up. The problem is not that the critic is conservative; the
problem is that its conservative estimate is allowed to be arbitrarily low.

So the value estimate needs two inequalities at once. It should remain below the learned policy's true
value, because I still need the CQL safety direction. But it also should not fall below the value of a
reference policy that I can trust. If the reference policy is worse than the learned policy, then keeping
the learned policy's estimated value above that reference prevents a worse online behavior from
appearing attractive just because the old estimate was too depressed. The condition I want is
`V^mu(s) <= E_{a~pi} Q_theta^pi(s,a) <= V^pi(s)` on the dataset states. The right side is the conservative
lower-bound idea; the left side is the new scale condition.

The reference has to be computable from the offline data. The behavior policy is the obvious candidate:
it generated the trajectories, and I can estimate its value without bootstrapping by taking discounted
return-to-go along each trajectory. A reverse scan gives me a per-transition scalar. Let me trace the
scan on the success trajectory above, rewards `[0,0,0,0,1]` with the terminal on the last step. Going
backwards: the running value starts at `0`, becomes `1` at the goal step, then `0.99`, `0.99^2`,
`0.99^3`, `0.99^4`, i.e. `[0.96, 0.97, 0.98, 0.99, 1.0]`. That matches the discounted goal-reward
exactly, so the reverse scan reproduces the behavior return I expect.

There is one trajectory shape where the naive scan misleads me, and I want to catch it before it
contaminates the floor. In an all-failure sparse trajectory the agent keeps receiving the same negative
reward `r_neg` until the episode-length timeout cuts it off — but the timeout is an artifact of the
logger, not a real terminal; the behavior policy would have kept failing forever. If I just reverse-scan
the logged window, I compute a truncated geometric sum. For `r_neg = -1`, `gamma = 0.99`, a 100-step
window gives `sum_{k<100} gamma^k (-1) = -63.4`, whereas the true forever-failing value is
`r_neg/(1-gamma) = -100`. The truncated number is far less negative, so it would raise the reference
floor too high and start propping up states that the behavior policy genuinely cannot escape — exactly
the overestimation I am trying to avoid. So in the all-failure case I use the constant
`r_neg / (1 - gamma)` rather than the partial sum. (A 700-step window already gives `-99.91`, confirming
the constant is just the limit the scan is converging to.) This gives a cheap empirical reference value
for each sampled state without inheriting the timeout artifact.

Now I need to put that scalar exactly where the excessive downward pressure enters. In CQL, the
push-down part is the policy-action side of the conservative regularizer. So I try masking that pressure
when the sampled policy-action value is already below the reference: replace the candidate value inside
the regularizer by `max(Q_theta(s,a), reference(s))`. The claim I am leaning on is that this `max` stops
the regularizer from dragging a sub-reference candidate any lower, so I should check that the gradient
actually behaves that way rather than just assert it. Take a floor `V^mu = -50` and a policy candidate
whose Q has been pushed to `-120`. The regularizer minimizes a log-sum-exp of candidates, so each
candidate feels a downward gradient. Differentiating `max(Q, -50)` at `Q = -120`: the active branch is
the constant `-50`, so `d/dQ = 0` — the candidate receives no push-down. At `Q = -10`, above the floor,
the active branch is `Q` itself and `d/dQ = 1` — ordinary CQL pressure. I also want to be sure the floor
does not accidentally silence the un-floored candidates sharing the same log-sum-exp. Putting a floored
policy candidate at `-120` and an un-floored random candidate at `-30` into the same
`logsumexp([q_rand, max(q_cur, -50)])` and backpropagating, the floored policy candidate gets gradient
`0` while the random candidate still gets a nonzero push-down. So the mask is local: it removes
conservative pressure only from the policy-action samples that already sit below the Monte Carlo
reference, and leaves everything else as standard CQL. That is the behavior I wanted from the floor, and
now I have actually watched it happen rather than hoped for it.

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
log-sum-exp. The floor goes on the current-policy and next-policy candidates; I deliberately leave the
random-action candidates un-floored, since the gradient check above showed the floored and un-floored
candidates coexist cleanly in the same log-sum-exp, and the random candidates are precisely the
unsupported actions I still want pessimism to suppress freely.

Online fine-tuning should not silently change the method into something else. I keep the same
reference-floor flag through both phases when it is enabled, mix offline and online samples according
to the chosen mixing ratio, and compute Monte Carlo returns for online rollouts before adding them to
the replay buffer in the supported sparse-reward environments. I can change the CQL weight at the
transition, and I can optionally turn off CQL itself, but I do not make "disable the floor online" a
separate step of the method.

Putting the pieces together: I keep CQL because I still need pessimism against unsupported actions, but
I replace the policy-action values inside the conservative regularizer by `max(Q, reference_return)` so
pessimism cannot drive the policy estimate below a behavior-reference scale. The change is small in code
but specific in placement — it is not a new actor, not a new replay rule, and not a new Bellman target,
but a mask on the CQL push-down pressure. With the floor calibrating the conservative critic to the
return scale, the depressed-value failure I started from no longer appears: the pretrained policy's
estimate stays on a scale the first online targets can meet, so the critic does not have to climb out of
a 120-unit hole before the actor stops abandoning its initialization. That is the property I was trying
to buy.

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
