I want the offline phase of reinforcement learning to hand the online phase something more useful than a high offline score. The setting is offline-to-online fine-tuning: train a policy and critic on a fixed dataset with no new interaction, then spend a small online budget improving that initialization. The online budget is short, so if the first few updates are consumed repairing the critic, the whole promise of pretraining is lost. The failure I need to explain is specific and counterintuitive — the strongest offline critic can make the policy worse immediately after interaction starts.

The conservative critic is the natural starting point, because it is built for exactly the offline danger. When an actor improves, it queries actions outside the dataset, and an ordinary critic overestimates those poorly-covered actions. Conservative Q-learning answers this by adding to the Bellman loss a regularizer that lowers values under policy-sampled actions and counterbalances it by raising values on dataset actions, $\alpha\,\big(\mathbb{E}_{s\sim D,\,a\sim\pi}\,Q(s,a) - \mathbb{E}_{(s,a)\sim D}\,Q(s,a)\big)$. In continuous control the first expectation is implemented as a temperature-scaled log-sum-exp over sampled candidate actions. Minimizing that term pushes down policy-sampled candidates while the dataset-action term keeps the observed actions from collapsing, and the actor is left with a pessimistic surface rather than an optimistic fantasy. The behavior-regularized and advantage-weighted alternatives — AWAC, TD3+BC, IQL — sidestep the worst extrapolation by staying near data-supported actions or using in-sample regression, but they tend to improve slowly when the online phase needs fast value-driven policy improvement; and pure online RL with the offline data merely in replay wastes its budget rediscovering behavior the dataset already demonstrates. So conservatism is the right ingredient, but it is not sufficient.

The reason is that pessimism has a magnitude, not only a sign. A value can be a correct lower estimate and still sit so far below any plausible return that it becomes a useless coordinate system for online learning. If the offline critic reports the current good policy as having value far below its actual returns, then the first online rollouts create Bellman targets on a much higher scale; the critic jumps upward to meet them; and during that jump some genuinely worse actions encountered online can momentarily look better than the depressed value of the pretrained policy, so the actor drifts away from the very initialization we paid for. The problem is not that the critic is conservative — it is that its conservative estimate is allowed to be arbitrarily low.

I propose Cal-QL, calibrated Q-learning. The idea is to demand two inequalities of the value estimate at once instead of one. On dataset states I want
$$V^\mu(s) \;\le\; \mathbb{E}_{a\sim\pi}\,Q_\theta^\pi(s,a) \;\le\; V^\pi(s).$$
The right inequality is the familiar conservative lower-estimate direction, which I still need against unsupported actions. The left inequality is new: it keeps the learned policy's estimated value above the value of a reference policy $\mu$ that I can trust to be no better than the learned policy — typically the behavior policy that generated the data. Holding the estimate above that reference is precisely what stops a worse online action from looking attractive merely because the old estimate was too depressed. The reference is the floor that calibrates the scale.

The reference must be computable from the offline data without bootstrapping, and the behavior policy supplies it directly: its value is the discounted return-to-go along each trajectory. For a transition at time $t$ this is $R_t = r_t + \gamma\,r_{t+1} + \gamma^2 r_{t+2} + \cdots$, obtained by a single reverse scan per episode until the terminal or episode boundary. This statistic is noisy but it never passes through a learned critic, so it is a clean Monte Carlo anchor. One subtlety matters in sparse-reward terminal domains: when a trajectory shows only the same failure reward until timeout, a short-horizon partial sum understates the true return-to-go, so for an all-failure sparse trajectory I substitute the constant $r_{\text{neg}}/(1-\gamma)$ — the value of receiving that same negative reward forever — rather than the truncated reverse-scan sum.

The decisive design choice is where to inject this scalar. The excessive downward pressure in CQL enters through exactly one place: the policy-action side of the conservative regularizer. So that is the only place I touch. I replace the candidate value inside the regularizer by $\max\!\big(Q_\theta(s,a),\,V^\mu(s)\big)$, giving the modified objective
$$\mathbb{E}_{s\sim D,\,a\sim\pi}\big[\max\!\big(Q_\theta(s,a),\,V^\mu(s)\big)\big] \;-\; \mathbb{E}_{(s,a)\sim D}\big[Q_\theta(s,a)\big].$$
When the candidate is above the reference, the CQL penalty is unchanged and pessimism operates exactly as before. When the candidate is already below the reference, the maximum makes the log-sum-exp see the constant floor instead, so it no longer emits a gradient that keeps dragging that candidate downward. The floor goes on the policy-sampled candidates — the current-policy actions at the current observations and the next-policy actions evaluated at the current observations — because those are the ones whose depressed values corrupt the scale of the policy we want to preserve. I deliberately do not floor the random-action candidates: those should remain free to be pushed down, since they are the out-of-distribution actions conservatism is meant to suppress. The dataset-action subtraction outside the log-sum-exp is left untouched as ordinary CQL.

I am careful not to overclaim the guarantee. In the tabular case, with a sufficiently large conservative coefficient, this masking prevents values from being pushed below the reference and can enforce the reference lower bound pointwise. With neural networks and sampled actions it is an optimization mechanism, not a pointwise certificate; what I can defend is the intended inequality in expectation over dataset states, and the exact loss behavior — the objective simply stops applying additional conservative pressure to any policy-action sample that already sits below its Monte Carlo reference. That is the whole insight, and it is small in code but specific in placement: not a new actor, not a new replay rule, not a new Bellman target, just a floor on the CQL push-down that keeps the offline critic conservative without letting its value scale become useless for online fine-tuning.

The rest is standard SAC+CQL machinery and must stay that way. The Bellman target uses twin target critics with clipped double-Q, optionally taking a max over several sampled next actions, and optionally subtracting the entropy term in the backup. The conservative term draws candidates from uniform random actions, current-policy actions, and next-policy actions, and applies importance corrections inside the log-sum-exp by subtracting $\log(0.5^{\,d})$ for the uniform samples of action dimension $d$ and the policy log-probabilities for the policy samples. Online fine-tuning does not silently become a different method: the same reference-floor flag is carried through both phases, offline and online batches are mixed by the chosen ratio, and online rollouts in the supported sparse-reward environments get their own Monte Carlo returns computed before they enter the replay buffer. The CQL weight may be changed at the transition and CQL itself may be disabled by configuration, but "turn off the floor online" is not a step of the method.

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
