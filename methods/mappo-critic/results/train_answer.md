I have a team of $n$ agents acting in a partially observable cooperative task — a DEC-POMDP with one shared reward $R(s,A)$ for the joint action $A=(a_1,\dots,a_n)$ — and I want to train them with plain on-policy policy gradients, because PPO is conceptually clean and I already trust it in the single-agent case. The trouble is that everyone believes on-policy methods are sample-inefficient here, and the cooperative benchmarks are won by off-policy machinery: value-decomposition $Q$-learning (VDN, QMIX) and centralized-$Q$ actor-critics (MADDPG, COMA). So if a simple PG method is going to be competitive, the win cannot come from a clever new objective; it has to come from one specific place. The policy gradient $\mathbb{E}[\sum_t \nabla\log\pi(a^t|o^t)\,G_t]$ has brutal variance because $G_t$ is a Monte-Carlo return, and the standard cure is to subtract a state-dependent baseline and form an advantage. Folding that into GAE, with $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ and $\hat A_t = \sum_{l\ge 0}(\gamma\lambda)^l\,\delta_{t+l}$, every term of the advantage is an *error of the value function* $V$. So the variance of my gradient is, term by term, controlled by how good $V$ is. The value function — the critic — is the lever, and "get it right" means drive down its variance contribution.

The crack worth prying open is that the actor must read only $o_i$ (that is all it has at execution time), but the critic is used *only during training*, in a simulator, and is discarded afterward. The baseline-correctness identity $\mathbb{E}[\nabla\log\pi(a|s)\,b(s)]=0$ says the baseline may depend on anything independent of the action given the state — it never said it had to be restricted to what the actor sees. This is exactly the CTDE principle: partial observability binds at execution, not at training, so a training-only baseline may use the global state the policy cannot. The actor stays decentralized; the critic goes centralized. The prior centralized methods, though, all over-answer the question by conditioning the critic on *actions*: MADDPG's per-agent $Q_i(x,a_1,\dots,a_n)$ pays with an input that grows with $n$ and a critic coupled to every agent's current policy; COMA's single $Q(s,A)$ plus counterfactual baseline must be re-evaluated once per candidate action of agent $i$; VDN and QMIX factorize a joint $Q$ and pay with a restricted representable class (additivity, monotonicity) — and all are off-policy, dragging replay buffers and target networks. But for my purpose I never need a $Q$: GAE needs only the *state value* $V(s_t)$ to form its residuals. So a centralized $V(s)$ buys the variance reduction while shedding all of that — no joint-action input, no scaling with $n$, no action enumeration, no factorization constraint.

I propose MAPPO (Multi-Agent PPO): single-agent PPO applied to the cooperative DEC-POMDP with exactly one thing changed, a centralized state-value critic $V_\phi$ used as the GAE baseline, while the decentralized actor $\pi_\theta(a_i|o_i)$ is untouched. The defining objectives are the standard clipped surrogate for the actor and a clipped value loss for the critic, summed over valid batch-time-agent positions $(b,t,i)\in M$ with importance ratio $r_{\theta,bti}=\pi_\theta(a_{bti}|o_{bti})/\pi_{\theta_{old}}(a_{bti}|o_{bti})$, entropy bonus $S$ and coefficient $\sigma$:
$$L(\theta) = \frac{1}{|M|}\sum_{(b,t,i)\in M}\min\!\big(r_{\theta,bti}\,\hat A_{bti},\;\mathrm{clip}(r_{\theta,bti},1-\epsilon,1+\epsilon)\,\hat A_{bti}\big)\;+\;\sigma\,\frac{1}{|M|}\sum_{(b,t,i)\in M}S[\pi_\theta(o_{bti})],$$
$$L(\phi) = \frac{1}{|M|}\sum_{(b,t,i)\in M}\max\!\big((V_\phi(x_{bti})-\hat R_{bti})^2,\;(V_{clip}-\hat R_{bti})^2\big),\quad V_{clip}=V_{old}(x_{bti})+\mathrm{clamp}\big(V_\phi(x_{bti})-V_{old}(x_{bti}),-\epsilon,\epsilon\big),$$
with $\hat R_{bti}$ the discounted reward-to-go and $\hat A$ the GAE advantage. The clip on the value mirrors the clip on the policy: neither $V$ nor $\pi$ may jump too far from the data-collecting batch in a single update.

The real design content is what $x_i$, the critic's input, looks at — and a diagnostic clue forces the choice. Independent local-only PPO (IPPO) is reported to be *surprisingly strong* on hard SMAC maps, sometimes beating centralized PPO that uses the environment's global state. That is backwards: a critic that sees strictly more should never produce a worse baseline, so the centralization must be feeding the wrong thing. Trace it. The environment-provided global state (EP), $x_i=s$, is one agent-agnostic vector — positions, health, shield, cooldown — but being the same for every agent it drops agent-specific local features (the agent's id, its available actions, its relative distances within sight). The value of a situation for agent $i$ often depends on exactly those features, so a critic on EP is systematically wrong in a way that varies by agent, and the GAE residuals inherit that as bias and variance; it can genuinely be worse than a local critic that at least sees $o_i$. The alternative, concatenating all local observations (CL), $x_i=(o_1,\dots,o_n)$, restores every local feature but its dimension grows linearly with $n$ (enormous on a 27-agent map, so value learning gets sample-hungry) and, since it is still a stack of observations, it can miss truly global structure no agent observes. So I stop choosing and concatenate both: the Agent-Specific (AS) global state $x_i=\mathrm{concat}(s,o_i)$ gives the critic the comprehensive global picture *and* the local features EP was dropping, without paying CL's full all-observations cost. The agent-specific component has a second payoff that resolves a decision I had left open. The agents are homogeneous, so I want parameter sharing — one critic network trained by every agent's data — but a single shared network fed the same $s$ for all agents can only emit one value, identical across agents. Because each agent feeds in *its own* $o_i$ (carrying its id and local features), one shared network produces a *different* value per agent. "Agent-specific input" and "share parameters" are the same insight from two sides: one cheap network that still distinguishes agents. Stripped to its minimum, the agent-specific signal is just an identity — append the one-hot $e_i$ to $s$, $x_i=\mathrm{concat}(s,e_i)$ — which is the cleanest thing to code and what the shared critic uses by default. One refinement: if $s$ and $o_i$ overlap (in SMAC they share enemy health, positions, and more), the AS concatenation double-counts those features, inflating the input dimension for no information gain; the Feature-Pruned (FP) variant $x_i=\mathrm{concat}(s,\mathrm{prune}(o_i))$ keeps $s$ and appends only the parts of $o_i$ not already in $s$ — same information, smaller input. The principle behind both is one sentence: give the value function both global and agent-specific local features, and carry no redundant dimensions. The network itself is deliberately generic — a shared MLP, two hidden layers of width 64 with ReLU and orthogonal initialization, ending in a scalar head — because the architecture is not the idea, the input is.

A baseline is only as good as it is well-trained, and three things in this setting will wreck $V$ if I am naive, each forcing a further design move. First, the value targets span orders of magnitude and drift as the policy improves — in the particle `Spread` task episode returns run from below $-200$ up toward $0$ — and a net regressing to such targets is unstable, with gradients dominated by the large-magnitude returns. I normalize the target by a running mean and standard deviation and learn in standardized space, denormalizing $V$ when I form $\delta_t$ for GAE. But naive renormalization is itself non-stationary: changing the statistics retroactively changes what the network's existing outputs mean. The PopArt resolution absorbs the statistics into the last linear layer — whenever $(\mu_{old},\sigma_{old})\to(\mu_{new},\sigma_{new})$, rescale $W \leftarrow W\,\sigma_{old}/\sigma_{new}$ and $b\leftarrow(\sigma_{old}b+\mu_{old}-\mu_{new})/\sigma_{new}$ — so the de-normalized output is preserved exactly across a statistics update while the targets are adaptively rescaled, with the statistics kept as a debiased EMA. Second, PPO reuses each batch over several epochs, and MARL non-stationarity (every agent is part of every other agent's environment) makes that reuse more dangerous: large per-update policy movement makes the value learning chase a moving target. So I dial PPO's "limit the change" instinct harder — clip $\epsilon<0.2$ (smaller on harder tasks), few epochs (roughly 5–15), at most two minibatches, and more data per gradient rather than more passes over the same data, since a larger batch lowers gradient variance exactly when the problem is non-stationary; the clipped value loss above applies the same conservatism to $V$. Third, on environments where agents die (SMAC), a dead agent's agent-specific local features go to zero while the agent-agnostic global features stay nonzero, so its critic input lurches into a rarely-visited region — a sharp distribution shift on the roughly fifth of timesteps that are post-death, fit poorly, with error that propagates through GAE and can corrupt the value representation for live states. The fix is death masking: replace a dead agent's input with a single fixed vector $0_a$, zeros with the agent id appended, so all of its post-death timesteps map to one input and the critic need only fit one quantity, the agent's average post-death value. The id is kept because that value differs by role; collapsing it would hurt on role-asymmetric maps. The tempting alternative — treat death as terminal, fold the accumulated post-death return $R_d=\sum_{t\ge d}\gamma^{t-d}r_t$ into the death step $d$, and skip value learning afterward — is theoretically clean but turns the 1-step GAE return at $d$ into a full $(T-d)$-step Monte-Carlo estimate, defeating exactly the bias-variance control I adopted GAE for. So masking beats both leaving the global state on and skipping. The constants follow PPO practice: $\gamma=0.99$, GAE $\lambda=0.95$, Adam with $\epsilon=1\mathrm{e}{-5}$, gradient-norm clipping at $10$, and a Huber value loss with $\delta=10$. The result is single-agent PPO with one thing changed — a centralized state-value critic over an agent-specific global state — dropped into the learner as a module returning a per-agent value.

```python
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CentralVCritic(nn.Module):
    """Shared central-V critic returning (B, T, n_agents, 1)."""

    def __init__(self, scheme, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.output_type = "v"

        input_shape = scheme["state"]["vshape"]
        if args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        if args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        input_shape += self.n_agents

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        inputs = []
        state = batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        inputs.append(state)

        if self.args.obs_individual_obs:
            obs = batch["obs"][:, ts].view(bs, max_t, -1)
            inputs.append(obs.unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        if self.args.obs_last_action:
            if t == 0:
                last_actions = th.zeros_like(batch["actions_onehot"][:, 0:1])
            elif isinstance(t, int):
                last_actions = batch["actions_onehot"][:, slice(t - 1, t)]
            else:
                last_actions = th.cat(
                    [th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]],
                    dim=1,
                )
            inputs.append(last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1))

        agent_id = th.eye(self.n_agents, device=batch.device)
        inputs.append(agent_id.unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat(inputs, dim=-1)

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

In the official on-policy stack, AS/FP/death-masked value inputs are already assembled as `share_obs` / `cent_obs`; the critic maps that centralized observation to one scalar value and the replay buffer stores `(T+1, rollout_threads, n_agents, 1)` value and return tensors.

```python
import torch.nn as nn
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.util import check
from onpolicy.utils.util import get_shape_from_obs_space


class R_Critic(nn.Module):
    def __init__(self, args, cent_obs_space, device):
        super().__init__()
        self._use_popart = args.use_popart
        self.base = MLPBase(args, get_shape_from_obs_space(cent_obs_space))
        self.rnn = RNNLayer(args.hidden_size, args.hidden_size, args.recurrent_N, args.use_orthogonal) \
            if args.use_recurrent_policy or args.use_naive_recurrent_policy else None
        self.v_out = PopArt(args.hidden_size, 1, device=device) if self._use_popart \
            else nn.Linear(args.hidden_size, 1)

    def forward(self, cent_obs, rnn_states, masks):
        features = self.base(check(cent_obs))
        if self.rnn is not None:
            features, rnn_states = self.rnn(features, check(rnn_states), check(masks))
        values = self.v_out(features)
        return values, rnn_states
```

```python
def cal_value_loss(values, value_preds_old, returns, clip_param, huber_delta, popart_head):
    value_pred_clipped = value_preds_old + (values - value_preds_old).clamp(-clip_param, clip_param)
    popart_head.update(returns)                            # W *= sigma_old/sigma_new
                                                           # b = (sigma_old*b + mu_old - mu_new)/sigma_new
    err_clipped  = popart_head.normalize(returns) - value_pred_clipped
    err_original = popart_head.normalize(returns) - values
    loss_clipped  = huber_loss(err_clipped,  huber_delta)
    loss_original = huber_loss(err_original, huber_delta)
    return th.max(loss_original, loss_clipped).mean()      # clipped value loss
```
