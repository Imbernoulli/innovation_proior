The goal I set myself is concrete: train a neural-network policy to maximize expected discounted return $\eta(\pi_\theta) = \mathbb{E}\!\left[\sum_t \gamma^t R_t\right]$ with a single method that is at once scalable, data-efficient, robust across both continuous-control robotics and discrete pixel games, and — the part I refuse to compromise on — first-order and easy to implement, needing nothing more exotic than the SGD/Adam machinery and automatic differentiation already in everyone's toolbox. None of the three credible families I have meets all of this. Deep Q-learning cracks Atari from pixels but is built around an argmax over actions, so it does not transfer cleanly to continuous control, and it is brittle and poorly understood — a change to reward scale or one hyperparameter can break it. Vanilla policy gradients are the opposite: dead simple and fully general across action spaces, but each sample feeds exactly one gradient step, so they throw data away, and the step size is treacherous — too large and a single update collapses the policy, and a collapsed policy does not come back. TRPO is the one I actually trust not to blow up, because it controls the policy drift with a hard KL trust region; but it does so only through heavy second-order machinery — conjugate gradient, Fisher-vector products, a backtracking line search — that resists parallelism, takes essentially one constrained step per batch (so it leaves data efficiency on the table), and turns hostile the moment the policy and value share a network or use stochastic units, because the KL is defined on the policy's output distribution and sharing smears exactly that distribution.

The tension is precise. Reusing one batch of on-policy data for multiple gradient steps is exactly what buys data efficiency, but it is also what makes naive policy-gradient ascent diverge. The surrogate $L^{PG}(\theta) = \hat{\mathbb{E}}_t\!\left[\log \pi_\theta(a_t|s_t)\,\hat A_t\right]$, whose gradient is the score-function estimator $\hat g = \hat{\mathbb{E}}_t\!\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\,\hat A_t\right]$, is only a first-order proxy valid *at* the current $\theta$; the advantages $\hat A_t$ were computed under the old policy's state visitation. Run several epochs of SGD on it and $\pi_\theta$ marches off into a region where the proxy has nothing to do with the true objective and the update is destructive. What governs when the surrogate stops being valid is distance between $\pi_\theta$ and $\pi_{\theta_\text{old}}$, and the reason is the conservative-policy-iteration analysis of Kakade and Langford (2002): $\eta(\pi) - \eta(\pi_\text{old})$ equals the expected advantage of $\pi_\text{old}$ taken under $\pi$'s *own* trajectory distribution, an exact identity that telescopes because $A_{\pi_\text{old}}(s,a) = \mathbb{E}[R + \gamma V_{\pi_\text{old}}(s') - V_{\pi_\text{old}}(s)]$ and the consecutive value terms cancel along a trajectory. That distribution is unknown — it is the thing being produced — so they evaluate the expected advantage under the *old* state distribution, giving a surrogate that matches $\eta$ to first order at $\pi = \pi_\text{old}$ and, crucially, is a genuine lower bound on true improvement whose error is bounded by a term growing with how far $\pi$ strays from $\pi_\text{old}$. Closeness is not a heuristic safety rail; it is the precise condition that makes the surrogate honest. Rewriting it for the samples I actually have, importance sampling introduces the probability ratio $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_\text{old}}(a_t|s_t)$ and turns the surrogate into $L^{CPI}(\theta) = \hat{\mathbb{E}}_t\!\left[r_t(\theta)\,\hat A_t\right]$, with $r_t(\theta_\text{old}) = 1$ at the start of each update. But importance sampling is also where the danger lives: as $\pi_\theta$ drifts, the ratios fan out, their variance grows, and the optimizer actively *seeks* large ratios on positive-advantage samples because inflating $r_t$ is the cheapest way to raise the surrogate — it need not find a genuinely better action, just push one number up. The ratio is the distance signal and the danger signal at once, and some mechanism must keep it near 1.

I propose PPO — Proximal Policy Optimization — which keeps the ratio near 1 not with a KL constraint and its second-order apparatus, nor with a KL penalty whose coefficient I would have to guess, but with a clip on the ratio that turns the trust region into a flat spot in a differentiable loss. The penalty form $\hat{\mathbb{E}}_t[r_t\hat A_t - \beta\,\mathrm{KL}]$ is the right *shape* — pure first-order — but the $\beta$ the bound hands you is enormous (it uses the max KL over states), and any fixed hand-chosen $\beta$ will not hold still: the right value depends on the advantage scale and on KL's sensitivity to a parameter step, both of which drift across tasks and across a single run. So I attack "stay close" differently: I refuse to give the optimizer any credit for moving $r_t$ outside a band $[1-\epsilon, 1+\epsilon]$ at all. The defining objective is

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\!\left[\min\!\big(r_t(\theta)\,\hat A_t,\ \mathrm{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\,\hat A_t\big)\right],$$

with $\epsilon = 0.2$. The clip alone flattens the term once the ratio leaves the band — for $\hat A_t > 0$ the term caps at $(1+\epsilon)\hat A_t$ and its gradient dies, removing the incentive to keep cranking a good action's probability up. But a *plain* clip is too forgiving, and this is the crux of why the $\min$ is there. Suppose an action is bad, $\hat A_t < 0$, but the new policy has already overshot its ratio to $r_t = 3$, far above the band — maybe from a noisy earlier step or from parameter coupling. The unclipped term $3\hat A_t$ is very negative and its gradient wants to pull $r_t$ back down, correcting the overshoot; but the plain clipped term $(1+\epsilon)\hat A_t$ sits in the flat region with zero gradient, freezing in the mistake. So I take the pessimistic *minimum* of the unclipped and clipped terms: keep whichever gives the objective less credit. Case by case this does exactly what I want. With $\hat A_t > 0$: for $r_t > 1+\epsilon$ the $\min$ picks the smaller capped $(1+\epsilon)\hat A_t$ (gradient off, no reward for going higher), but for $r_t < 1-\epsilon$ — a wrong-direction move that made a good action less likely — the clipped $(1-\epsilon)\hat A_t$ is the larger value, so the $\min$ keeps the unclipped $r_t\hat A_t$ and its gradient still pulls the probability back up. With $\hat A_t < 0$: for $r_t < 1-\epsilon$ the clipped $(1-\epsilon)\hat A_t$ is *more* negative, so the $\min$ picks it (flat, no reward for suppressing the bad action further); but for $r_t > 1+\epsilon$ — a bad-action overshoot — the unclipped term is more negative, so the $\min$ keeps it and the gradient corrects the overshoot. In every case the clip removes the incentive to push the ratio further in the direction the advantage already favors, but never discards the gradient that corrects a move in the wrong direction. $L^{CLIP}$ is thus a pessimistic lower bound on $L^{CPI}$: I ignore the full ratio change only when including it would make the objective look better. To first order at $\theta_\text{old}$, where every $r_t = 1$, the clip is inactive and $L^{CLIP} = L^{CPI}$, so the first epoch is ordinary policy-gradient ascent and the brake engages only as the policy moves. This is the entire trust region, and it is just a $\min$ of two products — no KL, no Fisher matrix, no line search.

The choice $\epsilon = 0.2$ is a sweet spot of the same kind a KL trust region has, but expressed on the unit-free ratio. Too small, say $0.05$, and the brake bites almost immediately, every update is microscopic, and the $K$-epoch reuse cannot move $\theta$ — TRPO with a tiny $\delta$. Too large, say $0.5$ or $1.0$, and the band is so wide the clip rarely engages, the ratios fan out, and the destructive-update failure mode returns. Around $0.2$ the policy moves a useful amount per iteration while the realized KL after a full update stays on the order of $0.01$–$0.02$, exactly the regime where the Kakade–Langford bound is tight; and because $\epsilon$ constrains the dimensionless ratio directly, one value is robust across MuJoCo and Atari where a fixed $\beta$ — which had to fight the advantage scale — could not be. With the brake in place I can finally do what started all this: $K$ epochs (ten on MuJoCo, three or four on Atari) of shuffled minibatch SGD on the same $N$-actor $\times$ $T$-step batch, the update self-limiting because the loss goes flat where it should.

The surrogate is half a step; the rest is advantage estimation, value fitting, and exploration, all chosen to stay first-order and to exploit the architecture freedom the clip buys me. For advantages I use truncated generalized advantage estimation: with the TD residual $\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)$, take $\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}$. The $\lambda$ knob trades bias for variance — $\lambda = 0$ gives $\delta_t$ (low variance, biased by $V$'s errors), and $\lambda = 1$ telescopes to the Monte-Carlo advantage $\sum_l \gamma^l R_{t+l} - V(s_t)$ (unbiased, high variance), with $\lambda \approx 0.95$, $\gamma = 0.99$ in the sweet spot — and it is computed as one reverse scan $\hat A_t = \delta_t + \gamma\lambda(1-\text{done})\hat A_{t+1}$ with a $(1-\text{done})$ mask zeroing the bootstrap across episode boundaries. Running $N$ short rollouts of fixed length $T$ in parallel, the way A2C does, decorrelates the data and gives an $NT$-sample batch. The value $V_\theta$ is the baseline that makes the advantage low-variance in the first place, regressed toward $V_t^{targ} = \hat A_t + V_\text{old}(s_t)$. Because my objective is just a differentiable loss, I can *share* a network between policy and value head and add the losses — the very thing TRPO's Fisher machinery made awkward — folding in an entropy bonus to sustain exploration, so each iteration I maximize

$$L^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t\!\left[L^{CLIP}_t(\theta) - c_1\big(V_\theta(s_t) - V_t^{targ}\big)^2 + c_2\, S[\pi_\theta](s_t)\right].$$

By symmetry with the policy clip — the value head rides the same drifting network through the same $K$ epochs — I clip the value loss too, taking the max of the unclipped squared error and a version where $V_\theta$ is held within $V_\text{old} \pm \epsilon$ before squaring, capping how far the prediction is rewarded for moving per update. The entropy bonus earns its keep on Atari with a categorical policy; on MuJoCo I often set $c_2 = 0$ because the Gaussian's own state-independent learned log-std is itself the exploration knob the optimizer tunes. A handful of details are load-bearing rather than cosmetic: orthogonal initialization with gain $\sqrt{2}$ on hidden layers (preserving signal through tanh), $0.01$ on the policy output (so the agent starts near-uniform, not over-confident), and $1.0$ on the value output; Adam with $\epsilon = 10^{-5}$ rather than $10^{-8}$, a larger denominator floor that keeps steps well-conditioned when RL gradients get small; linear learning-rate annealing to zero (big steps early, small steps late); per-minibatch advantage normalization to keep the effective step scale constant across reward scales; a global gradient-norm clip at $0.5$ as a cheap last line of defense against a spiky minibatch; and a state-independent log-std so exploration cannot silently die in particular states. I also keep an adaptive-KL-penalty variant — maximize $\hat{\mathbb{E}}_t[r_t\hat A_t - \beta\,\mathrm{KL}]$ and servo $\beta$ toward a KL target $d_\text{targ}$, doubling it when the realized KL exceeds $d_\text{targ}\cdot 1.5$ and halving it below $d_\text{targ}/1.5$ — as an honest first-order alternative that no longer guesses $\beta$. But clipping is the default: it carries no KL anywhere in the loss, no outer adaptation loop, and is at least as good. The whole loop is first-order and cheap — collect, estimate advantages, run $K$ epochs of minibatch Adam, set $\theta_\text{old} \leftarrow \theta$, repeat — with the entire trust region living inside one $\min$.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))  # state-independent

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        std = self.actor_logstd.expand_as(mean).exp()
        probs = Normal(mean, std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


agent = Agent(envs)
optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)

# ---- per iteration: rollout fills obs/actions/logprobs/rewards/dones/values for num_steps x num_envs ----

frac = 1.0 - (iteration - 1.0) / num_iterations
optimizer.param_groups[0]["lr"] = frac * 3e-4

# truncated GAE: A_t = delta_t + gamma*lambda*(1-done)*A_{t+1}
with torch.no_grad():
    next_value = agent.get_value(next_obs).reshape(1, -1)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values

# flatten the batch
b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
b_logprobs = logprobs.reshape(-1)
b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
b_advantages = advantages.reshape(-1)
b_returns = returns.reshape(-1)
b_values = values.reshape(-1)

# K epochs of minibatch updates on the same batch
b_inds = np.arange(batch_size)
for epoch in range(update_epochs):
    np.random.shuffle(b_inds)
    for start in range(0, batch_size, minibatch_size):
        mb_inds = b_inds[start:start + minibatch_size]

        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
            b_obs[mb_inds], b_actions[mb_inds])
        logratio = newlogprob - b_logprobs[mb_inds]
        ratio = logratio.exp()                      # r_t(theta) = pi_theta / pi_old

        mb_advantages = b_advantages[mb_inds]
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # clipped surrogate (minimize the negative => max of two negatives)
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # clipped value loss (symmetry with the policy clip)
        newvalue = newvalue.view(-1)
        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
        v_clipped = b_values[mb_inds] + torch.clamp(
            newvalue - b_values[mb_inds], -clip_coef, clip_coef)
        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)  # global norm cap
        optimizer.step()
```
