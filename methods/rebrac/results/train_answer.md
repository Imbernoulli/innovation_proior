We are handed a fixed dataset $D$ of transitions $(s, a, r, s')$ collected by some unknown behavior policy, and asked to learn the best continuous-control policy $\pi(s)$ we can without ever touching the environment. The trouble is precise. To fit the critic we regress $Q(s,a)$ onto a bootstrapped target $r + \gamma\, Q(s', a')$, where $a'$ is whatever the current policy proposes at $s'$. Online this is self-correcting: a bad $a'$ gets executed, earns a low reward, and the critic is corrected. Offline there is no execution. The proposed $a'$ is frequently an action the dataset never contains at $s'$, and a function approximator queried at such an out-of-distribution input returns an essentially arbitrary extrapolation — and worse than arbitrary, biased upward, because the actor is trained to *maximize* $Q$ and therefore actively seeks out the OOD actions the critic happens to overrate. That overrated value enters the next target, the next backup inherits it, the value surface inflates and detaches from reality, and the policy chases the inflation off into action space we have no data for. The run collapses. So the real problem is not "run RL on a frozen buffer"; it is to stop the critic from trusting out-of-distribution actions while still letting the policy exploit the in-distribution actions that are genuinely good.

A second pressure shapes the kind of answer I am willing to accept. The offline-RL literature has accumulated methods of escalating complexity — conservative critic penalties, large critic ensembles, learned generative behavior models, expectile value functions — and each tends to ship its headline idea bundled with a fistful of unablated "minor" choices: a different network depth, a bigger batch, normalization between layers, a tweaked learning rate, an actor pre-training phase. When such a method reports a gain, one genuinely cannot attribute it to the algorithm rather than the bundle. I want the opposite: a method that is minimal and honestly attributable, every piece individually justified and individually removable, no secondary networks, no compute overhead, and — with the parameter count capped against the strongest baseline — whatever I do has to live in the loss and the target construction, not in added capacity. Among the tools on the table, TD3 supplies the right online controls for the overestimation disease (clipped-min twin critics, target policy smoothing, delayed updates) but nothing that keeps the policy near a fixed dataset; the clipped minimum of two unconstrained off-distribution values is still an unconstrained value. TD3+BC adds a single behavior-cloning term to the actor and matches much heavier methods, but it regularizes only the *acting* policy and never the *bootstrap*. BRAC names the two places a behavior penalty can go but couples their strengths and, in practice, leans on a separately learned behavior model and a density divergence whose particular form never seems to matter. IQL sidesteps OOD queries entirely but cannot improve beyond the best in-support action. The gap is a minimal, deterministic, two-sided behavior regularizer.

I propose ReBRAC — "Revisited BRAC" — TD3 carrying a decoupled behavior-cloning penalty on *both* the actor loss and the critic bootstrap target, the TD3+BC value normalization on the actor, and a small set of architectural and optimization choices that are each derived rather than inherited. The starting point is the minimalist actor fix: replace the deterministic-policy-gradient actor objective with $\pi = \arg\max_\pi \mathbb{E}_{(s,a)\sim D}\big[ \lambda\, Q(s,\pi(s)) - (\pi(s)-a)^2 \big]$, maximize value but pay a squared penalty for straying from the logged action. The one subtlety that must be right is scale: the BC term $(\pi(s)-a)^2$ is bounded (actions in $[-1,1]^m$, so at most about $4$), while $Q$ scales with reward magnitude, which is arbitrary across tasks. Writing $Q-(\pi-a)^2$ would let $Q$ dwarf the penalty on high-reward tasks and the penalty dominate on low-reward ones. The fix is to normalize the value term by its own average magnitude, $\lambda = \mathrm{stopgrad}\big(1/\overline{|Q(s,\pi(s))|}\big)$, so one coefficient transfers across reward scales. Crucially $\lambda$ is a stop-gradient scalar — it rescales the loss, it does not bend the direction of the $Q$ gradient, since we differentiate $\pi$, not the normalizer.

But the actor penalty only guards the action taken at states we have seen; it says nothing about the next action the critic *assumes* it will take. The target is $r + \gamma\,\min_i Q(s',a')$ with $a' = \pi_{\text{target}}(s') + \text{noise}$, and nothing forces $\pi(s')$ to be in-distribution at $s'$. So the bootstrap can still select an OOD $a'$, overrate it, and reopen the exact loop one step downstream. The remedy is the second of the two penalty locations BRAC names: a value penalty inside the critic target, subtracting a divergence-from-behavior term from the bootstrapped value so the backup is pessimistic precisely where the policy departs from the data. BRAC wrote this with stochastic policies and density divergences requiring a learned behavior model — the bundled complexity I refuse. But my policy is deterministic; $\pi(s)$ is a point. The natural divergence between the policy's next action and the data's is simply the squared Euclidean distance to the dataset's *own recorded next action* $\hat a'$, which the buffer already hands me if I keep it. No behavior model, one subtraction. The bootstrap becomes, with target smoothing $a' = \mathrm{clip}\big(\pi_{\text{target}}(s') + \mathrm{clip}(\mathcal{N}(0,\sigma),-c,c),\,-1,1\big)$,
$$ q = \min_{i=1,2} Q_{\theta'_i}(s', a') - \beta_2 \,\lVert a' - \hat a' \rVert_2^2, \qquad y = r + \gamma\,(1-\text{done})\,q, $$
with the critic loss $\sum_i \mathbb{E}\big[(Q_{\theta_i}(s,a) - y)^2\big]$, and the actor loss
$$ \mathcal{L}_{\text{actor}} = \mathbb{E}\big[\, \beta_1 \lVert \pi(s)-a\rVert_2^2 - \lambda\, Q(s,\pi(s)) \,\big], \qquad \lambda = \mathrm{stopgrad}\!\left(\tfrac{1}{\overline{|Q(s,\pi(s))|}}\right). $$

The point I push on is the one BRAC left coupled: $\beta_1$ (actor) and $\beta_2$ (critic) should *not* share a value. The actor penalty controls how conservative the policy is when it acts; the critic penalty controls how distrustful the bootstrap is of off-distribution next-actions. These are different jobs with different right answers per environment — a broad, forgiving dataset wants a small actor penalty but a meatier critic guard, a narrow one the reverse — and forcing one coefficient onto a one-dimensional trade-off throws away a real degree of freedom. So I decouple them into two scalars tuned per environment. I expect the actor penalty to be the load-bearing one, standing between me and the OOD collapse, and the critic penalty to help less but complete the two-sided picture; decoupling is exactly what lets me not assume they move together.

With the losses settled, I confront the architecture and optimization bundle and insist each piece earn its place or be dropped. LayerNorm goes into the critic, between every hidden layer, because it provably addresses *my* disease. If the last hidden representation $\psi(s,a)$ feeding the output head $w$ is layer-normalized, its norm is a bounded constant for any input, OOD included, so by Cauchy–Schwarz and $\lVert\mathrm{relu}(x)\rVert \le \lVert x\rVert$,
$$ |Q(s,a)| = |w^\top \mathrm{relu}(\psi(s,a))| \le \lVert w\rVert\,\lVert \mathrm{relu}(\psi(s,a))\rVert \le \lVert w\rVert\,\lVert \psi(s,a)\rVert \le \lVert w\rVert. $$
The off-distribution value is hard-capped by the head weight norm, killing the runaway extrapolation that fuels the overestimation loop — and doing so without explicitly telling the policy to stay near the data. The actor is left without inter-layer normalization: it is not the surface that extrapolates dangerous values; its output is already bounded into the action box by a tanh and pulled to the data by $\beta_1$. The asymmetry is deliberate. Depth goes to three hidden layers of width $256$: a large static dataset rewards the extra capacity to fit the value and policy, the two-layer default is a holdover from online bases, and the gain saturates around three to five layers and drops if pushed to six. Batch size and learning rate go up on dense locomotion — batch $1024$, lr $10^{-3}$ — for lower-variance gradients and faster convergence within the one-million-step budget, with the learning rate scaled to the batch by the standard heuristic; on sparse-reward and harder domains, where a larger batch over-smooths and hurts, they stay small (AntMaze keeps batch $256$, actor lr $3\times10^{-4}$, critic lr as low as $5\times10^{-5}$). The discount is read off the horizon, not swept: on a length-$L$ episode with a single terminal reward, the signal reaches the start discounted by $\gamma^L$, and $0.99^{1000}\approx 4\times10^{-5}$ erases it while $0.999^{1000}\approx 0.37$ preserves better than a third — so $\gamma = 0.999$ on long sparse-reward tasks and $0.99$ on dense locomotion. Finally I drop the state-feature normalization TD3+BC inherited, both to keep the method runnable online later (a fixed dataset mean/std would go stale) and because its offline effect is small. The update schedule is TD3's two-timescale structure unchanged: the critic steps every iteration, the actor and the soft targets ($\tau = 5\times10^{-3}$) only every $\text{policy\_freq}=2$ steps, so critic error settles between policy moves. What results is TD3 plus two squared-distance penalties and a handful of derived design choices, every one individually justified and removable. The dataclass placeholder sets $\beta_1 = \beta_2 = 1.0$; actual runs use per-environment overrides — for instance `halfcheetah-medium-v2` uses $\beta_1 = 0.001$, $\beta_2 = 0.01$, lr $10^{-3}$, batch $1024$, $\gamma = 0.99$. The dataset must store the next action $\hat a'$ for the critic penalty; the `mc_returns` branch below is disabled by default and exists only to mirror an extra target floor present in the original reference repository.

```python
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeterministicActor(nn.Module):
    """pi(s) = max_action * tanh(net(s)). 3 x 256, ReLU, no LayerNorm."""

    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )

    def forward(self, state):
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state, device="cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().numpy().flatten()


class Critic(nn.Module):
    """Q(s, a). 3 x 256, ReLU, LayerNorm between layers (bounds OOD |Q|)."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class ReBRAC:
    def __init__(self, state_dim, action_dim, max_action,
                 actor_bc_coef=1.0, critic_bc_coef=1.0,     # beta_1, beta_2; YAML overrides per env
                 discount=0.99, tau=5e-3, lr=1e-3,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2,
                 normalize_q=True, use_mc_return_floor=False, device="cuda"):
        self.device = device
        self.beta_1, self.beta_2 = actor_bc_coef, critic_bc_coef
        self.discount, self.tau, self.max_action = discount, tau, max_action
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.policy_freq, self.normalize_q = policy_freq, normalize_q
        self.use_mc_return_floor = use_mc_return_floor
        self.total_it = 0

        self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_1_opt = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_opt = torch.optim.Adam(self.critic_2.parameters(), lr=lr)

    def train(self, batch):
        self.total_it += 1
        if self.use_mc_return_floor:
            states, actions, rewards, next_states, dones, next_actions_data, mc_returns = batch
            mc_returns = mc_returns.squeeze(-1)
        else:
            states, actions, rewards, next_states, dones, next_actions_data = batch
            mc_returns = None
        not_done = 1.0 - dones.squeeze(-1)
        rewards = rewards.squeeze(-1)

        # ---- critic update (every step) ----
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action)
            next_q = torch.min(self.critic_1_target(next_states, next_actions),
                               self.critic_2_target(next_states, next_actions))
            bc_penalty = ((next_actions - next_actions_data) ** 2).sum(-1)   # value penalty
            next_q = next_q - self.beta_2 * bc_penalty
            target_q = rewards + not_done * self.discount * next_q
            if mc_returns is not None:
                target_q = torch.maximum(target_q, mc_returns)               # DT6A repo variant

        critic_loss = (F.mse_loss(self.critic_1(states, actions), target_q)
                       + F.mse_loss(self.critic_2(states, actions), target_q))
        self.critic_1_opt.zero_grad(); self.critic_2_opt.zero_grad()
        critic_loss.backward()
        self.critic_1_opt.step(); self.critic_2_opt.step()

        # ---- delayed actor + target update ----
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(states)
            q = self.critic_1(states, pi)
            bc_mse = ((pi - actions) ** 2).sum(-1)                            # policy regularization
            lmbda = 1.0
            if self.normalize_q:
                lmbda = 1.0 / (q.abs().mean().detach() + 1e-8)                # TD3+BC normalization
            actor_loss = (self.beta_1 * bc_mse - lmbda * q).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            for net, tgt in ((self.critic_1, self.critic_1_target),
                             (self.critic_2, self.critic_2_target),
                             (self.actor, self.actor_target)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
```
