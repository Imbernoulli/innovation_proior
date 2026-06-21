DDPG behaved exactly as the single-critic diagnosis predicted. On the well-shaped tasks it is strong and tight — HalfCheetah at mean 11514 (seeds 12016 / 11387 / 11142) and Reacher at $-3.94$, both essentially solved — but on the higher-dimensional hidden Ant it falls apart: mean 1243 with seeds $\{585, 569, 2574\}$. Two seeds collapse to $\sim 570$ and one limps to 2574; that is not noise around a good policy, it is the single-critic overestimation loop biting where there is the most room for it. The actor ascends wherever the critic bulges upward, generates more data there, the critic inflates that region further, and the bootstrap reads the inflated value off a *single sharp target action* $\mu_{\text{targ}}(s')$. Compounding it, the deterministic actor explores nothing on its own — its only exploration is the fixed external $\sigma = 0.1\,a_{\max}$ noise — so once a seed commits to a bad basin near 570, the noise jitters around it but cannot re-open a genuinely different mode. So I want two things at once: stop the critic from overestimating, and give the policy an intrinsic drive to keep its mass spread out instead of locking onto one knife-edge action.

I propose SAC, a maximum-entropy stochastic actor-critic. Instead of maximizing $\sum_t \mathbb{E}[r(s_t,a_t)]$, maximize reward *plus* the policy's entropy at each step,

$$J(\pi) = \sum_t \mathbb{E}_{(s_t,a_t)\sim\rho_\pi}\big[r(s_t,a_t) + \alpha\,\mathcal{H}(\pi(\cdot|s_t))\big],\qquad \mathcal{H}(\pi(\cdot|s)) = \mathbb{E}_{a\sim\pi}[-\log\pi(a|s)],$$

with temperature $\alpha$ trading reward against randomness. As $\alpha\to 0$ this collapses to ordinary RL, so it is a generalization, not a different objective. It is the *right* generalization because an entropy-rewarded policy explores widely yet still abandons clearly bad actions (their low reward dominates), can hold multiple equally-good modes instead of arbitrarily forgetting one, and is known to be robust to model and estimation error — exactly the fragility I am fighting on Ant. Entropy is not an exploration hack; it is a more forgiving optimization landscape and a smoother target for the critic.

What makes it a real actor-critic is soft policy iteration, and each step is forced. For soft policy *evaluation*, define $V(s) = \mathbb{E}_{a\sim\pi}[Q(s,a) - \log\pi(a|s)]$ (the $-\log\pi$ is the entropy bookkeeping) and the soft backup $\mathcal{T}^\pi Q(s,a) = r(s,a) + \gamma\,\mathbb{E}_{s'}[V(s')]$. Pulling the entropy into the reward, $\mathcal{T}^\pi Q = r_\pi + \gamma\,\mathbb{E}[Q(s',a')]$ with $r_\pi = r + \gamma\,\mathbb{E}[-\log\pi(a'|s')]$, is an ordinary policy-evaluation backup with reward $r_\pi$, a $\gamma$-contraction in sup norm — the entropy terms cancel when two iterates are subtracted — so it has a unique fixed point $Q^\pi$. For soft policy *improvement* the theory points the new policy at the Boltzmann distribution $\pi \propto \exp(Q^{\pi_{\text{old}}})$, which is intractable to sample for an arbitrary neural $Q$ over continuous actions. So instead of *becoming* that energy-based distribution, *project onto* it: keep the policy in a tractable family $\Pi$ (Gaussians) and move it as close as $\Pi$ allows in KL,

$$\pi_{\text{new}} = \arg\min_{\pi'\in\Pi} D_{\mathrm{KL}}\!\Big(\pi'(\cdot|s)\,\big\|\,\tfrac{\exp(Q^{\pi_{\text{old}}}(s,\cdot))}{Z^{\pi_{\text{old}}}(s)}\Big).$$

The partition $Z$ depends only on $s$, so it is an additive constant in $\pi'$ and drops out of the argmin and every gradient — the intractable normalizer that forces energy-based methods into special samplers simply vanishes. And the projected step is still monotone: $\pi_{\text{old}}\in\Pi$ is feasible, so the minimizer can do no worse, and expanding the inequality and bootstrapping it through the soft Bellman equation yields $Q^{\pi_{\text{new}}}(s,a) \ge Q^{\pi_{\text{old}}}(s,a)$ everywhere. The fit-instead-of-become approximation costs nothing in monotonicity and converges to the best policy *in $\Pi$*.

Made concrete under the fixed 256-wide scaffold, the tabular steps become SGD on each objective. I do not need a separate value network: $V(s') = \mathbb{E}_{a'\sim\pi}[Q(s',a') - \log\pi(a'|s')]$ is estimated from a single fresh action sample at the next state, so the soft target is built directly from the critic and a sampled $a'\sim\pi(s')$. And I keep an overestimation guard, because the actor still ascends the critic and still chases whatever it overrates — DDPG's exact Ant failure. So twin critics $Q_1, Q_2$ with twin target copies, and the next-state value uses $\min(Q_{1,\text{targ}}, Q_{2,\text{targ}})$ before the entropy term:

$$y = r + \gamma(1-d)\big(\min_i Q_{i,\text{targ}}(s',a') - \alpha\log\pi(a'|s')\big),\qquad a'\sim\pi(s').$$

The $\min$ caps the bias at the worst case of a single critic and leans toward the safer underestimation side — directly what the DDPG floor lacked when two Ant seeds collapsed. The actor is the tractable family made concrete: a Gaussian is reparameterizable, $u = \mu_\phi(s) + \sigma_\phi(s)\odot\varepsilon$, $\varepsilon\sim\mathcal{N}(0,I)$, but physical actions live in a box, so squash through tanh, $a = \tanh(u)\cdot a_{\max}$. Squashing changes the density, so the log-prob carries the change-of-variables Jacobian, $\log\pi(a|s) = \log\mu(u|s) - \sum_i \log\!\big(a_{\max}(1 - \tanh^2 u_i) + \epsilon\big)$. Reparameterization is what lets $\nabla_a Q$ flow through the sampled action — the DDPG-style deterministic policy gradient generalized to a stochastic squashed policy — so the actor loss is $\big(\alpha\log\pi(\tilde a|s) - \min_i Q_i(s,\tilde a)\big).\text{mean}()$ with $\tilde a\sim\pi(s)$ reparameterized, the entropy term pulling the policy to spread out.

That leaves the temperature, the one knob that cannot be left to chance: too small and entropy dominates and the policy goes near-uniform and never exploits; too large and it collapses to near-deterministic early, losing the exploration I introduced it for. These three environments have very different reward scales, so rather than tune $\alpha$ per environment I auto-tune it. Hold a learned $\log\alpha$, set a target entropy $\mathcal{H}_{\text{target}} = -\dim(A)$ (one nat of spread per action dimension), and descend $-\log\alpha\cdot(\log\pi(a|s) + \mathcal{H}_{\text{target}})$, so $\alpha$ rises when the policy is below target entropy and falls when above. One self-adjusting mechanism for HalfCheetah, Reacher, and the hidden Ant is the right answer for a cross-environment board, and it subsumes the methods trace's reward-scaling knob.

In the scaffold, the `Actor` becomes the stochastic tanh-Gaussian — two 256 layers feeding a mean head and a log-std head ($\log\sigma$ tanh-bounded into $[-5, 2]$ for numerical safety), with `get_action` returning $(\text{action}, \log\text{-prob}, \text{mean\_action})$, which the loop's `eval_actor` unwraps. `OffPolicyAlgorithm` holds twin critics and twin target critics, no target *actor* (the next action is sampled fresh from the live policy), and `log_alpha` with its optimizer. In `update` I build the soft target under `no_grad`, regress both critics, take the reparameterized actor step and the $\alpha$ update every `policy_frequency=2` calls, and — deliberately against the delay — soft-update the two target critics *every* step, because the critics learn every step and their targets should track every step. I expect this to rescue Ant (twin-min cuts the overestimation, the entropy actor keeps exploring past the $\sim 570$ basin) but to risk an entropy tax where the task wants a sharp policy: HalfCheetah, which DDPG already maxes, and Reacher, a precision reach, may come in slightly below. If they do, the next rung is already written — I would want this overestimation control on a deterministic actor, with no entropy paid at evaluation.

```python
LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """Stochastic Tanh-Gaussian actor for SAC."""

    def __init__(self, obs_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", torch.tensor(max_action, dtype=torch.float32))

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale
        return action, log_prob, mean_action


class OffPolicyAlgorithm:
    """SAC — Soft Actor-Critic with automatic entropy tuning."""

    def __init__(self, obs_dim, action_dim, max_action, device, args):
        self.device = device
        self.max_action = max_action
        self.gamma = args.gamma
        self.tau = args.tau
        self.policy_frequency = args.policy_frequency
        self.total_it = 0

        self.actor = Actor(obs_dim, action_dim, max_action).to(device)

        self.qf1 = QNetwork(obs_dim, action_dim).to(device)
        self.qf2 = QNetwork(obs_dim, action_dim).to(device)
        self.qf1_target = QNetwork(obs_dim, action_dim).to(device)
        self.qf2_target = QNetwork(obs_dim, action_dim).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate)
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=args.learning_rate,
        )

        # Auto entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.learning_rate)

    def select_action(self, obs):
        obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            action, _, _ = self.actor.get_action(obs_t)
        return action.cpu().numpy().flatten()

    def update(self, batch):
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        # Update critics
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.get_action(next_obs)
            q1_next = self.qf1_target(next_obs, next_actions).view(-1)
            q2_next = self.qf2_target(next_obs, next_actions).view(-1)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi.view(-1)
            td_target = rewards + (1 - dones) * self.gamma * min_q_next

        q1 = self.qf1(obs, actions).view(-1)
        q2 = self.qf2(obs, actions).view(-1)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        # Update actor
        actor_loss_val = 0.0
        if self.total_it % self.policy_frequency == 0:
            pi, log_pi, _ = self.actor.get_action(obs)
            q1_pi = self.qf1(obs, pi).view(-1)
            q2_pi = self.qf2(obs, pi).view(-1)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * log_pi.view(-1) - min_q_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            # Update alpha
            with torch.no_grad():
                _, log_pi_alpha, _ = self.actor.get_action(obs)
            alpha_loss = (-self.log_alpha.exp() * (log_pi_alpha.view(-1) + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Update target networks
        soft_update(self.qf1_target, self.qf1, self.tau)
        soft_update(self.qf2_target, self.qf2, self.tau)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val, "alpha": self.alpha}
```
