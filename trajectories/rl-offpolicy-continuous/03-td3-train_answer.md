SAC confirmed the diagnosis and then handed me a sharper one. The Ant collapse is fixed decisively — from DDPG's $\{585, 569, 2574\}$ (mean 1243) to SAC's $\{3498, 3226, 4007\}$ (mean 3577) — the twin-min capped the overestimation that dragged DDPG's seeds into bad basins, and the entropy kept the policy from locking into the $\sim 570$ trap. But entropy cost real return on the tasks that did not need it. HalfCheetah dropped from DDPG's tight 11514 to 10360, and Reacher, a precision reach, *worsened* across every seed from $-3.94$ to $-4.76$: the maximum-entropy policy keeps action mass spread, and where the optimal behavior is a sharp deterministic reach, that residual stochasticity at evaluation is pure cost. So I have two methods each winning a different region — DDPG's deterministic actor where the task wants a sharp policy, SAC's twin-min-plus-entropy where overestimation kills you — and I want one method that takes the overestimation control onto a *deterministic* actor, so there is no entropy tax at evaluation.

To do that I go back to the DDPG floor and ask precisely *why* it overestimated. In the discrete world the max in $y = r + \gamma\max_{a'} Q(s',a')$ is the culprit: with $Q(s',a') = Q_{\text{true}} + \varepsilon_{a'}$, $\mathbb{E}[\varepsilon]=0$, taking expectations gives $\mathbb{E}[\max_{a'}(Q_{\text{true}}+\varepsilon)] \ge \max_{a'} Q_{\text{true}}$ — the max selects disproportionately the action whose error happened to be large and positive, so even zero-mean error biases the target high, and TD learning carries that bias backward through the reachable set. DDPG has no $\max$, only a soft gradient step, so I prove the bias survives the translation. Compare the actor's real step under the approximate critic $Q_\theta$ against the step it would take under the true critic $Q^\pi$. The approximate-gradient policy scores at least as high under $Q_\theta$, the true-gradient policy at least as high under $Q^\pi$; bridging them with the assumption that at the true policy the critic is on average not below the truth and chaining the inequalities yields $\mathbb{E}[Q_\theta(s,\pi_{\text{approx}})] \ge \mathbb{E}[Q^\pi(s,\pi_{\text{approx}})]$ — the value I believe is $\ge$ the truth of the policy I deployed, with no max anywhere, routed instead through the gradient that climbs wherever the critic bulges upward. And it accumulates: unrolling the TD-error residual through the bootstrap, the learned value equals the expected return minus the expected discounted sum of all future TD-errors, so its variance grows with the discounted future estimation errors, and with $\gamma$ near 1 the far-future error piles in at nearly full strength.

I propose TD3, the twin-delayed deterministic policy gradient, which attacks this with three drop-in cures on a deterministic actor-critic. First, the bias at the source. The discrete fix decouples selection from evaluation, but here the slow-moving policy makes $\pi_\phi(s')$ and $\pi_{\phi'}(s')$ pick nearly the same action, so that decoupling barely moves the bias, and two independent critics share a buffer and cross-built targets, so in the pockets where the second critic overshoots the first I get overestimation piled on overestimation — exactly where the policy is drawn. But the trouble is only ever when the second critic comes out *higher*; when it comes out lower it is doing its job. So I do not need the second critic unbiased, I need it never to make things worse than the already-overestimating first one — take the *minimum*:

$$y = r + \gamma(1-d)\,\min_{i=1,2} Q_{\theta'_i}(s', a').$$

If $Q_2 \ge Q_1$ the min is the standard single-critic target, no added bias beyond plain Q-learning; if $Q_2 < Q_1$ it is the Double-Q correction pulling down. It caps the worst case and leans toward underestimation, the *safer* direction here, because an overestimated action is selected by the actor and amplified through every policy update, while an underestimated action is merely avoided. This is the twin-min SAC used — the part that actually fixed Ant — now on a deterministic actor.

Second, the accumulation, which is variance through bootstrapping. A target network restrains it: freeze a slow copy so the critic fits a stationary objective over many steps instead of chasing a target it is simultaneously moving — `soft_update` with $\tau=0.005$ on both critics and the actor. And if policy updates against a still-high-error critic are what trigger divergence, I should not update the policy at the critic's cadence; let the critic settle first, updating the actor and the targets only every $d$ critic updates. The scaffold's `policy_frequency=2` gives two critic fits per actor move, so each policy step sees a critic that has had time to drive its error down — a two-timescale structure that is cheap and helps both bias and variance.

Third, a variance source peculiar to deterministic policies, the piece SAC got for free from being stochastic. The deterministic target action is a single point $\pi_{\phi'}(s')$, and the critic is a bumpy approximator, so the actor finds and sits on a narrow spurious peak and the target is read off that knife-edge. SAC's stochastic actor averaged the target over a distribution of actions, smoothing those peaks; on a deterministic actor I put that smoothing back by hand, fitting the value of a *neighborhood* of the target action:

$$y = r + \gamma(1-d)\,\min_i Q_{\theta'_i}\!\big(s',\, \text{clip}(\pi_{\phi'}(s') + \varepsilon)\big),\qquad \varepsilon\sim\text{clip}(\mathcal{N}(0,\tilde\sigma), -c, c).$$

The noise is clipped to a small band so the perturbed action stays a genuine neighbor, and averaging over minibatches does the smoothing. This is target-policy smoothing, a SARSA-flavored regularizer that denies the deterministic actor the sharp peaks SAC's entropy denied it stochastically — but without keeping the policy stochastic at evaluation, which is the crux of beating SAC on Reacher and HalfCheetah: the smoothing happens at training time, the deployed policy is sharp, no entropy tax on the tasks that want a precise reach or gait.

The three reinforce rather than collide. Form the target by taking the target actor's action, adding clipped noise, clipping back into range, evaluating under both target critics, and taking the min; regress both critics to it every step; and every $d$ steps update the actor by the deterministic policy gradient through the *first* critic, $-Q_1(s,\pi_\phi(s)).\text{mean}()$, and soft-update all targets. In the scaffold this is mostly a reversal of SAC: the `Actor` goes back to the rung-1 deterministic-tanh form returning a plain action, and `select_action` back to deterministic-plus-Gaussian-exploration-noise. `OffPolicyAlgorithm` keeps SAC's twin critics and twin target critics but adds back a *target actor* (a deterministic actor needs one — the next action comes from the slow, smoothed policy copy), drops the `log_alpha` machinery entirely, and sets `policy_noise=0.2`, `noise_clip=0.5`. One deliberate choice against the SAC fill: the target soft-updates here are gated by the policy delay — they run inside the `% policy_frequency` block alongside the actor — because for a deterministic target the target actor and target critics should move together on the slow timescale, which is what the delay is for. I expect TD3 to hold Ant near or above SAC (twin-min kept), beat SAC's $-4.76$ on Reacher back toward DDPG's $-3.94$, claw HalfCheetah back above 10360 toward 11514, and ideally be the only method strong on all three at once. The one residual I cannot fix from inside this design: TD3, like both rungs before it, still learns through *stale target networks* — the smoothed twin-min target is read off slow copies, and $\tau$ is still a knob balancing a deliberately-lagged signal — and if Ant variance stays wide, that staleness is the next thing to attack.

```python
class OffPolicyAlgorithm:
    """TD3 — Twin Delayed Deep Deterministic Policy Gradient."""

    def __init__(self, obs_dim, action_dim, max_action, device, args):
        self.device = device
        self.max_action = max_action
        self.gamma = args.gamma
        self.tau = args.tau
        self.exploration_noise = args.exploration_noise
        self.policy_frequency = args.policy_frequency
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.total_it = 0

        self.actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.target_actor = Actor(obs_dim, action_dim, max_action).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())

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

    def select_action(self, obs):
        obs_t = torch.tensor(obs.reshape(1, -1), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy().flatten()
        noise = np.random.normal(0, self.max_action * self.exploration_noise, size=action.shape)
        return np.clip(action + noise, -self.max_action, self.max_action)

    def update(self, batch):
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            ) * self.max_action
            next_actions = (self.target_actor(next_obs) + noise).clamp(
                -self.max_action, self.max_action
            )
            target_q1 = self.qf1_target(next_obs, next_actions).view(-1)
            target_q2 = self.qf2_target(next_obs, next_actions).view(-1)
            td_target = rewards + (1 - dones) * self.gamma * torch.min(target_q1, target_q2)

        q1 = self.qf1(obs, actions).view(-1)
        q2 = self.qf2(obs, actions).view(-1)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        actor_loss_val = 0.0
        if self.total_it % self.policy_frequency == 0:
            actor_loss = -self.qf1(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            soft_update(self.target_actor, self.actor, self.tau)
            soft_update(self.qf1_target, self.qf1, self.tau)
            soft_update(self.qf2_target, self.qf2, self.tau)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val}
```
