The advantage-weighted rung told me exactly what I feared. Pen came in at a mean of 67.1, but the per-seed spread is the tell — 106.7, 32.8, 61.8, a swing of seventy points — the sharp $\lambda=0.1$ reweighting reacting to a noisy advantage on twenty-five trajectories: when the critic happens to calibrate the demo advantages, the weighted MLE concentrates on the right actions and Pen flies; when it does not, the same weights point at the wrong demos and the policy is dragged off. Hammer sat at 1.05 (the floor, the policy never assembling the long precise contact sequence) and Door at 0.59, which is zero within noise. So the constraint lived only in the *actor*, and the calibration it depended on was exactly as noisy as one stochastic single-sample $V(s)$. I want the regularization moved into the *target construction* — make the critic itself refuse to trust an off-support next action — and I want a deterministic, lower-variance signal in place of the sampled advantage that produced the seventy-point Pen swing.

I propose ReBRAC — a revisited, minimalist TD3+BC with the behavior penalty added in *both* places the behavior-regularized framework allows. The base is TD3, every piece of it aimed at the same disease in its online form: overestimation. The root fact is that a greedy bootstrap is upward-biased under noise — for zero-mean error $\mathbb{E}[\max(Q+\varepsilon)] \ge \max Q$, the max hunts the luckiest positive error. TD3's answers are twin critics with a $\min$ target, $y = r + \gamma\,\min_i \bar{Q}_i(s', a')$, which biases toward *under*estimation and is self-correcting because the policy just avoids actions it underrates instead of chasing ones it overrates; clipped noise on the target action, $a' = \bar\pi(s') + \mathrm{clip}(\mathcal{N}(0,0.2), -0.5, 0.5)$, so the target cannot overfit a razor-thin $Q$ spike; and delayed actor/target updates every two steps, so critic error settles before each policy move. I keep all three. But none of them keep the policy near the data — offline, the actor walks off the demo tube and the $\min$ of two unconstrained off-support values is still garbage.

The cheapest offline fix is the minimalist one: add a behavior-cloning penalty to the actor so it pays a squared cost for deviating from the logged action. There is one scale subtlety to get right. The BC term is bounded (actions in $[-1,1]$, so at most $\sim 4$) but $Q$ scales with the arbitrary reward magnitude, so on a high-reward task $Q$ dwarfs the BC term and on a low-reward task the BC term dominates. The fix is to normalize the value term by the average magnitude of $Q$ itself, $\lambda = 1/\mathrm{mean}(|Q|)$, used as a *stop-gradient* scalar — it rescales the loss, it does not change the direction of the $Q$ gradient. So the actor objective is

$$\max_\pi \; \mathbb{E}\big[\, \lambda\, Q(s,\pi(s)) - \beta_{\text{actor}}\,\textstyle\sum (\pi(s)-a)^2 \,\big],$$

an *explicit* squared penalty on a *deterministic* policy — and that is my floor. The contrast with the previous rung is already sharp: deterministic removes the sampled-$V$ variance that swung Pen seventy points, and explicit lets me control conservativeness directly per task.

But one BC term on the actor is not the end, and the leak it leaves is the same one that floored the previous rung, just relocated. I regularized the actor at the training states $s$: at $s$ the policy is pulled toward $a$. But the critic target is $r + \gamma\,\min\bar{Q}(s', a')$ with $a' = \bar\pi(s') + \text{noise}$, generated at the *next* state $s'$, and nothing in the actor's BC term at $s$ guarantees $\pi(s')$ is in-distribution at $s'$. So the bootstrap can still pick an off-support $a'$, the critic can still overrate it, and the overestimation loop reopens one bootstrap step downstream from where I patched it. On Hammer this is exactly why the floor sticks: the long contact sequence means the target at every step depends on a next-action the actor BC never constrained. The actor penalty fixes "what action do I take at states I've seen"; it does nothing about "what action does the target assume I take next" — different leaks.

So I want a penalty *inside the critic target* too. The behavior-regularized actor-critic framing names exactly two places to inject a divergence $D(\pi(\cdot|s), \pi_\beta(\cdot|s))$: in the actor objective (the BC term I have) or in the critic target (a value penalty, subtracting $\alpha\,D$ from the bootstrap so it is pessimistic exactly where the policy departs from behavior). The two are complementary, and TD3+BC simply never took the value-penalty half. The general framework wrote $D$ as KL/MMD/Wasserstein and needed a learned $\pi_\beta$ — the very behavior model that is too hard to fit on twenty-five narrow trajectories. But my policy is *deterministic*: $\pi(s)$ is a point, not a distribution, so the natural divergence between the policy's action and the data's action is just the squared Euclidean distance. And the harness already hands me what I need — its dataset converter preserves the dataset's *own next action* $\hat a'$ in the batch $(s, a, r, s', \text{done}, \hat a')$. So the value penalty needs no behavior model at all. The critic target, written out, is a smoothed next action $a' = \mathrm{clip}(\bar\pi(s') + \mathrm{clip}(\mathcal{N}(0,0.2),-0.5,0.5), -1, 1)$, the bootstrap $q = \min_i \bar{Q}_i(s', a')$, the value penalty applied,

$$q \leftarrow \min_i \bar{Q}_i(s', a') - \beta_{\text{critic}}\,\textstyle\sum (a' - \hat a')^2, \qquad y = r + \gamma(1-\text{done})\,q.$$

This costs no extra network and nothing against the 256-width parameter budget — which matters, because that budget forbids me from buying capacity my way out of the problem.

The piece the framework left coupled is whether $\beta_{\text{actor}}$ and $\beta_{\text{critic}}$ should be the same number. They do different jobs: the actor penalty controls how conservative the policy is *when it acts*, how willing it is to leave the logged action to chase value; the critic penalty controls how distrustful the *bootstrap* is of off-support next-actions. Forcing one coefficient onto both gives every task a single point on a one-dimensional trade-off when the real trade-off is two-dimensional, so I decouple them and tune per task. This matters acutely here because the three Adroit datasets are *not* alike: Pen `human` is a tight near-expert tube, Hammer `human` a long narrow contact-heavy sequence, Door `cloned` a behavior-cloned mixture, broader and noisier. I set Pen $(\beta_{\text{actor}}, \beta_{\text{critic}}) = (0.1, 0.5)$, Hammer $(0.01, 0.5)$, Door (the `door-cloned` dataset) $(0.01, 0.1)$. The pattern is legible — Hammer and Door take a *tiny* actor penalty (let the policy move, because welding it to the demos was exactly what floored it before) while keeping a real critic penalty on Hammer (guard the long bootstrap) and relaxing both on the broader cloned Door data.

One architectural choice I will not inherit blindly, and that I expect to be load-bearing for *this* disease: LayerNorm in the critic. The whole problem is the critic extrapolating wildly on off-support actions. If the last hidden feature $\psi$ feeding the output head $w$ is layer-normalized, its norm is a bounded constant for *any* input, so by Cauchy-Schwarz $|Q(s,a)| = |w^\top \mathrm{relu}(\psi)| \le \|w\|\cdot\|\mathrm{relu}(\psi)\| \le \|w\|$ — a hard cap on the value of *any* action, including ones the twenty-five demos never contain. That kills the runaway-extrapolation engine that drove the previous rung's seed-to-seed instability, and it does so without telling the policy anything. So post-activation LayerNorm goes in the critic, between every hidden layer, and *not* in the actor — the actor is bounded into $[-1,1]$ by a tanh and pulled to data by $\beta_{\text{actor}}$; it is not the surface that extrapolates dangerous values. Asymmetric on purpose. I keep the TD3+BC value normalization $\lambda = 1/\mathrm{mean}(|Q|)$ on the actor, keep `policy_freq = 2`, $\tau = 5\times10^{-3}$, use CORL-style init (uniform fan-in hidden, small-uniform output), and run $\text{lr} = 3\times10^{-4}$. I do *not* turn on state normalization here, and I do *not* use a Monte-Carlo return floor.

My falsifiable expectations against the previous numbers: Pen should *tighten*, the seed spread well under the seventy-point swing even if the mean lands near or modestly above 67. Hammer I expect to stay low — a tiny actor penalty plus a guarded bootstrap may lift it off the absolute floor, but the long contact sequence is the hardest thing here. Door I expect to stay near zero on the cloned dataset. If Pen tightens but Hammer and Door barely move, that is the signal that the *explicit* TD3-style constraint, even decoupled, is still doing only one-step improvement off a deterministic policy — and the next move is a method that does genuine multi-step in-support dynamic programming without ever querying an unseen action.

```python
# EDITABLE region of custom_adroit.py — step 2: ReBRAC
# DeterministicActor body replaced (3x256, no LayerNorm, CORL-style init):
class DeterministicActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )
        import math
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                fan_in = layer.in_features
                if i < len(self.net) - 2:  # hidden layers
                    bound = math.sqrt(1.0 / fan_in)
                    nn.init.uniform_(layer.weight, -bound, bound)
                    nn.init.constant_(layer.bias, 0.1)
                else:  # output layer
                    nn.init.uniform_(layer.weight, -1e-3, 1e-3)
                    nn.init.uniform_(layer.bias, -1e-3, 1e-3)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    """Q with post-activation LayerNorm (ReBRAC critic_ln=True). 3x256 MLP."""

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 1),
        )
        import math
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Linear):
                fan_in = layer.in_features
                if i < len(self.net) - 1:  # hidden layers
                    bound = math.sqrt(1.0 / fan_in)
                    nn.init.uniform_(layer.weight, -bound, bound)
                    nn.init.constant_(layer.bias, 0.1)
                else:  # output layer
                    nn.init.uniform_(layer.weight, -3e-3, 3e-3)
                    nn.init.uniform_(layer.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class OfflineAlgorithm:
    """ReBRAC — TD3+BC with critic BC regularization in the Bellman target."""

    def __init__(self, state_dim, action_dim, max_action, replay_buffer=None,
                 discount=0.99, tau=5e-3, actor_lr=3e-4, critic_lr=3e-4,
                 alpha_lr=3e-4, orthogonal_init=True, device="cuda"):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0

        env_name = os.environ.get("ENV", "")
        if "hammer" in env_name:
            self.actor_bc_coef = 0.01
            self.critic_bc_coef = 0.5
        elif "door-cloned" in env_name:
            self.actor_bc_coef = 0.01
            self.critic_bc_coef = 0.1
        elif "door" in env_name:
            self.actor_bc_coef = 0.1
            self.critic_bc_coef = 0.1
        else:  # pen (default)
            self.actor_bc_coef = 0.1
            self.critic_bc_coef = 0.5
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.normalize_q = True

        self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        states, actions, rewards, next_states, dones, next_actions_data = batch
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)
        log_dict: Dict[str, float] = {}

        # Critic update
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions_policy = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)

            bc_penalty = ((next_actions_policy - next_actions_data) ** 2).sum(-1)
            target_q1 = self.critic_1_target(next_states, next_actions_policy)
            target_q2 = self.critic_2_target(next_states, next_actions_policy)
            next_q = torch.min(target_q1, target_q2)
            next_q = next_q - self.critic_bc_coef * bc_penalty
            target_q = rewards + (1.0 - dones) * self.discount * next_q

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Delayed actor update
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(states)
            bc_penalty_actor = ((pi - actions) ** 2).sum(-1)
            q_values = torch.min(self.critic_1(states, pi), self.critic_2(states, pi))

            lmbda = 1.0
            if self.normalize_q:
                lmbda = 1.0 / q_values.abs().mean().detach()

            actor_loss = (self.actor_bc_coef * bc_penalty_actor - lmbda * q_values).mean()
            log_dict["actor_loss"] = actor_loss.item()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict
```
