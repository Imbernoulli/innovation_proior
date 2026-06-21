TD3 won the board, and won it the way I hoped — strong on all three at once rather than trading one region for another: HalfCheetah recovered to 11055, Reacher snapped back to $-3.83$ (level with DDPG's $-3.94$), and Ant came in highest of all at 4642. The deterministic actor reclaimed the entropy tax where I predicted, and the twin-min carried the overestimation control that fixed Ant. But the residual I flagged is sitting in the Ant column: the per-seed returns are $\{4763, 3407, 5757\}$, a spread of over two thousand, the widest on the board, with one seed lagging the best by 40%. That is strong-on-average, not strong-and-stable, and I named the suspect already — every rung so far learns its critic through *stale target networks*. The smoothed twin-min target TD3 reads is computed off slow copies $Q_{\theta'}$, lagged by $\tau=0.005$, and a lagged bootstrap is a slower, noisier learning signal, most damaging on the high-dimensional task where the critic has the most to learn. So the move is to attack the target network itself.

First I re-derive *why* it is there, because "the bootstrap would chase itself" is not specific enough to attack. The critic minimizes $\big(Q_\theta(s,a) - [r + \gamma Q_\theta(s',a')]\big)^2$ with the next-state value already detached, so there is no degenerate shrink-the-target gradient path; the question is what diverges when that detached value comes from the *live* critic. Two things. One, the target value moves every step because $\theta$ moves — the DQN non-stationarity story. Two, and I think this is the one doing the damage, the input distributions at $(s,a)$ and $(s',a')$ are *not the same*: the actions $a$ at the current state come from the replay buffer, a mixture of stale policies, while the actions $a'$ at the next state come from the *current* policy. A deep net's behavior on one input cloud is unconstrained by its behavior on another, so if the critic drifts to different scales on the $(s,a)$ cloud versus the $(s',a')$ cloud — and nothing prevents it — the Bellman equation relates two numbers on different scales and the recursion amplifies the mismatch. The target network masks this by freezing one side so the mismatch cannot run away; but it does not address it, it slows everything enough to contain it, at the cost of the staleness now hurting Ant. That reframing is the lever: if the disease is a distribution mismatch between the current and next state-action inputs to one critic, the cure is not "freeze one side" but "make the two sides share a normalized distribution so the critic sees them as one population."

I propose CrossQ: normalize the mismatch away inside the critic and the target network becomes redundant. The standard tool whose whole job is to normalize a layer's inputs to a controlled distribution is Batch Normalization — but BatchNorm has a bad reputation in value learning, and the failure mode names the fix. Insert vanilla BatchNorm the obvious way: forward $(s,a)$ for the prediction, *separately* forward $(s',a')$ for the bootstrap, regress. The first pass normalizes by the $(s,a)$ batch's moments, the second by the $(s',a')$ batch's moments, and those statistics differ — exactly because the two batches have different distributions, the mismatch I am fixing — so the critic is literally a *different function* in the two passes, same weights, different effective transform, and the Bellman equation now relates the outputs of two non-identical functions, which is incoherent. That is precisely why naive BatchNorm destabilizes critics. So do not let the normalization see the two batches as two populations: *concatenate* them. Stack $(s,a)$ on top of $(s',a')$ into one batch of size $2N$, do a *single* forward pass through the normalized critic, split the output into current-state predictions and next-state values. The normalization now computes its moments from the *union* of both sub-batches — one shared distribution — so every input, current or next, is normalized identically and the critic is one consistent function across both, with prediction and bootstrap on the same scale by construction. This is almost free: one concatenation, one forward pass (cheaper than two separate passes), one split. The next-state half is detached before forming the target — it is a bootstrap value, no gradient flows into it — but it shares the forward pass and therefore the normalization with the current-state half. That shared normalization is what the target network was crudely approximating, and now I get it exactly, from the *live* critic, with no lag.

The target network played two roles — stationarity and cross-batch consistency — and the joint-batch normalization takes over both. It directly kills the cross-batch mismatch (both sides share moments) and substantially tames the non-stationarity, because fixing the scale and center of each layer's activations every step makes the representation the later layers see far more stationary than raw activations even as $\theta$ moves. So I delete the target network and the $\tau$ knob entirely; the bootstrap value is the live critic's next-state output, detached, normalized jointly with the prediction. One refinement, because plain BatchNorm is not quite enough under RL's non-stationarity: BatchNorm normalizes by *batch* statistics in training but by *running* statistics at inference, and that gap is harmful here — the policy changes, the replay distribution drifts, the running statistics chase a moving target, and a minibatch's stats can swing far from them. Batch Renormalization (Ioffe 2017) is the patch: keep normalizing by batch statistics but add a clipped affine correction $(r, d)$ tying the batch normalization back to the running statistics, with $r$ clamped to $[1/r_{\max}, r_{\max}]$, $d$ to $[-d_{\max}, d_{\max}]$, and both *detached* (constants, not differentiated), so $\hat x = \frac{x - \mu_{\text{batch}}}{\sigma_{\text{batch}}}\,r + d$. That keeps training-time and running-time normalization consistent under drifting data, which is the robustness the critic needs.

Everything else stays SAC, deliberately, because the contribution is a normalization change and I do not want to confound it: the stochastic tanh-Gaussian actor (reparameterized, with the squash log-prob correction, returning $(\text{action}, \log\text{-prob}, \text{mean})$ so the loop unwraps it), twin critics with a $\min$ next-state target minus the entropy term, and automatic temperature tuning to $\mathcal{H}_{\text{target}} = -\dim(A)$. The maximum-entropy objective is back, but unlike rung 2's concern, evaluation reads the policy *mean*, so the entropy serves exploration during training while evaluation stays the sharp mean — no entropy tax — and the real lever is the normalized, target-free critic, not the actor. Two harness facts I am honest about, because this configuration is not the paper's. The network dimensions are fixed by a parameter-count check, so the critic is 256-wide, not the paper's 2048; the paper's accuracy comes partly from that widening, which the normalization makes trainable, so I expect only the *mechanism's* benefit — un-stale bootstrap, consistent cross-batch scale — not the extra-capacity benefit. And with the target network gone there is no lagged target to settle behind, so I update the actor *every* step (`policy_frequency=1`) rather than every two. I also toggle the critics to `eval()` for the actor's Q-evaluation, since the actor reads the critic on a single current-state batch that should use running statistics rather than recompute one-sided batch stats. The bar is TD3's measured board, and the headline test is Ant: a target-free, jointly-normalized critic should *tighten* the $\{4763, 3407, 5757\}$ spread — the lagging seed comes up because an un-stale bootstrap learns the high-dimensional critic faster and more consistently — at a mean $\ge 4642$, while HalfCheetah and Reacher, near saturation, should hold rather than jump. Stripped of the paper's widening the honest expectation is parity-to-better with a tighter Ant; what this rung establishes for the ladder is that the target network — the one fixture every prior rung inherited unquestioned — is removable, and that the seed-to-seed instability still visible in the strongest baseline's Ant column is exactly what a target-free, jointly-normalized critic is built to cure.

```python
LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """Stochastic Tanh-Gaussian actor for CrossQ."""

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


class BatchRenorm1d(nn.Module):
    """Batch Renormalization (Ioffe, 2017) for 1-D inputs.

    More stable than standard BatchNorm for non-stationary data (e.g. RL)
    because it clips the correction terms r and d, preventing the running
    statistics from causing large deviations during training.
    """

    def __init__(self, num_features, momentum=0.01, eps=1e-5, rmax=3.0, dmax=5.0):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.rmax = rmax
        self.dmax = dmax
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_std = (batch_var + self.eps).sqrt()
            running_std = (self.running_var + self.eps).sqrt()

            # Renormalization corrections (clamped for stability)
            r = (batch_std / running_std).detach().clamp(1.0 / self.rmax, self.rmax)
            d = ((batch_mean - self.running_mean) / running_std).detach().clamp(-self.dmax, self.dmax)

            # Normalize using batch stats, then correct
            x_hat = (x - batch_mean) / batch_std * r + d

            # Update running statistics
            self.running_mean.mul_(1 - self.momentum).add_(batch_mean.detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(batch_var.detach() * self.momentum)
        else:
            # Inference: use running statistics (same as standard BN)
            x_hat = (x - self.running_mean) / (self.running_var + self.eps).sqrt()

        return self.weight * x_hat + self.bias


class QNetwork(nn.Module):
    """Q-function Q(s, a) -> scalar with BatchRenorm (CrossQ)."""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.bn1 = BatchRenorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = BatchRenorm1d(256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


class OffPolicyAlgorithm:
    """CrossQ — BatchRenorm-based off-policy actor-critic without target networks."""

    def __init__(self, obs_dim, action_dim, max_action, device, args):
        self.device = device
        self.max_action = max_action
        self.gamma = args.gamma
        self.policy_frequency = 1  # CrossQ paper default: update actor every step
        self.total_it = 0

        self.actor = Actor(obs_dim, action_dim, max_action).to(device)

        # Twin Q-networks with BatchRenorm — NO target networks
        self.qf1 = QNetwork(obs_dim, action_dim).to(device)
        self.qf2 = QNetwork(obs_dim, action_dim).to(device)

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

    def _q_forward_cross(self, qf, obs, actions, next_obs, next_actions):
        """Forward through Q-network on concatenated (current, next) batch.

        BatchRenorm sees both sub-batches, preventing bootstrap leakage.
        Returns (q_current, q_next) after splitting.
        """
        cat_obs = torch.cat([obs, next_obs], dim=0)
        cat_act = torch.cat([actions, next_actions], dim=0)
        q_all = qf(cat_obs, cat_act)
        q_current, q_next = torch.split(q_all, [obs.shape[0], next_obs.shape[0]], dim=0)
        return q_current, q_next

    def update(self, batch):
        self.total_it += 1
        obs, next_obs, actions, rewards, dones = batch

        # --- Critic update ---
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.get_action(next_obs)

        # CrossQ: forward current and next through Q-networks together
        self.qf1.train()
        self.qf2.train()
        q1_current, q1_next = self._q_forward_cross(self.qf1, obs, actions, next_obs, next_actions)
        q2_current, q2_next = self._q_forward_cross(self.qf2, obs, actions, next_obs, next_actions)

        # Detach q_next to prevent any gradient flow and free memory
        with torch.no_grad():
            min_q_next = torch.min(q1_next.detach().view(-1), q2_next.detach().view(-1)) - self.alpha * next_log_pi.view(-1)
            td_target = rewards + (1 - dones) * self.gamma * min_q_next

        critic_loss = F.mse_loss(q1_current.view(-1), td_target) + F.mse_loss(q2_current.view(-1), td_target)

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        # --- Actor update ---
        actor_loss_val = 0.0
        if self.total_it % self.policy_frequency == 0:
            pi, log_pi, _ = self.actor.get_action(obs)
            self.qf1.eval()
            self.qf2.eval()
            q1_pi = self.qf1(obs, pi).view(-1)
            q2_pi = self.qf2(obs, pi).view(-1)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * log_pi.view(-1) - min_q_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            # Update alpha (reuse log_pi from actor update, detached)
            alpha_loss = (-self.log_alpha.exp() * (log_pi.detach().view(-1) + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val, "alpha": self.alpha}
```
