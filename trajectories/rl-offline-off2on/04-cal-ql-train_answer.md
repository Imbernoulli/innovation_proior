IQL delivered what I predicted and showed me what is left on the table. Hammer-cloned climbed from AWAC's dead-flat 0.336 to a mean of 76.7 — and on *every* seed (69.0 / 68.4 / 92.7), not one lucky run: the in-support expectile backup did the stitching AWAC could not. Pen-cloned rose to 103.0, beating AWAC's 63.7, and hammer-expert held at 129.7. So IQL strictly dominates the other two baselines, and with a *no-op* transition — no schedule to tune. But that very property is the limit. IQL handles the offline→online handoff well precisely because it does *nothing special* there, which means it also does nothing to *exploit* the transition. Its value function is whatever the expectile regression produced; it is never deliberately *scaled* to make the first online updates cheap. The unclaimed axis is no longer "pretrain stably and not collapse" — IQL answers that — but "what property must the offline value function have so that online fine-tuning is not merely stable but *fast and monotone*, so the online budget is spent climbing, not repairing." To get there I go back to the one offline family the ladder skipped: conservative value regularization (CQL), the strongest *offline* method but a poor *initializer* — its $Q$ is a correct lower bound that is *uncalibrated in scale*, driven far below the true return, so at the online transition the first real returns tower over it, the critic lurches to repair the scale, and the policy dips before recovering.

I propose **Cal-QL** — calibrated Q-learning. Run CQL's transition in the head to locate the dip precisely. Offline, CQL's regularizer $\alpha\big(\mathbb{E}_{s\sim D}[\log\sum_a \exp Q(s,a)] - \mathbb{E}_{(s,a)\sim D}[Q(s,a)]\big)$ drives $Q$ down by a large, essentially arbitrary margin — far beneath, even far below zero, the true positive returns of these Adroit tasks. Switch online: real transitions arrive, the TD targets now contain actual sampled returns (large and positive), the TD error is enormous because the target towers over the depressed $Q$, the critic makes a big fast correction, and the greedy policy riding a $Q$ mid-lurch and temporarily incoherent across actions degrades. That is the dip — and crucially it is caused not by the lower-bound *direction* of CQL's conservatism (which I want to keep, for the sharp OOD suppression IQL never had) but by the unbounded *magnitude* of it. So the fix is to keep CQL's push-down direction and bound its magnitude: pin $Q$ to a sane scale.

What makes Cal-QL work is a *calibrated* value — a lower bound on the policy value *and* an upper bound on a trusted reference,

$$V^\mu(s) \;\le\; Q_{\text{offline}}(s, \pi(s)) \;\le\; Q^\pi(s).$$

The right inequality is CQL's lower bound; the left is the new constraint — do not let $Q$ drop below the reference. The reference must be computable from the dataset with no model and must be a meaningful floor, and the behavior policy is the natural choice: it generated the data, so its value is a sane scale and its returns are sitting in the dataset. For a transition at time $t$, the discounted sum of the rewards that actually followed, $R_t = \sum_{k\ge0}\gamma^k r_{t+k}$, is an unbiased Monte-Carlo sample of the behavior policy's return-to-go — free, by one reverse scan over each trajectory. So $V^\mu$ is the MC return-to-go, and the calibration floor is: never let the conservative $Q$ be pushed below $R_t$.

Where it enters is one line. CQL's push-down is the importance-sampled log-sum-exp over candidate actions (random + current-policy + next-policy); minimizing it lowers $Q$ on those candidates, and unchecked that is what drives $Q$ arbitrarily low. So before the log-sum-exp, floor the *policy-action* candidates by the reference,

$$\tilde Q(s,a) = \max\big(Q(s,a),\, R(s)\big).$$

Wherever the penalty would push a policy-action $Q$ below the behavior return, the $\max$ kills its gradient and the floor holds; above the floor CQL is unchanged. This only ever *raises* clamped entries toward the truth, so $Q$ stays a (looser) lower bound on $Q^\pi$ — offline safety preserved — while the floor enforces the $V^\mu$ upper bound. The value is boxed into $[V^\mu, Q^\pi]$, near the real scale, so when online data arrives the TD targets are already close to $Q$, the early correction is small, and the dip that plagued raw CQL is gone — while I kept the sharp push-down IQL never had. The calibration adds *zero* parameters; it is a `max`, which matters because the multi-action CQL critics plus the entropy `Scalar`s already sit just under the $1.2\times$ budget, so the gain has to be algorithmic.

The transition handling is now the *active* ingredient, the thing IQL left blank. Offline, the floor gives the good scale. Online, real feedback teaches the critic the true value, and the behavior reference is a *suboptimal* floor I now want to climb *above* — so at the handoff I **disable calibration** (drop the `max`, set $\alpha$ to its online value) and let the value rise past the behavior return as online returns warrant. The MC floor for newly collected online transitions is meaningless under the shifting distribution and is set to zero. And to keep the critic anchored to the offline scale while it learns online, I sample a fixed 50/50 mixing ratio of offline and online transitions during fine-tuning. So unlike IQL's no-op `on_online_start`, Cal-QL's transition is a deliberate switch: calibration did its one job — hand the online phase a well-scaled initialization — and then steps aside.

In this harness the Cal-QL fill is specific, because the scaffold exposes some of this and omits some. The MC returns are precomputed in `__init__` from the replay buffer by a reverse scan that detects episode boundaries from terminals *and* from state discontinuities ($\lVert s_{t+1} - s'_t \rVert$), since the harness's flat buffer carries no explicit trajectory index, and the maximum episode length is inferred from the data (Adroit $= 200$). The buffer's `sample` is *monkey-patched* to return the MC return as a 7th batch element and, when online, to draw the 50/50 offline/online mix; `add_transition` is patched to write $\texttt{mc} = 0$ for online entries. The actor is the template's $3\times256$ `TanhGaussianPolicy` but with learnable `log_std` multiplier/offset `Scalar`s and orthogonal init, supporting `repeat=n` to emit $n$ candidate actions; the critic is $3\times256$ with multi-action support (it reshapes $(B, n, \text{act})$ inputs). Automatic entropy tuning uses target entropy $-\dim(A)$, $\texttt{policy\_lr} = 1\mathrm{e}{-4}$, $\texttt{qf\_lr} = 3\mathrm{e}{-4}$. The calibration is applied to the current- and next-policy candidate Q-values via `torch.maximum(..., lower)`, exactly as derived; $\texttt{cql\_max\_target\_backup} = \text{True}$ takes the in-support max over $\texttt{cql\_n\_actions} = 10$ next actions; $\texttt{cql\_clip\_diff\_min} = -200$. `on_online_start` flips `_calibration_enabled` off and sets $\texttt{cql\_alpha} = \texttt{cql\_alpha\_online} = 1.0$, with `CONFIG_OVERRIDES = {"normalize": False}`. The harness exposes no separate larger online replay or per-task $\alpha$ schedule, so the online improvement rests entirely on calibration plus the 50/50 mixing — which is exactly the point: keep IQL's stability, add CQL's sharp push-down, and bound its magnitude with a free MC-return floor so the online phase climbs on purpose rather than by omission.

```python
CONFIG_OVERRIDES: Dict[str, Any] = {"normalize": False}


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def _calql_init_module_weights(module, orthogonal_init=False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class Actor(nn.Module):
    """TanhGaussianPolicy with learnable log_std multiplier/offset and orthogonal init."""

    def __init__(self, state_dim, action_dim, max_action,
                 log_std_multiplier=1.0, log_std_offset=-1.0, orthogonal_init=False):
        super().__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.base_network = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2 * action_dim),
        )
        if orthogonal_init:
            self.base_network.apply(lambda m: _calql_init_module_weights(m, True))
        else:
            _calql_init_module_weights(self.base_network[-1], False)
        self.log_std_multiplier = Scalar(log_std_multiplier)
        self.log_std_offset = Scalar(log_std_offset)
        self.log_std_min = -20.0
        self.log_std_max = 2.0

    def _get_dist(self, observations, repeat=None):
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        output = self.base_network(observations)
        mean, log_std = torch.split(output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return TransformedDistribution(
            Normal(mean, torch.exp(log_std)), TanhTransform(cache_size=1)
        ), mean

    def forward(self, observations, deterministic=False, repeat=None):
        dist, mean = self._get_dist(observations, repeat)
        action = torch.tanh(mean) if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return self.max_action * action, log_prob

    def log_prob(self, observations, actions):
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = torch.split(base_network_output, self.action_dim, dim=-1)
        log_std = self.log_std_multiplier() * log_std + self.log_std_offset()
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        dist = TransformedDistribution(
            Normal(mean, torch.exp(log_std)), TanhTransform(cache_size=1)
        )
        scaled = torch.clamp(actions / self.max_action, -1.0 + 1e-6, 1.0 - 1e-6)
        return dist.log_prob(scaled).sum(-1)

    @torch.no_grad()
    def act(self, state, device="cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        actions, _ = self(state, not self.training)
        return actions.cpu().data.numpy().flatten()


class Critic(nn.Module):
    """FullyConnectedQFunction — 3 hidden layers, multi-action support."""

    def __init__(self, state_dim, action_dim, orthogonal_init=False, n_hidden_layers=3):
        super().__init__()
        layers = [nn.Linear(state_dim + action_dim, 256), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(256, 256))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))
        self.network = nn.Sequential(*layers)
        if orthogonal_init:
            self.network.apply(lambda m: _calql_init_module_weights(m, True))
        else:
            _calql_init_module_weights(self.network[-1], False)

    def forward(self, observations, actions):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = torch.squeeze(self.network(torch.cat([observations, actions], dim=-1)), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values


class OfflineOnlineAlgorithm:
    """Cal-QL — Calibrated Q-Learning for offline-to-online RL."""

    def __init__(self, state_dim, action_dim, max_action, replay_buffer=None,
                 discount=0.99, tau=5e-3, actor_lr=3e-4, critic_lr=3e-4, device="cuda"):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0

        self.cql_n_actions = 10
        self.cql_temp = 1.0
        self.cql_alpha = 1.0
        self.cql_alpha_online = 1.0
        self.cql_importance_sample = True
        self.cql_max_target_backup = True
        self.cql_clip_diff_min = -200.0
        self.cql_clip_diff_max = float('inf')
        self.target_entropy = -float(action_dim)
        self.alpha_multiplier = 1.0
        self.use_automatic_entropy_tuning = True
        self.backup_entropy = False
        self.bc_steps = 0
        self.policy_lr = 1e-4
        self.qf_lr = critic_lr
        self.mixing_ratio = 0.5
        self._calibration_enabled = True
        self._offline_size = 0

        self._replay_buffer = replay_buffer
        if replay_buffer is not None:
            self._offline_size = replay_buffer._size
            self._setup_mc_returns(replay_buffer, discount)

        self.actor = Actor(state_dim, action_dim, max_action, orthogonal_init=True).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.policy_lr)

        self.critic_1 = Critic(state_dim, action_dim, orthogonal_init=True).to(device)
        self.critic_2 = Critic(state_dim, action_dim, orthogonal_init=True).to(device)
        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.qf_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.qf_lr)

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(self.log_alpha.parameters(), lr=self.policy_lr)
        else:
            self.log_alpha = None

    def switch_calibration(self):
        self._calibration_enabled = not self._calibration_enabled

    def _setup_mc_returns(self, buf, discount):
        n = buf._size
        rewards = buf._rewards[:n].squeeze(-1).cpu().numpy()
        dones = buf._dones[:n].squeeze(-1).cpu().numpy()
        states = buf._states[:n].cpu().numpy()
        next_states = buf._next_states[:n].cpu().numpy()

        _ep_lens, _el = [], 0
        for t in range(n):
            _el += 1
            if dones[t] or t == n - 1 or (t < n - 1 and np.linalg.norm(states[t + 1] - next_states[t]) > 1e-6):
                _ep_lens.append(_el)
                _el = 0
        _max_ep_steps = max(_ep_lens) if _ep_lens else 200

        mc_returns = np.zeros(n, dtype=np.float32)
        ep_start, ep_len, cur_rewards, terminals = 0, 0, [], []
        for t in range(n):
            cur_rewards.append(float(rewards[t]))
            terminals.append(float(dones[t]))
            ep_len += 1
            is_last_step = (
                (t == n - 1)
                or (t < n - 1 and np.linalg.norm(states[t + 1] - next_states[t]) > 1e-6)
                or ep_len == _max_ep_steps
            )
            if dones[t] or is_last_step:
                prev_return = 0.0
                for i in reversed(range(ep_len)):
                    cur_rewards[i] = cur_rewards[i] + discount * prev_return * (1 - terminals[i])
                    prev_return = cur_rewards[i]
                mc_returns[ep_start:ep_start + ep_len] = cur_rewards
                ep_start, ep_len, cur_rewards, terminals = t + 1, 0, [], []

        buf._mc_returns = torch.zeros((buf._buffer_size, 1), dtype=torch.float32, device=buf._device)
        buf._mc_returns[:n] = torch.tensor(mc_returns, dtype=torch.float32, device=buf._device).unsqueeze(-1)

        offline_size = n
        mixing_ratio = self.mixing_ratio

        def _sample_with_mc(batch_size, is_online=False):
            if is_online and buf._size > offline_size:
                n_offline = int(batch_size * mixing_ratio)
                n_online = batch_size - n_offline
                off_idx = np.random.randint(0, offline_size, size=n_offline)
                on_idx = np.random.randint(offline_size, buf._size, size=n_online)
                indices = np.concatenate([off_idx, on_idx])
            else:
                indices = np.random.randint(0, buf._size, size=batch_size)
            return [buf._states[indices], buf._actions[indices], buf._rewards[indices],
                    buf._next_states[indices], buf._dones[indices], buf._next_actions[indices],
                    buf._mc_returns[indices]]
        buf.sample = _sample_with_mc

        _orig_add = buf.add_transition

        def _add_with_mc(state, action, reward, next_state, done):
            idx = buf._pointer
            _orig_add(state, action, reward, next_state, done)
            buf._mc_returns[idx] = 0.0
        buf.add_transition = _add_with_mc

    def _alpha_and_alpha_loss(self, observations, log_pi):
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (log_pi + self.target_entropy).detach()).mean()
            alpha = torch.clamp(self.log_alpha().exp() * self.alpha_multiplier, min=1e-6, max=100.0)
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(self, observations, actions, new_actions, alpha, log_pi):
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(observations, actions)
            policy_loss = (alpha * log_pi - log_probs).mean()
        else:
            q_new_actions = torch.min(
                self.critic_1(observations, new_actions),
                self.critic_2(observations, new_actions),
            )
            policy_loss = (alpha * log_pi - q_new_actions).mean()
        return policy_loss

    def _q_loss(self, observations, actions, next_observations, rewards, dones, mc_returns, alpha, log_dict):
        q1_predicted = self.critic_1(observations, actions)
        q2_predicted = self.critic_2(observations, actions)

        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(next_observations, repeat=self.cql_n_actions)
            target_q_values, max_target_indices = torch.max(
                torch.min(
                    self.target_critic_1(next_observations, new_next_actions),
                    self.target_critic_2(next_observations, new_next_actions),
                ), dim=-1,
            )
            next_log_pi = torch.gather(next_log_pi, -1, max_target_indices.unsqueeze(-1)).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.actor(next_observations)
            target_q_values = torch.min(
                self.target_critic_1(next_observations, new_next_actions),
                self.target_critic_2(next_observations, new_next_actions),
            )

        if self.backup_entropy:
            target_q_values = target_q_values - alpha * next_log_pi

        target_q_values = target_q_values.unsqueeze(-1)
        td_target = (rewards + (1.0 - dones) * self.discount * target_q_values.detach()).squeeze(-1)
        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        batch_size = actions.shape[0]
        action_dim = actions.shape[-1]
        cql_random_actions = actions.new_empty(
            (batch_size, self.cql_n_actions, action_dim), requires_grad=False
        ).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.actor(observations, repeat=self.cql_n_actions)
        cql_next_actions, cql_next_log_pis = self.actor(next_observations, repeat=self.cql_n_actions)
        cql_current_actions = cql_current_actions.detach()
        cql_current_log_pis = cql_current_log_pis.detach()
        cql_next_actions = cql_next_actions.detach()
        cql_next_log_pis = cql_next_log_pis.detach()

        cql_q1_rand = self.critic_1(observations, cql_random_actions)
        cql_q2_rand = self.critic_2(observations, cql_random_actions)
        cql_q1_current_actions = self.critic_1(observations, cql_current_actions)
        cql_q2_current_actions = self.critic_2(observations, cql_current_actions)
        cql_q1_next_actions = self.critic_1(observations, cql_next_actions)
        cql_q2_next_actions = self.critic_2(observations, cql_next_actions)

        lower_bounds = mc_returns.reshape(-1, 1).repeat(1, cql_q1_current_actions.shape[1])

        if self._calibration_enabled:        # Cal-QL: floor policy-action Q by MC return
            cql_q1_current_actions = torch.maximum(cql_q1_current_actions, lower_bounds)
            cql_q2_current_actions = torch.maximum(cql_q2_current_actions, lower_bounds)
            cql_q1_next_actions = torch.maximum(cql_q1_next_actions, lower_bounds)
            cql_q2_next_actions = torch.maximum(cql_q2_next_actions, lower_bounds)

        if self.cql_importance_sample:
            random_density = np.log(0.5 ** action_dim)
            cql_cat_q1 = torch.cat([
                cql_q1_rand - random_density,
                cql_q1_next_actions - cql_next_log_pis.detach(),
                cql_q1_current_actions - cql_current_log_pis.detach(),
            ], dim=1)
            cql_cat_q2 = torch.cat([
                cql_q2_rand - random_density,
                cql_q2_next_actions - cql_next_log_pis.detach(),
                cql_q2_current_actions - cql_current_log_pis.detach(),
            ], dim=1)
        else:
            cql_cat_q1 = torch.cat([
                cql_q1_rand, torch.unsqueeze(q1_predicted, 1),
                cql_q1_next_actions, cql_q1_current_actions,
            ], dim=1)
            cql_cat_q2 = torch.cat([
                cql_q2_rand, torch.unsqueeze(q2_predicted, 1),
                cql_q2_next_actions, cql_q2_current_actions,
            ], dim=1)

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        cql_qf1_diff = torch.clamp(cql_qf1_ood - q1_predicted, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()
        cql_qf2_diff = torch.clamp(cql_qf2_ood - q2_predicted, self.cql_clip_diff_min, self.cql_clip_diff_max).mean()

        cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
        cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        log_dict.update(dict(
            qf1_loss=qf1_loss.item(), qf2_loss=qf2_loss.item(),
            cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
            cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
        ))
        return qf_loss

    def train(self, batch: TensorBatch, is_online: bool = False) -> Dict[str, float]:
        self.total_it += 1
        if is_online and self._replay_buffer is not None:
            batch = self._replay_buffer.sample(256, is_online=True)
            batch = [b.to(self.device) for b in batch]

        if len(batch) == 7:
            observations, actions, rewards, next_observations, dones, _next_act, mc_returns = batch
        elif len(batch) == 6:
            observations, actions, rewards, next_observations, dones, _next_act = batch
            mc_returns = torch.zeros_like(rewards)
        else:
            observations, actions, rewards, next_observations, dones = batch
            mc_returns = torch.zeros_like(rewards)

        new_actions, log_pi = self.actor(observations)
        alpha, alpha_loss = self._alpha_and_alpha_loss(observations, log_pi)
        policy_loss = self._policy_loss(observations, actions, new_actions, alpha, log_pi)

        log_dict = dict(log_pi=log_pi.mean().item(), policy_loss=policy_loss.item(),
                        alpha_loss=alpha_loss.item(), alpha=alpha.item())

        qf_loss = self._q_loss(observations, actions, next_observations, rewards, dones,
                               mc_returns, alpha, log_dict)
        log_dict["critic_loss"] = qf_loss.item()

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        soft_update(self.target_critic_1, self.critic_1, self.tau)
        soft_update(self.target_critic_2, self.critic_2, self.tau)
        return log_dict

    def select_action(self, state: np.ndarray) -> np.ndarray:
        return self.actor.act(state, self.device)

    def on_online_start(self):
        # Cal-QL: disable calibration at the offline-to-online transition
        self.switch_calibration()
        self.cql_alpha = self.cql_alpha_online
```
