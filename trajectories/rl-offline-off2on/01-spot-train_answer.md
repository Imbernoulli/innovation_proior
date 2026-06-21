The scaffold default for this task is a plain twin-critic actor-critic: the critics fit the Bellman target $r + \gamma\, Q_{\text{target}}(s', \pi(s'))$ and a deterministic actor maximizes $Q(s, \pi(s))$. On the Adroit `cloned-v1` mixtures — expert demonstrations diluted with noisy ones, where the behavior policy covers only a thin sliver of the 24-to-30-dimensional action cube — this diverges before any online interaction can help. The poison is in the target: $\pi(s')$ is the action the *actor* chose, not one the dataset contains, so the critic is queried off its training manifold where its extrapolation is essentially unbounded and skews *upward*, because the maximizing actor is actively attracted to whichever out-of-distribution action the critic happens to over-value. Those inflated values bootstrap back through the backup, the next actor update chases them harder, and the value function runs away — with no environment to punish the silly action offline, nothing stops it. The cure is not more capacity or a better optimizer; it is *forbidding the actor from leaving the region where the behavior policy actually put mass.*

I propose **SPOT** — support-constrained policy optimization built on TD3 with a VAE behavior model. The principled region to confine the actor to is the **support** of $\pi_\beta$: in state $s$, allow only actions with $\pi_\beta(a\mid s) > \varepsilon$. There is a clean reason to trust this rather than treat it as a hack. If I restrict the maximization inside the Bellman operator to the support set and call that backup $T_\varepsilon$, its fixed point $Q^*_\varepsilon$ is the supported optimal value, and writing $\alpha(\varepsilon) = \lVert T Q^* - T_\varepsilon Q^* \rVert_\infty$ for the worst-case one-step gap, the $\gamma$-contraction of $T_\varepsilon$ in sup-norm plus the triangle inequality gives

$$\lVert Q^* - Q^*_\varepsilon \rVert_\infty \le \gamma\,\lVert Q^* - Q^*_\varepsilon \rVert_\infty + \alpha(\varepsilon) \quad\Longrightarrow\quad \lVert Q^* - Q^*_\varepsilon \rVert_\infty \le \frac{\alpha(\varepsilon)}{1-\gamma}.$$

So confining myself to the support costs at most a controlled, contraction-amplified amount: a tighter support (bigger $\varepsilon$) shrinks the extrapolation risk but raises $\alpha(\varepsilon)$, a looser one does the reverse. Confinement is safe; the only question is how to enforce it.

What makes SPOT the right shape is how it enforces support. The parameterization camp (BCQ and descendants) bakes support into the architecture — fit a generative model of $\pi_\beta$, sample candidate actions, argmax $Q$ over them — which respects density by construction but welds the algorithm to a slow generative inference path at every action selection and makes the offline→online swap awkward. The regularization camp keeps the policy a plain network and adds a penalty, which is pluggable and one forward pass at inference. I want the regularization camp's pluggability, because the transition I care about is far easier when the offline and online objectives are *the same algorithm at two settings of one coefficient*. But every divergence-based regularizer (BEAR's MMD, TD3+BC's $(\pi(s)-a)^2$ cloning term) measures distributional *closeness*, not *support* — and on the broad, multimodal `cloned` mixtures, forcing $\pi$ close to $\pi_\beta$ in distribution drags the policy toward the noisy demonstrations even when the data plainly supports a sharper, stronger policy. The resolution is to stop measuring a divergence and instead evaluate the behavior density at the single action the policy takes and require it be large: $\pi_\beta(\pi(s)\mid s) > \varepsilon$. Relaxing the infinite per-state family to an average over the data states and Lagrangianizing gives the actor objective

$$J_\pi(\phi) = \mathbb{E}_{s\sim D}\big[\,{-}Q(s, \pi_\phi(s)) \;-\; \lambda \cdot \log \pi_\beta(\pi_\phi(s)\mid s)\,\big],$$

a pluggable extra term whose penalty *is* the behavior log-density — a direct support constraint, not a divergence. Here $\lambda$ plays the role of $\varepsilon$: large $\lambda$ pins the policy onto high-density actions, small $\lambda$ lets the value term lead.

I do not have $\pi_\beta$ in closed form, only the dataset actions, so I estimate the density with a conditional **VAE**, $\pi_\beta(a\mid s) \approx p_\psi(a\mid s)$ with a fixed $\mathcal N(0,I)$ prior — flexible enough to capture multimodal behavior a single Gaussian would smear into one blob, calling the valley between the modes "in support." The marginal is intractable, so I use the evidence lower bound: with an approximate posterior $q_\phi(z\mid a,s)$, Jensen gives $\log p_\psi(a\mid s) \ge \mathbb{E}_{q_\phi}[\log p_\psi(a\mid z,s)] - \mathrm{KL}(q_\phi \,\Vert\, p(z\mid s)) =: -\mathcal L_{\text{ELBO}}$. Two things make this the right object. First, $-\mathcal L_{\text{ELBO}}$ is a genuine *lower* bound (its gap is a nonnegative KL), so substituting it for $\log \pi_\beta$ in a constraint I want *above* a threshold is conservative in the right direction. Second, the KL stays analytic for Gaussian $q$ and prior, so a single latent sample gives a low-variance gradient. The penalty is $\texttt{neg\_log\_beta} = \mathcal L_{\text{ELBO}}(s, \pi(s)) = \texttt{recon} + \beta\cdot \mathrm{KL}$, with $\texttt{recon} = \text{mean}((\text{decode}(s,z)-a)^2)$ and $\beta = 0.5$ down-weighting the KL so the VAE spends capacity on faithful reconstruction.

The base off-policy algorithm is **TD3**, not SAC, and deliberately so: my enemy is OOD overestimation, and TD3 was built to suppress it — the $\min$ of twin critics caps the bootstrap, target policy smoothing keeps the critic from latching onto a sharp spurious peak, and delayed (every-`policy_freq`) actor updates let the value settle before the actor chases it. SAC is the opposite of what I want: its stochastic actor *samples* actions whose tails reach OOD and its entropy bonus actively rewards spreading toward the support's edge — exactly the behavior I am forbidding. One scale issue remains: the actor loss mixes $Q$ (on the scale of returns) with $\texttt{neg\_log\_beta}$ (on the scale of a log-density), so a single $\lambda$ would need re-tuning per task. I borrow TD3+BC's normalization, dividing the value term by the detached mean $|Q|$ over the minibatch, $\texttt{norm\_q} = 1/|Q|.\text{mean}().\text{detach}()$, so the actor loss becomes $-\texttt{norm\_q}\cdot Q.\text{mean}() + \lambda\cdot \texttt{neg\_log\_beta}.\text{mean}()$ — order-1 value gradient regardless of reward scale, same within-batch action ranking, so one $\lambda$ transfers across Pen, Door, and Hammer.

Two structural facts make this the offline→online method I want. First, at $\lambda = 0$ the loss collapses to $-\texttt{norm\_q}\cdot Q.\text{mean}()$ — plain TD3. There is no architectural gap between the offline algorithm and a well-grounded online one; the transition is just a knob that, at zero, returns ordinary online TD3. Second, that is exactly how I cross the handoff: offline I pretrain with a strong constraint (large $\lambda$) to keep the critic honest on the fixed data, but once fresh online interactions arrive the data distribution shifts toward the policy's own actions — the actions the critic can now learn about from real feedback — so the reason for the constraint erodes. I *cool* $\lambda$ linearly, $\lambda_t = \lambda \cdot \max(\lambda_{\text{end}},\, 1 - t_{\text{online}}/10^6)$, keeping a floor $\lambda_{\text{end}} > 0$ because on the hardest tasks bootstrap error stays dangerous even online and a residual constraint guards the critic mid-finetune. And I *freeze the VAE* during online fine-tuning: behavior models chase a moving, policy-dependent target online, so re-fitting would destroy the stable "where the offline data was" notion; the decaying $\lambda$ alone controls how much that notion still binds. The VAE is trained once, before Phase 1, in the harness's optional `pretrain` hook (`vae_iterations = 100k`, then `self.vae.eval()`), so the constraint is stationary from the first policy step. At `on_online_start` I reset all three optimizers — Adam moment estimates accumulated over 1M offline steps would otherwise inject stale momentum into the first online updates — and switch in the online discount. The actor and critic head weights start small (uniform $\pm 0.001$ on the actor head, $\pm 0.003$ on the critic heads) so the policy begins near-zero-action and the critics near-zero-value, keeping early backups gentle; the actor LR is dropped to $1\mathrm{e}{-4}$ for a slower, more conservative actor than the critic's $3\mathrm{e}{-4}$, and `normalize = False` matches the reference SPOT config for Adroit. The VAE is 750-wide with latent dim $2\cdot\dim(A)$ — the largest network in the whole task, which is why the budget sits at $1.2\times$ a CQL-plus-VAE architecture; I am at the edge of the cap, so any gain must be algorithmic, not capacity.

```python
CONFIG_OVERRIDES: Dict[str, Any] = {"normalize": False}


class VAE(nn.Module):
    """Variational Auto-Encoder for SPOT support constraint."""

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int,
                 max_action: float, hidden_dim: int = 750):
        super().__init__()
        self.encoder_shared = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )
        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self, state, action):
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        u = self.decode(state, z)
        return u, mean, std

    def encode(self, state, action):
        z = self.encoder_shared(torch.cat([state, action], -1))
        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        return mean, std

    def decode(self, state, z):
        return self.max_action * self.decoder(torch.cat([state, z], -1))


class Critic(nn.Module):
    """Q-function Q(s, a). 2x256 MLP, returns (batch, 1)."""

    def __init__(self, state_dim: int, action_dim: int, orthogonal_init: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class OfflineOnlineAlgorithm:
    """SPOT — Support constraint Policy Optimization via online Training (TD3 + VAE)."""

    def __init__(self, state_dim, action_dim, max_action, replay_buffer=None,
                 discount=0.99, tau=5e-3, actor_lr=3e-4, critic_lr=3e-4, device="cuda"):
        self.device = device
        self.discount = discount
        self.online_discount = 0.99
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0
        self.replay_buffer = replay_buffer

        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_freq = 2
        self.expl_noise = 0.1
        self.beta = 0.5
        self.lambd = 1.0
        self.lambd_cool = True
        self.lambd_end = 0.5
        self.num_samples = 1
        self.is_online = False
        self.online_it = 0
        self.max_online_steps = int(1e6)
        self.vae_iterations = 100_000
        self._actor_lr = 1e-4
        self._critic_lr = critic_lr

        latent_dim = 2 * action_dim
        self.vae = VAE(state_dim, action_dim, latent_dim, max_action, hidden_dim=750).to(device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        self._vae_trained = False

        self.actor = DeterministicActor(state_dim, action_dim, max_action).to(device)
        _actor_head = self.actor.net[-2]
        _actor_head.weight.data.uniform_(-0.001, 0.001)
        _actor_head.bias.data.uniform_(-0.001, 0.001)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self._actor_lr)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        _c1_head = self.critic_1.net[-1]
        _c1_head.weight.data.uniform_(-0.003, 0.003)
        _c1_head.bias.data.uniform_(-0.003, 0.003)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self._critic_lr)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        _c2_head = self.critic_2.net[-1]
        _c2_head.weight.data.uniform_(-0.003, 0.003)
        _c2_head.bias.data.uniform_(-0.003, 0.003)
        self.critic_2_target = deepcopy(self.critic_2)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self._critic_lr)

    def _elbo_loss(self, state, action):
        mean, std = self.vae.encode(state, action)
        N = self.num_samples
        mean_s = mean.repeat(N, 1, 1).permute(1, 0, 2)
        std_s = std.repeat(N, 1, 1).permute(1, 0, 2)
        z = mean_s + std_s * torch.randn_like(std_s)
        state_r = state.repeat(N, 1, 1).permute(1, 0, 2)
        action_r = action.repeat(N, 1, 1).permute(1, 0, 2)
        u = self.vae.decode(state_r, z)
        recon_loss = ((u - action_r) ** 2).mean(dim=(1, 2))
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        return recon_loss + self.beta * KL_loss

    def _vae_train_step(self, batch):
        state, action, *_ = batch
        recon, mean, std = self.vae(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + self.beta * KL_loss
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()
        return {"vae_loss": vae_loss.item(), "vae_recon": recon_loss.item()}

    def pretrain(self, replay_buffer, batch_size: int) -> Dict[str, float]:
        print(f"Pretraining SPOT VAE for {self.vae_iterations} steps")
        log_dict: Dict[str, float] = {}
        self.vae.train()
        for t in range(self.vae_iterations):
            batch = replay_buffer.sample(batch_size)
            batch = [b.to(self.device) for b in batch]
            log_dict = self._vae_train_step(batch)
            if (t + 1) % 1000 == 0:
                metrics_str = " ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in log_dict.items()
                )
                print(f"TRAIN_METRICS step=vae_{t+1} {metrics_str}", flush=True)
        self.vae.eval()
        self._vae_trained = True
        return log_dict

    def train(self, batch: TensorBatch, is_online: bool = False) -> Dict[str, float]:
        self.total_it += 1
        if not self._vae_trained:
            if self.replay_buffer is not None:
                self.pretrain(self.replay_buffer, batch[0].shape[0])
            else:
                self.vae.eval()
                self._vae_trained = True

        if is_online:
            self.online_it += 1
        state, action, reward, next_state, done, *_ = batch
        not_done = 1 - done
        log_dict: Dict[str, float] = {}

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        log_dict["critic_loss"] = critic_loss.item()

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            q = self.critic_1(state, pi)
            neg_log_beta = self._elbo_loss(state, pi)

            if self.lambd_cool:
                lambd = self.lambd * max(
                    self.lambd_end, (1.0 - self.online_it / self.max_online_steps)
                )
            else:
                lambd = self.lambd

            norm_q = 1.0 / q.abs().mean().detach()
            actor_loss = -norm_q * q.mean() + lambd * neg_log_beta.mean()
            log_dict["actor_loss"] = actor_loss.item()
            log_dict["lambd"] = lambd

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)

        return log_dict

    def select_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.tensor(
                state.reshape(1, -1), device=self.device, dtype=torch.float32
            )
            action = self.actor(state_t)
            noise = (torch.randn_like(action) * self.expl_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            action = (action + noise).clamp(-self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()

    def on_online_start(self):
        self.is_online = True
        self.discount = self.online_discount
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self._actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self._critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self._critic_lr)
```
