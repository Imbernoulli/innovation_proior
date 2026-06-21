GAIL came back exactly along the fault line I expected. HalfCheetah limped to a mean of $1646$ — alive, because that body has no terminal state, so even a partly-collapsed gait keeps accumulating return over a fixed-length episode. But Hopper cratered to $25.7$ and Walker2d to $77.8$, both essentially zero, both with the seed-to-seed jitter of a body that falls almost immediately. That is the saturation mechanism made visible: on these clean, tight demonstrations the discriminator *wins* the min-max game, its logits saturate, $-\log(1-D)$ flattens, and on the terminating bodies the policy gets no stable signal to stay upright. The diagnosis is not "tune the discriminator harder" — I already bumped the inner updates and the batch and added input normalization, and Hopper still died. The diagnosis is that the *reward itself* carries nothing usable once the game saturates: GAIL's discriminator at its optimum is $\tfrac12$ everywhere, an unstructured object from which no reward can be extracted, so when it saturates short of that optimum the policy ascends an arbitrary, shaped-by-accident signal. The next rung has to make the recovered reward a real reward function, not just a classifier score that happens to be high near the expert.

I propose AIRL, adversarial inverse RL. I keep the adversarial frame, because occupancy matching is still the right idea — it killed BC's compounding error — but I change *what the discriminator is*. In a GAN the optimal discriminator is $D^*=p_{\text{data}}/(p_{\text{data}}+q)$, and here I *know* $q$: the generator density is my policy, which I can evaluate as $\log\pi(a\mid s)$. So I plug $q$ in and let the discriminator model only the data density in Boltzmann form,

$$D = \frac{\exp f}{\exp f + \pi}, \qquad \text{logit} = f(s,a,s') - \log\pi(a\mid s),$$

a learned reward term $f$ minus the *filled-in* log policy density. This buys two things. First, the optimal discriminator is now independent of the generator — optimal exactly when $\exp f$ matches the data density up to the partition constant — so the discriminator is no longer chasing a moving target, and the adversarial game is far more stable than GAIL's. That alone should help where GAIL's instability was fatal. Second, and the whole point, I can read a reward back out of $f$: a free discriminator at the optimum is $\tfrac12$ and yields nothing, but this structured one keeps $f$.

What reward does it recover? At the GAN optimum the policy matches the expert, $D=\tfrac12$, and $\exp f/(\exp f+\pi_E)=\tfrac12$ forces $f^*=\log\pi_E(a\mid s)$. Under the maximum-entropy model the log of the optimal policy is the advantage, so $f^*=A^*(s,a)$ — the recovered reward is the expert's advantage. That is already strictly more than GAIL gives, but the advantage is *entangled*: $A^*=Q^*-V^*=r+\gamma V^*(s')-V^*(s)$ under deterministic dynamics, the true reward shaped by the value function. The raw $f$, free to be any function, can pour all its expressive power into matching the value-function component and very little into the part that actually distinguishes expert behavior — and when training saturates, that ill-conditioned, value-dominated signal is what the policy is stuck with. If I give the network an explicit place to *put* the value-function shaping, the remainder is forced to be a cleaner reward. The structure to impose is the only policy-invariant degree of freedom there is: potential-based shaping. The transform $r+\gamma\Phi(s')-\Phi(s)$ leaves the optimal policy unchanged for any potential $\Phi$, and without knowing the dynamics it is the only class of reward transformations that is policy-invariant. So I carve $f$ into a reward term and a potential-shaping term, each its own network,

$$f(s,a,s') = g(s,a) + \gamma\,h(s') - h(s).$$

Whatever shaping the optimization wants, it dumps into $h$; $g$ is left to be the unshaped reward. Working the algebra at the optimum (with the chaining lemma under decomposable dynamics) gives $h^*=V^*$ and $g^*=r$ up to constants — $h$ soaks up exactly the value-function shaping that made the advantage entangled, and $g$ comes out clean. This is what should stabilize the terminating bodies: $h$ absorbing the value gradient means $g$ does not have to, so the reward handed to the policy is better-behaved than GAIL's value-dominated mush.

Landing this in the scaffold forces several concrete departures I have to respect line by line. First, the logit needs $\log\pi(a\mid s)$, so the module must read the policy. The scaffold hands it in through `set_policy(policy, optimizer)`; I take the policy *reference* and ignore the optimizer, because the policy *learner* is the fixed PPO loop — I do not train the policy, PPO does. In `update()` I evaluate $\log\pi(a\mid s)$ under no-grad on both expert and policy batches and subtract it from $f$ to form the logit; expert label $1$, policy $0$, BCE. The $-\log\pi$ term is the whole reason the optimal discriminator is generator-independent. Second, the terminal-state subtlety, sharp on exactly the bodies GAIL killed. The shaping $\gamma h(s')-h(s)$ preserves the optimal policy only if I am honest about terminal states: at an episode's final transition there is no genuine $s'$, and the real value function sets a terminal state's future value to zero. If I let $h(s')$ fire on a terminal "next state" I add a phantom potential that breaks policy-invariance — and with Hopper and Walker2d terminating on falling, this is not a corner case but most of the interesting transitions. So I zero the shaping when the transition is terminal:

$$f(s,a,s',\text{done}) = g(s,a) + \gamma\,(1-\text{done})\,h(s') - h(s).$$

`raw_f` takes a `done` argument and multiplies $h(s')$ by $(1-\text{done})$. This done-aware shaping is the single most important reason AIRL should rescue the terminating bodies — it keeps $f$ a *valid* potential-shaped reward across variable-length episodes instead of one that paid the policy for phantom future value at the exact moment the body fell.

Third, the normalization layering, dictated by the fixed loop. The substrate applies its own running mean/std normalization to the buffer rewards before the PPO update, fixed and not editable. AIRL's raw shaped $f$ can have a large, drifting scale (it is a difference of three network outputs), and feeding that raw value into the fixed buffer normalization makes the running stats chase a moving target and either saturate or obliterate the signal. So I add a *second* RunningNorm on the reward net's *output*: `_out_rms` whitens $f$ so the value entering the fixed normalization is already roughly unit-variance, making that fixed step near-identity rather than destructive. `compute_reward` returns the normalized shaped $f$ under no-grad, with `done` unavailable at rollout time so the terminal correction is applied only during `update()` on the discriminator side. As in GAIL I keep a RunningNorm on the obs inputs (`_obs_rms`, refreshed each round) so the discriminator cannot cheat on raw observation scale. Three normalizations now coexist — obs-input (mine), reward-output (mine), buffer-reward (the fixed loop's) — and the middle one exists specifically to keep the third from collapsing the signal. Fourth, the budget knobs again: `irl_batch_size` and `n_irl_updates_per_round` are fixed and too few against a fast PPO policy, so I bump `_inner_updates=4` and `_batch_mult=4` inside `update()`, resampling fresh batches each inner step and refreshing `_out_rms` from the concatenated raw $f$. One honest concession the math does not force but the data does: the expert demos store no `done` flags, so I assume expert transitions are non-terminal (`expert_done_zeros`) — exact for HalfCheetah, mildly wrong for Hopper/Walker terminal states, but the reference imitation library also lacks expert dones by default. The architecture is $g$ as an MLP $[s,a]\to256\to256\to1$ and $h$ as $[s]\to256\to256\to1$, the largest reward net on the ladder, which is exactly why the scaffold's parameter budget was sized at $1.05\times$ this net.

The falsifiable expectations, read directly against GAIL's numbers: the structured, done-aware reward should help *most* exactly where GAIL failed worst — the terminating bodies — so I expect Hopper and Walker2d to climb decisively off GAIL's near-zero floor into the hundreds-to-low-thousands, and HalfCheetah, already surviving at $1646$ thanks to non-termination, to improve only modestly into the low-to-mid 2000s. The clean signature that would confirm the diagnosis is that AIRL beats GAIL on *every* environment with the largest margin on Hopper and Walker2d and the smallest on HalfCheetah — the inverse of where GAIL's terminal-driven collapse hit hardest. If instead AIRL also collapses on the terminating bodies, the problem is the adversarial frame itself on clean demos, and the next rung would have to abandon adversarial reward learning entirely.

```python
class _RunningMeanStd:
    """Welford running mean/std for input/output normalization."""

    def __init__(self, shape, device, eps=1e-4):
        self.mean = torch.zeros(shape, device=device, dtype=torch.float32)
        self.var = torch.ones(shape, device=device, dtype=torch.float32)
        self.count = eps

    @torch.no_grad()
    def update(self, x):
        if x.numel() == 0:
            return
        x = x.detach().to(self.mean.device, dtype=torch.float32).reshape(-1, self.mean.shape[-1]) \
            if self.mean.dim() else x.detach().to(self.mean.device, dtype=torch.float32).reshape(-1)
        batch_count = x.shape[0]
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False) if batch_count > 1 else torch.zeros_like(self.var)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot_count)
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


class RewardNetwork(nn.Module):
    """AIRL shaped reward: f(s,a,s',done) = g(s,a) + gamma * (1-done) * h(s') - h(s).

    Adds RunningNorm on the obs inputs (frozen during inference) and on
    the network output, mirroring imitation's tuned configs.
    """

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # g(s, a): reward approximator
        self.g_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        # h(s): potential-based shaping
        self.h_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.gamma = 0.99

        # Running normalization. Initialized lazily (we need a device).
        self._obs_rms = None
        self._out_rms = None
        self._update_norm = True

    def _ensure_rms(self, ref_tensor):
        if self._obs_rms is None:
            self._obs_rms = _RunningMeanStd((self.obs_dim,), ref_tensor.device)
            self._out_rms = _RunningMeanStd((1,), ref_tensor.device)

    def freeze_norm(self):
        self._update_norm = False

    def update_obs_norm(self, obs, next_obs):
        if self._obs_rms is None:
            self._ensure_rms(obs)
        if self._update_norm:
            self._obs_rms.update(obs)
            self._obs_rms.update(next_obs)

    def update_out_norm(self, raw_f):
        if self._out_rms is None:
            return
        if self._update_norm:
            self._out_rms.update(raw_f.unsqueeze(-1) if raw_f.dim() == 1 else raw_f)

    def _norm_obs(self, x):
        if self._obs_rms is None:
            return x
        return self._obs_rms.normalize(x)

    def _norm_out(self, y):
        if self._out_rms is None:
            return y
        return (y - self._out_rms.mean.squeeze()) / torch.sqrt(self._out_rms.var.squeeze() + 1e-8)

    def g(self, state, action):
        x = torch.cat([self._norm_obs(state), action], dim=-1)
        return self.g_net(x).squeeze(-1)

    def h(self, state):
        return self.h_net(self._norm_obs(state)).squeeze(-1)

    def raw_f(self, state, action, next_state, done=None):
        """Unnormalized shaped reward. done: float tensor of shape (batch,)."""
        h_next = self.h(next_state)
        if done is not None:
            h_next = h_next * (1.0 - done.float())
        return self.g(state, action) + self.gamma * h_next - self.h(state)

    def forward(self, state, action, next_state, done=None):
        """Shaped reward, with running output normalization applied."""
        f = self.raw_f(state, action, next_state, done)
        return self._norm_out(f)


class IRLAlgorithm:
    """AIRL — Adversarial Inverse Reinforcement Learning.

    Discriminator logits are raw_f(s,a,s',done) - log pi(a|s).
    The reward returned to PPO is the *normalized* shaped f, so the FIXED
    template-level reward normalization (running mean/std on buffer.rewards)
    becomes near-identity rather than collapsing the signal.
    """

    def __init__(self, reward_net, expert_demos, obs_dim, action_dim, device, args):
        self.reward_net = reward_net
        self.expert_demos = expert_demos
        self.device = device
        self.args = args
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.optimizer = optim.Adam(self.reward_net.parameters(), lr=args.irl_lr)
        self.total_updates = 0
        self._policy = None

        # Effective per-round disc updates and batch (args are FIXED outside
        # editable range — bump internally to match imitation tuned configs).
        self._inner_updates = 4   # multiplies args.n_irl_updates_per_round
        self._batch_mult = 4      # multiplies args.irl_batch_size

    def set_policy(self, policy, optimizer):
        del optimizer
        self._policy = policy

    def compute_reward(self, obs, acts, next_obs):
        """Normalized shaped reward for PPO. dones not available here, so
        the terminal correction is applied during update() instead."""
        with torch.no_grad():
            return self.reward_net(obs, acts, next_obs)

    def _log_policy_act_prob(self, obs, acts):
        if self._policy is None:
            raise RuntimeError("AIRL requires set_policy() before discriminator updates")
        with torch.no_grad():
            _, log_prob, _, _ = self._policy.get_action_and_value(obs, acts)
        return log_prob.detach()

    def update(self, policy_obs, policy_acts, policy_next_obs, policy_dones):
        """AIRL discriminator update using raw_f - log pi(a|s) on the BCE side."""
        self.total_updates += 1
        bs = self.args.irl_batch_size * self._batch_mult

        # Update obs running stats once per outer call using the freshest
        # generator rollout — keeps statistics close to the policy state dist.
        self.reward_net.update_obs_norm(policy_obs, policy_next_obs)

        n_expert = len(self.expert_demos["obs"])
        n_policy = len(policy_obs)
        # Expert "dones" are not stored in the demo dict — assume non-terminal
        # (correct for halfcheetah which has no terminal; mildly wrong for
        # hopper/walker terminal states but reference imitation also lacks
        # dones for expert demos by default).
        expert_done_zeros = torch.zeros(bs, device=self.device)

        last = {}
        for _ in range(self._inner_updates):
            # Resample fresh batches each inner step
            expert_idx = torch.randint(0, n_expert, (bs,))
            expert_obs = self.expert_demos["obs"][expert_idx]
            expert_acts = self.expert_demos["acts"][expert_idx]
            expert_next_obs = self.expert_demos["next_obs"][expert_idx]

            policy_idx = torch.randint(0, n_policy, (bs,))
            gen_obs = policy_obs[policy_idx]
            gen_acts = policy_acts[policy_idx]
            gen_next_obs = policy_next_obs[policy_idx]
            gen_dones = policy_dones[policy_idx].float()

            expert_f = self.reward_net.raw_f(
                expert_obs, expert_acts, expert_next_obs, expert_done_zeros
            )
            gen_f = self.reward_net.raw_f(
                gen_obs, gen_acts, gen_next_obs, gen_dones
            )

            expert_logits = expert_f - self._log_policy_act_prob(expert_obs, expert_acts)
            gen_logits = gen_f - self._log_policy_act_prob(gen_obs, gen_acts)

            logits = torch.cat([expert_logits, gen_logits], dim=0)
            labels = torch.cat([
                torch.ones(bs, device=self.device),
                torch.zeros(bs, device=self.device),
            ], dim=0)

            loss = F.binary_cross_entropy_with_logits(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.reward_net.update_out_norm(torch.cat([expert_f, gen_f], dim=0))
                acc = ((logits > 0).float() == labels).float().mean().item()
            last = {"irl_loss": loss.item(), "disc_acc": acc}

        return last
```
