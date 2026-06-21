AIRL did beat GAIL on every environment, and the margins fell where I predicted — HalfCheetah $1646\to2514$, Hopper $25.7\to840$, Walker2d $77.8\to358$ — but the seed-level numbers carry a sharper lesson than the means. Hopper's $840$ mean is $\{153, 169, 2198\}$: two seeds still essentially dead, one that found a good gait and ran. Walker2d's $358$ is $\{290, 654, 129\}$, all low, none near a real walking return. The done-aware shaping and the generator-independent discriminator made the adversarial frame *less* fragile than GAIL's, but not reliable: on the terminating bodies AIRL mostly produces a barely-alive policy with the occasional lucky seed masking it in the mean. Both methods inherit the same root liability — an adversarial reward that is *non-stationary by construction*. The discriminator moves every round, so the reward the policy ascends and the value function it fits are chasing a shifting target, and on a clean, tight expert distribution the discriminator's pull is strong enough that two of three seeds get stuck in a degenerate basin before a usable reward emerges. I have spent two rungs adding machinery to tame that non-stationarity — input normalization, output normalization, bumped inner updates, done-aware shaping — and the terminating bodies still mostly die. The honest read is that the remaining failure is not in the reward *structure* but in the *adversarial loop itself*. So the move is not more structure on the discriminator; it is to ask whether I need the discriminator at all.

Step back to what the task provides and scores. I have a fixed pile of expert $(s,a,s')$ transitions from a *competent* MuJoCo agent — clean, on-distribution, dense — and I am scored on the policy's true-reward return after training. The adversarial methods spend their whole budget learning a *reward* so PPO can then learn a *policy*: two coupled, non-stationary optimizations stacked on each other. But the demonstrations already contain, directly, the one thing I ultimately need — for each state the expert visited, the action a competent policy took. I propose BC, behavioral cloning by maximum likelihood: skip reward learning entirely and fit the policy to the expert actions by supervised learning,

$$\min_\theta\; -\,\mathbb{E}_{(s,a)\sim\text{expert}}\big[\log\pi_\theta(a\mid s)\big].$$

The lineage flagged BC's weakness up front — covariate shift, compounding $\varepsilon T^2$ error off the expert distribution — which is exactly why the ladder opened with the occupancy-matching methods meant to beat it. But I now have *measured* evidence that those methods, in this harness, are dominated by adversarial instability rather than by the covariate-shift advantage they were supposed to have. So the question is empirical, and running BC as the top rung is what settles it. Two things blunt BC's covariate-shift penalty here. First, the demos are dense (tens of thousands of transitions) and the experts competent, so the expert state distribution covers the locomotion manifold well and the learner does not have to extrapolate far. Second, and decisively, the *alternatives* are not paying a small drift cost — they are paying a large adversarial-instability cost, two-thirds of their seeds stuck in degenerate basins. Cloning has no discriminator to saturate, no reward to drift, no policy-density to evaluate, nothing to balance: it is a single stationary supervised loss. Trading a structural-but-bounded error for that stability can be a net win when the demos are clean.

The defining choice is what "clone" means here, because the scaffold forces a version quite different from the classical road-following cloner. The classical formulation carried a whole apparatus — a population-coded output bank read by center of mass for fine continuous control, synthetic perspective-transformed views to manufacture the off-center recovery examples the expert never demonstrates, pure-pursuit relabeling of those views, and a replay buffer with a mean-steering eviction rule to lock in symmetry. *None of that exists here.* The scaffold gives no road geometry to synthesize recovery views from, no population-coded output — the policy is a fixed Gaussian actor-critic with a mean head and a state-independent log-std — and no way to relabel manufactured states. What it does expose, which the classical cloner lacked, is the policy's *full action distribution*. So the task's BC is not "regress the action"; it is maximum-likelihood cloning of the Gaussian policy: minimize the negative log-probability of the expert's actions under $\pi(a\mid s)$, which trains the actor mean *and* the log-std at once and gives proper action coverage rather than a brittle point estimate. Regressing only the mean would throw away the variance the harness's Gaussian needs to represent action spread; the NLL objective fits both.

The mechanics follow from the rigid contract. The reward net is *unused* — BC learns no reward — so `RewardNetwork` is a dummy returning zeros, and `compute_reward` returns zeros for every transition. This is load-bearing for *why* BC stays stable: because the learned reward is identically zero, the fixed running-reward-normalization and the PPO update operate on a constant signal and contribute essentially nothing — they cannot inject the non-stationarity that destabilized the adversarial methods. PPO is strictly secondary and must not be allowed to fight the cloning. The actual learning happens in `update()`, and to train the *policy* there I need the policy reference *and* its optimizer — the scaffold hands both in through `set_policy(policy, optimizer)`, which BC stores (unlike AIRL, which kept only the reference; BC keeps the optimizer because *it*, not the fixed PPO step, does the gradient updates on the policy). Inside `update()` I sample expert $(s,a)$ minibatches, evaluate $\log\pi(a\mid s)$ via the policy's `get_action_and_value(obs, expert_acts)` — which returns the log-prob of the expert action under the current Gaussian — and minimize $-\mathbb{E}[\log\pi(a\mid s)]$, with a tiny entropy *penalty* ($-0.001\cdot\text{entropy}$, a small push against collapsing the policy's variance to zero) to keep the action distribution from degenerating. I run $20$ BC gradient steps per `update()` call so cloning dominates the secondary PPO updates, clip the policy grad-norm at $0.5$, and step the policy optimizer directly. BC's only optimization is the stationary supervised NLL on a fixed dataset; there is no game to lose.

The falsifiable expectations, read against the two rungs below: if the diagnosis is right — that the adversarial methods are dominated by min-max instability on clean demos, not by their occupancy-matching principle — then BC should beat AIRL on the aggregate, and most clearly where AIRL's instability bit hardest, the terminating bodies. I expect BC's Hopper and Walker2d to land in the low-thousands and low-to-mid thousands as consistently-alive cloned gaits, not AIRL's two-dead-one-alive pattern, because a supervised fit to a competent expert keeps the body upright without needing a reward to emerge first. On HalfCheetah, where AIRL already reached $2514$ and the body never terminates, I expect BC competitive but not dramatically ahead — possibly a wide seed spread, a runaway fast clone alongside an underfit high-dimensional gait. The clean signature that would confirm the whole ladder's story is BC strongest overall, winning Hopper and Walker2d decisively while roughly tying or modestly trailing AIRL on the non-terminal HalfCheetah: the simplest method beating the principled ones precisely because the principle's cost (instability) outweighs its benefit (occupancy matching) on clean demonstrations.

```python
class RewardNetwork(nn.Module):
    """Dummy reward network for BC (not used for reward shaping).

    BC does not learn a reward; this returns a constant so the PPO
    loop runs but does not meaningfully update from reward signal.
    The policy is trained via supervised loss in IRLAlgorithm.update().
    """

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # Unused parameters to keep interface consistent
        self.dummy = nn.Linear(1, 1)

    def forward(self, state, action, next_state):
        return torch.zeros(state.shape[0], device=state.device)


class IRLAlgorithm:
    """BC — Behavioral Cloning.

    Directly trains the policy network to mimic expert actions via
    supervised MSE loss. The reward network is unused.
    Policy is trained both via BC loss in update() and via PPO in the
    main loop (with near-zero reward), but BC dominates learning.
    """

    def __init__(self, reward_net, expert_demos, obs_dim, action_dim, device, args):
        self.reward_net = reward_net
        self.expert_demos = expert_demos
        self.device = device
        self.args = args
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.total_updates = 0
        # BC does not need a reward optimizer; policy is trained externally
        # We store a reference to policy that gets set during training
        self._policy = None
        self._policy_optimizer = None

    def set_policy(self, policy, optimizer):
        """Set reference to policy for BC updates."""
        self._policy = policy
        self._policy_optimizer = optimizer

    def compute_reward(self, obs, acts, next_obs):
        """BC uses constant reward (PPO loop is secondary)."""
        return torch.zeros(obs.shape[0], device=self.device)

    def update(self, policy_obs, policy_acts, policy_next_obs, policy_dones):
        """BC supervised update: minimize negative log-probability of expert actions.

        Uses the full policy distribution (mean + log_std) to compute log prob,
        matching the reference imitation library approach. This trains both the
        mean and the variance of the policy, giving better action coverage.
        """
        self.total_updates += 1

        if self._policy is None:
            return {"irl_loss": 0.0}

        batch_size = self.args.irl_batch_size

        # Sample expert data
        n_expert = len(self.expert_demos["obs"])

        total_bc_loss = 0.0
        n_bc_steps = 20  # more BC gradient steps per IRL update

        for _ in range(n_bc_steps):
            expert_idx = torch.randint(0, n_expert, (batch_size,))
            expert_obs = self.expert_demos["obs"][expert_idx]
            expert_acts = self.expert_demos["acts"][expert_idx]

            # Use get_action_and_value to get log_prob of expert actions
            # This trains both actor_mean and actor_logstd
            _, log_prob, entropy, _ = self._policy.get_action_and_value(
                expert_obs, expert_acts,
            )

            # Negative log-likelihood loss (matching reference BC)
            neglogp = -log_prob.mean()
            # Entropy bonus for exploration (prevents policy from collapsing)
            ent_bonus = -0.001 * entropy.mean()

            bc_loss = neglogp + ent_bonus

            self._policy_optimizer.zero_grad()
            bc_loss.backward()
            nn.utils.clip_grad_norm_(self._policy.parameters(), 0.5)
            self._policy_optimizer.step()
            total_bc_loss += bc_loss.item()

        return {"irl_loss": total_bc_loss / n_bc_steps}
```
