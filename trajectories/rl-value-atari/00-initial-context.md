## Research question

Value-based RL on Atari from raw pixels: learn an effective visual representation *and* a Q-value
estimate at the same time, off-policy, from a replay buffer, on $84\times84$ grayscale 4-frame stacks.
The one thing being designed is the **value head and its TD update** — how the per-action value is
parameterized, how the bootstrap target is constructed, and what loss trains it. Everything else (the
convolutional encoder, the replay buffer, $\epsilon$-greedy acting, the periodic target copy, the
evaluation protocol) is fixed. Capacity is *not* a lever: the encoder is frozen and a parameter-budget
check forbids buying improvement with a wider net, so every gain must be algorithmic — target
construction, head design, TD loss.

## Prior art before the first rung (value-based deep-RL lineage)

The first rung reacts to the standard deep Q-learning recipe; these are the methods that recipe
converged from, each with the gap it leaves open.

- **Deep Q-Network (DQN, Mnih et al. 2015).** Regress $Q(S_t,A_t;\theta)$ onto the bootstrap target
  $R_{t+1}+\gamma\max_a Q(S_{t+1},a;\theta^-)$ with a frozen target network $\theta^-$ and a uniform
  replay buffer, minimizing the squared TD error. Cracked Atari from pixels. Gap: the same estimates
  both *select* the greedy next action (the $\arg\max$) and *evaluate* it (the value read off), and
  because $\max$ is convex, any estimation error makes the target biased *upward* (Jensen) — the
  predicted value of the greedy policy runs above its realized return, and on some games diverges while
  the score collapses.
- **Dueling DQN (Wang et al. 2016).** Splits the head into a state-value stream $V(s)$ and an
  advantage stream $A(s,a)$, recombined as $Q=V+A-\operatorname{mean}_a A$, so the network can learn
  "this state is good" without committing to a per-action value. Gap: it improves the *mean* estimator
  architecturally but does nothing about the overestimation bias in the target, nor about the fact that
  $Q$ is still a single scalar collapsing the whole return distribution.
- **A distributional view (Bellemare, Dabney, Munos 2017).** The return $Z(x,a)$ is a random variable
  obeying a distributional Bellman recursion $Z\overset{D}{=}R+\gamma Z(X',A')$; learning its whole law
  (probabilities on a fixed atom grid $[V_{\min},V_{\max}]$, trained by KL after a projection of the
  shifted target) carries strictly more information than its mean and helps even with a mean-greedy
  policy. Gap: it must be told the return range $[V_{\min},V_{\max}]$ up front, it optimizes
  KL-after-projection rather than the Wasserstein metric the operator actually contracts in, and its
  resolution is capped at the $N$ atoms placed a priori.

These set up the ladder: fix the overestimation bias first (a target-construction change with no new
parameters), then enrich the value object into a distribution that needs no support bounds and gives
unbiased sample gradients.

## The fixed substrate

A standard off-policy value-based Atari loop is frozen and must not be touched. Atari preprocessing:
grayscale $84\times84$, frame-skip 4, 4-frame stack, no-op resets, terminal-on-life-loss and reward
clipping during training (the *eval* env drops both). A **fixed** Nature-DQN convolutional encoder maps
$(B,4,84,84)$ uint8 frames to a 512-dim feature vector and is shared by every method — only the head on
top of those 512 features may change. The loop runs a $10^6$-transition uniform replay buffer
(`optimize_memory_usage=True`), linear $\epsilon$-anneal $1\to0.01$ over the first 10% of steps,
`learning_starts=80000`, a training step every 4 environment steps with batch 32, $\gamma=0.99$, and a
periodic target-network sync every `target_network_frequency=1000` steps. A parameter-budget check
prints the head size against $1.05\times$ the largest baseline head (QR-DQN's $|\mathcal A|\times200$)
so capacity cannot be the contribution.

## The editable interface

Exactly one region is editable — the `QNetwork` head and the `ValueAlgorithm` class in
`custom_value_atari.py` (lines 186–249). The training loop calls a fixed contract:
`algorithm = ValueAlgorithm(envs, device, args)`; acting via `algorithm.select_action(obs, epsilon)`;
one gradient step per call via `metrics = algorithm.update(batch, global_step)` on a cleanrl
`ReplayBuffer` sample (`.observations`, `.next_observations`, `.actions`, `.rewards`, `.dones`); and
evaluation reads `algorithm.q_network(obs)` and takes its $\arg\max$. The algorithm **must** set
`self.q_network` and `self.target_network` to `nn.Module`s whose `forward(x)` returns per-action
$Q$-values, and must own its own optimizer and target-sync. Available to it: the fixed
`NatureDQNEncoder` (→512-d), `ENCODER_FEATURE_DIM = 512`, and `linear_schedule`.

The starting point is the scaffold default below — a single linear head and a stub `update` that does
nothing. Each rung replaces exactly this region.

```python
# EDITABLE region of custom_value_atari.py (lines 186-249) -- default scaffold fill
class QNetwork(nn.Module):
    """Q-network: NatureDQNEncoder (fixed) + head. Output: Q-values per action.

    The encoder is FIXED (Nature DQN CNN -> 512-dim features). Only the
    head layer(s) on top of the 512-dim features may be changed.
    """

    def __init__(self, envs):
        super().__init__()
        n_actions = envs.single_action_space.n
        self.encoder = NatureDQNEncoder()
        self.head = nn.Linear(ENCODER_FEATURE_DIM, n_actions)

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


class ValueAlgorithm:
    """Value-based algorithm for Atari -- implement your approach here."""

    def __init__(self, envs, device, args):
        self.device = device
        self.gamma = args.gamma
        self.tau = args.tau
        self.target_network_frequency = args.target_network_frequency

        self.q_network = QNetwork(envs).to(device)
        self.target_network = QNetwork(envs).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

    def select_action(self, obs, epsilon):
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return np.array([self.q_network.head.out_features])  # placeholder
        q_values = self.q_network(torch.Tensor(obs).to(self.device))
        return torch.argmax(q_values, dim=1).cpu().numpy()

    def update(self, batch, global_step):
        """Single gradient update. Returns a dict of scalar metrics."""
        return {"td_loss": 0.0, "q_values": 0.0}
```

## Evaluation settings

Three Atari games spanning the design space — **Breakout** (fast, dense, small action set),
**Seaquest** (longer-horizon, larger action set, harder credit assignment), and **Pong** (near-ceiling
once learned) — each over three seeds {42, 123, 456}, within a fixed interaction budget using the
benchmark Atari wrappers. One metric per game: mean episodic return over the evaluation episodes
(higher is better), evaluated at the greedy policy. A strong method should improve across games rather
than tune to one title.
