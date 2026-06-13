## Research question

Offline continuous control: learn a policy from a *fixed* dataset of transitions, with no environment
interaction, and beat the behavior policy that collected the data. The single failure that has to be
engineered around is **Q-value overestimation on out-of-distribution actions**. Bootstrapped
Q-learning forms its target by valuing the next action under a learned critic; offline, that next
action can be anything the policy proposes, and the critic — having never seen those actions — almost
always extrapolates *upward*. The inflated target backs up through the Bellman recursion, the policy
(which is defined by maximizing the critic) steers straight toward the over-valued actions, and the
error feeds itself into divergence. The thing being designed is the **offline RL algorithm itself** —
the losses, the target construction, the behavior regularization, the policy-extraction rule — that
suppresses this overestimation while still extracting an improved policy. Everything about capacity is
fixed; the contribution must be algorithmic.

## Prior art before the first rung (offline value-learning lineage)

The first rung reacts to a line of attempts at exactly this overestimation problem. These are the
ancestors the ladder climbs out of, each with the gap that motivates the next move.

- **Off-policy actor-critic offline (naive TD3/SAC on a static buffer).** Take a continuous-control
  actor-critic — a deterministic or Gaussian actor ascending a learned critic, with replay and target
  networks — and just run it on a fixed dataset. It collapses. The actor proposes actions outside the
  data, the critic over-values them (extrapolation error, Fujimoto et al. 2019), and there is no new
  experience to correct the mistake. Gap: unconstrained bootstrapping diverges offline.
- **Generative-model constraints (BCQ, BEAR; Fujimoto et al. 2019, Kumar et al. 2019).** Fit a model
  of the dataset's action distribution and only let the policy choose among actions that model deems
  in-support (BCQ), or constrain the policy's MMD to the behavior actions (BEAR). It works, but it
  bolts a generative model and extra sampling onto the agent — many moving parts, several
  hyperparameters that can only be tuned by interacting (which is forbidden offline). Gap: heavy
  machinery, hard to tune without a validation environment.
- **Conservative value penalties (CQL; Kumar et al. 2020).** Skip the policy constraint; instead add a
  term to the critic loss that pushes Q *down* on out-of-distribution actions and *up* on dataset
  actions, so the critic is pessimistic exactly where it has no data. Principled, but it has to
  explicitly query and sample OOD actions to push them down (a log-sum-exp over sampled actions), which
  is extra compute and an extra temperature knob; the conservatism can also be too blunt, flattening
  useful in-distribution structure. Gap: still queries OOD actions; one more sensitive coefficient.
- **Behavior-regularized actor-critic (BRAC, AWAC; Wu et al. 2019, Nair et al. 2020).** Keep standard
  value learning but add a divergence penalty (KL, MMD) between the policy and the behavior policy, or
  extract the policy by advantage-weighted regression so it never strays far from the data. A unifying
  view, but the divergence estimate needs a behavior model again, and a single global penalty trades
  exploitation against caution clumsily across states. Gap: divergence estimation + one global dial.

The substrate below is what these converged on as the *minimal* place to stand: a twin-critic
actor-critic with clipped double-Q and Polyak targets, fixed at 256 hidden units, on which each rung is
a different choice of loss / target / extraction.

## The fixed substrate

A single offline-RL training file is frozen except for one region. The fixed loop: load a D4RL MuJoCo
dataset into a `ReplayBuffer`, **state-normalize** features with exact dataset mean/std, then for
`max_timesteps = 1e6` steps sample a `batch_size` minibatch
`[states, actions, rewards, next_states, dones, next_actions]` and call `trainer.train(batch)`; every
`eval_freq = 5e3` steps roll out `trainer.actor` for `n_episodes = 10` and report the **D4RL normalized
score** (0 = random, 100 = expert). The buffer also exposes `next_actions` — the dataset action at the
*next* timestep — for SARSA-style and next-action-regularized targets, and the whole dataset via
`replay_buffer._states/_actions/_rewards` for global statistics. Fixed utilities are provided:
`soft_update` (Polyak), `compute_mean_std`/`normalize_states`, `init_module_weights` (orthogonal init),
and a `_mlp()` factory **locked at hidden width 256**. Hyperparameter defaults live in `TrainConfig`
(`discount = 0.99`, `tau = 5e-3`, `actor_lr = critic_lr = 3e-4`, `batch_size = 256`, `normalize = True`,
`orthogonal_init = True`); a method may override a whitelisted subset via `CONFIG_OVERRIDES`
(`normalize`, `normalize_reward`, `actor_lr`, `critic_lr`, `tau`, `discount`, `batch_size`).

Two hard constraints make this an *algorithmic* benchmark, not a capacity one. **All hidden widths are
256** — the `_mlp()` factory asserts it and custom network classes must obey it. And **total trainable
parameters are capped at 1.05× the largest baseline architecture**, checked at runtime before training;
a method that tries to win by adding networks (large critic ensembles, a learned encoder) is rejected.
So every rung must pay for its gains in *math*, not parameters.

## The editable interface

Exactly one region of `custom.py` is editable: the network classes (`DeterministicActor`, `Actor`,
`Critic`, `ValueFunction`) and the `OfflineAlgorithm` class — its `__init__` (build the nets and
optimizers) and its `train(batch)` (one gradient update, returning a scalar log dict). The loop only
hands the trainer the environment dimensions and the `TrainConfig` learning rates; everything
algorithm-specific is built inside. The contract the loop relies on: `self.actor` must be an
`nn.Module` with an `.act(state, device)` method (this is what evaluation rolls out), and
`train(batch)` must consume the six-tensor batch and step the optimizers.

The starting point is the scaffold default: a placeholder `OfflineAlgorithm` that builds a Gaussian
`Actor`, two `Critic`s and their targets, three Adam optimizers, and a `train` that does **nothing** —
it returns zero losses without updating. Each rung replaces this class (and, where the algorithm wants
a different architecture, the relevant network classes) and nothing else.

```python
# EDITABLE region of custom.py — default fill (placeholder, no learning)
class OfflineAlgorithm:
    """Offline RL algorithm -- implement your approach here.

    The training loop calls:
        trainer = OfflineAlgorithm(state_dim, action_dim, max_action, **kwargs)
        log_dict = trainer.train(batch)          # called at every timestep
        eval_actor(env, trainer.actor, ...)      # called every eval_freq steps
    You MUST set self.actor to an nn.Module that has an .act(state, device) method.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        replay_buffer: "ReplayBuffer" = None,
        discount: float = 0.99,
        tau: float = 5e-3,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        orthogonal_init: bool = True,
        device: str = "cuda",
    ):
        self.device = device
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.total_it = 0
        self.replay_buffer = replay_buffer

        # Build networks -- modify or replace as needed
        self.actor = Actor(state_dim, action_dim, max_action, orthogonal_init).to(device)
        self.critic_1 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_2 = Critic(state_dim, action_dim, orthogonal_init).to(device)
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_2_target = deepcopy(self.critic_2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        """Update networks on one batch. batch = [states, actions, rewards,
        next_states, dones, next_actions] (torch.Tensor, on device)."""
        self.total_it += 1
        states, actions, rewards, next_states, dones, next_actions = batch

        # -- Placeholder: replace with your algorithm --
        log_dict: Dict[str, float] = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
        }
        return log_dict
```

## Evaluation settings

Three D4RL MuJoCo `medium-v2`/`-v1` datasets spanning the difficulty range — **HalfCheetah**
(`halfcheetah-medium-v2`), **Maze2D** (`maze2d-medium-v1`, a goal-reaching navigation task that rewards
trajectory *stitching*), and **Walker2d** (`walker2d-medium-v2`) — each over three seeds {42, 123, 456}.
One metric per dataset, higher is better: the **D4RL normalized score** (0 = random, 100 = expert),
averaged over the 10 evaluation rollouts at the final budget. A strong method generalizes across all
three rather than relying on a dataset-specific quirk; the task score is the geometric mean across the
three datasets.
