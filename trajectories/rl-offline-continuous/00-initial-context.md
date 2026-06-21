## Research question

Offline continuous control: learn a policy from a *fixed* dataset of transitions with no environment interaction and beat the behavior policy that collected the data. The central failure to engineer around is **Q-value overestimation on out-of-distribution actions**. Bootstrapped Q-learning builds its target from the value of the next action proposed by the policy; offline, that next action can lie far from the data, and the critic—having never seen those actions—extrapolates upward. The inflated target propagates through the Bellman recursion, the policy (defined by maximizing the critic) drifts toward the over-valued actions, and the error feeds on itself. The design target is the **offline RL algorithm itself**—its losses, target construction, behavior regularization, and policy-extraction rule—subject to fixed capacity. The contribution must be algorithmic, not architectural.

## Prior art / Background / Baselines

These are the main existing answers to the overestimation problem and the gap each leaves.

- **Naive off-policy actor-critic on a static buffer.** Run a standard continuous-control actor-critic with replay and target networks on a fixed offline dataset. The actor quickly proposes actions outside the data distribution; the critic over-values them, and with no new experience the errors accumulate. Gap: unconstrained bootstrapping diverges offline.

- **Generative-model constraints (BCQ, BEAR).** Fit a model of the dataset's action distribution and restrict the policy to actions the model deems in-support, or bound the policy's distance to the behavior actions. They avoid some OOD proposals but add a generative/density model, extra sampling, and several hyperparameters that are difficult to tune without environment interaction. Gap: heavy machinery and hard-to-tune components.

- **Conservative value penalties (CQL).** Add a critic-loss term that pushes Q-values down on OOD actions and up on dataset actions. It lowers overestimation, but it must explicitly sample and query OOD actions and introduces an additional temperature/coefficient. Gap: still requires OOD queries and another sensitive hyperparameter.

- **Behavior-regularized actor-critic (BRAC, AWAC).** Keep standard value learning and penalize the policy's divergence from the behavior policy, or extract the policy via advantage-weighted regression so it stays near the data. The divergence estimate again requires a behavior model, and a single global penalty trades exploitation against caution across all states. Gap: behavior-policy estimation plus a global regularization dial.

## Fixed substrate / Code framework

A single offline-RL training file is frozen except for one region. The fixed loop loads a D4RL MuJoCo dataset into a `ReplayBuffer`, **state-normalizes** features with the exact dataset mean/std, then for `max_timesteps = 1e6` steps samples a `batch_size` minibatch `[states, actions, rewards, next_states, dones, next_actions]` and calls `trainer.train(batch)`. Every `eval_freq = 5e3` steps it rolls out `trainer.actor` for `n_episodes = 10` and reports the **D4RL normalized score** (0 = random, 100 = expert). The buffer exposes `next_actions`—the dataset action at the next timestep—for SARSA-style and next-action-regularized targets, and the whole dataset via `replay_buffer._states/_actions/_rewards` for global statistics. Fixed utilities are provided: `soft_update` (Polyak), `compute_mean_std`/`normalize_states`, `init_module_weights` (orthogonal init), and a `_mlp()` factory **locked at hidden width 256**. `TrainConfig` defaults are `discount = 0.99`, `tau = 5e-3`, `actor_lr = critic_lr = 3e-4`, `batch_size = 256`, `normalize = True`, `orthogonal_init = True`; a method may override a whitelisted subset via `CONFIG_OVERRIDES`.

Two constraints keep this algorithmic, not a capacity contest. All hidden widths are 256—the `_mlp()` factory asserts it and custom network classes must obey it. Total trainable parameters are capped at 1.05× the largest baseline architecture, checked at runtime before training.

## Editable interface

Only one region of `custom.py` is editable: the network classes (`DeterministicActor`, `Actor`, `Critic`, `ValueFunction`) and the `OfflineAlgorithm` class—its `__init__` (build nets and optimizers) and its `train(batch)` (one gradient update returning a scalar log dict). The loop hands the trainer only the environment dimensions and `TrainConfig` learning rates; everything algorithm-specific is built inside. The contract: `self.actor` must be an `nn.Module` with an `.act(state, device)` method, and `train(batch)` must consume the six-tensor batch and step the optimizers.

The starting point is a placeholder `OfflineAlgorithm` that builds a Gaussian `Actor`, two `Critic`s and their targets, three Adam optimizers, and a `train` that does **nothing**—it returns zero losses without updating. Each method replaces this class (and any network classes it needs) and nothing else.

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

Three D4RL MuJoCo `medium-v2`/`-v1` datasets—**HalfCheetah** (`halfcheetah-medium-v2`), **Maze2D** (`maze2d-medium-v1`, a goal-reaching navigation task that rewards trajectory *stitching*), and **Walker2d** (`walker2d-medium-v2`)—each over three seeds {42, 123, 456}. One metric per dataset, higher is better: the **D4RL normalized score** (0 = random, 100 = expert), averaged over the 10 evaluation rollouts at the final budget. A strong method generalizes across all three; the task score is the geometric mean across the datasets.
