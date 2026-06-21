## Research question

Meta-reinforcement learning: I am given a distribution `p(T)` over related MDPs that share a state and action space but differ in their reward and/or transition functions, and I must build an agent that, after meta-training across the family, walks into a held-out task and is good at it from a handful of interactions. Concretely I design two coupled objects. The **agent** performs task inference — it encodes the experience it has collected so far (the context) into a compact task representation and conditions its policy on that representation — and the **algorithm** meta-trains the agent across tasks so the inference and the conditioned policy generalize to unseen tasks. Everything reduces to three knobs: how to encode context into a task variable, how to condition control on it, and how to optimize the whole thing across tasks so it transfers.

## Prior art / Background / Baselines

These are the methods the design reacts to; each is here for the gap it leaves.

- **RL² / learning-to-RL (Duan et al. 2016; Wang et al. 2016).** Make the policy a recurrent net, feed it each observation together with the previous action, reward, and done flag, and carry the hidden state across the episode; ordinary RL trains the whole recurrence to maximize trial return. *Gap:* it is on-policy and sample-hungry (~1e8 steps), and the hidden state is an opaque vector with no explicit task representation or notion of uncertainty.
- **MAML / ProMP (Finn et al. 2017; Rothfuss et al. 2018).** Meta-learn an initialization such that a few policy-gradient steps adapt the policy to a new task. *Gap:* both inner and outer loops are on-policy, and the exploration data is collected under the un-adapted policy and therefore separated from exploitation by construction — the adaptation data is never optimized for online return.
- **Soft Actor-Critic (Haarnoja et al. 2018).** An off-policy, maximum-entropy actor-critic for single-task continuous control that uses twin Q-networks, a target value network, a reparameterized squashed-Gaussian actor, and a replay buffer. *Gap:* it has no task variable and no mechanism to adapt across a family of tasks; it serves as the control backbone that will be conditioned on `z`.
- **Prototypical networks (Snell et al. 2017).** Few-shot classification by embedding support examples with a shared network, representing each class as the mean of its embeddings, and classifying by distance. *Gap:* it is a supervised-classification method with labeled support sets and discrete classes, not a sequential decision-making algorithm that must infer tasks from reward sequences.

The fixed substrate below is an off-policy SAC actor-critic conditioned on a task variable `z`, with a context encoder and a meta-gradient step left open to design.

## Fixed substrate / Code framework

A meta-RL harness is frozen and must not be touched. It provides:

- **Environments and configs.** Three task families (`cheetah-vel`, `sparse-point-robot`, `point-robot`), each with its own `algo_params`: number of outer iterations, per-task collection budgets (`num_steps_prior`, `num_steps_posterior`, `num_extra_rl_steps_posterior`), gradient steps per iteration, `meta_batch`, RL `batch_size`, `embedding_batch_size`, `discount`, `reward_scale`, `max_path_length`, and `sparse_rewards`.
- **Network building blocks.** `build_mlp(in, out, hidden=200, n_layers=3)`; `build_policy(obs, act, latent, net_size=300)` (a `TanhGaussianPolicy` over `(obs, z)`); `build_qf(...)` and `build_vf(...)` (`FlattenMlp` heads over `(obs, act, z)` / `(obs, z)`).
- **Replay + sampling.** `create_replay_buffers(env, tasks)` returns an RL buffer and a separate *encoder* buffer; `sample_context_from_buffer(enc_buf, indices, embedding_batch_size, ...)` returns a context tensor `(num_tasks, embedding_batch_size, context_dim)` where each row is `(o, a, r)` (and `next_o` if `use_next_obs_in_context`); `sample_sac_batch(buf, indices, batch_size)` returns a decorrelated RL batch with a leading task dimension; `collect_data(...)` rolls out the sampler into both buffers and (when posterior-conditioned) re-infers `z` online; `InPlacePathSampler` is the rollout engine.
- **The evaluation protocol.** For each held-out task: clear the context, collect `num_exp_traj_eval` exploration trajectories with the context accumulating, call `agent.adapt()`, then continue with a **deterministic** policy; `meta_test_return` is the average final-trajectory return over held-out tasks. Higher is better.
- **The outer loop.** Collect an initial pool; then per iteration put the nets in train mode, call `algorithm.train_iteration(it)`, switch to eval mode, and run evaluation.

## Editable interface

Exactly one region is editable — the `CustomMetaRLAgent` and `CustomMetaRLAlgorithm` classes in `custom_meta_rl.py`.

The **agent** must implement `get_action(obs, deterministic=False) -> (action_np, info)` (act conditioned on the current task belief); `update_context(transition)` (accumulate one online transition); `adapt()` (infer the task from accumulated context, after exploration); `clear_context(num_tasks=1)` (reset context and belief); `infer_posterior(context_tensor)` (encode a context batch drawn from the buffer, used in training); a `context` property; a `z` attribute (the latent task variable); and a `networks` property (the `nn.Module` list for GPU transfer). The **algorithm** must implement `collect_initial_data()`; `train_iteration(iteration_idx) -> dict` (one meta-training iteration = collection + gradient updates); and a `networks` property.

The starting point is the scaffold default: an MLP `TanhGaussianPolicy` with a dummy zero `z`, **no context encoder, and a no-op training step** — `infer_posterior`/`adapt`/`train_iteration` do nothing, so `z` never carries task information and no gradients are taken. The implementation replaces exactly these two classes with a real encoder, a real `z`-conditioned actor-critic, and a real meta-gradient step.

```python
# EDITABLE region of custom_meta_rl.py — default fill (no task inference, no training)
class CustomMetaRLAgent(nn.Module):
    """Placeholder: MLP policy, dummy zero z, no context encoder."""

    def __init__(self, obs_dim, action_dim, latent_dim=5, net_size=300,
                 reward_dim=1, use_next_obs_in_context=False, **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.use_next_obs_in_context = use_next_obs_in_context
        self.policy = build_policy(obs_dim, action_dim, latent_dim, net_size)
        self.register_buffer('z', torch.zeros(1, latent_dim))      # dummy: never informative
        self._context = None

    def clear_context(self, num_tasks=1):
        self.z = ptu.zeros(num_tasks, self.latent_dim)
        self._context = None

    def clear_z(self, num_tasks=1):
        self.clear_context(num_tasks)

    def sample_z(self):
        pass

    @property
    def context(self):
        return self._context

    def update_context(self, inputs):                              # append one online transition
        o, a, r, no, d, info = inputs
        o = ptu.from_numpy(o[None, None, ...]); a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...]); no = ptu.from_numpy(no[None, None, ...])
        data = torch.cat([o, a, r, no], dim=2) if self.use_next_obs_in_context \
            else torch.cat([o, a, r], dim=2)
        self._context = data if self._context is None else torch.cat([self._context, data], dim=1)

    def adapt(self):                                               # task inference — override
        pass

    def infer_posterior(self, context):                           # encode replay context — override
        pass

    def get_action(self, obs, deterministic=False):
        in_ = torch.cat([ptu.from_numpy(obs[None]), self.z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def detach_z(self):
        self.z = self.z.detach()

    @property
    def networks(self):
        return [self.policy]


class CustomMetaRLAlgorithm:
    """Placeholder: collects an initial pool; the training step is a no-op."""

    def __init__(self, agent, env, train_tasks, eval_tasks,
                 replay_buffer, enc_replay_buffer, config):
        self.agent = agent; self.env = env
        self.train_tasks = train_tasks; self.eval_tasks = eval_tasks
        self.replay_buffer = replay_buffer; self.enc_replay_buffer = enc_replay_buffer
        self.config = config
        self.sampler = InPlacePathSampler(env=env, policy=agent,
                                          max_path_length=config['max_path_length'])

    def collect_initial_data(self):
        n = self.config.get('num_initial_steps', 200)
        for idx in self.train_tasks:
            self.env.reset_task(idx)
            collect_data(self.agent, self.env, self.sampler,
                         self.replay_buffer, self.enc_replay_buffer,
                         idx, n, 1, np.inf, add_to_enc_buffer=True, config=self.config)

    def train_iteration(self, iteration_idx):
        return {}                                                  # no gradient updates

    @property
    def networks(self):
        return self.agent.networks
```

## Evaluation settings

Three meta-RL families spanning the difficulty range, each over three seeds {42, 123, 456}:

- **`cheetah-vel`** — 30 train / 10 test tasks; target velocities in `[0, 3]` m/s; obs dim 20, action dim 6; **dense** reward from velocity matching. High-dimensional observations stress the encoder.
- **`sparse-point-robot`** — 40 train / 10 test tasks; goals on a half-circle; **sparse** reward (+1 near goal, 0 otherwise); obs dim 2, action dim 2. A return of 0 means no goal was reached in the budget; sparsity makes task inference and exploration the whole game.
- **`point-robot`** — 40 train / 10 test tasks; goals in `[-1, 1]²`; **dense** reward (negative L2 distance); obs dim 2, action dim 2. The basic-quality check.

The metric on each is `meta_test_return` — average held-out-task return after meta-training under the adapt-then-deterministic protocol; higher is better on all three. The training budget is intentionally short (20-ish outer iterations, ~1 hour wall time per environment), so only the *relative* ordering within this benchmark is meaningful.
