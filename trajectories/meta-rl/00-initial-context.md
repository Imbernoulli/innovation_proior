## Research question

Meta-reinforcement learning: an agent meta-trained on a *family* of related tasks — the same
cheetah at different target velocities, the same arena with the goal at different places — is
dropped into an unseen task and must get good in a handful of trajectories. The only window onto
"which task am I in" is the experience it collects: a stream of transition tuples
`(state, action, reward, next_state)`. The single thing being designed is the **context encoder**
— the module that eats those tuples and turns them into a latent task representation `z` the policy
conditions on. Everything else about the agent — the SAC policy/value backbone, the replay buffers,
the task sampling, the outer training loop, and the way per-transition encoder outputs are fused
into the task posterior — is fixed.

## Prior art before the first rung (meta-RL adaptation lineage)

The fixed substrate the first rung plugs into is PEARL's off-policy, probabilistic-latent agent;
the encoder slot is what the ladder fills. These are the methods that precede the ladder — the
lineage the first encoder reacts to.

- **RL² / recurrent meta-RL (Duan et al. 2016; Wang et al. 2016).** Feed an RNN the stream of
  `(state, action, reward)` and let adaptation live entirely in the recurrent dynamics: a single
  deterministic hidden state, trained only through the return, *is* the agent's memory of the task.
  General and end-to-end, but on-policy and sample-hungry, and the hidden state is never made to be
  an honest, calibrated representation of *which task* or *how uncertain* — it is whatever the return
  gradient shapes it into. Gap: expensive to meta-train, no uncertainty handle for exploration.
- **MAML / gradient-based meta-RL (Finn et al. 2017).** Meta-learn an initialization from which a
  *single policy-gradient step* adapts to a new task. Elegant and policy-agnostic, but on-policy
  (adapts from freshly collected data and discards it), and adaptation yields a *point* estimate of
  the task — one gradient step, one new policy — with no representation of uncertainty to drive
  directed exploration. Gap: data-inefficient, deterministic point adaptation.
- **PEARL (Rakelly et al. 2019).** Disentangle *task inference* from *control*: an encoder maps
  context transitions to a posterior `q(z|c)` over a low-dimensional latent task, an SAC policy and
  critic condition on samples of `z`, and the two are fed different data — the encoder near-on-policy
  recent context, the actor/critic the whole off-policy buffer — which is what lets meta-training be
  cheap. The latent is probabilistic, so its spread *is* the exploration mechanism (posterior
  sampling). This is the fixed substrate below. The one slot it leaves open is the encoder
  architecture, and the per-transition encoder outputs are fused into the posterior by a fixed
  **product of Gaussians** in the agent. Gap (the ladder's target): the *shape* of the encoder — how
  a transition becomes a Gaussian factor, and whether order or cross-transition structure should
  matter — is exactly what is unspecified.

## The fixed substrate

A PEARL meta-RL loop is frozen and must not be touched. An SAC actor-critic (`TanhGaussianPolicy`,
two Q-heads, a V-head, all `net_size`-wide FlattenMLPs) conditions on `[obs, z]`; a `PEARLAgent`
owns the encoder and turns its output into the task posterior. Concretely the agent, in
`infer_posterior`, calls the encoder on a context tensor shaped `(task, seq, feat)` —
`feat = obs_dim + action_dim + 1` (reward), or `+ next_obs` when `use_next_obs_in_context` — then
reshapes the encoder output to `(task, -1, output_size)` and, because `use_information_bottleneck`
is on, reads the first `latent_dim` columns of each per-transition row as a mean `μ_n` and softplus
of the rest as a variance `σ_n²`, and **fuses them by a product of Gaussians**:
`σ² = 1/Σ_n σ_n^{-2}`, `μ = σ²·Σ_n μ_n σ_n^{-2}` (precisions add, so belief sharpens with evidence).
The encoder loss is the SAC critic's Bellman gradient plus a `kl_lambda·KL(q(z|c)‖N(0,I))`
information bottleneck — *not* a reconstruction/ELBO objective. So whatever the encoder emits per
context row is interpreted as a Gaussian factor and combined by the loop; the encoder cannot change
the aggregation, the loss, or the bottleneck.

The harness exposes one structural lever the encoder can flip: a class attribute
`IS_RECURRENT = True` tells the launcher to set `recurrent=True`, which makes the loop sample context
as an *ordered sequence* and truncate backprop through it. A recurrent encoder therefore collapses
the `seq` dimension to a single per-task output (one Gaussian, no product), and the launcher caps the
context batch at 100 transitions for it (the LSTM is otherwise ~10x over the time budget). A
non-recurrent encoder keeps one output per transition for the product of Gaussians.

## The editable interface

Exactly one region is editable — the `CustomContextEncoder` class (and a custom-import line) in
`oyster/custom_encoder.py`. Every method on the ladder is a fill of this same contract:
`__init__(hidden_sizes, input_size, output_size, **kwargs)` — must call
`self.save_init_params(locals())` first, set `self.output_size`, and extend `PyTorchModule`;
`forward(input, return_preactivations=False)` returning a tensor whose last dim is `output_size`
(`output_size = 2·latent_dim`: means and pre-softplus variances); and `reset(num_tasks=1)` to clear
any stateful component (the recurrent hidden state). `hidden_sizes` is `[200, 200, 200]` and the
input/output dims are computed by the fixed launcher.

The starting point is the scaffold default: a single linear layer from the transition features
straight to `output_size`. Each method replaces exactly this class.

```python
# EDITABLE region of oyster/custom_encoder.py — default fill (single linear layer placeholder)
class CustomContextEncoder(PyTorchModule):
    """Context encoder for PEARL meta-RL task inference.

    Input:  (*, input_size) transition features (obs, action, reward [, next_obs])
    Output: (*, output_size) Gaussian parameters (mean and log_variance)
    """
    def __init__(self, hidden_sizes, input_size, output_size,
                 init_w=3e-3, hidden_activation=F.relu, **kwargs):
        self.save_init_params(locals())
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Default: single linear layer (baseline placeholder)
        self.fc = nn.Linear(input_size, output_size)
        self.fc.weight.data.uniform_(-init_w, init_w)
        self.fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        output = self.fc(input)
        if return_preactivations:
            return output, output
        return output

    def reset(self, num_tasks=1):
        pass
```

## Evaluation settings

Three meta-RL task families spanning the difficulty range, each over three seeds {42, 123, 456}:
**Half-Cheetah Velocity** (`cheetah-vel`: 30 train / 10 test tasks, target speeds in [0,3] m/s,
obs dim 20, action dim 6, dense velocity-matching reward — the headline, high-dimensional continuous
task distribution); **Sparse Point Robot** (`sparse-point-robot`: 40 train / 10 test tasks, goals on
a half-circle, sparse +1-inside-radius reward, obs/action dim 2 — tests extracting task identity from
a binary signal); and **Point Robot** (`point-robot`: 40 train / 10 test tasks, goals uniform in
[-1,1]², dense negative-L2 reward, obs/action dim 2 — a simpler dense distribution). The single
metric is `meta_test_return`, the average return on held-out test tasks after meta-training under a
deliberately short fixed budget (20 outer iterations; far below the 500+ of the source papers, so
absolute returns are not comparable to them — only the relative ordering across encoders is
meaningful). Higher is better. On `sparse-point-robot`, a 0 means no goal was reached within the
budget rather than an algorithmic failure.
