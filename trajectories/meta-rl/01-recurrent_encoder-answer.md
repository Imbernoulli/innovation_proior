**Problem.** Meta-RL task inference: map an agent's stream of `(s, a, r [, s'])` transitions to a
latent task `z` the fixed PEARL policy conditions on, so the agent adapts fast to an unseen task.
The encoder is the only open slot; the SAC backbone, the product-of-Gaussians fusion, and the
`KL(q‖N(0,I))` bottleneck are fixed.

**Key idea (recurrence as the belief accumulator).** Bayes-optimal adaptation is online belief
updating: the posterior `p(m|τ_{:t})` over the task is a function of the whole *ordered* history,
updated by each new transition (today's prior is yesterday's posterior). That is exactly a recurrent
network's job — carry a hidden state `h_t = f(h_{t-1}, x_t)` as the running sufficient statistic of
the past and read the latent off it. So embed each transition with a per-transition MLP, run a
one-layer LSTM over the ordered context, take the last hidden state, and project to
`2·latent_dim` (mean and pre-softplus variance). Setting `IS_RECURRENT = True` flips the loop into
ordered-sequence mode (and caps the context batch at 100).

**Why start here.** The recurrent encoder is the most direct expression of "inference = online
belief updating," but it imposes a *sequence* structure the Markov inference problem may not need
(each transition independently identifies the task, so order carries no extra task information), and
it must *learn* approximate order-invariance from data — costly under a 20-iteration budget with
recurrent backprop eating wall-clock. It also cannot use the loop's product-of-Gaussians sharpening
(it collapses the sequence to one per-task Gaussian). It is the floor by construction.

**What the harness omits.** The full recurrent-belief method trains the latent with a variational
ELBO whose decoder reconstructs past *and future* transitions and whose KL chains each belief to the
previous one. None of that is editable here: the loop trains the encoder through the SAC critic's
Bellman gradient plus a fixed bottleneck KL, no reconstruction. So this is recurrence as a PEARL
encoder (RL²-flavored), not the full decoder method.

**Scaffold edit / hyperparameters.** `hidden_sizes = [200, 200, 200]`, one-layer LSTM of width 200,
fan-in init, biases 0.1, small (`±3e-3`) output-head init, defensive hidden-state reset on a
task-dim mismatch (the loop's `evaluate()` does not `clear_z` before `infer_posterior`).

```python
def _identity(x):
    return x


class CustomContextEncoder(PyTorchModule):
    """PEARL recurrent encoder matching oyster.rlkit.torch.networks."""
    IS_RECURRENT = True

    def __init__(self, hidden_sizes, input_size, output_size,
                 init_w=3e-3, hidden_activation=F.relu,
                 output_activation=_identity, hidden_init=ptu.fanin_init,
                 b_init_value=0.1, **kwargs):
        self.save_init_params(locals())
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        in_size = input_size
        self.fcs = []
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))
        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim,
            num_layers=1, batch_first=True,
        )

    def forward(self, input, return_preactivations=False):
        # Oyster's recurrent path supplies ordered context as (task, seq, feat).
        task, seq, feat = input.size()
        out = input.view(task * seq, feat)

        for fc in self.fcs:
            out = self.hidden_activation(fc(out))
        out = out.view(task, seq, -1)

        # Defensive resize: oyster's evaluate() with dump_eval_paths=False
        # never calls clear_z before infer_posterior, leaving hidden sized for
        # the last training meta_batch. Reset when task dim mismatches.
        if self.hidden.size(1) != task:
            self.reset(task)

        zeros = torch.zeros(self.hidden.size()).to(ptu.device)
        out, (hn, cn) = self.lstm(out, (self.hidden, zeros))
        self.hidden = hn
        out = out[:, -1, :]

        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)

        if return_preactivations:
            return output, preactivation
        return output

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)
```
