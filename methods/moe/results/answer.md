# The Sparsely-Gated Mixture-of-Experts Layer

## Problem

A network's capacity to absorb information is bounded by its parameter count, but
in a dense network every parameter fires on every example, so capacity and
compute-per-example rise together — a quadratic blow-up in total training cost as
model and dataset grow. The goal is to decouple them: a model with an enormous
parameter count whose per-example compute stays roughly fixed, trainable
end-to-end, efficient on branch-averse, bandwidth-limited GPU clusters, and
demonstrated on data large enough to train billions of parameters (language
modeling, machine translation).

## Key idea

A general-purpose layer holding `n` expert sub-networks `E_1 … E_n` (here
identical feed-forward nets with separate parameters) plus a trainable gating
network `G` that emits a **sparse** weight vector over the experts:

    y = Σ_{i=1}^{n} G(x)_i · E_i(x)

Because `G(x)` is mostly zero, wherever `G(x)_i = 0` the expert `E_i` is not
computed. With only `k ≪ n` experts active per example, parameter count scales
with `n` while compute scales with `k` — a >1000× capacity increase at nearly
constant compute.

## Noisy Top-K gating

Start from softmax gating `G_σ(x) = Softmax(x · W_g)` (dense, no savings) and add
sparsity and noise:

    H(x)_i  = (x · W_g)_i + StandardNormal() · Softplus((x · W_noise)_i)
    KeepTopK(v, k)_i = v_i if v_i is in the top k of v, else −∞
    G(x)    = Softmax(KeepTopK(H(x), k))

- **Top-k → −∞ → softmax** gives an exactly `k`-sparse, normalized gate.
- **`k > 1`** so the surviving gate values have nonzero gradient w.r.t. `W_g`;
  trained by plain backprop, not REINFORCE.
- **Tunable Gaussian noise** (scale `Softplus(x · W_noise)`, kept nonnegative and
  differentiable by Softplus) provides exploration and makes the load estimate
  below differentiable.

## Balancing expert utilization

Trainable gates collapse onto a few experts (favored experts train faster, so are
chosen more — self-reinforcing). Two scale-free `CV²` penalties counteract it
(`CV(z)² = var(z)/mean(z)²`, zero iff all entries equal):

**Importance** — equalize total gate weight per expert:

    Importance(X)_i = Σ_{x ∈ X} G(x)_i
    L_importance(X) = w_importance · CV(Importance(X))²

**Load** — equalize the *number of examples* per expert. The count is discrete, so
use the noise to define a smooth probability that expert `i` is selected for `x`
(fresh noise on `i`, others fixed). `G(x)_i > 0` iff `H(x)_i` exceeds the `k`-th
greatest of the other components, `kth_excluding(H(x), k, i)`:

    P(x, i) = Φ( ( (x·W_g)_i − kth_excluding(H(x), k, i) ) / Softplus((x·W_noise)_i) )
    Load(X)_i = Σ_{x ∈ X} P(x, i)
    L_load(X) = w_load · CV(Load(X))²

with `Φ` the standard normal CDF. Either loss alone reaches similar quality; no
loss is far worse. Both are used with modest, hand-tuned weights (e.g. 0.1 for
language modeling, 0.01 for translation). Initialize `W_g = W_noise = 0` so the
load starts balanced (only noise distinguishes experts) before the losses engage.

## Scaling and engineering

- **Shrinking batch:** each expert sees ~`kb/n` examples per batch. Keep standard
  layers + gate **data-parallel** but hold one shared, **model-parallel** copy of
  each expert; pooling examples across `d` synchronous devices gives each expert
  ~`kbd/n` examples. Adding experts ⇔ adding devices, at constant per-device cost.
- **Convolutional application:** apply the same MoE to all timesteps at once for a
  further batch-size multiple.
- **Bandwidth:** experts are stationary, so only inputs/outputs cross the network.
  A one-hidden-layer expert's compute-to-io ratio equals its hidden size, so
  widening experts keeps the layer compute-bound.
- **Hierarchical MoE** for very large `n`: a primary gate over `a` groups, each a
  secondary MoE over `b` experts, with `y_H = Σ_{i,j} G_primary(x)_i G_i(x)_j E_{i,j}(x)`;
  load uses `Load_H_{i,j} = Load_primary_i · Load_i(X^{(i)})_j / |X^{(i)}|` so the
  gradient reaches the primary gate.

## Code

```python
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class SparseDispatcher:
    """Build a compact input batch per expert from the nonzero gate entries and
    recombine outputs weighted by the gates. Only examples with gates[b,i] > 0 are
    sent to expert i, so the sparsity yields a real compute saving."""

    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1),
                            requires_grad=True, device=stitched.device)
        return zeros.index_add(0, self._batch_index, stitched.float())

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class MoE(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size,
                 noisy_gating=True, k=4):
        super().__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList(
            [Expert(input_size, output_size, hidden_size) for _ in range(num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)
        gates = torch.zeros_like(logits, requires_grad=True).scatter(
            1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev,
                                       top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss
```

The returned `loss` (the sum of the two `CV²` balancing terms, scaled by
`loss_coef = w_importance = w_load`) is added to the model's task loss; the layer
is dropped between other layers (e.g. between stacked LSTMs) and applied at every
sequence position.
