How much a network can know is bounded by how many parameters it holds, and on the large text corpora — a billion words, tens of billions of words — quality keeps climbing as I add parameters. The data has more to teach than my models can currently store. But there is a wall: in an ordinary dense network every parameter fires on every example, so doubling the parameters doubles the work per example, and since more capacity demands more training signal, the total training cost grows as roughly the *product* of model size and dataset size — a quadratic blow-up as I scale both axes together. Hardware is not keeping up with that, so I cannot simply keep making a dense model bigger. What I need is to break the link between parameter count and compute-per-example: a model that *holds* an enormous bank of parameters but only *touches* a tiny fraction of them per example, routing each input to a few specialist subnetworks and leaving the rest dormant.

The wish for "conditional computation" is old, and the reasons it has not been cashed out at scale are concrete and constrain everything. GPUs are far faster at dense arithmetic than at branching, so a gating decision must switch a *large* chunk of computation on or off to pay for itself — that argues for coarse units, whole subnetworks rather than individual neurons. Large batches are what make these devices efficient, but routing examples conditionally means each active chunk sees only the examples sent to it, a smaller batch — a real tension with the batching that makes hardware fast. Inter-device bandwidth can be thousands of times smaller than aggregate compute, so any scheme that ships parameters or activations per example (an embedding lookup is the classic case) becomes bandwidth-bound and the compute sits idle. The hard-binary-gate line (Bengio and colleagues) needed three auxiliary losses and a REINFORCE-style estimator, which means high-variance gradients for the router. And the existing demonstrations were on small image sets of a few hundred thousand examples, which plausibly cannot supply enough signal to train millions, let alone billions, of parameters. The natural shape — a bank of experts plus a router — already exists as the adaptive mixtures of local experts of Jacobs, Jordan, Nowlan and Hinton (1991): expert networks $E_1 \dots E_n$ and a gating network $G$ producing a weight per expert, with output $y = \sum_i G(x)_i\, E_i(x)$ and $G(x) = \mathrm{Softmax}(x \cdot W_g)$. But classically the mixture is the whole model and the gate is *dense* — the softmax gives every expert nonzero weight, so all $n$ experts run for every example. That is no savings at all; capacity and compute rise together, the same wall. Eigen, Ranzato and Sutskever used mixtures as *components* inside a deeper network, the right granularity for a drop-in layer, and noted the gate could be made sparse — but kept it dense.

I propose the Sparsely-Gated Mixture-of-Experts layer. The move is in the equation: wherever $G(x)_i = 0$ the term contributes nothing and I never have to compute $E_i(x)$ at all, so if I make $G(x)$ *sparse* — mostly zeros, only $k$ nonzero entries — I run only $k$ experts per example no matter how large $n$ is. Capacity is $n$ experts' worth of parameters, compute is $k$ experts' worth; set $n$ to thousands and $k$ to a handful and I have a thousand-fold capacity increase at nearly constant compute. The sparsity of the gate *is* the conditional computation. To get sparsity I keep the top $k$ gating logits and force the rest to $-\infty$ before the softmax, so the $-\infty$ entries exponentiate to zero and the gate is supported on exactly $k$ experts while still summing to one:
$$G(x) = \mathrm{Softmax}(\mathrm{KeepTopK}(H(x), k)), \qquad \mathrm{KeepTopK}(v,k)_i = \begin{cases} v_i & v_i \text{ among top } k \text{ of } v \\ -\infty & \text{otherwise.}\end{cases}$$
Top-$k$ is discontinuous — as $x$ moves, the membership of the top-$k$ can jump — but the boundary set has measure zero and the softmax over the surviving $k$ is smooth, so in practice this causes no trouble. The choice of $k$ matters for training: with $k=1$ the softmax over a single surviving entry is the constant $1$, giving no gradient to the gate; with $k>1$ the softmax over the surviving logits has a genuine nonzero derivative with respect to each, hence with respect to $W_g$, so the gating network learns by plain backprop to up-weight experts that helped and down-weight those that hurt. This is why I take $k>1$, and it is a clean win over the boolean-gate line: the selected experts are differentiable, so I get low-variance backprop gradients straight through the gate rather than a high-variance REINFORCE estimate of a hard choice.

The danger that quietly kills naive trainable gating is collapse. Train $G$ and the experts together and a small initial accident makes the gate prefer a few experts; those get more examples, hence more gradient, hence improve faster, hence get picked even more — a self-reinforcing run-away in which a small clique dominates and the rest, never selected, never train and can never be picked. So I add a force that spreads load. Over a batch $X$ define the *importance* of expert $i$ as the total gate weight it receives, $\mathrm{Importance}(X)_i = \sum_{x \in X} G(x)_i$. I want a penalty that is zero when these are equal and grows as they spread, and is scale-free since absolute magnitude depends on batch size and $k$ — the squared coefficient of variation $\mathrm{CV}^2 = \mathrm{var}/\mathrm{mean}^2$ is exactly that, smooth, differentiable, and zero iff all entries are equal:
$$L_{\text{importance}}(X) = w_{\text{importance}} \cdot \mathrm{CV}\big(\mathrm{Importance}(X)\big)^2.$$
But equal importance is not equal *example count*: one expert could get a few examples with large gate values while another gets many with small values, identical importance but wildly different load, and load is what a distributed system feels — an overloaded expert is a straggler that needs more activation memory and bottlenecks the step. So I add a second penalty on the number of examples per expert. The obvious count $\sum_x \mathbf{1}[G(x)_i > 0]$ is a discrete integer with zero gradient almost everywhere; I need a smooth surrogate. The fix reuses noise that I want for another reason anyway: a deterministic gate gives the collapse dynamic no escape, since a disfavored expert is never tried, whereas injecting randomness into the logits lets a normally-disfavored expert occasionally get bumped into the top-$k$, earn some examples and gradient, and stay alive while the balancing loss does its work. So I add tunable Gaussian noise to each logit before the top-$k$:
$$H(x)_i = (x \cdot W_g)_i + \mathrm{StandardNormal}() \cdot \mathrm{Softplus}\big((x \cdot W_{\text{noise}})_i\big),$$
where the noise scale is a per-expert, per-input standard deviation set by a second trainable matrix $W_{\text{noise}}$; Softplus is used because a standard deviation must be nonnegative, and it maps any real to a positive value smoothly, staying differentiable everywhere unlike a hard ReLU clamp. With noise in $H$, whether expert $i$ lands in the top-$k$ is now a random event with a smooth *probability*. Taking a fresh draw of the noise on component $i$ while holding the others fixed, $G(x)_i$ is nonzero exactly when $H(x)_i$ exceeds the $k$-th greatest of the *other* components, $\mathrm{kth\_excluding}(H(x), k, i)$. The only random quantity is the standard normal, and the probability it exceeds a value $t$ is $\Phi(-t)$, so
$$P(x,i) = \Phi\!\left( \frac{(x \cdot W_g)_i - \mathrm{kth\_excluding}(H(x), k, i)}{\mathrm{Softplus}\big((x \cdot W_{\text{noise}})_i\big)} \right),$$
with $\Phi$ the standard normal CDF. This slides smoothly from 0 to 1 and is differentiable in both $W_g$ and $W_{\text{noise}}$; the noise standard deviation in the denominator is precisely what turns a hard step into a soft one, and with zero noise it collapses back to the non-differentiable indicator — which is exactly why the noise had to be there for the load estimate. Then $\mathrm{Load}(X)_i = \sum_{x \in X} P(x,i)$ and $L_{\text{load}}(X) = w_{\text{load}} \cdot \mathrm{CV}(\mathrm{Load}(X))^2$, the same $\mathrm{CV}^2$ shape as importance. Either loss alone reaches similar quality and no balancing loss at all is far worse, so I keep both with modest hand-tuned weights (around 0.1 for language modeling, 0.01 for translation). A subtlety pins down the threshold: an expert not currently in the top-$k$ must beat the $k$-th best value overall to get in, while one currently in the top-$k$ faces a bar at the $(k+1)$-th best, since excluding it drops the bar by one rank — so I compute the top $k+1$ noisy values and, in 0-based code, use `top[k-1]` as the "if I were out" threshold and `top[k]` as the "if I were in" threshold. One final guard: I initialize $W_g = W_{\text{noise}} = 0$, so at step zero the clean logits are all equal and only the noise distinguishes experts — pure symmetric randomness gives roughly balanced load from the outset, before the balancing losses engage, avoiding an early out-of-memory crash from an accidental overload.

This still leaves the shrinking-batch wall: routing $k$ of $n$ experts per example gives each expert only about $kb/n$ examples from a batch of $b$, starving the experts of the big batches the hardware needs. The trick is to combine two kinds of parallelism: keep the standard layers and the gating network *data-parallel* as usual, but hold only one shared copy of each expert *model-parallel*, sharded across the $d$ devices, so that for the MoE layer each expert pools the examples routed to it from all $d$ synchronous input batches and sees about $k\,b\,d/n$ — a factor of $d$ larger. Adding experts then means adding devices in proportion, keeping per-expert batch, per-device memory, per-device bandwidth, and step time constant: capacity grows with the cluster at fixed cost per device. Applying the same MoE to all timesteps of a sequence at once gives a further batch multiple for free. The bandwidth wall is handled because the experts are stationary, so only their inputs and outputs cross the network, not their parameters; for a one-hidden-layer feed-forward expert the compute-to-io ratio equals the hidden size, so widening the experts — the same knob that adds capacity — keeps the layer compute-bound. For very large $n$, a hierarchical MoE folds the experts into a primary gate over $a$ groups, each a secondary MoE over $b$ experts, with $y_H = \sum_{i,j} G_{\text{primary}}(x)_i\, G_i(x)_j\, E_{i,j}(x)$ and load weighted as $\mathrm{Load}_H(X)_{i,j} = \mathrm{Load}_{\text{primary}}(X)_i \cdot \mathrm{Load}_i(X^{(i)})_j / |X^{(i)}|$ so the gradient reaches the primary gate. The implementation realizes the saving with a sparse dispatcher that gathers a compact per-expert batch from the nonzero gate entries, runs each expert once on its own slice, and scatter-adds the gate-weighted outputs back — never evaluating every expert on every example, which would defeat the entire purpose. The returned `loss` (the sum of the two $\mathrm{CV}^2$ terms scaled by `loss_coef` $= w_{\text{importance}} = w_{\text{load}}$) is added to the model's task loss, and the layer drops in between other layers (e.g. between stacked LSTMs) at every sequence position.

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
        # Tensor2Tensor orders pairs from transpose(gates): expert-major,
        # then batch-major within each expert.
        where = torch.nonzero(gates.t() > 0, as_tuple=False)
        self._expert_index = where[:, 0]
        self._batch_index = where[:, 1]
        self._part_sizes = (gates > 0).sum(0).tolist()
        self._nonzero_gates = gates[self._batch_index, self._expert_index].unsqueeze(1)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = stitched.new_zeros(self._gates.size(0), stitched.size(1))
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
        if x.numel() <= 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        x = x.float()
        return x.var(unbiased=False) / (x.mean() ** 2 + eps)

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
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        gates = torch.zeros_like(clean_logits).scatter(1, top_k_indices, top_k_gates)
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
