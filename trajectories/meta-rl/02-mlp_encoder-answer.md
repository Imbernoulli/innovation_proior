**Problem (from step 1).** The recurrent encoder failed exactly where task inference is load-bearing
(`cheetah-vel` −185, `sparse-point-robot` 0.62 with two dead seeds) and only held on the trivial
dense game — because it imposed a *sequence* structure the Markov inference problem does not have,
had to learn approximate order-invariance from data under a 20-iteration budget, and never used the
loop's product-of-Gaussians fusion.

**Key idea (per-transition MLP + product of Gaussians).** The context is an *unordered set* of
Markov transitions — each one independently identifies the task — so build permutation-invariance in
instead of learning it: encode each transition independently with a stateless MLP that emits a
Gaussian factor `N(μ_n, σ_n²)`, and let the fixed agent fuse them by product, `1/σ*² = Σ_n 1/σ_n²`,
`μ* = σ*²·Σ_n μ_n/σ_n²`. The precisions add, so the belief sharpens with every transition (the
sharpening the recurrent path threw away); the mean is precision-weighted. Output is
`2·latent_dim` (means and pre-softplus variances); the agent applies softplus and floors variances at
`1e-7`.

**Why it works.** Permutation-invariance is correct by construction (multiplication commutes), so no
budget is spent learning to ignore an uninformative ordering; the loop's fusion gives free Bayesian
evidence accumulation; and the encoder is *simpler* than the LSTM, so the budget goes to useful
gradient steps. The `kl_lambda·KL(q‖N(0,I))` bottleneck (a bound on `I(z;c)`) handles overfitting, so
a modest 3×200 MLP suffices.

**What the harness omits.** PEARL's encoder is trained through the critic's Bellman gradient with
context sampled from recent data while the RL batch comes from the whole off-policy buffer — all
fixed in the loop. This rung fills only the per-transition `f_φ`; it is the original PEARL encoder in
its native slot.

**Scaffold edit / hyperparameters.** `hidden_sizes = [200, 200, 200]`, ReLU, `fanin_init` on hidden
weights, biases 0.1, small (`±3e-3`) output-head init; stateless (`reset` is a no-op).

```python
class CustomContextEncoder(PyTorchModule):
    """Original PEARL MLP context encoder (3-layer, 200 units)."""
    def __init__(self, hidden_sizes, input_size, output_size,
                 init_w=3e-3, hidden_activation=F.relu, **kwargs):
        self.save_init_params(locals())
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation

        in_dim = input_size
        self.fcs = nn.ModuleList()
        for h_dim in hidden_sizes:
            fc = nn.Linear(in_dim, h_dim)
            ptu.fanin_init(fc.weight)
            fc.bias.data.fill_(0.1)
            self.fcs.append(fc)
            in_dim = h_dim
        self.last_fc = nn.Linear(in_dim, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        preactivation = self.last_fc(h)
        output = preactivation
        if return_preactivations:
            return output, preactivation
        return output

    def reset(self, num_tasks=1):
        pass
```
