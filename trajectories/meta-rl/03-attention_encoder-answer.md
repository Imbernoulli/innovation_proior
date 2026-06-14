**Problem (from step 2).** The MLP encoder fixed the structure (sparse-point lifted to 3.80 all-seed
non-zero; cheetah-vel to −141.86) but its product-of-Gaussians fusion treats transitions as
*conditionally independent given `z`* — each factor is computed in isolation. On `cheetah-vel`, where
the task (target velocity) is a *trajectory* property revealed by the *relations between*
transitions, that independence leaves the most on the table (still −141, far from adapted).

**Key idea (self-attention as a cross-transition aggregator).** Let the transitions in a context
*talk to each other* before they become Gaussian factors. Self-attention —
`softmax(Q Kᵀ/√d_k) V` with `Q,K,V` all projections of the context — lets each transition attend to
every other by content, in one parallel operation, and is *permutation-equivariant*: it preserves one
output per transition, order-independent, exactly what the loop's product of Gaussians needs. So:
per-transition MLP embedding → one multi-head self-attention sub-layer (4 heads over the 200-wide
embedding) with a residual and layer norm → per-transition output head to `2·latent_dim`. The
`√d_k` scaling (internal to `nn.MultiheadAttention`) keeps logits at unit scale so the softmax does
not saturate.

**Why it works.** Cross-transition mixing is the upgrade to the MLP that *keeps* the fusion the
recurrent encoder broke. The residual *adds* gathered context to each transition's own embedding
(a transition that needs nothing from neighbors keeps its factor); layer norm (not batch norm) has
per-position statistics, so it works for a variable-sized context group. No positional encoding —
order is not information here and would break the permutation-equivariance.

**What the harness omits / shrinks.** The full attention encoder stacks `N=6` blocks of
self-attention *plus* a wide (`d_ff=2048`) per-position feed-forward sub-layer, `d_model=512`, `h=8`.
None of that depth is affordable in a 20-iteration budget over short contexts, so this rung keeps the
*one* idea (transitions attend to each other) in a single minimal block: no 6-deep stack, no FFN
sub-layer, `h=4`. The forward must handle both 2-D `(batch, feat)` and 3-D `(task, seq, feat)` input
and must *preserve per-transition output* (no pooling), or the agent's product of Gaussians breaks.

**Scaffold edit / hyperparameters.** `hidden_sizes = [200, 200, 200]`, ReLU, `fanin_init`, biases 0.1;
`nn.MultiheadAttention(200, num_heads=4, batch_first=True)`; one `LayerNorm(200)`; small (`±3e-3`)
output-head init; stateless (`reset` is a no-op, `IS_RECURRENT` off).

```python
class CustomContextEncoder(PyTorchModule):
    """Self-attention context encoder for cross-transition reasoning."""
    def __init__(self, hidden_sizes, input_size, output_size,
                 init_w=3e-3, hidden_activation=F.relu, **kwargs):
        self.save_init_params(locals())
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.hidden_dim = hidden_sizes[-1]

        # Per-transition MLP embedding
        in_dim = input_size
        self.fcs = nn.ModuleList()
        for h_dim in hidden_sizes:
            fc = nn.Linear(in_dim, h_dim)
            ptu.fanin_init(fc.weight)
            fc.bias.data.fill_(0.1)
            self.fcs.append(fc)
            in_dim = h_dim

        # Self-attention for cross-transition reasoning
        self.attn = nn.MultiheadAttention(
            self.hidden_dim, num_heads=4, batch_first=True,
        )
        self.ln = nn.LayerNorm(self.hidden_dim)

        # Output projection
        self.last_fc = nn.Linear(self.hidden_dim, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        # Handle both 2D (batch, feat) and 3D (task, seq, feat) input
        needs_reshape = (input.dim() == 2)
        if needs_reshape:
            input = input.unsqueeze(0)

        task, seq, feat = input.size()
        h = input.view(task * seq, feat)

        # Per-transition MLP embedding
        for fc in self.fcs:
            h = self.hidden_activation(fc(h))
        h = h.view(task, seq, -1)

        # Self-attention + residual + layer norm
        attn_out, _ = self.attn(h, h, h)
        h = self.ln(h + attn_out)

        # Per-transition output (compatible with product-of-Gaussians)
        preactivation = self.last_fc(h)
        output = preactivation

        if needs_reshape:
            output = output.squeeze(0)
            preactivation = preactivation.squeeze(0)

        if return_preactivations:
            return output, preactivation
        return output

    def reset(self, num_tasks=1):
        pass
```
