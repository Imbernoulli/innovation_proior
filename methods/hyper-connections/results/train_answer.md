The residual stream in a deep Transformer has to do two things at once: give gradients a clean identity highway through many layers, and decide how strongly each new layer output should influence the hidden state. In Pre-Norm the highway is clean because `h = h + T(LN(h))` adds the branch with coefficient one, but the stream keeps accumulating unnormalized content, so deep branch contributions shrink relative to the growing sum and neighboring layers become too similar. Post-Norm rebalances the forward representation by normalizing after each addition, yet that puts normalization repeatedly on the gradient path and makes deep training unstable. A single scalar branch weight cannot preserve a long clean route and give recent outputs a strong local influence when those two demands point in opposite directions, so the Pre-Norm versus Post-Norm trade-off is structural rather than just an initialization issue.

Earlier fixes on this ladder tried to control the depth flow. A predefined warmup schedule for residual branch weights helps early conditioning but expires back into the unit-weight accumulator. Learnable per-layer scalars plus a direct embedding re-injection make the weights persistent and counter token dilution, yet they are still a rank-one scalar knob that mixes every layer the same way for every token. Attention over coarse depth blocks lets the model choose sources content-dependently, but it coarsens the depth axis and still keeps only one residual stream, so the fundamental seesaw between highway and strong local writes remains. What is missing is room for more than one depth pattern to exist at the same time.

The method I propose is Hyper-Connections. It replaces each single residual stream with `n` parallel hidden streams, represented as a hyper-hidden matrix `H` of shape `(B, T, n, D)`, and wraps every attention and feed-forward residual site with a small `(n+1) x (n+1)` connection matrix. The sublayer itself is untouched: it still receives one `D`-dimensional input and returns one `D`-dimensional output. The connection matrix is factored into a width connection that mixes the streams into the sublayer input and a depth connection that writes the sublayer output back into the streams. With `n > 1`, one stream can stay a clean identity highway while another carries a more strongly written local pattern, breaking the single-stream seesaw. The same matrix also spans sequential and parallel layer arrangements, so the network can learn a soft mixture of depth layouts.

The static form has matrices `B` in `R^{1 x n}`, `A_m` in `R^{n x 1}`, and `A_r` in `R^{n x n}`. For one site, `h_0 = A_m^T H` becomes the single sublayer input, `H' = A_r^T H` carries and mixes the streams forward, and `H_hat = B^T T(h_0)^T + H'` writes the branch output back into every stream. The dynamic extension, DHC, adds a small input-dependent correction: the streams are normalized, projected, passed through `tanh`, and scaled by a small learnable scalar. The static base is initialized so the model starts as ordinary Pre-Norm. Specifically, `B` is all ones, `A_r` is the identity, and `A_m` is a one-hot vector that cycles through the streams with the site index. The dynamic projection weights are initialized to zero, so the correction is exactly zero at step zero. Because all streams are identical copies at initialization and are updated identically under the static base, summing the streams reproduces the Pre-Norm behavior up to the scale invariance of the final normalization.

The extra cost is small. For expansion rate `n`, each static module has `n(n+2)` parameters, and each dynamic module adds `O(d n)` parameters plus two scalar scales. The main added computation is a width matmul of cost `O(d n^2)` per token, which is negligible next to attention and the feed-forward network for small `n`. In practice `n = 2` already breaks the seesaw on modest budgets, while `n = 4` is the standard setting for large language-model experiments. The connection parameters are gains rather than weight matrices, so they should be placed in a no-weight-decay optimizer group.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm


class HyperConnection(nn.Module):
    """One residual site. h: (B, T, n, D). Static base = Pre-Norm on n copies;
    dynamic projections zero-init -> exactly Pre-Norm at initialization."""

    def __init__(self, dim, rate, site_id, dynamic=True):
        super().__init__()
        self.rate = rate
        self.dynamic = dynamic

        # B = 1_{1 x n}: write the full sublayer output into every stream.
        self.static_beta = nn.Parameter(torch.ones(rate))

        # [A_m | A_r] = [e_{site_id mod n} | I_n]
        init_alpha0 = torch.zeros(rate, 1)
        init_alpha0[site_id % rate, 0] = 1.0
        self.static_alpha = nn.Parameter(
            torch.cat([init_alpha0, torch.eye(rate)], dim=1)
        )  # (n, n+1)

        if self.dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, rate + 1))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1) * 0.01)
            self.layer_norm = LayerNorm(dim, bias=False)

    def width_connection(self, h):
        if self.dynamic:
            norm_h = self.layer_norm(h)
            wc = torch.tanh(norm_h @ self.dynamic_alpha_fn) * self.dynamic_alpha_scale
            alpha = wc + self.static_alpha  # (B, T, n, n+1)
            dc = torch.tanh(norm_h @ self.dynamic_beta_fn) * self.dynamic_beta_scale
            beta = dc + self.static_beta    # (B, T, n)
        else:
            alpha = self.static_alpha[None, None, ...]
            beta = self.static_beta[None, None, ...]
        mix_h = alpha.transpose(-1, -2) @ h  # (B, T, n+1, D)
        return mix_h, beta

    def depth_connection(self, mix_h, h_o, beta):
        # h_o: (B, T, D); distribute it over the n streams, then add carried streams.
        return torch.einsum('btd,btn->btnd', h_o, beta) + mix_h[..., 1:, :]


class GPTWithHyperConnections(nn.Module):
    """Example wiring inside a GPT-like model. CausalSelfAttention, MLP, LayerNorm,
    and block definitions are kept unchanged; HyperConnection drives them from the
    forward loop over an n-stream tensor H of shape (B, T, n, D)."""

    def __init__(self, config):
        super().__init__()
        # ... standard embeddings, transformer blocks, final ln, and head ...
        self.config = config
        self.hc_rate = 2  # expansion rate n; n=2 already breaks the seesaw.
        self.hc = nn.ModuleList([
            HyperConnection(config.n_embd, self.hc_rate, site_id=k, dynamic=True)
            for k in range(2 * config.n_layer)  # attention and MLP are two sites per layer
        ])

    def forward(self, idx):
        # ... token + position embeddings ...
        x = self.transformer.wte(idx) + self.transformer.wpe(
            torch.arange(idx.size(1), device=idx.device)
        )
        x = self.transformer.drop(x)

        # Lift the single stream into n identical copies.
        H = x.unsqueeze(-2).expand(-1, -1, self.hc_rate, -1).contiguous()

        s = 0
        for block in self.transformer.h:
            # Attention residual site.
            mix_h, beta = self.hc[s].width_connection(H)
            h0 = block.attn(block.ln_1(mix_h[..., 0, :]))
            H = self.hc[s].depth_connection(mix_h, h0, beta)
            s += 1

            # MLP residual site.
            mix_h, beta = self.hc[s].width_connection(H)
            h0 = block.mlp(block.ln_2(mix_h[..., 0, :]))
            H = self.hc[s].depth_connection(mix_h, h0, beta)
            s += 1

        # Reduce the n streams back to one vector for final normalization and head.
        x = H.sum(dim=-2)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        hc_ids = {id(p) for p in self.hc.parameters()}
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and id(p) not in hc_ids]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and id(p) not in hc_ids]
        hc_params = [p for n, p in param_dict.items() if id(p) in hc_ids]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
            {'params': hc_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
```
