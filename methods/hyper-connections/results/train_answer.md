The residual stream in a deep Transformer has to do two things at once: give gradients a clean identity highway through many layers, and decide how strongly each new layer output should influence the hidden state. In Pre-Norm the highway is clean because `h = h + T(LN(h))` adds the branch with coefficient one, but the stream keeps accumulating unnormalized content, so deep branch contributions shrink relative to the growing sum and neighboring layers become too similar. Post-Norm rebalances the forward representation by normalizing after each addition, yet that puts normalization repeatedly on the gradient path and makes deep training unstable. A single scalar branch weight cannot preserve a long clean route and give recent outputs a strong local influence when those two demands point in opposite directions, so the Pre-Norm versus Post-Norm trade-off is structural rather than just an initialization issue.

Earlier fixes on this ladder tried to control the depth flow. A predefined warmup schedule for residual branch weights helps early conditioning but expires back into the unit-weight accumulator. Learnable per-layer scalars plus a direct embedding re-injection make the weights persistent and counter token dilution, yet they are still a rank-one scalar knob that mixes every layer the same way for every token. Attention over coarse depth blocks lets the model choose sources content-dependently, but it coarsens the depth axis and still keeps only one residual stream, so the fundamental seesaw between highway and strong local writes remains. What is missing is room for more than one depth pattern to exist at the same time.

The method I propose is Hyper-Connections. It replaces each single residual stream with `n` parallel hidden streams, represented as a hyper-hidden matrix `H` of shape `(B, T, n, D)`, and wraps every attention and feed-forward residual site with a small `(n+1) x (n+1)` connection matrix. The sublayer itself is untouched: it still receives one `D`-dimensional input and returns one `D`-dimensional output. The connection matrix is factored into a width connection that mixes the streams into the sublayer input and a depth connection that writes the sublayer output back into the streams. With `n > 1`, one stream can stay a clean identity highway while another carries a more strongly written local pattern, breaking the single-stream seesaw. The same matrix also spans sequential and parallel layer arrangements, so the network can learn a soft mixture of depth layouts.

The static form has matrices `B` in `R^{1 x n}`, `A_m` in `R^{n x 1}`, and `A_r` in `R^{n x n}`. For one site, `h_0 = A_m^T H` becomes the single sublayer input, `H' = A_r^T H` carries and mixes the streams forward, and `H_hat = B^T T(h_0)^T + H'` writes the branch output back into every stream. The dynamic extension, DHC, adds a small input-dependent correction: the streams are normalized, projected, passed through `tanh`, and scaled by a small learnable scalar. The static base is initialized so the model starts as ordinary Pre-Norm. Specifically, `B` is all ones, `A_r` is the identity, and `A_m` is a one-hot vector that cycles through the streams with the site index. The dynamic projection weights are initialized to zero, so the correction is exactly zero at step zero. Because all streams are identical copies at initialization and are updated identically under the static base, summing the streams reproduces the Pre-Norm behavior up to the scale invariance of the final normalization.

The extra cost is small. For expansion rate `n`, each static module has `n(n+2)` parameters, and each dynamic module adds `O(d n)` parameters plus two scalar scales. The main added computation is a width matmul of cost `O(d n^2)` per token, which is negligible next to attention and the feed-forward network for small `n`. In practice `n = 2` already breaks the seesaw on modest budgets, while `n = 4` is the standard setting for large language-model experiments. The connection parameters are gains rather than weight matrices, so they should be placed in a no-weight-decay optimizer group.

The module below implements this exactly. `dim` is the model width `d`, `rate` is the expansion `n`, and `layer_id` is the residual-site index `k` that fixes which one-hot stream `A_m` reads from at initialization; passing `dynamic=False` collapses the module to the pure static base. `width_connection` normalizes `h`, forms the dynamic corrections through `dynamic_alpha_fn`/`dynamic_beta_fn` and a bounded `tanh`, adds them onto the static `alpha`/`beta`, and returns the mixed tensor `mix_h` together with the write weights `beta`; the sublayer then runs on `mix_h[..., 0, :]` exactly as it would on an ordinary Pre-Norm input. `depth_connection` distributes the sublayer's output across the streams with `beta` and adds it to the carried streams `mix_h[..., 1:, :]`, producing the updated `H` for the next residual site. Wiring one of these modules around attention or the feed-forward network is three lines each — call `width_connection` for the branch input, run the sublayer and its own normalization on that single vector, then call `depth_connection` on the sublayer's output — and every Transformer layer holds two such modules, one for attention and one for the feed-forward network. Only after the last residual site are the `n` streams reduced, `H.sum(dim=-2)`, and only then does the ordinary final normalization and output head run:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm


# h: hyper hidden matrix (BxLxNxD)
class HyperConnection(nn.Module):
    def __init__(self, dim, rate, layer_id, dynamic, device=None):
        super(HyperConnection, self).__init__()

        self.rate = rate
        self.layer_id = layer_id
        self.dynamic = dynamic

        self.static_beta = nn.Parameter(torch.ones((rate,), device=device))

        init_alpha0 = torch.zeros((rate, 1), device=device)
        init_alpha0[layer_id % rate, 0] = 1.0
        self.static_alpha = nn.Parameter(
            torch.cat([init_alpha0, torch.eye((rate), device=device)], dim=1)
        )

        if self.dynamic:
            self.dynamic_alpha_fn = nn.Parameter(torch.zeros((dim, rate + 1), device=device))
            self.dynamic_alpha_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
            self.dynamic_beta_fn = nn.Parameter(torch.zeros((dim,), device=device))
            self.dynamic_beta_scale = nn.Parameter(torch.ones(1, device=device) * 0.01)
            self.layer_norm = LayerNorm(dim)

    def width_connection(self, h):
        if self.dynamic:
            norm_h = self.layer_norm(h)

        if self.dynamic:
            wc_weight = norm_h @ self.dynamic_alpha_fn
            wc_weight = F.tanh(wc_weight)
            dynamic_alpha = wc_weight * self.dynamic_alpha_scale
            alpha = dynamic_alpha + self.static_alpha[None, None, ...]
        else:
            alpha = self.static_alpha[None, None, ...]

        if self.dynamic:
            dc_weight = norm_h @ self.dynamic_beta_fn
            dc_weight = F.tanh(dc_weight)
            dynamic_beta = dc_weight * self.dynamic_beta_scale
            beta = dynamic_beta + self.static_beta[None, None, ...]
        else:
            beta = self.static_beta[None, None, ...]

        mix_h = alpha.transpose(-1, -2) @ h
        return mix_h, beta

    def depth_connection(self, mix_h, h_o, beta):
        h = torch.einsum("blh,bln->blnh", h_o, beta) + mix_h[..., 1:, :]
        return h
```
