# Hyper-Connections

Hyper-Connections replace each single residual stream with `n` hidden streams and wrap each attention
or FFN residual site with a small connection matrix. The sublayer itself is unchanged: it still receives
one `d`-dimensional input and returns one `d`-dimensional output.

## Core Update

Initialize `H_0 = [h_0; ...; h_0] in R^{n x d}`. For one residual site with sublayer `T`,

```text
         | 0    B   |
HC   =   | A_m  A_r |       B in R^{1 x n}, A_m in R^{n x 1}, A_r in R^{n x n}

h_0^T = A_m^T H
H'    = A_r^T H
Hhat  = B^T T(h_0)^T + H'
```

`A_m` reads the streams into the sublayer input, `A_r` carries/mixes streams, and `B` writes the
sublayer output back into the streams. At the end of the model, sum the `n` streams, then apply the
usual final normalization and output head.

## Dynamic Form

DHC adds a normalized, tanh-bounded, small-scaled correction to the static connection:

```text
Hbar   = norm(H)
B(H)   = s_beta  * tanh(Hbar W_beta)^T + B        W_beta in R^d
A_m(H) = s_alpha * tanh(Hbar W_m)      + A_m      W_m in R^d
A_r(H) = s_alpha * tanh(Hbar W_r)      + A_r      W_r in R^{d x n}
```

In the Appendix J implementation, `W_m` and `W_r` are packed as `dynamic_alpha_fn` with shape
`(d, n + 1)`: the first output column corrects `A_m`, and the remaining `n` columns correct `A_r`.
`dynamic_beta_fn` has shape `(d,)`.

## Initialization

For residual site index `k`, initialize

```text
B^k   = 1_{1 x n}
A_m^k = e_{k mod n}
A_r^k = I_n
W_beta = W_m = W_r = 0
s_beta = s_alpha = 0.01
```

With this base, all streams start identical and remain identical at initialization. The row-sum before
the final normalization is `n` times the ordinary Pre-Norm hidden vector. The final normalization removes
that global scale in the mathematical normalization rule, up to epsilon-level numerical differences in
implementation, so the initialized model starts as the Pre-Norm baseline.

## Cost

Per hyper-connection module:

```text
SHC params = n(n + 2)
DHC params = |theta_norm| + d_model(n + 2) + n(n + 2) + 2
```

Each Transformer layer uses two modules, one for attention and one for FFN. The main extra compute is
the width connection, `O(d_model * n * (n + 1))` per token, which is small compared with attention
projection and FFN costs for the small expansion rates used in the main experiments. The main LLM
setting is DHC with `n = 4`.

## Reference Code

Appendix J gives the canonical PyTorch-like implementation for one residual site:

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

Transformer wiring:

```python
# Attention block
mix_h, beta = atten_hyper_connection.width_connection(h)
h = attn_norm(mix_h[..., 0, :])
h = self_attention(h)
h = atten_hyper_connection.depth_connection(mix_h, dropout(h), beta)

# FFN block
mix_h, beta = ffn_hyper_connection.width_connection(h)
h = ffn_norm(mix_h[..., 0, :])
h = ffn(h)
h = ffn_hyper_connection.depth_connection(mix_h, dropout(h), beta)
```

The stream tensor is reduced only after the residual-site loop: `h = H.sum(dim=-2)`, then final norm
and output head.
