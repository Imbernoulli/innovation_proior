The problem is to build a subquadratic sequence mixer that can replace softmax attention in a decoder-only language model. Softmax attention is quadratic in sequence length and stores an unbounded key-value cache, so a family of linear-RNN alternatives keeps a fixed-size matrix state and reads it with a query. The simplest version, additive linear attention, writes every new key-value pair into that state and never forgets; once the sequence grows past the key dimension, old associations interfere with new ones and retrieval breaks. Two fixes have been explored separately. Gating multiplies the state by a data-dependent decay before each write, which lets stale context fade but cannot remove a specific association that a new key collides with. The delta rule writes by one gradient step on the retrieval error, so it overwrites exactly the colliding direction, but as usually stated it has no decay at all and therefore cannot do rapid global forgetting. The remaining question is whether a single layer can have both capabilities without sacrificing the stable, hardware-efficient chunkwise training form that makes these models practical at scale.

The method is Gated DeltaNet. It composes the two mechanisms on orthogonal axes of one recurrence: S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t v_t k_t^T, with output o_t = S_t q_t. Here alpha_t in (0,1] is a per-head scalar data-dependent decay that provides the global eraser, and beta_t = sigma(W_beta x_t) in (0,1) is the learned writing strength of the error-correcting delta rule, the local scalpel. Setting alpha_t = 1 for all t recovers DeltaNet; setting beta_t = 0 drops the delta write and leaves the pure scalar-gated decay skeleton that underlies Mamba2 and scalar-gated linear attention. Because it contains both parents as special cases, Gated DeltaNet can be at least as good as whichever is better for a given task, and it can solve tasks that genuinely need both global fading and targeted updates.

A scalar alpha_t is chosen deliberately rather than a per-channel diagonal gate. The delta rule already supplies fine-grained, content-addressed subspace control, so the gate only needs to do uniform global fading; keeping it scalar prevents the two mechanisms from overlapping and, crucially, lets the scalar telescope cleanly across time. That telescoping is what allows DeltaNet's chunkwise UT-transform algorithm to survive: the decay folds into a chunk-local cumulative sum of log-gates, and every heavy operation remains a matrix multiplication. The gate is parameterized Mamba2-style through a log-decay g_t = -exp(A_log) softplus(a_proj(x_t) + dt_bias) <= 0, so alpha_t = exp(g_t). The dt_bias is initialized so alpha_t starts near one, giving a long-memory prior and preventing the state from halving every step before training begins.

Stability follows from the factorization. With L2-normalized keys, the contractive factor along the key direction is alpha_t(1 - beta_t) and along every orthogonal direction is alpha_t; both lie in [0,1]. The layer keeps DeltaNet's stabilizers: SiLU plus L2 normalization on queries and keys, a learned beta_t per head, and causal depthwise short convolutions of kernel 4 on q/k/v to support local token comparisons. Because the data-dependent decay makes per-head outputs vary in scale, the output is routed through a gated RMSNorm followed by a swish output gate, restoring the nonlinearity that softmax normally provides. Training uses the chunkwise form below; inference uses the same recurrence in a constant-memory recurrent mode.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def chunk_gated_delta_rule(q, k, v, beta, g, chunk_size=64):
    """Chunkwise UT-transform form with scalar decay folded into chunk-local cumsum(g).
    q,k: (B,H,T,d_k); v: (B,H,T,d_v); beta,g: (B,H,T); g <= 0 is log-decay."""
    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    q = q * (d_k ** -0.5)
    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad:
        q, k, v_beta, k_beta = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v_beta, k_beta))
        g = F.pad(g, (0, pad))
    q, k, v_beta, k_beta, g = (rearrange(x, 'b h (n c) ... -> b h n c ...', c=chunk_size)
                               for x in (q, k, v_beta, k_beta, g))
    decay = g.cumsum(-1)
    dexp = decay.exp()[..., None]
    Lm = (decay[..., :, None] - decay[..., None, :]).tril().exp().tril()
    eye = torch.eye(chunk_size, dtype=q.dtype, device=q.device)
    mask0 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    attn = -((k_beta @ k.transpose(-1, -2)) * Lm).masked_fill(mask0, 0)
    for i in range(1, chunk_size):                        # forward substitution -> (I + L)^{-1}
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone()
                                               * attn[..., :, :i].clone()).sum(-2)
    attn = attn + eye
    u = attn @ v_beta                                     # pseudo-values
    w = attn @ (k_beta * dexp)                            # decay-weighted pseudo-keys
    S = q.new_zeros(B, H, d_k, d_v)
    o = torch.zeros_like(v_beta)
    mask1 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    for i in range(q.shape[2]):
        q_i, k_i = q[:, :, i], k[:, :, i]
        intra = (q_i @ k_i.transpose(-1, -2) * Lm[:, :, i]).masked_fill(mask1, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S                 # corrected write for carried state
        o_inter = (q_i * dexp[:, :, i]) @ S               # read carried state, query scaled by decay
        o[:, :, i] = o_inter + intra @ u_i
        d_last = decay[:, :, i, -1]
        S = S * d_last[..., None, None].exp() + \
            (k_i * (d_last[..., None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ u_i
    o = rearrange(o, 'b h n c d -> b h (n c) d')[:, :, :T]
    return o


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet layer: data-dependent scalar decay + delta-rule write."""

    def __init__(self, hidden_size, num_heads=6, head_dim=128, expand_v=2.0,
                 conv_size=4, norm_eps=1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * head_dim
        self.value_dim = num_heads * self.head_v_dim
        self.use_pos_emb = False

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)       # writing strength beta_t
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)       # gate input -> Delta_t
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)  # swish output gate

        A = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        dt = torch.exp(torch.rand(num_heads) * (torch.log(torch.tensor(0.1))
                       - torch.log(torch.tensor(1e-3))) + torch.log(torch.tensor(1e-3)))
        dt = torch.clamp(dt, min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.dt_bias._no_weight_decay = True

        self.q_conv = nn.Conv1d(self.key_dim, self.key_dim, conv_size,
                                groups=self.key_dim, padding=conv_size - 1, bias=False)
        self.k_conv = nn.Conv1d(self.key_dim, self.key_dim, conv_size,
                                groups=self.key_dim, padding=conv_size - 1, bias=False)
        self.v_conv = nn.Conv1d(self.value_dim, self.value_dim, conv_size,
                                groups=self.value_dim, padding=conv_size - 1, bias=False)
        self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.conv_size = conv_size

    def _short_conv(self, x, conv):
        T = x.shape[1]
        return F.silu(conv(x.transpose(1, 2))[..., :T].transpose(1, 2))

    def forward(self, x):
        q = self._short_conv(self.q_proj(x), self.q_conv)
        k = self._short_conv(self.k_proj(x), self.k_conv)
        v = self._short_conv(self.v_proj(x), self.v_conv)
        q, k = (rearrange(t, 'b s (h d) -> b h s d', d=self.head_k_dim) for t in (q, k))
        v = rearrange(v, 'b s (h d) -> b h s d', d=self.head_v_dim)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        beta = rearrange(self.b_proj(x).sigmoid(), 'b s h -> b h s')
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)
        g = rearrange(g, 'b s h -> b h s')
        o = chunk_gated_delta_rule(q, k, v, beta, g)
        o = self.o_norm(o)
        o = rearrange(o, 'b h s d -> b s (h d)')
        o = o * F.silu(self.g_proj(x))
        return self.o_proj(o)
```
