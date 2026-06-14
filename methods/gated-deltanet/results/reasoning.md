Let me start from the two things that already work and the gap between them, because the gap is the
whole opportunity. I want a subquadratic sequence mixer with a fixed-size matrix state `S`, trained by a
matmul-rich chunkwise algorithm, that matches softmax attention on language modeling and especially on
recall. I have two mechanisms in hand that each fix one disease of plain additive linear attention
(`S_t = S_{t-1} + k_t^T v_t`, which never forgets and whose Hebbian write cross-talks once the sequence
outruns the key dimension), and I want to see whether they are rivals or partners.

The first is *gating*. Multiply the state by a data-dependent decay before the additive write,
`S_t = Diag(alpha_t) S_{t-1} + k_t^T v_t` with `alpha_t in (0,1)` from `x_t`. This gives the model a
content-chosen forgetting rate — it can let stale context fade — and, crucially, a scalar or diagonal
decay keeps the chunkwise matmul training form alive because the per-step gates telescope into a
cumulative product that preconditions Q and K. What it cannot do is targeted removal. The decay is
elementwise: it scales the whole state (per channel) uniformly, so when a new key `k_t` arrives that
collides with an association I already stored, the gate has no way to say "erase *that* association and
leave the rest." Read the additive store back with a probe key, `S k_j = v_j (k_j^T k_j) + sum_{i != j}
(k_i^T k_j) v_i`: the second term is interference from every non-orthogonal key, and a global decay
shrinks the signal and the interference together. Gating controls *how fast* memory fades, not *what*
fades.

The second is the *delta rule*. Treat `S` as a regressor mapping `k_t` to `v_t` and write by one
gradient step on the squared error `1/2 ||S k_t - v_t||^2`, giving
`S_t = S_{t-1} - beta_t (S_{t-1} k_t - v_t) k_t^T = S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T`,
with a learned writing strength `beta_t = sigma(W_beta x_t) in (0,1)`. The write is proportional to the
*error*: retrieve the old value `v_t^old = S_{t-1} k_t`, blend `v_t^new = beta_t v_t + (1-beta_t)
v_t^old`, swap it in. This is content-addressed deallocation — it removes exactly the stale association
for the colliding key. With L2-normalized keys the transition `I - beta_t k_t k_t^T` has its only
non-unit eigenvalue at `1 - beta_t`, and at `beta_t = 1` it is an orthogonal projection that annihilates
precisely the `k_t` direction and leaves the other `d-1` dimensions untouched. Surgical. And it is
trainable at scale because the state stays additive in pseudo-values `u_t = beta_t(v_t - sum_{i<t} u_i
(k_i^T k_t))`, the Householder products have a WY representation, and a chunk's writes come out of a
single triangular (UT-transform) inverse — dense matmuls, `O(L/C)` sequential steps. But stare at the
recurrence: there is *no decay term*. The transition is `I - beta_t k_t k_t^T`, identity minus a
rank-one, with eigenvalue 1 on the entire orthogonal complement of `k_t`. An association written into
some direction, if no later key directly overwrites that direction, persists *forever*. The delta rule
controls *what* is written and removed, precisely, but it cannot let memory globally fade.

So the two mechanisms are not rivals — they are doing different jobs on different axes. Gating is a
multiplicative pre-factor on `S_{t-1}`; the delta rule is the additive transition structure of the
write. There is no reason I cannot have both, and the failure each one has is exactly the capability the
other supplies: gating gives rapid global erasure, the delta rule gives targeted local update. Let me
just write down the combined recurrence and then *check* that it is well-posed rather than assume it. I
want the previous state first decayed by a data-dependent scalar `alpha_t in (0,1]`, then have the
delta-rule erase-and-write applied:
`S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t v_t k_t^T`, read out `o_t = S_t q_t`.

Before anything else, verify it is a genuine generalization of both parents, because that is what makes
it a safe bet — it should reduce to each in a limit, so it cannot be strictly worse than the better one
unless the extra freedom is mis-trained. Set `alpha_t = 1` for all `t`: the pre-factor disappears and I
recover `S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t v_t k_t^T`, exactly the delta rule. Set
`beta_t = 0`: the delta write vanishes and the Householder collapses to the identity, leaving
`S_t = alpha_t S_{t-1}`, the pure scalar-gated *decay skeleton* that the Mamba2 / scalar-gated-linear-
attention family is built on (that family adds its own additive write back on top; the `beta_t = 0`
limit isolates only the gating mechanism). So the combined rule sits above both as special cases; it is strictly more general, and it can
be better wherever a task needs *both* a global eraser and a local scalpel.

Now the design choices, each of which I have to justify, because a combined rule opens new failure
modes. First, what shape is `alpha_t`? The gating literature offers per-channel diagonal decay; I will
use a single *scalar per head* instead. Two reasons. The delta rule already provides all the
fine-grained, per-direction control I could want — the Householder picks out the exact subspace to erase
— so I do not need the gate to *also* be fine-grained; I need it to do the one thing the delta rule
cannot, uniform global fading, and a scalar does exactly that and nothing more, keeping the two
mechanisms on non-overlapping jobs. And, decisively for training, a scalar pulls cleanly out of the
chunkwise cumulative product — the same property that made fixed-scalar decay parallelizable — so the
combined rule keeps the delta rule's UT-transform chunk algorithm intact, with the decay folded in as a
chunk-local cumulative sum of log-gates. A diagonal gate combined with the Householder would break the
clean telescoping that each parent relied on; the scalar is what lets the two compose without losing the
matmul form.

Second, how to parameterize `alpha_t` so it trains? A naive sigmoid gate sits near 0.5 at init — the
state halves every step, long memory dead before training starts. I want a near-1 prior. Borrow the
Mamba2 discretization, which is built for exactly this: a positive timescale `Delta_t = softplus(W_a x_t
+ dt_bias)` with `dt_bias` initialized so `Delta_t` starts small, and a per-head positive rate
`A = exp(A_log)`; the log-decay is `g_t = -A * Delta_t <= 0`, so `alpha_t = exp(g_t) in (0,1]`, near 1
at initialization (slow forgetting), data-dependent through `W_a x_t`. This is the well-conditioned,
small-magnitude log-decay the chunkwise stable form wants. (I keep `A_log` and `dt_bias` out of weight
decay; they set timescales, not feature weights.)

Third, stability — derive it, do not assume. The combined transition along the key direction scales by
`alpha_t (1 - beta_t)`, and along every orthogonal direction it scales by `alpha_t`. With L2-normalized
keys (so `1 - beta_t in [0,1]` for `beta_t in (0,1)`) and `alpha_t in (0,1]`, both factors are in
`[0,1]` — inside the unit disk — so the recurrence is stable by the DeltaNet argument plus a
strictly-contractive scalar. I keep DeltaNet's stabilizers: SiLU then L2-normalization on q and k (SiLU
keeps sign and is smooth, L2 makes the projection exact), a learned `beta_t = sigma(W_beta x_t)` per
head, and a lightweight depthwise short convolution (kernel 4) on the q/k/v projections, which
generalizes the shift operator and lets the layer do precise local token comparisons that pure
content-addressing is bad at. An output normalization per head before the projection completes it; and
because re-introducing a data-dependent decay gives the per-head outputs head-varying, content-dependent
scale, I route the output through a *gated* RMSNorm with a swish output gate — the standard linear-
attention output recipe that closes the gap, the same nonlinearity-restoration deleting softmax forces.

Now make the chunkwise training algorithm concrete, because the whole claim is that the scalar decay
folds into DeltaNet's chunk form without breaking it. Within a chunk, let `g` be the per-position
log-decay and `decay = cumsum(g)` the chunk-local accumulated log-decay. The delta-rule pseudo-value
solve is the same triangular UT inverse as DeltaNet, but the key-key similarities that build it must be
weighted by the relative decay between positions: the strictly-lower-triangular matrix uses
`(k_beta @ k^T) * exp(decay_i - decay_j)` for `i > j`, so the closed-form `T = (I + L)^{-1}` absorbs the
gate. The carried chunk state is decayed across the chunk by `exp(decay_last)` and the chunk's keys are
weighted by `exp(decay_last - decay)` before they fold into the next state; the inter-chunk read scales
the query by `exp(decay)`. Every cumulative product spans at most one chunk, so it stays bounded, and
every heavy operation is a matmul — exactly the DeltaNet hardware profile with the scalar gate threaded
through the cumsum. Recompute the chunk states in the backward to save memory.

Let me write the reference recurrence first — it is the cleanest statement of the model and exactly what
the chunk kernel parallelizes — then the chunk algorithm that does it in matmuls, then the layer module
that wires up the Mamba2-style gate, the learned writing strength, the short conv, and the gated output
norm.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def naive_recurrent_gated_delta_rule(q, k, v, beta, g, scale=None):
    """Reference recurrence the chunk kernel parallelizes.
    q, k: (B, H, T, d_k); v: (B, H, T, d_v); beta, g: (B, H, T).
    g = log of the per-head scalar decay (<= 0), so alpha_t = exp(g_t) in (0,1].
    Implements  S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t v_t k_t^T,  o_t = S_t q_t."""
    B, H, T, d_k = q.shape
    d_v = v.shape[-1]
    scale = d_k ** -0.5 if scale is None else scale
    q = q * scale
    S = q.new_zeros(B, H, d_k, d_v)                      # matrix-valued state, the memory
    o = torch.zeros_like(v)
    for t in range(T):
        k_t, v_t = k[:, :, t], v[:, :, t]
        S = S * g[:, :, t].exp()[..., None, None]        # alpha_t S_{t-1}  (global scalar decay)
        v_old = (S * k_t[..., None]).sum(-2)             # S_{t-1}^(decayed) k_t  = old value
        u_t = beta_t = beta[:, :, t]                     # writing strength (alias for readability)
        delta = (v_t - v_old) * beta_t[..., None]        # beta_t (v_t - v_old) : error-correcting write
        S = S + k_t[..., None] * delta[..., None, :]     # + k_t (delta)^T   (rank-one delta update)
        o[:, :, t] = (q[:, :, t][..., None] * S).sum(-2) # o_t = S_t q_t
    return o


def chunk_gated_delta_rule(q, k, v, beta, g, chunk_size=64):
    """Chunkwise (matmul-rich) form: DeltaNet's UT-transform with the scalar decay folded into the
    chunk-local cumulative log-decay. q,k: (B,H,T,d_k); v: (B,H,T,d_v); beta,g: (B,H,T)."""
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
    decay = g.cumsum(-1)                                  # chunk-local accumulated log-decay
    dexp = decay.exp()[..., None]
    # decay-weighted, strictly-lower key-key similarities -> the UT triangular system
    Lm = (decay[..., :, None] - decay[..., None, :]).tril().exp().tril()
    eye = torch.eye(chunk_size, dtype=q.dtype, device=q.device)
    mask0 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 0)
    attn = -((k_beta @ k.transpose(-1, -2)) * Lm).masked_fill(mask0, 0)
    for i in range(1, chunk_size):                        # forward substitution -> (I + L)^{-1}
        attn[..., i, :i] = attn[..., i, :i] + (attn[..., i, :, None].clone()
                                               * attn[..., :, :i].clone()).sum(-2)
    attn = attn + eye                                    # this is T = (I + L)^{-1}
    u = attn @ v_beta                                   # pseudo-values
    w = attn @ (k_beta * dexp)                          # decay-weighted pseudo-keys
    S = q.new_zeros(B, H, d_k, d_v)
    o = torch.zeros_like(v_beta)
    mask1 = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), 1)
    n_chunks = q.shape[2]
    for i in range(n_chunks):
        q_i, k_i = q[:, :, i], k[:, :, i]
        intra = (q_i @ k_i.transpose(-1, -2) * Lm[:, :, i]).masked_fill(mask1, 0)
        u_i = u[:, :, i] - w[:, :, i] @ S               # effective write, corrected for carried state
        o_inter = (q_i * dexp[:, :, i]) @ S             # read carried state, query scaled by exp(decay)
        o[:, :, i] = o_inter + intra @ u_i
        d_last = decay[:, :, i, -1]
        S = S * d_last[..., None, None].exp() + \
            (k_i * (d_last[..., None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ u_i
    o = rearrange(o, 'b h n c d -> b h (n c) d')[:, :, :T]
    return o


class GatedDeltaNet(nn.Module):
    """Gated delta rule: data-dependent scalar decay (Mamba2-style) composed with the delta-rule
    error-correcting write; trained with the chunkwise UT-transform form."""

    def __init__(self, hidden_size, num_heads=6, head_dim=128, expand_v=2.0,
                 conv_size=4, norm_eps=1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = num_heads * head_dim
        self.value_dim = num_heads * self.head_v_dim
        self.use_pos_emb = False                         # decay + recurrence encode relative position

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)    # writing strength beta_t
        self.a_proj = nn.Linear(hidden_size, num_heads, bias=False)    # gate input  -> Delta_t
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)  # swish output gate
        # Mamba2 decay parameters: alpha_t = exp(-exp(A_log) * softplus(a_proj(x) + dt_bias)) in (0,1]
        A = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A));            self.A_log._no_weight_decay = True
        dt = torch.exp(torch.rand(num_heads) * (torch.log(torch.tensor(0.1)) -
                                                torch.log(torch.tensor(1e-3))) + torch.log(torch.tensor(1e-3)))
        dt = torch.clamp(dt, min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)));  self.dt_bias._no_weight_decay = True
        # depthwise short convolutions (SiLU) on q/k/v projections
        self.q_conv = nn.Conv1d(self.key_dim, self.key_dim, conv_size, groups=self.key_dim, padding=conv_size - 1, bias=False)
        self.k_conv = nn.Conv1d(self.key_dim, self.key_dim, conv_size, groups=self.key_dim, padding=conv_size - 1, bias=False)
        self.v_conv = nn.Conv1d(self.value_dim, self.value_dim, conv_size, groups=self.value_dim, padding=conv_size - 1, bias=False)
        self.o_norm = nn.RMSNorm(self.head_v_dim, eps=norm_eps)        # per-head output norm (gated below)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.conv_size = conv_size

    def _short_conv(self, x, conv):                      # x: (B, T, D) -> SiLU(depthwise conv), causal
        T = x.shape[1]
        y = conv(x.transpose(1, 2))[..., :T].transpose(1, 2)
        return F.silu(y)

    def forward(self, x):                                # x: (B, T, d)
        q = self._short_conv(self.q_proj(x), self.q_conv)
        k = self._short_conv(self.k_proj(x), self.k_conv)
        v = self._short_conv(self.v_proj(x), self.v_conv)
        q, k = (rearrange(t, 'b s (h d) -> b h s d', d=self.head_k_dim) for t in (q, k))
        v = rearrange(v, 'b s (h d) -> b h s d', d=self.head_v_dim)
        q = F.normalize(q, dim=-1, p=2)                  # SiLU already applied; L2-normalize q, k
        k = F.normalize(k, dim=-1, p=2)
        beta = self.b_proj(x).sigmoid()                  # (B, T, H) writing strength
        beta = rearrange(beta, 'b s h -> b h s')
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)  # (B, T, H) log-decay <= 0
        g = rearrange(g, 'b s h -> b h s')

        # training: the chunk kernel; for clarity the recurrence below computes the same outputs.
        o = chunk_gated_delta_rule(q, k, v, beta, g)     # (B, H, T, d_v)

        o = self.o_norm(o)
        o = rearrange(o, 'b h s d -> b s (h d)')
        o = o * F.silu(self.g_proj(x))                   # swish output gate
        return self.o_proj(o)
```

So the chain, end to end. Plain additive linear attention never forgets and cross-talks; the two cures
each fix one axis — gating adds a data-dependent decay that fades memory globally but cannot localize
removal, and the delta rule adds an error-correcting, content-addressed write that removes exactly the
colliding association but has no global decay. They are complementary, not rival, so I compose them into
one recurrence, `S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t v_t k_t^T`, which reduces to the
delta rule at `alpha_t = 1` and to the pure scalar-gated decay skeleton (the gating part of the Mamba2 /
gated-linear-attention family) at `beta_t = 0`, so it is a strict generalization of both. The decay is a *scalar per head* — the smallest gate that does only global
fading, which keeps the two mechanisms non-overlapping and, being scalar, telescopes cleanly so the
delta rule's UT-transform chunk algorithm survives with the decay folded into a chunk-local
cumulative-sum; it is parameterized Mamba2-style (`alpha_t = exp(-exp(A_log) softplus(a_proj(x) +
dt_bias))`) for a near-1 long-memory prior. L2-normalized keys make the combined contractive factor
`alpha_t(1 - beta_t)` and the orthogonal factor `alpha_t` both land in `[0,1]`, so the recurrence is
stable; SiLU + short conv, a learned `beta_t`, a per-head gated RMSNorm output and a swish gate complete
a layer that drops into a pre-norm transformer block with no absolute position embeddings.
