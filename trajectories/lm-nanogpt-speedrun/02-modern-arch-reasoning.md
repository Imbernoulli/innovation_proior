Muon got the body of the network learning fast — 22.3 minutes, 6200 steps. With the optimizer now
condition-blind and nearly free per step, the next thing pinning the step count is the *architecture* it's
optimizing. The body is still a faithful GPT-2 block: GELU MLPs, the original normalization placement,
a fused QKV projection, and the conservative defaults Karpathy inherited. Each of those was the right
choice in 2019; by now several have better-behaved replacements that should let the same Muon+AdamW recipe
reach 3.28 in fewer steps. Let me go through the block piece by piece and ask, for each, what is costing me
optimization speed.

Start with the MLP nonlinearity. GELU is smooth and fine, but it saturates softly and its gradient is
small over a wide region. There's a cheaper, sharper alternative that's been reported to train language
models slightly better: squared ReLU, x ↦ ReLU(x)². It's exactly zero for negative inputs (hard
sparsity, like ReLU) but grows quadratically for positive inputs, so the active units have a larger,
position-dependent gradient — the activation is more selective and the gradient signal through the MLP is
stronger where it fires. It costs nothing extra (a relu and a square). Worth swapping GELU → ReLU².

Now the attention QK path. Rotary already replaced the learned absolute positions, good. But there's a
stability issue Muon makes more visible: the queries and keys can grow in norm during training, and since
the attention logits are qᵀk, a few large-norm query/key dimensions can blow the logits up and make the
softmax peaky and the gradients spiky. The fix is to normalize the per-head query and key vectors before
the dot product — RMS-norm q and k along the head dimension. Then the attention score is essentially a
cosine-like similarity at a controlled scale, the logits can't run away, and training is steadier. "QK
norm." Apply rotary first, then norm q and k.

While I'm in the attention, reconsider the fused QKV. The baseline packs Q, K, V into one 3·n_embd-wide
`c_attn` matrix. That was a tiny matmul-efficiency convenience, but it forces Q, K, V to share a single
weight matrix that the optimizer treats as one object — and with Muon orthogonalizing matrices, a stacked
[3n, n] block is awkward (it's why the Muon step has that special "split grouped QKV" branch). If instead
I give Q, K, V their own `c_q`, `c_k`, `c_v` matrices, each is a clean n×n matrix that Muon orthogonalizes
directly, and the three roles get independent updates. So split the fused projection into three separate
linear layers.

Heads. The baseline uses 12 heads of dim 64 at width 768. Head dim 64 is small; modern practice and some
of the rotary/attention-kernel work prefer wider heads. Going to head dim 128 means 6 heads at width 768.
Wider heads give each attention head a richer subspace to compute similarity in, and 128 is a friendly
size for the attention kernels. So n_head 12 → 6 (head_dim 64 → 128).

Now normalization, and this is where I want to be careful. GPT-2 uses LayerNorm with a learned gain and
bias. But the gain/bias add parameters and a per-feature affine that, empirically, the network often
doesn't need once the block has skip connections — what's actually doing the work is the *normalization*
(rescaling to unit RMS), not the learned affine. RMSNorm without any learnable parameters — just
x / RMS(x) — is cheaper, has no extra parameters for the optimizer to chase, and works as well. The
baseline already had a hand-rolled `rmsnorm`; I'll standardize on the parameter-free `F.rms_norm(x,
(x.size(-1),))` everywhere normalization appears (pre-attention, pre-MLP, on q/k, and the final norm
before the head).

Initialization of the projections. This one interacts with the skip connections. The block is
x ← x + attn(norm(x)) and x ← x + mlp(norm(x)) — residual. At initialization, what should each sublayer
contribute? If I *zero-initialize the output projection* of both the attention (`c_proj`) and the MLP
(`c_proj`), then at step zero every block is exactly the identity: attn and mlp output zero, the residual
passes the input straight through. The network starts as a clean identity stack and the blocks have to
*earn* their contribution from zero. This is a muP-flavored idea — it makes the early dynamics far gentler
(no random sublayer outputs fighting the residual stream) and removes the need for the depth-dependent
`1/√(2·n_layer)` attention-scale fudge the baseline used to keep the residual variance under control.
With zero-init output projections, I can delete that `attn_scale` entirely; the residual stream variance
is controlled because the blocks contribute nothing until they learn to.

Last, a pure-efficiency detail that's free loss-wise but real on wallclock: the GPT-2 vocab is 50257
tokens, an ugly number for the head matmul and the embedding. Padding the vocabulary up to the nearest
multiple of 128 — 50304 — makes the large head/embedding matmuls land on tensor-core-friendly tile sizes.
The extra 47 rows are never targets, so they cost nothing in loss; they just make the matmul faster.

None of these is a new idea on its own — they're the accumulated "modern transformer" deltas (rotary was
already in; now ReLU², QK-norm, parameter-free RMSNorm, separated QKV, head dim 128, zero-init residual
projections, padded vocab). The bet is that together they make the *same* Muon+AdamW optimization reach
3.28 in noticeably fewer steps: ReLU² and wider heads give a better-conditioned function to fit, QK-norm
and zero-init residuals make the early optimization stable enough to push the learning rate, and the
padded vocab shaves the per-step matmul time. The risk is that stacking this many changes at once makes it
hard to attribute, and that some of them (zero-init especially) need the learning rate re-tuned or they
stall — so I'll re-tune the schedule (the warmdown and total iteration count) along with this.

Here is the modernized block and config — the deltas against the prior Muon script.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)   # split QKV
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))  # QK norm
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # zero init

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))   # no attn_scale: zero-init projections handle it
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

@dataclass
class GPTConfig:
    vocab_size : int = 50304   # 50257 padded up to a multiple of 128
    n_layer : int = 12
    n_head : int = 6           # head dim 128
    n_embd : int = 768
```

The chain: GELU → ReLU² and head dim 64 → 128 give a function that fits faster; QK-norm and parameter-free
RMSNorm steady the attention logits and remove unused affine params; zero-initializing the residual output
projections starts the net as an identity stack and lets the `1/√(2·n_layer)` attn-scale be dropped, which
makes the early dynamics gentle enough to keep the learning rate high; splitting QKV gives Muon three clean
n×n matrices; padding the vocab to 50304 speeds the head matmul. Together these should cut the step count
from 6200 toward ~5100 at the same 3.28 bar.
