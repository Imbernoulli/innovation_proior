Muon got the body of the network learning fast — the orthogonalized-momentum update reaches the 3.28 bar in 22.3 minutes over 6200 steps, down from the ~31-minute AdamW baseline. With the optimizer now condition-blind and nearly free per step, the thing pinning the step count is the *architecture* Muon is optimizing. The body is still a faithful 2019 GPT-2 block — GELU MLPs, a fused QKV projection, head dim 64, LayerNorm-style normalization with learned affine, and a depth-dependent attention-scale fudge — each the conservative choice Karpathy inherited, and each with a better-behaved 2024 replacement that should let the same Muon+AdamW recipe reach the bar in fewer steps. None of the deltas is a new idea on its own; the bet is that the accumulated "modern transformer" changes, applied together, give Muon a function it can fit faster and a more stable early phase that tolerates a higher learning rate.

I propose a **modernized block** that bundles seven changes. The MLP nonlinearity goes from GELU to **ReLU²**, $x \mapsto \mathrm{ReLU}(x)^2$. GELU is smooth but saturates softly with a small gradient over a wide region; ReLU² is exactly zero for negative inputs (hard sparsity, like ReLU) but grows quadratically for positive ones, so active units carry a larger, position-dependent gradient — the activation is more selective and the gradient through the MLP is stronger where it fires — and it costs nothing extra, a relu and a square. In the attention QK path I add **QK-norm**: queries and keys can grow in norm during training, and since the logits are $q^\top k$ a few large-norm dimensions can blow the logits up and make the softmax peaky and the gradients spiky, which Muon makes more visible. RMS-normalizing per-head $q$ and $k$ along the head dimension (applied *after* rotary) turns the score into a cosine-like similarity at a controlled scale, so the logits cannot run away. I also **split the fused QKV** into separate `c_q`, `c_k`, `c_v` linears: the packed $[3n, n]$ block forces $Q$, $K$, $V$ to share one weight matrix and is exactly what made Muon need its special "split grouped QKV" branch; three clean $n\times n$ matrices give Muon three objects to orthogonalize directly and the three roles independent updates. I widen the heads from dim 64 to **head dim 128** (n_head 12 → 6), giving each head a richer subspace to compute similarity in and a kernel-friendly size.

Two of the changes are about stability rather than capacity. I standardize on **parameter-free RMSNorm** — `F.rms_norm(x, (x.size(-1),))` — everywhere normalization appears (pre-attention, pre-MLP, on $q$/$k$, and the final norm). What actually does the work in a residual block is the rescale to unit RMS, not the learned gain/bias; dropping the affine removes parameters the optimizer would otherwise chase and works as well. The subtle one is initialization. The block is residual, $x \leftarrow x + \mathrm{attn}(\mathrm{norm}(x))$ and $x \leftarrow x + \mathrm{mlp}(\mathrm{norm}(x))$, and if I **zero-initialize the output projection** (`c_proj`) of *both* the attention and the MLP, then at step zero every block is exactly the identity: the sublayers output zero and the residual passes the input straight through. The network starts as a clean identity stack and the blocks have to *earn* their contribution from zero — a muP-flavored idea that makes the early dynamics far gentler, with no random sublayer outputs fighting the residual stream. Crucially this lets me delete the depth-dependent $1/\sqrt{2\,n_{\text{layer}}}$ attention-scale fudge the baseline used to keep the residual variance bounded; with zero-init projections the residual variance is controlled because the blocks contribute nothing until they learn to. The last change is pure efficiency: the GPT-2 vocab of 50257 is an ugly number for the head and embedding matmuls, so I **pad the vocab to 50304**, the nearest multiple of 128, landing those large matmuls on tensor-core-friendly tile sizes. The extra 47 rows are never targets, so they cost nothing in loss and only speed the matmul. Together, ReLU² and wider heads give a better-conditioned function to fit, QK-norm and zero-init residuals make the early optimization stable enough to push the learning rate, and the padded vocab shaves per-step matmul time — so the same recipe reaches 3.28 in noticeably fewer steps. Because several of these (zero-init especially) shift the early dynamics, I re-tune the schedule — the warmdown and total iteration count — alongside.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head; self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)   # split QKV
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()                              # zero init
        self.rotary = Rotary(self.head_dim)
    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))   # QK norm
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()                              # zero init
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()                                       # ReLU² > GELU
        return self.c_proj(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config); self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))              # no attn_scale
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

@dataclass
class GPTConfig:
    vocab_size : int = 50304   # padded to multiple of 128
    n_layer : int = 12
    n_head : int = 6           # head dim 128
    n_embd : int = 768
```
