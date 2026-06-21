Plain RoPE is the strongest rung so far, and the way it won tells me where the remaining loss is hiding. Stripping the QK-Norm half edged the combined model out exactly as predicted: validation loss $2.2570$ against RoPE + QK-Norm's $2.2589$, WikiText-2 $43.17$ vs $43.44$, LAMBADA $65.81$ vs $67.20$ — plain RoPE best on every language-modeling metric, with the downstream split I flagged also showing up (RoPE + QK-Norm kept ARC-Easy and PIQA, plain RoPE took HellaSwag and WinoGrande). So plain RoPE is the strongest language model and the right rung to build on. But step back and look at what *every* rung so far has in common: QK-Norm changed the logit *scale*, RoPE changed how *position* enters the logit, and neither touched the *shape of the attention distribution itself* — all three end in the same operation, a single softmax over the context, then a weighted average of values. That single softmax has a structural defect no amount of better position or scaling can fix. The attention weights are $a_{m,n} = \exp(s_{m,n}) / \sum_{n'}\exp(s_{m,n'})$, and every $\exp(\cdot)$ is strictly positive, so no finite logit can drive a weight to exactly zero — the softmax can make an irrelevant token's weight small, but never zero, and it has no mechanism to take mass *back off* a position. Over a 1024-token context, even if every irrelevant token gets a weight of order $1/T$, the *aggregate* of those weights is most of the mass, so the output $o_m = \sum_n a_{m,n} v_n$ is part signal and part a low-level average of the irrelevant remainder. RoPE put the *peak* attention in the right relative location; it did nothing about the noise floor smeared under that peak, because the floor is a property of softmax positivity, not of where the peak sits. The $2.2570$ is, in part, the cost of averaging junk into every token's representation.

I propose the **Differential Transformer** (Ye et al. 2024): change the operator so it can *subtract* mass from irrelevant positions, which is the one thing a single positive softmax cannot do. The principle is a differential amplifier, which rejects the voltage common to its two inputs and amplifies only their difference, cancelling common-mode noise. Form *two* softmax maps over the same context from two different query/key projections and take their signed difference:
$$ \mathrm{DiffAttn}(X) = \big(\,\mathrm{softmax}(Q_1 K_1^\top/\sqrt{d}) - \lambda\,\mathrm{softmax}(Q_2 K_2^\top/\sqrt{d})\,\big)\,V, \qquad Q = [Q_1; Q_2],\; K = [K_1; K_2], $$
with a learnable scalar $\lambda$. Both maps see the same content, so over the irrelevant tail their floor patterns are *correlated* — the common-mode part — and $A_1 - \lambda A_2$ drives it toward zero; on the relevant tokens the model has an incentive to make the first map spike where the second does not, so the difference is large there. The resulting weights no longer sum to one and are no longer all positive — they are *signed*, which is exactly the new power: the operator can push an irrelevant value's contribution to zero or below instead of being stuck with the positive floor. The trivial solution $\lambda = 0$ is not the optimum, because cancelling the floor genuinely lowers loss, so the optimizer has a real reason to use the subtraction. This is the representational lever QK-Norm and RoPE both lacked, because both ended in one positive softmax.

Three engineering pieces make it a fair, stable drop-in rather than a doubled-cost, destabilizing one. First, $\lambda$ is not a free scalar (badly conditioned, can drift the subtraction into a wild magnitude); I re-parameterize it as
$$ \lambda = \exp(\lambda_{q1}\!\cdot\!\lambda_{k1}) - \exp(\lambda_{q2}\!\cdot\!\lambda_{k2}) + \lambda_\text{init} $$
with four learnable `head_dim` vectors initialized $\mathcal{N}(0, 0.1)$, so at init the two exponentials are $\approx 1$, roughly cancel, and $\lambda \approx \lambda_\text{init}$ with well-scaled signed gradients. Second, the budget: doubling the queries and keys is paid for by *halving the head dimension* — each logical head uses $\text{head\_dim} = n_\text{embd}/n_\text{head}/2 = 32$, with $2\,n_\text{head}$ query/key sub-heads of that dimension and $n_\text{head}$ value heads of dimension $2\cdot\text{head\_dim} = 64$. Total q/k width $2\,n_\text{head}\cdot\text{head\_dim} = n_\text{embd}$ and total v width $n_\text{head}\cdot 2\,\text{head\_dim} = n_\text{embd}$ — exactly the vanilla widths, so the fused `c_attn` projection is unchanged and the compute is matched; the doubling is absorbed by the halving, and any gain is purely from the changed attention *shape*, not added capacity. Third, scale: the subtraction makes heads heterogeneous and shrinks the operator gain roughly by $(1 - \lambda)$, so I per-head normalize each head's $2\cdot\text{head\_dim}$ output (a per-head RMSNorm, the GroupNorm-across-heads discipline) and then rescale by the *fixed* constant $(1 - \lambda_\text{init})$ — fixed, not the learned $\lambda$, so the gain compensation is a stable constant and the normalization carries the rest. That fixed compensation is what lets the frozen GPT-2 Medium optimizer, learning rate, and schedule transfer unchanged, which matters because the loop is fixed and I cannot retune.

Two real compromises are forced by this edit surface, and I name them honestly. The editable region is only `CausalSelfAttention(config)`, and `config` does not carry the *layer index*. The full method's depth-scaled $\lambda_\text{init} = 0.8 - 0.6\exp(-0.3(l-1))$ needs $l$, which I do not have — every one of the 24 blocks constructs its attention from the same `config` with no $l$ — so I set a single fixed $\lambda_\text{init} = 0.8$, the schedule's deep-layer asymptote and the operating point the reparameterization is centered on. This is a genuine omission: the early layers will cancel harder than the schedule would prescribe. The second compromise: the fused SDPA path returns only the final averaged output, not the two softmax maps I need to subtract, so I cannot use Flash here. I take the manual masked-softmax path the scaffold already provides as the non-flash fallback, compute the $2\,n_\text{head}$ attention maps explicitly, reshape to $(B, n_\text{head}, 2, T, T)$, subtract, and matmul with $v$ — which costs the memory of the explicit $T\times T$ maps, unavoidable because the differential subtraction lives *between* the softmax and the value-average, exactly where the fused kernel hides its internals. Position stays as the strongest baseline's: RoPE (`use_pos_emb = False`, the split-half rotation from the previous rung) applied to the doubled $\text{head\_dim}$-wide q/k sub-heads, so this rung is "the strongest position scheme, with the single softmax replaced by a differential one." The bar to clear is plain RoPE's $2.2570$ / WikiText-2 $43.17$ / LAMBADA $65.81$; because the noise floor is a real fraction of attention mass on a 1024-token context, I expect `val_loss` and both perplexities below the RoPE numbers, LAMBADA most of all since its long-passage last-word prediction is exactly the retrieval-under-noise setting the cancellation targets. If `val_loss` does *not* drop below $2.2570$, the honest reading is that at 355M params, 7.1B tokens, and a 1024-token context the attention-noise floor is not yet the binding constraint — the differential mechanism's advantage grows with scale and context length, and this is a bet that pays off at larger scale than this task runs.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 33–70) — finale: Differential Transformer + RoPE
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # Differential attention: halve the per-head dim so the doubled q/k stays budget-matched.
        # 2*n_head query/key sub-heads of dim head_dim; n_head value heads of dim 2*head_dim.
        self.head_dim = config.n_embd // config.n_head // 2
        self.scaling = self.head_dim ** -0.5
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        self.use_pos_emb = False  # RoPE replaces learned position embeddings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
        # config carries no layer index -> use the depth schedule's deep asymptote as a fixed init.
        self.lambda_init = 0.8
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        # Per-head normalization over the 2*head_dim attention output (the GroupNorm/sub-LN).
        self.subln = nn.RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def _apply_rope(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat([y1, y2], dim=-1).type_as(x)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # 2*n_head query/key sub-heads of dim head_dim; n_head value heads of dim 2*head_dim.
        q = q.view(B, T, 2 * self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, 2 * self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, 2 * self.head_dim).transpose(1, 2)
        # Relative position on the query/key sub-heads.
        q = self._apply_rope(q, T)
        k = self._apply_rope(k, T)
        # Two softmax maps (manual path: the fused kernel hides the maps the subtraction needs).
        att = (q @ k.transpose(-2, -1)) * self.scaling
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        att = att.view(B, self.n_head, 2, T, T)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        att = att[:, :, 0] - lambda_full * att[:, :, 1]            # (B, n_head, T, T), signed
        y = att @ v                                                # (B, n_head, T, 2*head_dim)
        y = self.subln(y) * (1.0 - self.lambda_init)               # per-head norm + fixed gain comp.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
```
