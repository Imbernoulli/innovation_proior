The slot I have to fill is deceptively small and almost entirely decided by the structure of its input. A weather model has tokenized each meteorological variable independently, so at every spatial patch it hands me a set of $V$ variable tokens, each a $D$-vector, giving a tensor $x:[B,V,L,D]$. Before the shared Vision-Transformer backbone runs, these per-variable tokens must collapse into a single token per location, an aggregator $[B,V,L,D]\to[B,L,D]$ reducing over the variable axis. Three constraints fall straight out of the domain: there is no canonical ordering of temperature, geopotential, humidity and the surface mask, so the reduction must be permutation-invariant in the variables; $V$ is read from the input at runtime and may vary, so the module must accept any $V$; and the fused token feeds a downstream stack tuned to a single-token scale, so the reduction should keep the output on that scale. This is a set-to-vector problem, and the right way to attack it is to ask which class of set-to-vector functions to draw from and then pick the most expressive member that honors the constraints.

The cleanest characterization of permutation-invariant set functions writes them as $\rho(\sum_v \phi(x_v))$: encode each element with a shared $\phi$, combine with the symmetric sum, decode with $\rho$. That is universal in principle, but it commits to a fatal property here — each element is encoded *in isolation* before the sum, so the elements never see one another during encoding. Every reduction already on the table is this skeleton in a bare form. The uniform mean $\frac{1}{V}\sum_v x_v$ is $\phi$ the identity with a fixed average; it is parameter-free and scale-preserving but content-blind, weighting every variable identically at every location and state. The learned weighted sum $\sum_v \mathrm{softmax}(a)_v\,x_v$ adds one learnable scalar per variable, but its weighting is a single global distribution, one compromise forced to serve every location. Even the single-query cross-attention that is the ClimaX default shares the blind spot in a subtler place: a single learnable query scores each variable token against itself and pools by those scores, but the score on each variable is computed without reference to the other variables' contents — the query asks each token independently "how much do you match my fixed question?" None of the existing reductions lets the variables interact before being summarized.

That interaction is exactly the structure this domain has. The right combination of variables at a grid cell is relational, not separable across variables: a mid-tropospheric wind token may carry decisive information only when the geopotential token indicates a sharp gradient, and a humidity token's relevance depends on the temperature token's regime. These are inter-variable correlations — the pooling weight that belongs on variable $i$ depends on what variable $j$ is currently saying — and a pooling that scores each variable independently, whether by a fixed scalar or by a fixed query over the raw token, simply cannot express "upweight wind *because* geopotential is in this state." To capture it, the variables have to read each other before the summary is formed.

I propose the Set Transformer pooling for this slot: a Set Attention Block (SAB) that lets the variable tokens attend to one another, followed by Pooling by Multihead Attention (PMA) with a single learnable seed that summarizes the now context-aware tokens into one vector. It is the "SAB + PMA" architecture restricted to one encoder block and one seed — the minimal form that adds, over the single-query cross-attention baseline, the two levers that baseline lacks: multihead attention pooling, and, decisively, self-attention among the variables before pooling.

Both blocks are built from one well-behaved primitive, the Multihead Attention Block (MAB), a Transformer encoder block adapted for sets — multihead attention plus a feed-forward with residuals and layer norm, but with no positional encoding and no dropout, because positions would break the permutation symmetry I depend on and stochastic dropout would break the determinism of the set map. For a query set $X$ and key/value set $Y$ it is

$$H = \mathrm{LayerNorm}\big(X + \mathrm{Multihead}(X,Y,Y)\big),\qquad \mathrm{MAB}(X,Y) = \mathrm{LayerNorm}\big(H + \mathrm{rFF}(H)\big),$$

where $\mathrm{rFF}$ is a row-wise feed-forward applied identically to every element and $\mathrm{Multihead}$ is the usual $h$-head scaled dot-product attention. Two properties carry the construction: the attention output is a weighted average over the keys $Y$, so MAB is invariant to the order of $Y$; and it processes the query rows symmetrically, so MAB is equivariant in $X$. The $1/\sqrt{d}$ scaling inside the softmax is load-bearing — for query/key components roughly independent with unit variance, the dot product over $d$ dimensions has variance $d$, so unscaled logits grow like $\sqrt{d}$, saturate the softmax toward one-hot, and kill its gradient; dividing by $\sqrt{d}$ keeps the softmax responsive.

From MAB the two blocks fall out directly. Self-attention within the variable set is $\mathrm{SAB}(X) := \mathrm{MAB}(X,X)$: the tokens are simultaneously queries, keys and values, so each variable token's new representation reads from all the others by content compatibility. This is precisely the inter-variable interaction the lower rungs lack, and it is the mechanism that resists the degenerate trap of this kind of design. If I let an encoder be optimized only to make pooling easy, the cheapest solution can be a collapse — map every token to the same vector and pooling becomes trivial. Self-attention defeats that, because each token's output is a content-weighted combination of all the tokens' values *with a residual connection carrying its own information forward*, so folding every token to a constant would have to fight the residual rather than ride a free gradient down to it. SAB is also permutation-equivariant, exactly the property I need so that a permutation-invariant pooling on top yields a permutation-invariant whole. One SAB block already gives the pairwise mixing the domain argument calls for; its cost is $O(V^2)$ per location, but with $V = 48$ that is a $48\times48$ attention per cell, trivial, and emphatically not the $O((V\cdot h\cdot w)^2)$ blowup that motivated aggregation in the first place — the backbone still sees only the $h\cdot w$ aggregated sequence. So I can afford the full SAB rather than the inducing-point ISAB approximation, which exists only for large sets where $V^2$ would hurt.

The pooling is $\mathrm{PMA}_1(Z) := \mathrm{MAB}(S, Z)$ with one learnable seed $S \in \mathbb{R}^{1\times D}$. The seed is a trainable query that asks the encoded set "what is your summary?", and the answer is a content-dependent weighted combination of the encoded variable values; because the attention is multihead, the seed can pose several sub-questions at once — a thermodynamic summary, a dynamical summary — and fuse them through the block's output projection. One seed yields exactly one summary token per location, the $[B,L,D]$ the contract wants; $k>1$ seeds are for problems wanting several correlated outputs, like clustering, and would need a trailing SAB among the seeds to model their interaction, which is irrelevant here. The pooling is permutation-invariant in $Z$ since the softmax runs over the set of keys, so composing the equivariant SAB encoder with the invariant PMA pooling makes the whole aggregator permutation-invariant, as required.

Checking this against the rung directly below confirms I am adding the right thing and not just machinery. PMA with one seed and a single head, applied to the raw tokens with no encoder, is essentially the single-query cross-attention — so the ClimaX default is the $k=1$, no-encoder, single-head special case of this construction. The two things added over it are exactly the two levers it leaves unpulled: the pooling is multihead, and there is a self-attention encoder over the variable set before pooling, so the summary is formed over context-aware variable representations rather than raw, independently-scored ones. That second lever is the inter-variable-correlation modelling the domain needs. There is direct evidence that attending among elements before pooling captures structure that fixed reductions cannot: on a max-value-regression toy task, the attentive self-attention-plus-pooling construction matches the max-pooling oracle while mean and sum pooling fail badly, because finding the governing element requires the elements to be compared against each other — the same relational structure I expect among meteorological variables.

A few remaining choices settle the design. Whether PMA pools $Z$ or $\mathrm{rFF}(Z)$: the canonical definition prepends a row-wise feed-forward, $\mathrm{PMA}_k(Z) = \mathrm{MAB}(S, \mathrm{rFF}(Z))$, but that leading $\mathrm{rFF}$ is dropped when the preceding block already ends in a feed-forward — and SAB does end in its $\mathrm{rFF}$ sublayer, so $\mathrm{PMA}_1$ applied directly to the SAB output is the standard, non-redundant form. LayerNorm stays on in both blocks, since this sits inside a deep ViT pipeline fine-tuned from pretrained weights and LayerNorm in the residual blocks is what keeps activations well-conditioned during fine-tuning. Heads use the pipeline's $\mathrm{num\_heads}=16$, so $d_k = D/16 = 64$ per head, the same granularity as the rest of the model; with $D = 1024$ this is consistent throughout. The seed $S$ is a learnable parameter, Xavier-initialized so the attention starts with sensible-magnitude logits.

The implementation lives or dies on the indexing. The reduction is per-location and per-example and identical at every location, so each $(\text{example},\text{location})$ pair is an independent $V$-element set: permute to $[B,L,V,D]$ to bring the variable axis adjacent to $D$, fold $B$ and $L$ into one batch $[B\cdot L, V, D]$, apply SAB ($\mathrm{MAB}(X,X)$) to make the variables context-aware, expand the single seed to $[B\cdot L, 1, D]$, apply PMA ($\mathrm{MAB}(S, Z)$), squeeze the seed axis and unfold back to $[B,L,D]$. Crucially $V$ never enters any weight matrix — every $\mathrm{Linear}$ inside MAB maps $D\to D$ and the seed is $1\times1\times D$ — so any $V$ is accepted and $V$ only sizes the runtime attention. I implement MAB faithfully to its canonical reference rather than reach for a library layer, because the residual placement and the $\mathrm{rFF}$ are part of the block's identity: it projects $Q,K,V$ to width $D$, splits each into heads of width $D/\mathrm{num\_heads}$ stacked along the batch dimension so one batched matmul does all heads at once, computes $A = \mathrm{softmax}(Q_\_ K_\_^\top / \sqrt{D})$ over the key axis, forms the residual attention output $Q_\_ + A V_\_$, reassembles the heads along the feature axis, applies the first LayerNorm, then the feed-forward residual $O + \mathrm{ReLU}(\mathrm{fc\_o}(O))$, then the second LayerNorm.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    """Multihead Attention Block (Set Transformer, Lee et al. 2019, eqs. 6-7):
        H        = LayerNorm(X + Multihead(X, Y, Y))
        MAB(X,Y) = LayerNorm(H + rFF(H)),   rFF = row-wise Linear+ReLU residual.
    No positional encoding, no dropout (both would break the set symmetry).
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=True):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.ln0 = nn.LayerNorm(dim_V) if ln else None
        self.ln1 = nn.LayerNorm(dim_V) if ln else None
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if self.ln0 is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if self.ln1 is None else self.ln1(O)
        return O


class VariableAggregator(nn.Module):
    """Set Transformer pooling over the V variable tokens at each spatial location.

    SAB (self-attention among the variables) then PMA with one learnable seed (k=1).

    Input  x: [B, V, L, D]
    Output:   [B, L, D]
    """

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars
        # SAB encoder: variables attend to each other (MAB(X, X)).
        self.sab = MAB(embed_dim, embed_dim, embed_dim, num_heads, ln=True)
        # PMA pooling: one learnable seed cross-attends over the encoded set (MAB(S, Z)).
        self.seed = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.seed)
        self.pma = MAB(embed_dim, embed_dim, embed_dim, num_heads, ln=True)

    def forward(self, x):
        b, v, l, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * l, v, d)   # [B*L, V, D]: each (ex, loc) is one set
        z = self.sab(x, x)                               # [B*L, V, D]: variables read each other
        s = self.seed.expand(b * l, -1, -1)              # [B*L, 1, D]: the shared learnable seed
        out = self.pma(s, z)                             # [B*L, 1, D]: seed pools the encoded set
        out = out.squeeze(1).reshape(b, l, d)            # [B, L, D]: one token per location
        return out
```
