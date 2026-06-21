The cross-attention rung confirmed the whole ladder's logic. Against the weighted sum's 345.79 / 2.5934 / 3.4245, the learnable-query cross-attention landed 255.85 / 2.1214 / 3.1455 — a large drop on every target, and crucially it pulled wind10m-7day back under the mean's floor (3.1455 < 3.3991), exactly the falsifiable bar I had set. So the diagnosis held: the weighted sum's wind regression was the cost of a frozen, content-independent weighting, and letting the weight on each variable react to the token contents fixed it. But I have to ask what cross-attention itself still leaves on the table. Look at *how* each weight is computed: the score on variable $v$ is a compatibility between the fixed query and $x_v$ *alone*. The query asks the same fixed question of every token independently — the score on the geopotential token does not depend on what the wind token is saying, and vice versa. The variable tokens never read each other before they are pooled. So cross-attention reacts to each variable's content in isolation but cannot express a *relational* rule like "upweight the wind token *because* the geopotential token indicates a sharp gradient here." That inter-variable conditioning is exactly the structure I expect to matter most on wind10m-7day: it is the longest lead (7 days) and the smallest residual headroom (3.1455), and at long lead the informative combination of variables is plausibly the most state-coupled — which variable carries the signal depends on the joint atmospheric configuration, not on any one token in isolation. Every rung so far has left this lever unpulled: the mean averages each variable in isolation, the weighted sum scales each in isolation, the cross-attention scores each in isolation. None lets the $V$ tokens *interact* before the summary is formed.

I propose **Set Transformer pooling** (Lee et al., ICML 2019): let the $V$ variable tokens self-attend so each is represented in the context of the others, then pool the context-aware tokens with a learnable-seed multihead attention. The construction is two stages — a **SAB** (set attention block) encoder followed by a **PMA** (pooling by multihead attention) — both built from one well-behaved primitive, the multihead attention block $\mathrm{MAB}(X, Y)$ for a query set $X$ over a key/value set $Y$:
$$H = \mathrm{LayerNorm}\big(X + \mathrm{Multihead}(X, Y, Y)\big), \qquad \mathrm{MAB}(X, Y) = \mathrm{LayerNorm}\big(H + \mathrm{rFF}(H)\big),$$
with $\mathrm{rFF}$ a row-wise feed-forward and the usual $1/\sqrt{d}$ scaling so the softmax does not saturate (the same responsiveness argument that mattered at the cross-attention rung). Critically, MAB carries **no positional encoding and no dropout**: a position would break the permutation symmetry I rely on, and stochastic dropout would break the determinism of the set map. From MAB the two blocks fall out. Self-attention within the variable set is $\mathrm{SAB}(X) = \mathrm{MAB}(X, X)$: tokens are queries, keys, and values at once, so each variable reads from all the others by content compatibility — the inter-variable interaction the lower rungs lack. The pooling is $\mathrm{PMA}(Z) = \mathrm{MAB}(S, Z)$ with one learnable *seed* vector $S$, a trainable query that asks the encoded set for its summary and returns one token, multihead so it can pose several summary sub-questions (the thermodynamic and dynamical summaries) and combine them through the block's output projection.

There is a collapse trap to navigate. "Let the elements interact, then pool" can fail if the interaction stage is optimized only to make pooling easy: the cheapest solution might be an encoder mapping every variable token to the same vector — trivially poolable, useless. The interaction mechanism must genuinely mix information across the set and resist that trivial collapse, and self-attention does exactly this. Each token's new representation is a content-weighted combination of all the tokens' values *with a residual* that carries the token's own information forward, so folding every token to a constant would have to fight the residual rather than ride a free gradient down to it — the identity-collapse is not the cheap optimum. And SAB is permutation-*equivariant* (permute the inputs, the outputs permute the same way), while one-seed PMA is permutation-*invariant*, so the composition equivariant SAB $\circ$ invariant PMA is an invariant aggregator — the set semantics the contract demands.

The consistency check that makes this the right next rung and not a sideways move: a one-seed, single-head PMA on the *raw* tokens with no encoder is essentially the cross-attention rung — a learnable query pooling the variable tokens. So the ClimaX cross-attention is the no-encoder, $k=1$, single-head special case of this construction, and the two things I add over it are exactly the two it lacks — the pooling is multihead, and (decisively) there is a self-attention encoder over the variable set *before* pooling, so the summary is formed over context-aware variable representations rather than independently-scored raw ones. That is the inter-variable-correlation modelling I argued wind10m-7day most needs. It is not idle: the attend-among-elements-then-pool construction is known to match a max-pooling oracle on a max-value-regression task where mean and sum pooling fail badly, because identifying the governing element requires the elements to be compared against each other — the same relational structure I expect among meteorological variables.

The design choices are sized to the problem and the pretrained backbone. SAB is $O(V^2)$ per location, but $V = 48$, so this is a $48 \times 48$ attention at each of the 512 locations, tiny, and it does *not* reintroduce the $O((V\!\cdot\!h\!\cdot\!w)^2)$ blowup that motivated aggregation (the backbone still sees only the $h\!\cdot\!w$ aggregated sequence). Because the set is small I can afford the full SAB rather than the inducing-point bottleneck (ISAB), which is meant for large sets; and one seed gives exactly the one token per location the contract wants, so I do not need $k > 1$ seeds, which are for problems wanting several correlated outputs. I keep LayerNorm on in both blocks — inside a deep ViT fine-tune, LayerNorm in the residual blocks is what keeps activations well-conditioned and training stable — and use the pipeline's $\text{num\_heads} = 16$ ($d_k = 64$ per head). The seed is Xavier-initialized so the pooling attention starts with sensible-magnitude logits. Unlike the cross-attention rung I cannot zero-initialize to recover the mean, because the SAB encoder makes the map nonlinear in the tokens — but the residual structure and LayerNorm keep the starting point well-behaved, and fine-tuning from pretrained ClimaX weights means the backbone adapts to the new aggregator's output distribution. I also drop the leading $\mathrm{rFF}(Z)$ that canonical PMA applies before pooling, since SAB already ends in a feed-forward.

The shapes follow the same per-location folding as the cross-attention rung: permute $x: [B, V, L, D]$ to $[B, L, V, D]$ and fold to $[B\!\cdot\!L, V, D]$ so each (example, location) is an independent $V$-set; apply SAB to get $[B\!\cdot\!L, V, D]$ with the variables now context-aware; expand the seed to $[B\!\cdot\!L, 1, D]$ and apply PMA to get $[B\!\cdot\!L, 1, D]$; squeeze and unfold to $[B, L, D]$. $V$ never enters any weight — every `Linear` inside MAB maps $D \to D$ and the seed is $1\times 1\times D$ — so the module accepts any $V$. MAB is implemented faithfully to its canonical reference: three projections to width $D$, heads split and stacked along the batch dimension for one `bmm`, residual $Q_- + AV_-$, LayerNorm, then the feed-forward residual $O + \mathrm{ReLU}(\text{fc\_o}(O))$, LayerNorm; SAB calls $\mathrm{MAB}(x, x)$ and PMA calls $\mathrm{MAB}(\text{seed}, z)$.

The bar this has to clear, against the cross-attention numbers, is 255.85 / 2.1214 / 3.1455, and the decisive test is again wind10m-7day — the longest-lead, most state-coupled target and the smallest residual headroom, exactly where inter-variable correlation should matter most. If Set Transformer pooling cannot get wind10m below 3.1455, then attending-among-variables-before-pooling is not what that target needed and the extra block is unjustified. On z500-3day and t850-5day I expect smaller relative gains than cross-attention already delivered (their biggest defect, the frozen global split, was already fixed by content-dependent pooling) but still a measurable improvement if relational structure among the dynamical and thermodynamic variables is real. The risk I would watch is overfitting: SAB roughly triples the aggregator's parameters over the single cross-attention layer, and on a single-seed, single-block budget fine-tuned from pretrained weights, a too-expressive aggregator could fit noise and fail to clear the bar on the easier targets even while helping wind10m. Clear it on all three — and especially wind10m — and the ladder's final step, from independent-scoring pooling to interaction-then-pooling, is earned.

```python
# EDITABLE region of custom_forecast.py (lines 310-351) — finale: Set Transformer pooling (SAB + PMA)
import math


class MAB(nn.Module):
    """Multihead Attention Block (Set Transformer, eqs. 6-7):
        H        = LayerNorm(X + Multihead(X, Y, Y))
        MAB(X,Y) = LayerNorm(H + rFF(H)),   rFF = row-wise Linear+ReLU residual.
    No positional encoding, no dropout (both break the set symmetry).
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
    """Set Transformer pooling: SAB over the V variable tokens, then PMA with one seed.

    Args:
        embed_dim (int): Embedding dimension D.
        num_heads (int): Number of attention heads.
        num_vars (int): Number of input variables V.
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
        """
        Args:
            x: [B, V, L, D] — per-variable patch embeddings.
        Returns:
            [B, L, D] — aggregated representation.
        """
        b, v, l, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * l, v, d)   # B*L, V, D
        z = self.sab(x, x)                               # B*L, V, D: variables read each other
        s = self.seed.expand(b * l, -1, -1)              # B*L, 1, D: the shared learnable seed
        out = self.pma(s, z)                             # B*L, 1, D: seed pools the encoded set
        out = out.squeeze(1).reshape(b, l, d)            # B, L, D
        return out
```
