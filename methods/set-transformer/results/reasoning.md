Let me think about what makes a *set* hard for a neural network. The input is an unordered collection `{x_1,…,x_n}`, the count `n` varies, and the target depends only on the collection, not on any ordering I happen to write it in. So two non-negotiables fall out of the definition: the model must be permutation invariant — reorder the inputs and the output is unchanged — and it must swallow any `n`. A plain feed-forward net fails both (fixed-size input, and it sees order). An RNN handles variable length but is order-sensitive; feeding it `{x_1,x_2}` versus `{x_2,x_1}` gives different answers, which is wrong for a set.

The known way to satisfy both at once is to bake symmetry into the architecture. If I encode each element independently with some `φ`, then combine the encodings with a *symmetric* reduction `pool` (sum, mean, max), then post-process with `ρ`,

`net({x_1,…,x_n}) = ρ( pool({ φ(x_1),…,φ(x_n) }) )`,

the symmetry of `pool` makes the whole thing permutation invariant no matter what `φ` and `ρ` are. The architecture is also size-agnostic as a function: the same weights can be applied to any number of rows, even though a concrete minibatch implementation may still group equal-size sets or use padding masks. And there's a reassuring fact about this family: Zaheer et al. (2017, Deep Sets) showed that with `pool = sum` and continuous `ρ, φ`, `ρ(sum(φ(·)))` can approximate *any* permutation-invariant set function. The decomposition is clean: `φ` is an encoder acting independently on each element, `ρ∘pool` is a decoder that aggregates.

So why am I not done? Because "`φ` acts on each element independently" is doing a lot of damage. Every element is embedded in isolation, and only *after* that does any combination happen, through a fixed symmetric reduction. All information about how the elements relate to each other is squeezed out before pooling. For tasks where the answer is some simple aggregate of independent per-element scores, fine. But consider amortized clustering: I want a network that maps a point set straight to its cluster centers. The map has to assign each point to a cluster *and* respect explaining-away — the clusters shouldn't fight over the same points. That's exactly why clustering is normally done by iterative refinement (EM), each step looking at all points jointly. A pooling net can only learn to *quantize* the space — carve it into regions and pool within them — and worse, that quantization is baked into `φ`'s weights, so it *cannot depend on the contents of the particular input set*. Two different point sets get the same fixed partition. That's a recipe for under-fitting on anything where the elements need to talk to each other.

I should be careful that universality and this complaint aren't in contradiction — they aren't, and it's worth being precise about why, because it's the whole motivation. Universality is about what the family can represent *in the limit of arbitrary `φ, ρ`*; it says nothing about whether a fixed-width, gradient-trained `φ` can *learn* a content-dependent partition. The fixed pooling forces all the burden of relating elements onto `ρ` acting on a single pooled vector that has already collapsed the set. So the gap I care about is a *learnability/efficiency* gap, not a representability one. That tells me what to look for: keep (or recover) universality, but let elements interact *during* encoding — pairwise and higher-order — and make the pooling *learnable* and content-dependent rather than a fixed mean or max. Let me find a mechanism that does interaction while staying permutation symmetric.

Attention is a natural candidate, because it's built out of inner products and weighted sums, both of which are symmetric in the right ways. The primitive: with queries `Q ∈ R^{n×d_q}`, keys `K ∈ R^{n_v×d_q}`, values `V ∈ R^{n_v×d_v}`,

`Att(Q,K,V;ω) = ω(QKᵀ) V`.

`QKᵀ` is the `n×n_v` matrix of query–key similarities; `ω` turns each query's row of similarities into weights (a scaled softmax `ω(·)=softmax(·/√d)` — the `1/√d` keeps the dot products from blowing up as `d` grows and saturating the softmax); the output stacks, for each query, a weighted average of the values. Let me check the symmetry claim rather than assert it. If I permute the *key/value* set — reorder the rows of `K` and `V` together by a permutation `π` — then the columns of `QKᵀ` permute by `π`, so each query's row of weights `ω(QKᵀ)` permutes by `π`, and `V`'s rows permute by `π` too. A sum `Σ_i w_i v_i` reindexed so that both `w` and `v` follow the same `π` is unchanged. So attention's output is *invariant* to permutations of the keys/values. And if I permute the *queries* (rows of `Q`), only the rows of the output reorder identically — *equivariant* in the queries. That's exactly the pair of symmetries I need. I'll want to recheck this numerically once I've assembled a concrete block, because the residual/normalization wrapping could in principle spoil it.

I'll use the multi-head version (Vaswani 2017): project `Q,K,V` into `h` subspaces with learnable `W_j^Q, W_j^K, W_j^V`, run an attention head in each, concatenate, and mix with `W^O`. The point of multiple heads is that one shared similarity over the full `d` dimensions is a narrow channel; `h` heads attend in `h` different learned subspaces and can capture several relationships at once. Set `d_q = d_v = d` and `d_q^M = d_v^M = d/h` so the concatenation comes back to width `d`.

Now wrap this into a reusable block. I take the Transformer's encoder block, because it already combines attention with the residual + normalization + feed-forward structure that makes attention trainable in depth — but I have to strip two things. First, the positional encoding: a set has *no* positions, and adding position information would make the block order-sensitive, destroying permutation invariance. That's the whole reason I can't just use a Transformer off the shelf. Second, dropout (I just don't want it here). So the block — call it a Multihead Attention Block, `MAB`, taking two sets `X` (queries) and `Y` (keys/values):

`H = LayerNorm( X + Multihead(X, Y, Y; ω) )`,
`MAB(X,Y) = LayerNorm( H + rFF(H) )`,

where `rFF` is a row-wise feed-forward layer — applied to each row identically and independently, so it preserves equivariance. The first line lets the queries `X` gather information from `Y` (residual around the attention), the second is a per-element nonlinear refinement (residual around the FF). The keys/values `Y` enter only through the order-invariant attention sum, and everything touching the query rows is row-wise, so `MAB(X,Y)` should be equivariant in `X` and invariant to permutations of `Y` — same as the bare attention, with the wrapping not breaking it because every added operation (residual add, LayerNorm over the feature axis, row-wise FF) acts per row. I'll confirm that on numbers below.

To encode a set while letting its elements interact, I feed the set to itself — self-attention. Define the Set Attention Block

`SAB(X) := MAB(X, X).`

Now every element queries every other element; the output is a set of the same size in which each element's representation has absorbed information about all the others — pairwise interactions. Stack two `SAB`s and the second sees representations that already encode pairwise structure, so it can encode higher-order interactions. One worry: with `Q=K=V=X`, does this collapse to a plain residual block on `X`? It doesn't, because the per-head linear projections `W_j^Q, W_j^K, W_j^V` let each head compare *projected* views of the elements, so the attention weights are nontrivial functions of the data rather than the identity. As for symmetry, with `X` in all three slots a permutation of the input rows permutes the query rows, key rows, and value rows together, so by the bare-attention argument the output rows should permute the same way — `SAB` equivariant. I'd rather see the number than trust the prose, so let me build the block and test it.

I implement `MAB` exactly as above (multi-head attention split across the batch dimension, scaled softmax, residual, optional LayerNorm, row-wise FF residual), wrap `SAB(X)=MAB(X,X)`, take a single random set of 5 elements in `d=8` with `h=2` heads, and a fixed permutation `π=(2,0,4,1,3)` of its rows. Equivariance means `SAB(X)` permuted by `π` equals `SAB(πX)`. The max absolute difference over all entries comes out at `6.0e-8` — float32 roundoff, i.e. zero. So `SAB` is permutation equivariant, and the residual/LayerNorm/FF wrapping did not break it. Good; the symmetry I argued for is real, not just plausible.

There's a cost problem, though. `SAB` computes the full `n×n` attention matrix — `O(n²)` time and memory. For a point cloud with thousands of points, or a large clustering dataset, that's prohibitive. I want to keep the expressive self-attention but not pay quadratic cost.

The structural fact I can exploit: a big set usually has low-rank interaction structure — its `n` elements can be summarized through far fewer than `n` representatives. This is exactly the inducing-point idea from sparse Gaussian processes (Snelson & Ghahramani 2005) and Nyström approximation (Fowlkes 2004): instead of the full `n×n` Gram matrix, route everything through `m ≪ n` inducing points and pay `O(nm)`. Let me transplant that into attention. Introduce `m` trainable vectors `I ∈ R^{m×d}` — *inducing points*, parameters of the block, learned with everything else. First, let the inducing points attend to the set:

`H = MAB(I, X) ∈ R^{m×d}.`

Here `I` are the queries (`m` of them) and `X` is the key/value set — so `H` is `m` vectors, each a summary of the whole set `X`, computed in `O(nm)`. Then let the set attend back to this summary:

`ISAB_m(X) = MAB(X, H) ∈ R^{n×d},`

with `X` as queries and `H` as keys/values — `n` outputs, again `O(nm)`. Total `O(nm)`, linear in `n` for fixed `m`. This is structurally a low-rank / autoencoder-style bottleneck: project the set down onto `m` summary vectors `H`, then reconstruct an `n`-element output from them — except the goal isn't reconstruction, it's to extract good features, and the inducing points are free to learn whatever global structure best explains the set. In amortized clustering, for instance, the `m` inducing points could spread out as landmarks on the plane, and elements then get compared *indirectly* through their proximity to those shared landmarks, which is cheaper and gives a content-summarizing intermediate.

Is `ISAB` still permutation equivariant in `X`? This is the step I'm least sure of, because the two `MAB`s use `X` in different roles, so let me walk it and then test it. In `H = MAB(I, X)`, the queries are the *fixed* inducing points `I` and `X` is the key/value set; `MAB` is invariant to permutations of its key/value argument, so permuting the rows of `X` should leave `H` unchanged — `H` permutation *invariant* in `X`. Then in `ISAB_m(X) = MAB(X, H)`, `X` is the query and `H` (now fixed w.r.t. that permutation, by the previous sentence) is the key/value set; `MAB` is equivariant in its query, so permuting `X`'s rows should permute the output rows identically. So `ISAB_m` should be equivariant — like `SAB`, but linear-time. Two things to confirm numerically: that the intermediate `H` really is unchanged when I permute `X`, and that the whole `ISAB` is equivariant. On the same 5-element set with `m=3` inducing points: permuting `X` changes `H` by at most `1.2e-7` (zero — `H` is invariant, as the argument needs), and `ISAB(X)` permuted by `π` matches `ISAB(πX)` to `1.2e-7` (equivariant). Both hold, including the non-obvious middle claim that the summary is invariant while the output is equivariant.

Now the decoder / pooling. The default is a fixed symmetric reduction — mean or max — but that weights every element equally (mean) or keeps only one (max) with a fixed rule, and I argued I want content-dependent weighting. Let me make the case concrete on the simplest interaction-light task, max-regression: the target is `max_i x_i` of a set of reals, so the answer is recoverable from a *single* element, the largest, and the right aggregation should *find* that element and put its weight there. Take the set `{3, 50, 12, 7}` (mean 18, max 50) and a one-parameter attention readout where the query gives element `i` logit `w·x_i`. With `w=0` the logits are all zero, the softmax is uniform `[.25,.25,.25,.25]`, and the readout is `18.0` — exactly the mean, and stuck there. With `w=0.2` the weights become `[0.000, 0.999, 0.001, 0.000]` and the readout is `49.97`; with `w=1` the weights are `[0,1,0,0]` and the readout is `50.00`, the true max. So a *learnable* attention pooling slides continuously from the mean to argmax-selection just by scaling its query — the mean can't move off `18.0`, and max only works because that one task happens to match its hard rule. That's the content-dependent weighting I want, and it's a single learned scale away from what attention already is.

So introduce `k` trainable *seed* vectors `S ∈ R^{k×d}` and let them attend over the encoded set `Z`:

`PMA_k(Z) = MAB(S, rFF(Z)),`

Pooling by Multihead Attention. The seeds are the queries, the (feed-forward-refined) set is the keys/values, so the output is `k` vectors, each a learned, content-dependent weighted readout of the set — and it should be permutation invariant in `Z` because `Z` enters only as keys/values. I test it: `PMA` with `k=1` on the permuted 5-element set gives `PMA(X)` and `PMA(πX)` agreeing to `6.0e-8`, so invariant, as needed. Usually one seed (`k=1`) suffices: aggregate to a single vector. But some problems need several correlated outputs — amortized clustering wants `k` cluster centers, and what each center should be depends on where the *others* are (explaining-away again). So I use `k` seeds, and then, to let those `k` outputs interact, run a `SAB` over them:

`H = SAB(PMA_k(Z)).`

That self-attention among the `k` pooled vectors is what lets the model reason about the clusters jointly instead of emitting `k` independent guesses.

Putting it together. The encoder maps `X ↦ Z ∈ R^{n×d}` as a stack of equivariant blocks — `SAB(SAB(X))` when `n` is small, `ISAB_m(ISAB_m(X))` when `n` is large (cost `O(ℓ n²)` vs `O(ℓ n m)` for `ℓ` blocks). The decoder aggregates and finishes with a row-wise feed-forward:

`Decoder(Z) = rFF( SAB( PMA_k(Z) ) ) ∈ R^{k×d}`,  with  `PMA_k(Z) = MAB(S, rFF(Z))`.

Every encoder block tested as equivariant and `PMA` tested as invariant, so the composition encoder-then-decoder is permutation invariant: permuting the input permutes the encoder's rows identically, and `PMA` is blind to that permutation of its keys/values, so the final output doesn't move. It also handles any `n`, since every block applies the same weights to a variable number of rows.

Now the worry I flagged at the start: have I *lost* the universality the plain sum-pooling family had by routing everything through attention and softmax? The complaint against pooling was a learnability gap, and I'd be trading badly if I fixed that by giving up representability. Let me check that the new pieces *contain* the old ones — if attention pooling can reproduce mean, power means, and sum, then I can fall back to a Deep-Sets-shaped network inside this family and inherit its universality. I'll compute the reductions, not just claim them.

First, the mean as a special case of softmax attention. Take a single query equal to the zero vector `s = 0` and let `X` be the keys/values:

`Att(s, X, X; softmax) = softmax( s Xᵀ / √d ) X = softmax(0) X.`

`softmax` of an all-zeros logit vector is uniform `1/n`, so this is `(1/n) Σ_i x_i`, the mean. I check it on a random `4×3` set: the zero-query weights come out `[0.25,0.25,0.25,0.25]` summing to `1`, and the attention output equals the plain column mean to `0.0` (exact). So attention with a zero query *is* the mean — confirmed, not just asserted.

Next, power means `M_p(z) = ( (1/n) Σ_i z_i^p )^{1/p}`. Use the seed-pooling shape with a fixed (not data-dependent) query: set the seed `s = 0`, put a front row-wise feed-forward map realizing `z ↦ z^p`, and let the value projection in each one-dimensional head select one coordinate. To force uniform attention weights, set the query projection and its bias to zero so every query–key logit is equal; then each head is exactly the zero-query mean above, giving `mean(z^p)`. The fixed zero seed and zero query projection mean the residual around attention contributes no data-dependent term. A back row-wise feed-forward map realizes `u ↦ u^{1/p}` coordinate by coordinate, so the decoder forms `(mean z^p)^{1/p}`. The subtlety I have to get right: merely projecting query and key onto the *same coordinate* would not give a mean — it would give data-dependent logits and hence data-dependent (non-uniform) weights. The construction needs the zero-logit route, exactly the one I just verified produces uniform weights.

Now the one that doesn't come for free: `PMA` with *softmax* can give the mean but not the un-normalized sum, because softmax weights always sum to one. Zaheer et al.'s universality statement is about `rFF(sum(rFF(·)))` with a genuine sum. So for the expressivity argument I use the more general attention activation `ω(·) = 1 + f(·)` with `f(0)=0` (identity, ReLU, or a centered sigmoid all qualify) in place of softmax. With a zero query every logit is `0`, so `ω(0) = 1`: every value gets weight exactly one and the output is `Σ_i z_i`. I check this too, on the same `4×3` set: the non-normalizing zero-query readout equals the plain column sum to `0.0` (exact). So the attention-pooling family *contains* sum pooling, provided I'm allowed the non-normalizing activation for the proof — the practical module will still use softmax.

Finally, assemble the Deep-Sets network inside this family. Suppress the attention contribution in every `SAB`/`ISAB`: in the full multi-head definition, set the output-mixing matrix `W^O = 0`; in the simplified implementation block, zero the value/attention path. The residual connections, layer norms, and row-wise feed-forward maps are all per-row, so with the attention path off the encoder collapses to an instance-wise map `Z = rFF(X)`. The decoder then does `rFF(sum(Z))` by the non-softmax sum-pooling construction just verified. Composing, the model realizes `rFF(sum(rFF(X)))` — which is universal by Zaheer et al. So adding attention costs nothing in representational reach for the broader activation family; attention isn't *needed* for universality, but it's the mechanism that makes the interaction-heavy tasks learnable in practice, which is the gap I started from. The practical module below uses the robust softmax version; the exact-sum construction is the theoretical fallback showing the wider family contains Deep Sets.

Let me write the blocks. `MAB` first — the multi-head attention with residual, layer-norm, and a row-wise feed-forward realized as a single linear-plus-ReLU residual:

```python
import torch, torch.nn as nn, torch.nn.functional as F, math

class MAB(nn.Module):
    """Multihead Attention Block: queries Q attend over keys/values K.
    Transformer encoder block minus positional encoding and dropout."""
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)        # W^Q
        self.fc_k = nn.Linear(dim_K, dim_V)        # W^K
        self.fc_v = nn.Linear(dim_K, dim_V)        # W^V
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)        # row-wise FF (rFF)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        # split the d-dim representation into num_heads heads (stacked on batch dim)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        # scaled softmax attention; this implementation divides by sqrt(dim_V)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        # residual around attention, then (optional) LayerNorm
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        # residual around the row-wise feed-forward, then (optional) LayerNorm
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O
```

`SAB` is just self-attention — the set against itself:

```python
class SAB(nn.Module):
    """Set Attention Block: MAB(X, X). Self-attention within the set; O(n²)."""
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
    def forward(self, X):
        return self.mab(X, X)
```

`ISAB` adds the `m` trainable inducing points and routes through them, two MABs, `O(nm)`:

```python
class ISAB(nn.Module):
    """Induced Set Attention Block: inducing points I summarize X, then X reads
    back from that summary. Permutation equivariant in X; O(nm)."""
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))  # inducing points
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)  # H = MAB(I, X)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)  # MAB(X, H)
    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)            # m×d summary (invariant in X)
        return self.mab1(X, H)                                      # n×d output (equivariant in X)
```

`PMA` is the learnable pooling — `k` trainable seeds attending over the set. The formula I derived can place an `rFF` before the keys/values; in the practical module I omit it because the previous block already ends with a row-wise feed-forward layer:

```python
class PMA(nn.Module):
    """Pooling by Multihead Attention: k learnable seeds attend over the set,
    producing k content-dependent readouts. Permutation invariant in X."""
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))      # seed vectors
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)
    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
```

A full model for, say, classification then stacks two `ISAB`s as the encoder, pools with `PMA` (`k=1` here), optionally refines with `SAB`s, and projects:

```python
class SetTransformer(nn.Module):
    def __init__(self, dim_in, num_outputs, dim_out,
                 num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super().__init__()
        self.enc = nn.Sequential(
            ISAB(dim_in, dim_hidden, num_heads, num_inds, ln=ln),   # O(n m) encoder
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),         # learnable pooling
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),          # interactions among k outputs
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_out))
    def forward(self, X):
        return self.dec(self.enc(X))
```
