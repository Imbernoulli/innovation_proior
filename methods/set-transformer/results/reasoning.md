Let me think about what makes a *set* hard for a neural network. The input is an unordered collection `{x_1,…,x_n}`, the count `n` varies, and the target depends only on the collection, not on any ordering I happen to write it in. So two non-negotiables fall out of the definition: the model must be permutation invariant — reorder the inputs and the output is unchanged — and it must swallow any `n`. A plain feed-forward net fails both (fixed-size input, and it sees order). An RNN handles variable length but is order-sensitive; feeding it `{x_1,x_2}` versus `{x_2,x_1}` gives different answers, which is wrong for a set.

The known way to satisfy both at once is to bake symmetry into the architecture. If I encode each element independently with some `φ`, then combine the encodings with a *symmetric* reduction `pool` (sum, mean, max), then post-process with `ρ`,

`net({x_1,…,x_n}) = ρ( pool({ φ(x_1),…,φ(x_n) }) )`,

the symmetry of `pool` makes the whole thing permutation invariant no matter what `φ` and `ρ` are. The architecture is also size-agnostic as a function: the same weights can be applied to any number of rows, even though a concrete minibatch implementation may still group equal-size sets or use padding masks. And this isn't just a hack: with `pool = sum` and continuous `ρ, φ`, this family is a universal approximator of permutation-invariant set functions (Zaheer et al. 2017, Deep Sets). So `ρ(sum(φ(·)))` can in principle represent *any* set function. The decomposition is clean: `φ` is an encoder acting independently on each element, `ρ∘pool` is a decoder that aggregates.

So why am I not done? Because "`φ` acts on each element independently" is doing a lot of damage. Every element is embedded in isolation, and only *after* that does any combination happen, through a fixed symmetric reduction. All information about how the elements relate to each other is squeezed out before pooling. For tasks where the answer is some simple aggregate of independent per-element scores, fine. But consider amortized clustering: I want a network that maps a point set straight to its cluster centers. The map has to assign each point to a cluster *and* respect explaining-away — the clusters shouldn't fight over the same points. That's exactly why clustering is normally done by iterative refinement (EM), each step looking at all points jointly. A pooling net can only learn to *quantize* the space — carve it into regions and pool within them — and worse, that quantization is baked into `φ`'s weights, so it *cannot depend on the contents of the particular input set*. Two different point sets get the same fixed partition. That's a recipe for under-fitting on anything where the elements need to talk to each other.

What I actually want is for the elements to interact *during* encoding — pairwise and higher-order — and for the pooling to be *learnable* and content-dependent rather than a fixed mean or max. Let me find a mechanism that does interaction while staying permutation symmetric.

Attention is the natural candidate, because it's built out of inner products and weighted sums, both of which are symmetric in the right ways. The primitive: with queries `Q ∈ R^{n×d_q}`, keys `K ∈ R^{n_v×d_q}`, values `V ∈ R^{n_v×d_v}`,

`Att(Q,K,V;ω) = ω(QKᵀ) V`.

`QKᵀ` is the `n×n_v` matrix of query–key similarities; `ω` turns each query's row of similarities into weights (a scaled softmax `ω(·)=softmax(·/√d)` — the `1/√d` keeps the dot products from blowing up as `d` grows and saturating the softmax); the output stacks, for each query, a weighted average of the values. The thing to notice is the symmetry: if I permute the *key/value* set — reorder the rows of `K` and `V` together — then the columns of `QKᵀ` permute and the rows of `V` permute the same way, and a softmax-weighted sum over those rows is invariant to their order. So attention's output is *invariant* to permutations of the keys/values and *equivariant* to permutations of the queries (permute the queries, the output rows permute identically). That's exactly the pair of symmetries I need.

I'll use the multi-head version (Vaswani 2017): project `Q,K,V` into `h` subspaces with learnable `W_j^Q, W_j^K, W_j^V`, run an attention head in each, concatenate, and mix with `W^O`. The point of multiple heads is that one shared similarity over the full `d` dimensions is a narrow channel; `h` heads attend in `h` different learned subspaces and can capture several relationships at once. Set `d_q = d_v = d` and `d_q^M = d_v^M = d/h` so the concatenation comes back to width `d`.

Now wrap this into a reusable block. I take the Transformer's encoder block, because it already combines attention with the residual + normalization + feed-forward structure that makes attention trainable in depth — but I have to strip two things. First, the positional encoding: a set has *no* positions, and adding position information would make the block order-sensitive, destroying permutation invariance. That's the whole reason I can't just use a Transformer off the shelf. Second, dropout (I just don't want it here). So the block — call it a Multihead Attention Block, `MAB`, taking two sets `X` (queries) and `Y` (keys/values):

`H = LayerNorm( X + Multihead(X, Y, Y; ω) )`,
`MAB(X,Y) = LayerNorm( H + rFF(H) )`,

where `rFF` is a row-wise feed-forward layer — applied to each row identically and independently, so it preserves equivariance. The first line lets the queries `X` gather information from `Y` (residual around the attention), the second is a per-element nonlinear refinement (residual around the FF). Because the keys/values `Y` only enter through the order-invariant attention sum, and everything touching the query rows is row-wise, `MAB(X,Y)` is equivariant in `X` and invariant to permutations of `Y`.

To encode a set while letting its elements interact, I feed the set to itself — self-attention. Define the Set Attention Block

`SAB(X) := MAB(X, X)`.

Now every element queries every other element; the output is a set of the same size in which each element's representation has absorbed information about all the others — pairwise interactions. Stack two `SAB`s and the second sees representations that already encode pairwise structure, so it can encode higher-order interactions. Note that with `Q=K=V=X` this looks like it might collapse to a plain residual block on `X`, but it doesn't, because the per-head linear projections `W_j^Q, W_j^K, W_j^V` let each head compare *projected* views of the elements, so it learns genuinely more than identity-mixing. And `SAB(X)` is permutation equivariant: if I permute the input rows, the query rows, key rows, and value rows all permute together; the attention matrix is conjugated by that same permutation, the weighted sums follow the query rows, and every residual, layer norm, and row-wise feed-forward step preserves the row permutation.

There's a cost problem, though. `SAB` computes the full `n×n` attention matrix — `O(n²)` time and memory. For a point cloud with thousands of points, or a large clustering dataset, that's prohibitive. I want to keep the expressive self-attention but not pay quadratic cost.

The structural fact I can exploit: a big set usually has low-rank interaction structure — its `n` elements can be summarized through far fewer than `n` representatives. This is exactly the inducing-point idea from sparse Gaussian processes (Snelson & Ghahramani 2005) and Nyström approximation (Fowlkes 2004): instead of the full `n×n` Gram matrix, route everything through `m ≪ n` inducing points and pay `O(nm)`. Let me transplant that into attention. Introduce `m` trainable vectors `I ∈ R^{m×d}` — *inducing points*, parameters of the block, learned with everything else. First, let the inducing points attend to the set:

`H = MAB(I, X) ∈ R^{m×d}`.

Here `I` are the queries (`m` of them) and `X` is the key/value set — so `H` is `m` vectors, each a summary of the whole set `X`, computed in `O(nm)`. Then let the set attend back to this summary:

`ISAB_m(X) = MAB(X, H) ∈ R^{n×d}`,

with `X` as queries and `H` as keys/values — `n` outputs, again `O(nm)`. Total `O(nm)`, linear in `n` for fixed `m`. This is structurally a low-rank / autoencoder-style bottleneck: project the set down onto `m` summary vectors `H`, then reconstruct an `n`-element output from them — except the goal isn't reconstruction, it's to extract good features, and the inducing points are free to learn whatever global structure best explains the set. In amortized clustering, for instance, the `m` inducing points could spread out as grid points on the plane, and elements then get compared *indirectly* through their proximity to those shared landmarks, which is cheaper and gives a content-summarizing intermediate.

Is `ISAB` still permutation equivariant in `X`? Walk it through. In `H = MAB(I, X)`, the queries are the *fixed* inducing points `I` and `X` is the key/value set; `MAB` is invariant to permutations of its key/value argument, so permuting the rows of `X` leaves `H` unchanged — `H` is permutation *invariant* in `X`. Then in `ISAB_m(X) = MAB(X, H)`, `X` is the query and `H` (now fixed w.r.t. that permutation) is the key/value set; `MAB` is equivariant in its query, so permuting `X`'s rows permutes the output rows identically. So `ISAB_m` is permutation equivariant — exactly like `SAB`, but linear-time.

Now the decoder / pooling. The default is a fixed symmetric reduction — mean or max — but that weights every element equally (mean) or keeps only one (max) with a fixed rule, and I argued I want content-dependent weighting. Think about a target that is the max of a set of numbers: the answer is recoverable from a *single* element, the largest, so the right aggregation would *find* that element and attend to it — impossible for mean, and max only works because that task happens to match its hard rule. In general the influence of each element on the target is unequal and depends on the set. Attention is the tool again: introduce `k` trainable *seed* vectors `S ∈ R^{k×d}` and let them attend over the encoded set `Z`:

`PMA_k(Z) = MAB(S, rFF(Z))`,

Pooling by Multihead Attention. The seeds are the queries, the (feed-forward-refined) set is the keys/values, so the output is `k` vectors, each a learned, content-dependent weighted readout of the set — and it's permutation invariant in `Z` because `Z` enters only as keys/values. Usually one seed (`k=1`) suffices: aggregate to a single vector. But some problems need several correlated outputs — amortized clustering wants `k` cluster centers, and what each center should be depends on where the *others* are (explaining-away again). So I use `k` seeds, and then, to let those `k` outputs interact, run a `SAB` over them:

`H = SAB(PMA_k(Z))`.

That self-attention among the `k` pooled vectors is what lets the model reason about the clusters jointly instead of emitting `k` independent guesses.

Putting it together. The encoder maps `X ↦ Z ∈ R^{n×d}` as a stack of equivariant blocks — `SAB(SAB(X))` when `n` is small, `ISAB_m(ISAB_m(X))` when `n` is large (cost `O(ℓ n²)` vs `O(ℓ n m)` for `ℓ` blocks). The decoder aggregates and finishes with a row-wise feed-forward:

`Decoder(Z) = rFF( SAB( PMA_k(Z) ) ) ∈ R^{k×d}`,  with  `PMA_k(Z) = MAB(S, rFF(Z))`.

Since every encoder block is permutation equivariant, the encoder map `X ↦ Z` is equivariant; since `PMA` is permutation invariant, the composition encoder-then-decoder is permutation invariant. The whole model is invariant by construction, and handles any `n`.

I should make sure I haven't *lost* expressive power by adding all this attention. Can this still represent any permutation-invariant function, the way plain sum-pooling can? Let me check that the new pieces *contain* the old ones, then lean on the Deep Sets universality result.

First, the mean is a special case of softmax attention. Take a single query equal to the zero vector `s = 0` and let `X` be the keys/values:
`Att(s, X, X; softmax) = softmax( s Xᵀ / √d ) X = softmax(0) X = (1/n) Σ_i x_i`,
because `softmax` of an all-zeros logit vector is uniform `1/n`. So attention with a zero query *is* the mean.

Next, the decoder can express any element-wise power mean `M_p(z) = ( (1/n) Σ_i z_i^p )^{1/p}`. Use the seed-pooling shape, not a data-dependent query: set the seed `s = 0`, let a front row-wise feed-forward map realize `z ↦ z^p`, and let the value projection in each one-dimensional head select one coordinate. To force the attention weights to be uniform, set the query projection and its bias to zero, or otherwise make every query-key logit equal; then every head is exactly the zero-query mean above. The fixed zero seed and zero query projection mean the residual around attention contributes no data-dependent term. A back row-wise feed-forward map can realize `u ↦ u^{1/p}` coordinate by coordinate, so the decoder can form `(mean z^p)^{1/p}`. The important correction is that merely projecting query and key onto the same coordinate would *not* make a mean; it would make data-dependent logits. I need the zero-logit construction.

Now the key one: `PMA` can express plain sum pooling, but not with the default softmax normalization. Softmax with a zero seed gives the mean above. For the expressivity construction I use the more general attention activation `ω(·) = 1 + f(·)` with `f(0)=0`, such as identity, ReLU, or a centered sigmoid. Then with a zero query every attention logit is `0`, so `ω(0) = 1`: every value gets weight exactly one, and the output is `Σ_i z_i`. So the attention-pooling family contains sum pooling when I choose a non-normalizing activation for this proof.

Finally, Zaheer et al. proved `rFF(sum(rFF(·)))` is a universal approximator of permutation-invariant functions. So: in every `SAB` and `ISAB`, suppress the attention contribution. In the full multi-head definition that can be done by setting the output-mixing matrix `W^O = 0`; in the simplified implementation-style block, the equivalent is to zero the value/attention path. The residual connections, layer norms, and row-wise feed-forward maps are all per-row, so the encoder collapses to an instance-wise map `Z = rFF(X)`. The decoder can do `rFF(sum(Z))` by the non-softmax sum-pooling construction. Composing, the model can realize `rFF(sum(rFF(X)))`, which is universal. So adding attention costs nothing in representational reach for the broader attention family; attention is not *needed* for universality, but it is the mechanism that makes interaction-heavy tasks learnable in practice, since the pooling baseline that's universal in theory under-fits in practice. The practical module I will write below uses the robust softmax version, while the exact sum construction is the theoretical fallback showing the wider family contains Deep Sets.

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

Tracing the chain back: I needed a permutation-invariant, size-agnostic set model that also captures inter-element interactions, which the per-element-encode-then-fixed-pool baseline structurally cannot. Attention gives the right symmetries — invariant in its keys/values, equivariant in its queries — so I built a `MAB` (a Transformer encoder block with positional encoding removed because a set has no order), used self-attention `SAB = MAB(X,X)` to make the encoder model pairwise and, when stacked, higher-order interactions, replaced its quadratic cost with `ISAB`'s inducing-point bottleneck (`O(nm)`), and replaced fixed pooling with `PMA`, learnable seed-driven attention pooling that weights elements by content and, with `k` seeds plus a following `SAB`, emits `k` jointly-reasoned outputs. Mean and power means sit inside the softmax attention version, and exact sum pooling sits inside the wider non-normalizing attention family, so the family stays a universal approximator of permutation-invariant functions while gaining the interaction modeling the pooling baseline lacked.
