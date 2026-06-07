OK, let me think about what I actually have. The pair representation `z_ij` is a vector for every ordered pair of residues `(i, j)`. Semantically it's supposed to carry the *relation* between `i` and `j` — eventually their distance and relative orientation, because the thing I read it out into is a distogram: a predicted distribution over how far apart `i` and `j` are. So `z` is, really, an edge feature on the complete graph whose nodes are the residues. It gets born from a relative-position encoding plus an outer sum of the per-residue embeddings, and it keeps getting refreshed from the alignment by the outer-product mean, which pours coevolutionary signal into each `(i, j)`. Fine. The question is how to *update* `z` block after block so it gets better.

The dumb thing to do is update each edge on its own — run a little MLP on `z_ij`, or do attention along one residue axis. Let me actually picture the axial option, because that's the strong baseline. Axial attention on a 2D grid: for a fixed row `i`, let `z_ij` attend over all the other entries in that row, `z_ik` for every `k`; then for a fixed column `j`, let `z_ij` attend over `z_kj` for every `k`. That's cheap — `O(N)` interactions per axis instead of `O(N²)` over all pairs of cells — and it's exactly the cost reduction that makes row/column mixing on large alignment-like grids tractable. So why isn't it enough?

Stare at what the edge `(i, j)` is allowed to see. Row attention lets it mix with `(i, k)`: same starting residue `i`, sweeping the other end. Column attention lets it mix with `(k, j)`: same ending residue `j`, sweeping the other end. But here's the thing: at no point does it put the two sides connected to the same third residue into the same message. It can see `(i, k)` in one operation and `(k, j)` in another, but it does not bind them as a triangle indexed by the same `k`. Those two sides are precisely the ones that constrain `(i, j)`. Why? Because of geometry.

Let me make sure I'm not hand-waving. The pair map is only worth anything if the distances it implies can actually be realized by points in 3D. And an arbitrary symmetric matrix of "distances" is almost never embeddable in three dimensions; metric geometry does not let me choose every `d_ij` independently. The very first necessary condition is the triangle inequality: for any three residues `i, j, k`, the distance `d_ij` can't exceed `d_ik + d_kj`. Higher-order Euclidean distance-matrix constraints make the story stricter, but the triangle inequality is the cleanest local piece. So `d_ij` is not a free number. It is squeezed by the two other sides of the triangle `{i, j, k}`, and this is true for every third residue `k`. The edge `(i, j)` sits in `N - 2` triangles, and each one gives it a constraint.

Now I can say cleanly why axial attention has a blind spot: it can route information along a triangle's sides one side at a time, but it never closes the triangle in a single message. To use the constraint on `(i, j)`, I need the two sides connected to `k` present together, combined, and then aggregated over `k`. Row attention has `(i, k)`. Column attention has `(k, j)`. Neither directly conditions on the pair tied to the same `k`. That's the gap. I need an update where the message into edge `(i, j)` is built out of the other two sides of each triangle, aggregated over the third node.

So the shape of the operation is forced: for target edge `(i, j)`, walk every third residue `k`, grab the two directed tensor entries that represent the other two sides of the triangle, combine them into a message, and sum those messages over all `k`. This is just message passing, but on a triangle: an edge gets messages routed through a third node. The only thing left undecided is the combine function and the aggregation, and I want both to be cheap, because this is going to run inside a deep stack on sequences of hundreds to thousands of residues, and any "for all `k`" already costs me a factor of `N` beyond the `N²` edges, so `O(N³)`, and I must not make the constant worse than it has to be.

What's the cheapest symmetric way to take two `c`-dimensional edge vectors, combine them, and sum over `k`? My first instinct is attention again: compute a content-based weight between the two sides and do a weighted sum. But that's expensive and I haven't even established I need content selection yet; I just need the two sides to meet. Because the tensor is directed, the geometric side between `j` and `k` can be read as `z_jk` or `z_kj`. Start with the `z_jk` readout. The frugal move is a bilinear one: form `a_{ik} ⊙ b_{jk}`, an elementwise product of two learned projections, and sum it over `k`. Channel `c` of the message into `(i, j)` is `Σ_k a_{ik,c} · b_{jk,c}`. That's an outer-product-style contraction, the same flavor as the outer-product mean that already feeds `z` from the MSA, so it's a natural primitive here. It's one `einsum`, it's `O(N³ c)`, and it puts two sides of every directed triangle into every message. Let me write it as the contraction it is:

`update_ij = Σ_k a_{ik} ⊙ b_{jk}`,  i.e. the einsum `ikc, jkc -> ijc`.

Wait — I need to be careful about *which* two edges I'm pairing, because `z` is not symmetric. `z_ij ≠ z_ji` in general (it carries directed relational info), so the choice of which index of each incident edge is the shared apex `k` actually matters. Let me look hard at the indices in `Σ_k a_{ik} ⊙ b_{jk}`. The two edges I'm combining are `(i, k)` and `(j, k)`. In both of those, `k` is the *second* index — `k` is the common *target*, while `i` and `j` are the two *sources* that point at the same `k`. Picture it: from node `i` an edge goes out to `k`, from node `j` an edge goes out to `k`; both edges *leave* their source and share their destination `k`. Call this the "outgoing" pairing — `i` and `j` are joined through a `k` they both emit toward.

But geometry doesn't privilege "both point at `k`." I could equally pair the two edges that *come from* a shared `k`: `(k, i)` and `(k, j)`, where now `k` is the common *source* and `i, j` are the two targets. That's a different contraction:

`update_ij = Σ_k a_{ki} ⊙ b_{kj}`,  einsum `kjc, kic -> ijc`.

Call it "incoming": both edges enter `i` and `j` from a shared origin `k`. Because `z` is directed, these two contractions read genuinely different entries of the pair map. Outgoing pulls from the `(i, .)` and `(j, .)` rows; incoming pulls from the `(. , i)` and `(. , j)` columns. They are not the same operation, and neither alone covers both orientations of the triangle relative to the directed edge. The fix is obvious once I see they're complementary: do both, as two consecutive updates in the block. Outgoing then incoming. Together they give the edge `(i, j)` both directed ways to read the third-residue evidence.

Now, raw `Σ_k a_{ik} ⊙ b_{jk}` has problems I should fix before it's usable. First, magnitude: I'm summing `N` products. If the products are roughly centered and weakly correlated, the variance of the sum grows with `N`, so the standard deviation grows like `sqrt(N)`; if they have a nonzero mean, the mean grows like `N`. Either way the message scale is at the mercy of sequence length. I need to renormalize after the sum. A LayerNorm on the contracted result does exactly that: it centers and scales the aggregated message back to a controlled magnitude regardless of how many `k` contributed. So there are going to be two normalizations, one on the input `z` before I project it to keep the projections well-conditioned, and one on the summed result, the center norm, to kill the length dependence. Let me not forget the input one either: I LayerNorm `z_ij` first, then form `a` and `b` from the normalized thing.

Second problem: the "for all `k`" is indiscriminate. The hard padding mask can remove nonexistent residues, but it cannot know which real residue pairs carry useful evidence. The soft, learnable stand-in is a gate: multiply each projected edge by a per-channel sigmoid gate before the contraction, so the network can learn to damp edges that shouldn't carry signal into the product. So `a_{ij} = sigmoid(Linear(z_ij)) ⊙ Linear(z_ij)` and likewise `b_{ij} = sigmoid(Linear(z_ij)) ⊙ Linear(z_ij)`, each gate and each projection its own linear map. The gate is a differentiable selection knob on the edge features.

Third: once I have a clean aggregated, normalized message and project it back up to `c_z` with a `Linear`, should I just dump it into `z_ij`? No — some edges `(i, j)` are already confident (strong coevolution signal, say) and I don't want a triangle aggregate to stomp on them, while others are uncertain and should absorb a lot. So gate the *output* too: `g_ij = sigmoid(Linear(z_ij)) ∈ ℝ^{c_z}`, and the final update is `g_ij ⊙ Linear(centernorm(Σ_k a_{ik} ⊙ b_{jk}))`. The output gate lets the target edge decide how much of the triangle-routed update it actually takes — and since the whole thing is added residually, that's exactly the knob you want: confident edges set `g` low.

Let me assemble the outgoing version end to end and check it makes sense. I start with `z_ij ← LayerNorm(z_ij)` to condition the input. From that normalized edge I make two gated hidden projections, `a_ij = sigmoid(Linear(z_ij)) ⊙ Linear(z_ij)` and `b_ij = sigmoid(Linear(z_ij)) ⊙ Linear(z_ij)`, both in `ℝ^c`. I'll take `c = 128` here; this multiplicative update is the cheap `O(N³)` workhorse of the pair stack, so it earns the wide hidden width. I also make the output gate `g_ij = sigmoid(Linear(z_ij)) ∈ ℝ^{c_z}`. The outgoing update is then `z̃_ij = g_ij ⊙ Linear(LayerNorm(Σ_k a_{ik} ⊙ b_{jk}))`.

Incoming is identical except step 4 sums `a_{ki} ⊙ b_{kj}`. And in the block these are two residual sub-layers, outgoing first, then incoming.

Now, is the multiplicative update enough? Look at what it can and can't do. It puts the two other directed sides into every message, so it has the right inputs for triangle consistency. But every `k` contributes through a fixed, content-blind path: the only thing that down-weights a `k` is the per-edge gate on `a` and `b`, which is a function of those edges alone, not of how well edge `(i, k)` actually matches edge `(i, j)`. There's no mechanism for edge `(i, j)` to say "residue `k` is especially informative for me specifically." For some triangles I'd want the update to concentrate on the few `k` that are geometrically decisive and ignore the rest. That's a content-dependent, query-dependent selection, which is exactly what attention does and the Hadamard sum does not.

So I want a second operation: let `(i, j)` *attend* over the third node, choosing which `k` matter by query–key similarity, while still feeling the third edge of the triangle. Plain row attention already lets `(i, j)` attend over `(i, k)` (same starting node `i`) — query from the central edge `ij`, keys and values from the left edges `ik`. But that's the axial baseline again: it couples only two sides of the triangle, `(i, j)` and `(i, k)`. The missing side is `(j, k)` — the edge between the two "other" endpoints. So inject it as a bias on the attention logit: project the third edge `z_{jk}` to a scalar (per head) and *add* it to the `q·k` affinity. Now the decision of how much `k` matters is made over the *whole* triangle — `q^h_{ij}·k^h_{ik}` brings in sides `ij` and `ik`, and `+ b^h_{jk}` injects the missing side `jk`:

`a^h_{ijk} = softmax_k( (1/√c) q^h_{ij}·k^h_{ik} + b^h_{jk} )`,
`o^h_{ij} = g^h_{ij} ⊙ Σ_k a^h_{ijk} v^h_{ik}`,  then `z̃_ij = Linear(concat_h o^h_{ij})`.

The `1/√c` is the usual dot-product temperature so the softmax doesn't saturate when `c` is moderate; here `c = 32` per head, `4` heads. And again there's a sigmoid output gate `g^h_{ij}` on the per-head value sum, the same "how much should this edge absorb" knob as before. This is the "starting node" version: every edge in the attention shares the starting residue `i` (the central edge `ij` and the left edges `ik` both start at `i`), and the bias edge `jk` closes the triangle. Its sibling shares the *ending* node: query `ij`, keys/values from `kj` (sharing the ending residue `j`), bias from the edge `ki`, i.e. `a^h_{ijk} = softmax_k( (1/√c) q^h_{ij}·k^h_{kj} + b^h_{ki} )` with values `v^h_{kj}`. Same reason as before for doing both — one orientation of the shared node isn't enough on a directed pair map; do starting-node then ending-node.

Why keep both families, the multiplicative update and the attention, rather than just the richer attention? Because they trade off. The multiplicative update is cheap, fully symmetric in how it treats the `k`'s because there is no query, and it's the natural frugal contraction; it gives every block a broad triangle-shaped mixing pass at low constant cost. The attention is the content-routed refinement that can sharpen onto the decisive `k`, but it's the more expensive operation, so it runs at the narrower width (`c = 32`, 4 heads) while the multiplicative workhorse runs wide (`c = 128`). Stacking them, two multiplicative updates, then two attentions, then a transition MLP, gives both the cheap symmetric consistency pass and the selective refinement. The order is deliberate: broad triangle mixing first, content routing next, then a 4x-wide transition MLP digests the result.

Let me sanity-check the attention's triangle-closure claim once more, because it's the subtle part. In ordinary self-attention on the pair map, the logit for "how much edge `ik` informs edge `ij`" would be `q_{ij}·k_{ik}` — a function of edges `ij` and `ik` only. Two sides of triangle `{i, j, k}`. The third side `jk` is invisible, so the attention can't tell apart two different `k` that have identical `ik` edges but wildly different `jk` edges — yet geometrically those are completely different triangles. Adding `+ b^h_{jk}` is precisely what breaks that degeneracy: now the affinity depends on all three sides, so the model attends to a `k` based on the full triangle it forms, not on a single edge. That's the whole point, and it's why this is *triangular* attention rather than vanilla axial attention with a bias bolted on for no reason.

One more shape worry on the attention: the bias `b^h_{jk}` ranges over `(j, k)`. For fixed starting node `i`, query index `j`, and key index `k`, the bias is read from the `j`-th row and `k`-th column of the per-head scalar projection of `z`. The attention itself is row attention on `z`, so implementation-wise the bias has shape `[*, 1, H, J, K]`: it broadcasts over the starting node `i`, while its two residue axes align with the query residue `j` and key residue `k`. For the ending-node version everything transposes: operate on columns of `z`, which can be implemented as a `transpose(-2, -3)` before the same row-attention machinery and a transpose back after. Same operation, mirror index.

Now to the code. I'll mirror a clean implementation of both operators. The multiplicative update first.

```python
import torch
import torch.nn as nn
from functools import partialmethod

# Linear, LayerNorm, Attention, DropoutRowwise, DropoutColumnwise,
# PairTransition, permute_final_dims, and is_fp16_enabled are trunk primitives.


class TriangleMultiplicativeUpdate(nn.Module):
    """Update z_ij from the other two edges of every triangle {i,j,k}.
    _outgoing=True : message Σ_k a_ik ⊙ b_jk  (i,j share the target k)
    _outgoing=False: message Σ_k a_ki ⊙ b_kj  (i,j share the source k)
    """
    def __init__(self, c_z, c_hidden, _outgoing=True):
        super().__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden          # c = 128: the wide, cheap workhorse
        self._outgoing = _outgoing

        self.layer_norm_in = LayerNorm(c_z)        # condition the input z
        self.layer_norm_out = LayerNorm(c_hidden)  # renormalize the sum over k

        self.linear_a_p = Linear(c_z, c_hidden)                  # left projection
        self.linear_a_g = Linear(c_z, c_hidden, init="gating")   # left edge gate
        self.linear_b_p = Linear(c_z, c_hidden)                  # right projection
        self.linear_b_g = Linear(c_z, c_hidden, init="gating")   # right edge gate

        self.linear_z = Linear(c_hidden, c_z, init="final")      # project message up
        self.linear_g = Linear(c_z, c_z, init="gating")          # output gate g_ij
        self.sigmoid = nn.Sigmoid()

    def _combine_projections(self, a, b):
        # a, b: [*, N, N, c]. Contract over the shared node k.
        # outgoing -> 'ikc,jkc->ijc' ; incoming -> 'kjc,kic->ijc'
        if self._outgoing:
            a = permute_final_dims(a, (2, 0, 1))   # [*, c, i, k]
            b = permute_final_dims(b, (2, 1, 0))   # [*, c, k, j]
        else:
            a = permute_final_dims(a, (2, 1, 0))   # [*, c, k, i]
            b = permute_final_dims(b, (2, 0, 1))   # [*, c, k, j]
        p = torch.matmul(a, b)                     # sum over k -> [*, c, i, j]
        return permute_final_dims(p, (1, 2, 0))    # [*, i, j, c]

    def forward(self, z, mask=None):
        # z: [*, N, N, c_z]
        if mask is None:
            mask = z.new_ones(z.shape[:-1])
        mask = mask.unsqueeze(-1)

        z = self.layer_norm_in(z)                              # step 1
        a = mask * self.sigmoid(self.linear_a_g(z)) * self.linear_a_p(z)  # gated left
        b = mask * self.sigmoid(self.linear_b_g(z)) * self.linear_b_p(z)  # gated right
        if is_fp16_enabled():
            a_std = a.std()
            b_std = b.std()
            if a_std != 0.0 and b_std != 0.0:
                a = a / a_std
                b = b / b_std
            with torch.cuda.amp.autocast(enabled=False):
                x = self._combine_projections(a.float(), b.float())
        else:
            x = self._combine_projections(a, b)                # Σ_k a ⊙ b
        x = self.layer_norm_out(x)                             # kill N-dependence
        x = self.linear_z(x)                                   # back to c_z
        g = self.sigmoid(self.linear_g(z))                     # output gate g_ij
        return x * g                                           # step 4


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):   # Σ_k a_ik ⊙ b_jk
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)

class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):   # Σ_k a_ki ⊙ b_kj
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)
```

The `permute → matmul → permute` is just how the `einsum` over `k` is realized efficiently: stack `c` as a batch dimension and do a plain matrix multiply `[i,k]·[k,j]→[i,j]`, which *is* `Σ_k a_{ik}·b_{jk}` per channel. The outgoing/incoming difference is entirely in which axis gets put where before the matmul — outgoing contracts the second index of both edges, incoming the first. The mask zeros out padded residues so they never enter the sum over `k`.

Now the attention sibling. The whole triangle-closure lives in one extra additive bias derived from `z` itself.

```python
class TriangleAttention(nn.Module):
    """starting=True : attend over edges ik (share start i), bias from edge jk.
       starting=False: attend over edges kj (share end j),   bias from edge ki."""
    def __init__(self, c_in, c_hidden, no_heads, starting=True, inf=1e9):
        super().__init__()
        self.starting = starting
        self.inf = inf
        self.layer_norm = LayerNorm(c_in)
        # per-head scalar bias read off the pair map -> closes the triangle
        self.linear = Linear(c_in, no_heads, bias=False, init="normal")
        # standard gated multi-head attention with the 1/sqrt(c) scaling and sigmoid gate
        self.mha = Attention(c_in, c_in, c_in, c_hidden, no_heads)

    def forward(self, x, mask=None):
        # x: [*, I, J, c_in]
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        if not self.starting:                 # ending-node = operate on columns
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        x = self.layer_norm(x)
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]   # mask padded keys
        # the third-edge bias: project z to a per-head scalar, broadcast as a logit
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1)).unsqueeze(-4)
        biases = [mask_bias, triangle_bias]   # logit = q·k/sqrt(c) + bias_jk + mask

        x = self.mha(q_x=x, kv_x=x, biases=biases)
        if not self.starting:
            x = x.transpose(-2, -3)
        return x


TriangleAttentionStartingNode = TriangleAttention

class TriangleAttentionEndingNode(TriangleAttention):
    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
```

Inside `Attention`, the per-head logit is `(1/√c) q^h·k^h` plus the additive `biases`; the `triangle_bias` is the `b^h_{jk}` term, and the module also applies the `sigmoid` output gate `g^h_{ij}` and the final `concat_h → Linear` back to `c_z`. So the attention's logit is exactly `(1/√c) q^h_{ij}·k^h_{ik} + b^h_{jk}`, with the bias supplying the edge that vanilla row attention would never see.

And they sit together in the pair stack like this, each a residual sub-layer. The ending-node attention gets columnwise dropout in the original orientation; one PyTorch implementation obtains the same effect by transposing, applying rowwise dropout, and transposing back.

```python
class PairStack(nn.Module):
    def __init__(self, c_z, pair_dropout=0.25):
        super().__init__()
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, 128)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, 128)
        self.tri_att_start = TriangleAttentionStartingNode(c_z, 32, 4)
        self.tri_att_end = TriangleAttentionEndingNode(c_z, 32, 4)
        self.pair_transition = PairTransition(c_z, n=4)
        self.dropout_row = DropoutRowwise(pair_dropout)
        self.dropout_col = DropoutColumnwise(pair_dropout)

    def forward(self, z, pair_mask):
        z = z + self.dropout_row(self.tri_mul_out(z, mask=pair_mask))
        z = z + self.dropout_row(self.tri_mul_in(z, mask=pair_mask))
        z = z + self.dropout_row(self.tri_att_start(z, mask=pair_mask))
        z = z + self.dropout_col(self.tri_att_end(z, mask=pair_mask))
        z = z + self.pair_transition(z, mask=pair_mask)
        return z
```

Causal chain, start to finish: `z_ij` is an edge feature, and a set of edges is only physical if every triple obeys the triangle inequality, so the constraint on `(i, j)` lives in the other two sides of each triangle; axis-local updates can see those sides only separately. Routing a message into `(i, j)` from two directed reads of those sides, summed over the third residue `k`, forces a third-node contraction; the cheapest symmetric form is the gated Hadamard product `Σ_k a ⊙ b`, in two flavors, share the target `k` and share the source `k`, because the pair map is directed. A center LayerNorm tames the `N`-term sum and an output gate controls absorption. Because that contraction is content-blind, a triangular attention sits on top: `(i, j)` picks the decisive `k` by `q·k` similarity while a bias `b_{jk}` or `b_{ki}` injects the missing third edge so the choice is made over the whole triangle. Two multiplicative updates, two attentions, a transition: each block nudges the pair representation toward a geometry that can actually be built in 3D.
