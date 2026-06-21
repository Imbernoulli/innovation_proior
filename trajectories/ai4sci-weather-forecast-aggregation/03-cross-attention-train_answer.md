The learned weighted sum did what I bet it would and then one thing I didn't, and the surprise is the most informative part. Against the mean's 353.50 / 2.6032 / 3.3991, the global per-variable softmax landed 345.79 / 2.5934 / 3.4245 — it beat the mean on z500-3day (the largest absolute drop, exactly where I predicted the headroom) and shaved t850-5day, confirming the core diagnosis that per-variable contributions are *not* uniform. But on wind10m-7day it went the *wrong* way, 3.3991 → 3.4245, slightly worse than even uniform weighting. That regression is the tell. The weighting the weighted sum learns is *fixed*: a single global distribution over the 48 variables, one compromise that has to serve z500 at 3 days, t850 at 5 days, *and* 10 m wind at 7 days at once. Whatever split most reduces the geopotential and temperature loss is apparently a hair worse for the longest-lead wind target — the global weighting overfit to the targets where it helped and mis-served the one where the right variables differ. A content- and location-independent weighting cannot be right everywhere at once; the weighting needs to *react* rather than be legislated. And both lower rungs share a second defect: they just rescale-and-add the *raw* variable tokens — a geopotential token, a humidity token, and a wind token are differently grounded objects, and adding them even with good weights hands the backbone a muddied vector.

So the question is no longer "which variables matter on average" — the weighted sum answered that — but "which variables matter *here*, given what the tokens at this location currently say." I want a reduction whose weights are content-dependent and whose values are re-expressed in a shared space before mixing. Writing down what that forces at one location keeps circling the same object. I want output $= \sum_v \alpha_v \cdot (\text{something I do to token } v)$ where the weights $\alpha_v$ are a *function of the tokens themselves*, normalized to a convex combination so the output stays a proper pooling on the single-token scale (the scale discipline the mean taught me and the weighted sum kept) and accepts any $V$. "Normalized, data-dependent convex combination over a set" is the shape of a softmax; "the weight comes from how well each token matches some reference" is a compatibility score. That is **cross-attention**, and it is not one option among several — it is the thing that matches every requirement the wind regression handed me at once: react to content, stay a convex combination, re-project the values.

I propose the learnable-query cross-attention aggregator (the ClimaX default). Keys and values are the $V$ variable tokens at a location with learned projections, so values become $W^V x_v$ — there is the re-projection into a shared space, the muddied-vector fix for free — and keys become $W^K x_v$. The query is the subtle part. In a sequence model each output position carries its own query, but here I want exactly *one* output token per location summarizing the whole set, and there is no natural source for it: the output is not a transformed version of any particular variable token, it is a summary of all of them. So I let the query be a single **learnable** vector, reused at every spatial location and example, that the network trains to ask the right question of the variable set — the same trainable-query-summarizes-a-set move a ViT's class token makes over the spatial set, pointed here at the variable axis. Per head with $d_k = D/\text{num\_heads}$,
$$\alpha_v = \mathrm{softmax}_v\!\left(\frac{(W^Q q)\cdot(W^K x_v)}{\sqrt{d_k}}\right), \qquad \text{head} = \sum_v \alpha_v\,(W^V x_v),$$
the heads concatenated and passed through $W^O$. The softmax runs over the *set* of variable tokens, so it is invariant to their order and defined for any $V$ — exactly the set semantics the contract demands.

Two pieces are load-bearing. The $1/\sqrt{d_k}$ is not decoration: for query/key components roughly independent with unit variance, the dot product $\sum_{i=1}^{d_k} q_i k_i$ has variance $d_k$, so its magnitude grows like $\sqrt{d_k}$; left unscaled, large logits push the softmax toward one-hot, where its Jacobian collapses and the gradient through the attention weights becomes tiny. Dividing by $\sqrt{d_k}$ keeps the logits unit-variance and the softmax responsive — and responsiveness is the whole point of this rung, since the wind regression was caused by a weighting that *couldn't* respond. The multi-head choice is the other one: a single query-key softmax produces one weighting pattern over the variables, one answer to "which variables matter." But that question has several simultaneous answers — the thermodynamic story and the dynamical story are different — and a single softmax forces one compromise, exactly the compromise that sank wind10m at the previous rung. Several heads, each with its own projections to a $d_k = D/\text{num\_heads}$ subspace, keep distinct "which variables matter for *this* aspect" patterns distinct instead of averaging them into one; $W^O$ mixes the heads' subspaces back into one $D$-vector. Multi-head is the structural antidote to the one-compromise-weighting failure I measured.

One query already turns the set into exactly one token per location — the $[B, L, D]$ the contract wants — and a single cross-attention layer already does it; stacking more would re-attend a length-one sequence into the variable set again for little gain. So one layer, one query: minimal. The cost ledger is favorable, which matters inside an already-large fine-tuned backbone: the reduction is $O(V)$ per location, linear in $V$ like the weighted sum, and crucially it does *not* reintroduce the $O((V\!\cdot\!h\!\cdot\!w)^2)$ blowup that motivated aggregation in the first place — that came from running the backbone's self-attention over the full variable-and-space sequence, whereas here the cross-attention is a tiny softmax over $V$ keys with one query and the backbone still sees only the $h\!\cdot\!w$-length aggregated sequence.

Initialization closes the loop on the whole ladder. A learnable query starting badly could corrupt the pretrained ClimaX features the lower rungs respected, but the lower rungs handed me the discipline: zero-initialize the query, and with the attention biases at zero every score is $(W^Q\!\cdot\!0)\cdot(W^K x_v)/\sqrt{d_k} = 0$, so at step zero the softmax over the $V$ variables is uniform — the layer begins with the mean's equal-weighting pattern applied to the projected value tokens, the safe prior the pretrained backbone expects, and gradient descent specializes the query from there. This also exhibits the two lower rungs as degenerate cases of this one: the mean is this layer with the softmax forced uniform ($\mathrm{softmax}(0)$) and identity projections; the learned weighted sum is this layer with logits that are learned *constants* rather than functions of the tokens; the full cross-attention is the same machine with the weights free to depend on the data and the values re-projected. The simpler things fall out of it, which is the test that I picked the right object — and the diagnosis is clean: the weighted sum's wind regression was the cost of frozen logits, and this rung unfreezes them.

The shapes follow from treating every (example, location) pair as an independent $V$-element set: permute $x: [B, V, L, D]$ to $[B, L, V, D]$ to bring the variable axis next to $D$, fold $B$ and $L$ together into $[B\!\cdot\!L, V, D]$, expand the one learnable query $[1, 1, D]$ to $[B\!\cdot\!L, 1, D]$, run multi-head cross-attention with that query against the $V$ tokens as keys and values to get $[B\!\cdot\!L, 1, D]$, drop the length-one axis to $[B\!\cdot\!L, D]$, and unfold back to $[B, L, D]$. In the scaffold this is a single `nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)` plus one `nn.Parameter` query, with $D = 1024$ and $\text{num\_heads} = 16$ so $d_k = 64$ per head — the standard head width. $V$ never enters the module's weights; it is read from the input shape, which is the entire point of the contract.

The sharp, falsifiable test against the two runs I have is wind10m-7day: that was the regression where the weighted sum's 3.4245 was worse than even the mean's 3.3991, so if content-dependent multi-head attention is the right diagnosis, wind10m should drop back *below the mean's floor* 3.3991, not merely below 3.4245. On z500-3day and t850-5day, where the global split already helped, I expect content dependence to push further below 345.79 and 2.5934, with z500 again showing the largest absolute gain since it carries the most headroom and leans hardest on a state-dependent dynamical subset.

```python
# EDITABLE region of custom_forecast.py (lines 310-351) — step 3: ClimaX cross-attention
class VariableAggregator(nn.Module):
    """Cross-attention variable aggregation (ClimaX default).

    A learnable query token attends to all V variable tokens at each spatial
    location via multi-head cross-attention, producing one token per location.

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
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        """
        Args:
            x: [B, V, L, D] — per-variable patch embeddings.
        Returns:
            [B, L, D] — aggregated representation.
        """
        b, v, l, d = x.shape
        x = x.permute(0, 2, 1, 3)   # B, L, V, D
        x = x.reshape(b * l, v, d)  # B*L, V, D

        query = self.var_query.expand(b * l, -1, -1)  # B*L, 1, D
        out, _ = self.var_agg(query, x, x)             # B*L, 1, D
        out = out.squeeze(1)                            # B*L, D

        out = out.reshape(b, l, d)  # B, L, D
        return out
```
