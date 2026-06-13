The learned weighted sum did what I bet it would, and then it did one thing I didn't, and the surprise is
the most informative part. Against the mean's 353.50 / 2.6032 / 3.3991, the global per-variable softmax
landed 345.79 / 2.5934 / 3.4245 — it beat the mean on z500-3day (353.50 → 345.79, the largest absolute
drop, exactly where I predicted the headroom was) and shaved t850-5day (2.6032 → 2.5934), confirming the
core diagnosis: the per-variable contributions are *not* uniform, a fixed non-uniform split helps, and the
gain is biggest on the geopotential target whose evolution leans on a few dynamical fields. So the
"variables matter unequally" hypothesis is validated. But on wind10m-7day it went the *wrong* way:
3.3991 → 3.4245, very slightly worse than the mean. That regression is the tell I have to read carefully,
because it is precisely the failure mode I flagged when I built that rung — the weighting it learns is
*fixed*. Once trained, variable `v` gets the same share at every grid cell and in every atmospheric state,
no matter what its token says there. A single global distribution over the 48 variables is one compromise
weighting that has to serve z500 at 3 days, t850 at 5 days, *and* 10 m wind at 7 days at once. Whatever
fixed split most reduces the geopotential and temperature loss is apparently a hair *worse* for the
longest-lead wind target than even uniform weighting — the global split overfit to the targets where it
helped and mis-served the one where the right variables differ. That is the wall: a content- and
location-independent weighting cannot be right everywhere at once, and the wind regression proves the
weighting needs to *react* rather than be legislated.

So the question is no longer "which variables matter on average" — the weighted sum answered that — but
"which variables matter *here*, given what the tokens at this location currently say." I want a reduction
whose weights are content-dependent: the weight on each variable token decided by the token contents at
that location, not by a global parameter. And while I'm at it, I should fix the second defect both lower
rungs share and that I noted at the very bottom: the mean and the weighted sum both just rescale-and-add
the *raw* variable tokens. A geopotential token, a humidity token, and a wind token are differently
grounded objects; adding them, even with good weights, hands the backbone a muddied vector. A good
reduction should re-express the tokens in a shared space before mixing them. Let me write down what a
content-dependent, re-projecting reduction looks like at one location and see what it forces.

At one location I have `V` value vectors and I want output `= Σ_v α_v · (something I do to token v)`, where
the weights `α_v` are a *function of the tokens themselves* and normalized to a convex combination so the
output stays a proper pooling on the single-token scale (the scale discipline the mean taught me and the
weighted sum kept — I will not give that up) and accepts any `V`. "Normalized, data-dependent convex
combination over a set" is the shape of a softmax. "The weight comes from how well each token matches some
reference" is a compatibility score. I keep circling the same object: this is attention. The output is a
weighted average of values; the weights are a softmax of query-key compatibilities; the softmax runs over
the *set* of variable tokens so it is invariant to their order and defined for any `V` — the exact set
semantics the contract demands. Attention is not one option among several here; it is the thing that
matches every requirement the wind regression just handed me — react to content, stay a convex
combination, re-project the values — at once.

Attention needs a query, keys, and values. Keys and values are obvious: the `V` variable tokens at this
location, with learned projections, so values become `W^V x_v` (there is the re-projection into a shared
space, for free — the muddied-vector fix) and keys become `W^K x_v`. The query is the subtle part. In a
sequence model each output position has its own query from somewhere. Here I want exactly *one* output
token per location summarizing the whole set, and there is no natural source for it — the output isn't a
transformed version of any particular variable token, it's a summary of all of them. So let the query be a
free parameter: a single learnable vector, reused at every spatial location and every example, that the
network trains to ask the right question of the variable set. One trainable query, cross-attending into
the `V` keys and values. (This is the trainable-query-summarizes-a-set pattern — the same move a ViT's
class token makes over the spatial set, pointed here at the variable axis.) Concretely, per head with
`d_k = D / num_heads`, `score_v = (W^Q q)·(W^K x_v)/√d_k`, `α = softmax_v(score)`, and the head output is
`Σ_v α_v (W^V x_v)`; the heads concatenate and pass through `W^O`.

Two pieces of that are load-bearing and I should justify them rather than copy them. The `1/√d_k` is not
decoration: for query/key components roughly independent with unit variance, the dot product
`Σ_{i=1}^{d_k} q_i k_i` has variance `d_k`, so its magnitude grows like `√d_k`; left unscaled, large
logits push the softmax toward one-hot, where its Jacobian collapses and the gradient through the
attention weights becomes tiny. Dividing by `√d_k` keeps the logits unit-variance and the softmax
responsive — and "responsive" is the whole point of this rung, since the wind regression was caused by a
weighting that *couldn't* respond. The multi-head choice is the other one: a single query-key softmax
produces one weighting pattern over the variables — one answer to "which variables matter." But "which
variables matter" has several simultaneous answers — the thermodynamic story and the dynamical story are
different, and a single softmax forces one compromise, which is exactly the compromise that sank wind10m at
the previous rung. Several heads, each with its own projections to a `d_k = D/num_heads` subspace, keep
distinct "which variables matter for *this* aspect" patterns distinct instead of averaging them into one;
`W^O` then mixes the heads' subspaces back into one `D`-vector. Multi-head is the structural antidote to
the one-compromise-weighting failure I just measured.

How many such layers, and how many queries? One query already turns the set into exactly one token per
location — exactly the `[B, L, D]` the contract wants — and a single cross-attention layer already does
it. Stacking more would re-attend a length-one sequence into the variable set again, buying little for the
cost. So one layer, one query: minimal. And the cost ledger is favorable, which matters because this sits
inside an already-large fine-tuned backbone. The reduction is `O(V)` per location, `O(V·h·w)` total —
linear in `V`, like the weighted sum — and crucially it does *not* reintroduce the `O(V²)` blowup that
motivated aggregation in the first place: that blowup came from running the backbone's self-attention over
the full `V·h·w` sequence; here the cross-attention is a tiny softmax over `V` keys with a single query,
and the backbone still sees only the `h·w`-length aggregated sequence. So I get content-dependent,
re-projecting, multi-head mixing at linear-in-`V` cost.

Now initialization, because a learnable query starting badly could corrupt the pretrained ClimaX features
the lower rungs were careful to respect — and the lower rungs handed me the right discipline. If I
zero-initialize the query and the attention biases start at zero, every score is
`(W^Q·0)·(W^K x_v)/√d_k = 0`, so at step zero the softmax over the `V` variables is uniform — the layer
begins with the same equal-weighting pattern as the mean, applied to the projected value tokens. That is
the safe prior I want for a fine-tune-from-pretrained run: before learning anything, "treat all variables
equally" is the right default, and it is the exact starting point the mean and the zero-initialized
weighted sum both used and that produced sensible numbers. From there gradient descent specializes the
query toward content-dependent pooling. This also closes the loop on the whole ladder: the mean is this
layer with the softmax forced uniform (`softmax(0)`) and identity projections; the learned weighted sum is
this layer with logits that are learned *constants* rather than functions of the tokens; the full
cross-attention is the same machine with the weights set free to depend on the data and the values
re-projected. The two rungs I climbed are the degenerate cases of this one — which is the test that I
picked the right object: the simpler things fall out of it, they don't sit beside it. And the diagnosis is
clean: the weighted sum's wind regression was the cost of frozen logits; this rung unfreezes them.

Let me make the shapes concrete, since the indexing is the only fiddly part and it must land the literal
edit. The tokenizer hands me `x: [B, V, L, D]`. The reduction is per-location, per-example, and identical
at every location, so I treat every `(example, location)` pair as an independent `V`-element set: permute
to `[B, L, V, D]` to bring the variable axis next to `D`, then fold `B` and `L` together into `[B·L, V, D]`
— `B·L` independent set-reduction problems. The query is the one learnable vector `[1, 1, D]`, expanded
across the batch to `[B·L, 1, D]`. Run multi-head cross-attention with that query against the `V` tokens as
keys and values: out `[B·L, 1, D]`. Drop the length-one query axis to `[B·L, D]`, unfold back to
`[B, L, D]`. In the scaffold this is a single `nn.MultiheadAttention(embed_dim, num_heads,
batch_first=True)` plus one `nn.Parameter` query, with `embed_dim = D = 1024` and `num_heads = 16` so
`d_k = 64` per head — the standard head width. `V` never enters the module's weights; it is read from the
input shape, which is the entire point of the contract. The query is zero-initialized, as argued. This is
exactly the ClimaX default fill the scaffold ships, and now I've derived *why* each piece is there rather
than inherited it. (The full scaffold module is in the answer.)

So the delta from step 2 is precise: where the weighted sum held a single global softmax distribution over
the 48 variables — a frozen weighting that helped z500 and t850 but mis-served wind10m — I now let one
learnable query compute, at each location, a content-dependent multi-head softmax over the *projected*
variable tokens, so the weighting reacts to what the tokens say there and the values are re-expressed in a
shared space before mixing.

Let me close on the falsifiable expectations against the two runs I have. I expect this to beat the
weighted sum on *every* target, and the cleanest test is wind10m-7day: that was the regression — the
weighted sum's 3.4245 was worse than even the mean's 3.3991 — and if content-dependent, multi-head
attention is the right diagnosis of that regression, then wind10m should drop back below the mean's floor,
i.e. below 3.3991, not merely below 3.4245. That is the sharp, falsifiable bar: if cross-attention cannot
get wind10m under 3.3991, then content dependence is not what the longest-lead wind target needed and the
regression had another cause. On z500-3day and t850-5day, where the global split already helped (345.79,
2.5934), I expect the content-dependent version to push further — the right variables at a synoptically
active cell differ from a quiescent one, and only this rung can express that — so I expect both to drop
below 345.79 and 2.5934, with z500 again showing the largest absolute gain since it carries the most
headroom and leans hardest on a state-dependent dynamical subset. If all three land below the weighted
sum's numbers and wind10m clears the mean's 3.3991, the ladder's logic holds end to end: uniform → fixed
global weighting → content-dependent re-projecting attention, each rung fixing the defect the previous
one's numbers exposed.
