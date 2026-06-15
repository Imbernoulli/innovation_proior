Block AttnRes is the strongest rung in the ladder and its number tells me both that the depth-flow idea
was right and exactly where it stopped short. Validation loss fell to 2.2544 — the first sub-2.26
result, a clear step below learned-scaling's 2.2680 — and the perplexities moved with it the way the
"per-token choice of which deep block to read" story predicted: WikiText-2 dropped to 41.82 (from 43.91)
and LAMBADA fell to 64.32, finally below both prores's 67.21 and learned-scaling's 68.76. So the jump
from a rank-one scalar mix to a content-dependent, full-rank mix over depth was the largest single move
on the ladder, and it added genuine capacity, not just conditioning. But read what it cost to fit the
budget, because that cost is the seam. To survive a fixed 2-GPU, micro-batch-32 run, Block AttnRes had
to *coarsen* the depth axis: it kept ordinary unweighted residuals *within* each block of 4 layers and
only ran the learned attention at the 6 block boundaries. So inside every block of 4 I am right back to
the rigid unit-weight accumulator — the very thing the whole ladder has been fighting — and the dynamic
routing only acts at 5 seams plus a readout. The depth-flow rule got smarter at the coarse scale and
stayed dumb at the fine scale. And there is a deeper structural fact Block AttnRes never addresses, one
that sits underneath *every* rung so far: no matter how I weight or attend over the *single* residual
stream, that stream is forced to serve two conflicting jobs at once. It must stay a clean identity
highway so gradients reach the shallow layers, *and* it must carry each deep layer's output strongly
enough to keep that layer's representation distinct. Those two demands pull the one stream's coefficient
in opposite directions. ProRes, learned-scaling, Block AttnRes — all of them tune *how* the one stream
is written; none of them gives the network *more than one* stream to write into. That is the move I have
not made, and it is the one that attacks the conflict at its root rather than at one scale of it.

Let me name the conflict precisely, because the fix has to be derived from it. Trace Pre-Norm again:
`h_k = h_{k-1} + T(Norm(h_{k-1}))`, branch input normalized, raw output added to an unnormalized stream
whose norm climbs with depth. As it climbs, a fresh layer's output is a shrinking fraction of the total,
so deep layers' contributions wash out and adjacent deep features collapse toward each other — exactly
the representation redundancy I suspected back at the vanilla floor and which Block AttnRes only partly
relieves at block boundaries. The naive counter is Post-Norm — normalize after the addition so each
output stays a meaningful fraction — but that puts a normalization Jacobian on the highway and the
gradients vanish with depth. This is a seesaw, and the key realization is that it is *structural for a
single stream*: one coefficient governs both "how much of the past survives" (the gradient route) and
"how much of this layer enters" (the depth influence), and one number cannot satisfy both. Every scalar
fix I have tried — and the ones I considered and rejected, ReZero's learned scalar, the depth-aware
inits — slides along this seesaw without tipping it off. So the structural escape is not a better scalar;
it is to stop having one stream.

What if the residual stream carried `n` parallel copies instead of one? Replicate the embedding into `n`
copies to form a hyper-hidden *matrix* `H ∈ R^{n×d}` (here, in this task, a tensor `(B, T, n, D)`), and
let every layer operate on the whole matrix, summing the `n` copies back into one vector only at the very
top, right before the final LayerNorm and the head. The point of `n` copies is decoupling: now one copy
can act as the clean Pre-Norm gradient highway while another carries a strongly-written, distinct deep
output, so I do not have to make a single coefficient serve both jobs. With `n > 1` the two demands live
in different copies. And I can convince myself this is structural and not just extra parameters: with a
single stream the conflict provably persists no matter how the routing is learned (the `n = 1` case does
not improve over baseline), whereas `n > 1` lets the network *reserve multiple patterns* of connecting to
preceding layers simultaneously. That is the difference between sliding the seesaw and breaking it.

Now derive the routing rule rather than guess it, because the overhead has to stay negligible or this
cannot run at GPT-2 Medium scale on a fixed budget. A layer reads the matrix `H`, must form a single
input for its sublayer, run the sublayer, and write the result back into the `n` copies. There are two
distinct axes of connection. *Depth*: how the new sublayer output is distributed into the copies, and how
each copy carries forward — the generalization of the residual skip. *Width*: how the copies exchange
information within a layer, and how they are mixed to form the sublayer's single input. Collect every
weight into one small `(n+1)×(n+1)` matrix where index 0 is the sublayer-output slot and `1..n` are the
copies. Its first row `β` distributes the single output back into the copies (depth write); its first
column `A_m` mixes the copies into the sublayer input `h_0 = A_m^T H` (width read); its `n×n` block `A_r`
carries the copies to the new copies `H' = A_r^T H` (the stream's own recombination). The whole layer is
then: read the input via `A_m`, carry the streams via `A_r`, run the sublayer on `h_0`, and write its
output back via `β` plus the carry. One matrix of scalars per residual site, the attention and MLP
untouched, `n` streams in and `n` streams out. Two structural bonuses fall out for free, and they are why
this is more than a re-parameterization: this matrix *contains* Pre-Norm and Post-Norm as the
non-trainable `n = 1` special cases (so I lose nothing), and at `n = 2` specific integer matrices express
both the ordinary sequential residual *and* a parallel-block arrangement — so learning the matrix learns
a soft, even dynamic, blend of sequential and parallel depth that the fixed residual can never reach.

I want the routing to depend on the input, not be a fixed learned constant, because the right depth/width
mix surely differs token to token — this is the same instinct that made Block AttnRes's per-token
attention beat the static scalars. So make the matrix entries functions of `H`: keep the static matrix as
a base and *add* a small dynamic correction predicted from `H` — normalize `H`, take a linear projection,
squash with `tanh`, scale by a small learnable factor, add to the static base, separately for `β`, `A_m`,
`A_r`. Each piece earns its place. The norm-before-projection keeps the depth-growing stream scale out of
the routing predictor (the disease I am curing must not re-enter through the router — the same reason
Block AttnRes RMS-normed its keys). The tanh bounds the correction so a runaway logit cannot blow up the
connection weights and destabilize a 13.5k-step run I cannot afford to lose. The small init scale means
the dynamic part starts negligible and the network has to *earn* its way off the static matrix.

Initialization is the part I cannot get wrong, and it is the cleanest argument for trying this here: I
can make the very first forward pass behave *exactly* like the Pre-Norm residual that the vanilla floor
already trains cleanly, so the model is never worse than baseline at step zero and only improves. Two
requirements. The dynamic projections start at zero, so `tanh(0) = 0` and the dynamic correction is
exactly nothing at init — the layer begins as pure static hyper-connection. And the static matrix encodes
Pre-Norm-on-`n`-copies: `β = 1` writes the full output into every copy (as Pre-Norm adds the full
branch), `A_r = I` carries each copy through unchanged (each stream a clean identity highway), and `A_m`
is a one-hot `e_{k mod n}` that reads the sublayer input from a single copy, rotated by layer index so
the copies are used round-robin and none is privileged. With this base and zero dynamics, summing the `n`
rows at the top is *exactly* Pre-Norm. So unlike Block AttnRes — which started at a uniform block-average,
a different operating point than vanilla — hyper-connections start at *bit-for-bit Pre-Norm* and bend
away only as the dynamics learn. That is the safest possible starting point on this budget.

Now the budget check, because it decides the expansion rate. Static parameters per site: the
`(n+1)×(n+1)` matrix, `O(n^2)` scalars — nothing. Dynamic parameters per site: projections of size
`O(nd)` plus two scalars and a norm — tiny against the `O(d^2)` of attention and the MLP. The main new
compute is the width matmul `A^T H`, an `(n+1)×n`-by-`n×d` per token, a rounding error for small `n`.
Memory is the real cost: `n` streams cost `O(n·s·b·d)` activation per layer. For `n = 2` that is a few
percent; the ablations say `n = 4` is where dynamic clearly beats static and `n = 1` is actually *worse*
than baseline (a single dynamic stream has no room to reserve patterns and the dynamic noise just hurts).
On *this* 24-layer, micro-batch-32, 2-GPU budget, where Block AttnRes already had to coarsen to fit, I
will not assume `n = 4` is free — but `n = 2` is comfortably affordable and is the point where the seesaw
provably breaks, so it is the honest default to land for this task, with `n = 4` the reach if memory
allows. The streams sum to one vector before `ln_f`, so the head and the loss are untouched.

Place it in the edit surface, because the fit is exact and worth being precise about. The contract keeps
`CausalSelfAttention`, `MLP`, `LayerNorm` fixed and asks `Block.forward(x) → x` and `GPT.forward(idx,
targets) → (logits, loss)`. Hyper-connections need two residual sites per layer (attention and MLP),
each with its own static matrix, dynamic projections, and norm — and they operate on the `n`-stream
tensor, not the single `x`. So I keep the `Block` as the vanilla container of `ln_1, attn, ln_2, mlp` but
do not call its `forward`; instead, as Block AttnRes already did, I drive the sublayers directly from the
`GPT.forward` loop. In `GPT.__init__` I build `2·n_layer` `HyperConnection` modules (one per site, the
static one-hot indexed by site id `k mod n`), set the expansion rate `n`, and that is the only structural
addition. In the forward loop I lift the embedding `x` into `n` copies (`H` of shape `(B, T, n, D)`),
then for each layer run the attention site — width connection gives me the layer input in row 0 and the
carried streams in the rest, I norm row 0 with `ln_1`, run `attn`, and the depth connection writes the
output back via `β` and adds the carry — then the same for the MLP site with `ln_2` and `mlp`. After all
layers I sum the `n` streams into one vector and pass it to `ln_f` and the head. In `configure_optimizers`
the new parameters (static matrices, dynamic projections, scales, site norms) are mostly 1-D or small;
following the pattern the ladder established for leveraged routing parameters (Block AttnRes's `0.1×`
query group), I route the hyper-connection parameters into their own no-decay group — the static matrices
and scales are gains, not weight matrices, and weight decay would just pull them back toward the Pre-Norm
init I deliberately chose. The LR schedule and `CONFIG_OVERRIDES` stay default. The full scaffold module
is in the answer.

So the delta from the strongest baseline is precise and it is the move no rung in the ladder made: Block
AttnRes kept one residual stream and got smarter about *attending over its coarsened history*;
hyper-connections keep the *attention idea's* per-token dynamic routing but apply it to a fundamentally
richer object — `n` parallel streams with a full static-plus-dynamic `(n+1)×(n+1)` routing matrix at
*every* sublayer, fine-grained where Block AttnRes was coarse, and breaking the gradient-vs-collapse
seesaw at its root instead of relieving it at 5 seams.

Here is the bar this has to clear and what I would validate, stated against the real numbers and with no
invented ones. The number to beat is 2.2544 validation loss, with WikiText-2 41.82, LAMBADA 64.32,
ARC-Easy 55.01, HellaSwag 34.05. The mechanism predicts hyper-connections should clear it: it adds
fine-grained dynamic depth routing that Block AttnRes gave up to fit the budget, it starts at exactly the
Pre-Norm operating point (so it never pays the start-from-a-different-point tax), and it directly targets
the adjacent-deep-layer feature collapse that the per-block-only routing leaves untouched inside each
block. The signature I would look for, and the falsifiable test of the whole story: if hyper-connections
are working, the cosine similarity between adjacent layers' features should *drop* relative to Pre-Norm —
that is the direct readout of "deep layers made distinct," and it is the diagnostic that distinguishes a
real seesaw-break from a mere reshuffle of the loss. On the headline metrics I would expect the
deep-layer-sensitive ones to move most: LAMBADA below 64.32 and WikiText-2 below 41.82, with validation
loss into the low-2.25s or below. The honest risks I would watch are two. First, the expansion rate: at
`n = 2` the seesaw breaks but the capacity gain is smaller than `n = 4` would give, so if memory forces
`n = 2` the win over Block AttnRes may be narrow rather than decisive — and `n = 1` would be *worse* than
even vanilla, so the rate must stay above 1. Second, the short horizon: hyper-connections' reported
advantage grows over long training (its convergence-speed gain compounds), and 13.5k steps on 7B tokens
is a brief run, so I would not be surprised if the measured margin is smaller than the large-scale
reports — the test is whether it beats 2.2544 cleanly and bends the adjacent-layer-similarity curve down,
not whether it reproduces a 1.8× convergence speedup at this scale.
