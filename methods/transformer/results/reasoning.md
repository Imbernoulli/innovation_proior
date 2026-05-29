Okay, let me start from the thing that actually hurts, because the worst long-sentence quality bottleneck is no longer the main pain. I want to train sequence-to-sequence translation models, and Bahdanau et al. (2014) let the decoder look anywhere in the source, so long sentences stopped collapsing into a single fixed-vector bottleneck. What's killing me now is *speed*, and not the kind of speed I can buy back with a bigger machine. The recurrent encoder-decoder computes its hidden states as `h_t = f(h_{t-1}, x_t)`. Stare at that recurrence: `h_t` literally cannot be computed until `h_{t-1}` exists. So inside one training sentence of length `n`, the work is a chain of `n` steps that must happen strictly one after another — `O(n)` sequential operations per layer. On a GPU, a machine that wants one enormous matrix multiply rather than a thousand tiny dependent ones, this is the worst possible shape. And the obvious dodge — "just batch more sentences together to fill the machine" — dies at exactly the length where I need it: long sequences eat memory, so I can't batch many of them, so the intra-sentence parallelism I lost is gone for good. Kuchaiev & Ginsburg's (2017) factorization tricks and Shazeer et al.'s (2017) conditional computation shave the constant in front, but the `O(n)` sequential *depth* just sits there untouched. That depth is the enemy.

So what already works that I can build from? Attention, and I should be precise about why it's the right primitive and not just a fashionable one. Before Bahdanau, Sutskever et al. (2014) crammed the entire source into one fixed-length vector — the encoder LSTM's final state — and the decoder had to reconstruct everything from that single vector. That's an information bottleneck; the longer the sentence, the more gets crushed; they even reversed the source word order as a hack to shorten the path between aligned words, which tells me the *path between related positions* was already the thing biting them. Bahdanau dissolved the bottleneck: at each decoder step `i`, instead of one frozen context, compute a *fresh* weighted average over all encoder annotations `h_j`. The weights come from an alignment score, then a softmax, then a convex combination: `e_ij = vᵀ tanh(W_s s_{i-1} + W_h h_j)`, `α_ij = softmax_j(e_ij)`, `c_i = Σ_j α_ij h_j`. The beautiful structural property is that every output position reaches every input position in *one hop*. Distance stopped mattering for routing. This is content-based addressing: the query `s_{i-1}` decides where to look by *what it is*, not by a fixed offset, and the whole thing is differentiable — a soft, learnable lookup over the source.

But Bahdanau scores with a little tanh-MLP, one evaluation per query-key pair. Luong et al. (2015) noticed you don't need the MLP: you can score with a plain dot product `q·k` (or `qᵀWk`). Same theoretical cost as additive, but — and this is the part that matters for my speed problem — a batch of dot products *is* a single dense matrix multiply, and matrix multiply is precisely what the hardware is built to do at full throughput. Additive scoring forces an elementwise tanh on a `(n_q × n_k × d)` tensor before reducing; dot-product scoring collapses to one `(n_q × d)·(d × n_k)` GEMM. If I'm going to use attention *everywhere*, hundreds of times in a deep stack, I want the version that is one GEMM, not the version that materializes a giant intermediate and runs a nonlinearity over it. So dot-product attention, decisively, for hardware reasons.

Here's the tension I keep circling, though. Attention gives me `O(1)` path length — gorgeous — but everyone *bolts it onto an RNN*. The `h_j` being averaged and the `s_{i-1}` doing the querying are still produced by recurrence, so I keep the `O(n)` sequential chain and merely decorate it with attention. The convolutional people went the other way: ByteNet (Kalchbrenner et al. 2017) and ConvS2S (Gehring et al. 2017) killed recurrence entirely, computing all positions in parallel with CNNs. That fixes the sequential-depth axis. But look at path length again. A conv with kernel `k` only sees `k` neighbors per layer; to connect two positions `D` apart you need a *stack* — `O(D/k)`, linear in distance, for ConvS2S; `O(log_k n)` for ByteNet with dilation. They traded the sequential tax for a *path-length* tax. The number of operations to relate two distant positions grows with the distance again — the exact thing attention had just abolished. Hochreiter et al. (2001) is the relevant warning here: the longer the forward/backward path between two positions, the harder the dependency is to learn. So convolution didn't win; it moved the cost.

Let me write the three numbers down side by side. Per layer, on a sequence of length `n` with width `d`, a recurrent layer costs `O(n·d²)` compute because each of `n` steps is a `d×d` matmul, has `O(n)` sequential ops, and has `O(n)` max path length. A non-dilated convolutional layer with kernel `k`, the ConvS2S case, costs `O(k·n·d²)` compute and has `O(1)` sequential ops, but it needs `O(n/k)` layers to connect positions across the whole sequence. A dilated convolutional stack like ByteNet keeps the per-layer work parallel and uses dilation to cut that maximum path length to `O(log_k n)`, but the path still grows with distance.

And what would a layer made *purely of attention over the sequence itself* be — self-attention, every position attending to every position via dot products? Compute: I form the full score matrix `QKᵀ`, which is `n²` dot products of `d`-vectors, so `O(n²·d)`; then mix the values, again `O(n²·d)`. Sequential ops: it's a couple of big matmuls and a softmax with no step-to-step dependency, so `O(1)`. Max path length: one hop, always, `O(1)`. There it is — constant sequential depth *and* constant path length at once, which nothing else on the table has. The compute is `O(n²·d)` instead of `O(n·d²)`, so self-attention is the cheaper layer exactly when `n < d`. With subword vocabularies (BPE, word-piece) a sentence is a few dozen to low-hundreds of tokens while `d = 512`, so `n < d` is the normal regime. I'll pay the `n²` gladly to get `O(1)` on both axes I care about. (And if `n` ever blows past `d` — very long documents — I can fall back to restricting each query to a local neighborhood of size `r`, which makes compute `O(r·n·d)` and path length `O(n/r)`; a knob for later, not now.)

There's a second, subtler reason to prefer self-attention that the complexity table doesn't show. In a recurrent or convolutional layer, the information position `i` receives from the rest of the sequence is compressed into a single `d`-vector *before* `x_i` ever gets to filter it by content — `y_i = F(i, x_i, G(i, {x_{j≠i}}))` with `G` a fixed-width summary. That bottleneck forces a lot of potentially-relevant context to be thrown away before the query even looks. In self-attention the aggregation happens *after* the content-based weighting — the query picks what to pull, then pulls it — so relevant information isn't crushed by an unfiltered summary. That's an independent argument for the same primitive.

So the real question crystallizes: can I build the *entire* encoder and decoder out of self-attention and nothing else — no recurrence, no convolution anywhere? Parikh et al. (2016) already did attention-without-recurrence, but for sentence-pair classification, not autoregressive generation. Nobody's built a full transduction stack on pure attention. Let me actually try to build it and watch where it breaks. This is exactly the empty `SequenceModel` slot I left in the harness — the thing between embedding and the output projection — and I'm now going to fill it.

Start with the primitive in matrix form so it's one matmul. Queries packed into `Q`, keys `K`, values `V`. For self-attention all three come from the *same* place — the previous layer's representations — each linearly projected. Scores are all pairwise dot products `QKᵀ`; softmax each row to get a distribution over positions; mix the values: `Attention(Q,K,V) = softmax(QKᵀ) V`. Clean, and it's one GEMM for the scores and one for the mix.

But there's a known wart with raw dot-product scoring, and I need to understand it before it bites me. People observed (Britz et al. 2017) that *unscaled* dot-product attention does fine for small key dimension `d_k` but gets *worse* than additive attention as `d_k` grows. Why? Let me do the variance honestly. Take a query component `q_i` and a key component `k_i` as independent random variables with mean 0 and variance 1 — which is roughly what they look like coming out of well-normalized layers. The score is `q·k = Σ_{i=1}^{d_k} q_i k_i`. Its mean: `E[Σ q_i k_i] = Σ E[q_i]E[k_i] = 0` by independence. Its variance: since the mean is 0, `Var = E[(q·k)²]`. Expand the square — the cross terms `E[q_i k_i q_j k_j]` for `i≠j` factor into `E[q_i]E[k_i]E[q_j]E[k_j] = 0`, so only the diagonal survives: `Σ_i E[q_i²]E[k_i²] = Σ_i (1)(1) = d_k`. So the dot product has variance `d_k`, meaning its typical magnitude grows like `√d_k`.

And that's the disaster — though the precise failure is in the gradient, not the forward pass. With `d_k = 64`, scores routinely sit around ±8, ±10. Feed logits that far apart into a softmax and it saturates — it collapses onto essentially a one-hot, dumping almost all the mass on the single largest score. Write the softmax Jacobian: `∂softmax_i/∂logit_j = softmax_i(δ_ij − softmax_j)`. When the softmax output is near one-hot — say `softmax ≈ (1, 0, …, 0)` — every entry of that Jacobian is a product with a factor near 0: the diagonal term `p_i(1−p_i)` is `≈ 1·0 = 0` for the winner and `≈ 0·1 = 0` for the losers, and the off-diagonal `−p_i p_j` has at least one near-zero factor. So the Jacobian is `≈ 0` everywhere. The gradient flowing back through the attention weights is microscopic, and the model can't *learn* who should attend to whom; it's frozen in a near-argmax with no signal to move it. That is exactly why large-`d_k` dot-product attention degraded while additive (whose tanh keeps the pre-softmax scores `O(1)`) didn't.

The fix falls straight out of the variance I computed. The thing blowing up is variance `d_k`; I want it back at 1; the standard deviation I'm fighting is `√d_k`, so divide the scores by `√d_k`. Then `Var(q·k / √d_k) = d_k / d_k = 1`, the logits stay `O(1)`, the softmax stays in its responsive region, and the Jacobian stays healthy. So the primitive is `Attention(Q,K,V) = softmax(QKᵀ / √d_k) V`. The `1/√d_k` isn't a fudge factor I tuned — it's the unique rescaling the variance argument demands: not `1/d_k`, because that would over-correct (it normalizes the *variance's* magnitude, but what I want is the standard deviation back at 1, and `std(q·k) = √d_k`). Wall cleared, and cleared for a reason I can write down.

Now the second worry about a single attention head. One softmax distribution per query position is one "summary" of the sequence, and an average, by its nature, blurs. If position `i` needs to simultaneously track the subject for agreement *and* the verb for tense *and* a nearby modifier, one averaging operation smears all of those into one blob — and worse, a single head produces a single weighting pattern, so it can attend to one *kind* of relation at a time. This is precisely the cost I flagged for self-attention earlier: reduced effective resolution because everything goes through one averaging. I need resolution back without giving up the `O(1)` reach. So don't do one attention over the full `d`-dimensional vectors; do *several* in parallel over lower-dimensional projections, each free to attend to a different place for a different reason. Project `Q,K,V` with `h` different learned matrices down to `d_k = d/h` dimensions, run scaled dot-product attention in each subspace, and gather the `h` outputs: `head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)`. Now one head's `W^Q_i, W^K_i` can carve out the dimensions that encode syntactic role, so that head attends grammatically; another head's projections can emphasize semantic dimensions, so it attends by meaning. Crucially, because each head has its *own* softmax over its *own* projected scores, the heads don't average over each other — the blurring happens *within* a head but not *across* them, so distinct relations stay distinct.

After running the heads I have `h` outputs sitting in disjoint coordinate blocks — head 3's vector knows nothing of head 5's, and the residual stream they have to rejoin is `d_model`-wide, not `h·d_v`-wide. Concatenating them just stacks the blocks; the heads never get to *talk* to each other, so "the syntactic head found X and the semantic head found Y" can never be composed into a single conclusion. What I need is a learned linear that both maps `h·d_v → d_model` and mixes the per-head perspectives into one integrated vector — and one matrix does both jobs at once. So `MultiHead(Q,K,V) = Concat(head_1,…,head_h) W^O`. The `W^O` is the integration step, not a formality; without it the heads stay siloed.

Now, how expensive is this? Each head works in `d_k = d/h` dimensions, so the score matmul per head is `O(n²·d_k)` and there are `h` of them: `h · n² · (d/h) = n²·d` — identical to a single full-width attention. The projections `W^Q_i,…` together are `d × d` worth of parameters, same as one `d×d` projection. So slicing the budget into `h` specialists costs essentially what one full-dimensional head costs — resolution comes back for free, which is exactly why I tie `d_k = d_v = d_model/h` rather than letting each head keep the full `d` (that would cost `h×` as much and throw away the whole point). With `d = 512`, set `h = 8`, giving `d_k = d_v = 64`.

This same `MultiHead` block, by choosing where `Q, K, V` come from, gives me all three kinds of routing from one piece of machinery. In encoder self-attention, `Q, K, V` are all the previous encoder layer — every source position attends to every source position. In encoder-decoder (cross) attention, the queries come from the decoder while keys and values come from the encoder output — which is exactly Bahdanau's setup, the decoder querying over all source annotations, now expressed in the very same dot-product mechanism. And in decoder self-attention, all three come from the previous decoder layer — but the decoder must stay autoregressive, and that needs care.

The autoregressive constraint: the prediction for position `i` may depend only on outputs at positions `< i`, because at inference those are all I'll have generated. The matrix form makes the fix trivial. I'm computing the full `n×n` score matrix `QKᵀ/√d_k` in one shot, including the illegal entries where query `i` looks at key `j > i` — the future. I don't want to *skip* them, because skipping would break the single-matmul parallelism that is the entire reason I'm here. I want to *neutralize* them. So before the softmax I set every future entry to `−∞`. Then `exp(−∞) = 0`, those positions get exactly zero weight, and position `i` attends only to `≤ i` — while every row is still computed in parallel. In practice `−∞` is a large negative constant like `−1e9` added to the disallowed logits. This is the extra target-side mask the harness left undecided: the target mask is now the padding mask *and* this causal upper-triangular mask, together. Plus I offset the target by one position so position `i` predicts token `i` from tokens `< i`. That preserves the auto-regressive property without serializing training.

Now I hit the deepest wall. Pure attention is *permutation-equivariant*. Look hard at `softmax(QKᵀ/√d_k) V`: it is all dot products and weighted sums over the *set* of positions. There is no term anywhere that knows which position is which. If I shuffle the input rows, the output rows shuffle the same way and nothing else changes — the operation has no idea what *order* the tokens came in. The model literally cannot distinguish "the dog bit the man" from "the man bit the dog." Recurrence got order for free from the time axis; convolution gets it from kernel offsets. I threw both away. If I do nothing, I've built a sophisticated bag-of-words. I must *inject* position.

The cleanest move is to give each token a position-dependent vector and combine it with the embedding at the bottom of the stack. Two questions: how to combine, and what function of position to use.

Take the combination first — add or concatenate? Concatenation feels safer (it keeps content and position in separate, non-overlapping channels), so let me see whether that safety is real. The very first thing that touches the combined vector is a *learned linear projection*. A learned linear over the concatenation `[e ; p]` is `W[e ; p] = W_e e + W_p p` — it can already treat the two parts completely separately if it wants to, via the two blocks `W_e, W_p`. But a learned linear over the *sum* `e + p` is `W(e + p) = W e + W p` — the same two terms, with the constraint that `W_e = W_p = W`. So concatenation-then-projection is just addition-then-projection with the freedom to use *different* projections for content and position. Is that extra freedom worth its price? Its price is steep: if I concatenate a `p`-dimensional position code, every downstream weight — `W^Q, W^K, W^V` in every head of every layer, and the FFN — grows to `(d+p)` input columns, a width tax paid on every matrix, forever. And the freedom it buys is marginal: the embedding `e` is itself a learned `d`-vector and the position code lives in the same `d` dimensions; downstream linears can learn to allocate some directions to "mostly position" and others to "mostly content," and a sum is easily disentangled when the position code occupies a recognizable subspace. So I add the positional vector to the embedding. (This is also why both must be dimension `d`.)

Now what function of position. I could *learn* an embedding per absolute position, like ConvS2S — a fine option, and I should keep it in mind as a baseline. But a learned position table has a hard cap: it has no entry for any position beyond the longest sequence seen in training, so it cannot extrapolate to longer inputs at all. Before settling, let me think about what property I actually want from these vectors. What attention does with them is take dot products, and implicitly differences — I want a head to be able to learn "look 3 tokens back," and to apply that *same* relative rule regardless of where it currently is in the sequence. So I want a positional code in which a shift by a fixed offset `k` is a *simple, position-independent* transformation — then a head can learn one operator that means "+k everywhere." Sinusoids do exactly this. Define, for position `pos` and dimension index `i`:
`PE(pos, 2i) = sin(pos / 10000^{2i/d})`, `PE(pos, 2i+1) = cos(pos / 10000^{2i/d})`.
Each pair of dimensions is a sine/cosine at its own frequency `ω = 1/10000^{2i/d}`, the wavelengths sweeping a geometric progression from `2π` up to `10000·2π`. Why does this give relative-shift-as-a-linear-map? Take one frequency `ω` and the 2-vector `[sin(ω·pos), cos(ω·pos)]`, and shift by `k`:
`sin(ω(pos+k)) = sin(ω·pos)cos(ωk) + cos(ω·pos)sin(ωk)`,
`cos(ω(pos+k)) = cos(ω·pos)cos(ωk) − sin(ω·pos)sin(ωk)`.
That is exactly a rotation:
`[sin(ω(pos+k)); cos(ω(pos+k))] = [[cos ωk, sin ωk]; [−sin ωk, cos ωk]] · [sin(ω·pos); cos(ω·pos)]`.
The rotation matrix depends only on `k`, *not on `pos`*. So `PE_{pos+k}` is a fixed linear function of `PE_{pos}` for every offset `k`, the same linear map at every absolute position. A head can implement "attend `k` positions away" with one learned linear that works everywhere — precisely the relative-offset behavior I wanted. The geometric spread of frequencies matters too: high-frequency dimensions resolve fine, nearby distinctions; low-frequency ones (wavelength up to `10000·2π`) stay nearly constant over a whole sentence and so encode coarse, long-range position. Together they give a multi-scale ruler. And because `sin` and `cos` are defined for *any* real `pos`, the code exists for positions longer than any seen in training, so the model can in principle extrapolate — the thing the learned table simply cannot do. That's why I pick sinusoids over a learned table even though, for in-distribution lengths, a learned table would do about as well. Wall cleared, and cleared in a way that's strictly better on extrapolation.

One scaling subtlety surfaces here. The sinusoidal components are `O(1)` — sines and cosines live in `[−1, 1]`. Meanwhile a freshly-initialized embedding row, with the usual variance-preserving init, has components of size roughly `O(1/√d)`. So the raw embedding would be a *whisper* next to the `O(1)` positional sinusoids, and position would drown content at the very bottom of the stack. To put the two signals on the same footing I multiply the embedding by `√d_model`: each component becomes `O(1)`, amplitude-matched to `PE`, so the sum carries both at similar strength rather than one swamping the other. That's the reason for the `√d_model` embedding scaling — amplitude-matching, not arbitrary.

While I'm on the embeddings, a parameter-count thought: the input embedding maps a token id to a `d`-vector, and the pre-softmax projection in the generator scores a decoder state against each token's vector — these are the *same* token↔vector correspondence used in two directions (lookup vs scoring). Since they're the same geometry, I can tie the input embedding, the target embedding, and the pre-softmax projection to one shared matrix (following Press & Wolf 2016). That cuts a large block of parameters — the vocabulary is tens of thousands of rows times `d` — and ties the two views of each token together, which regularizes. The `√d_model` scaling is applied at the embedding *forward* call so the lookup is amplitude-matched, while the tied matrix in the output projection sees the unscaled geometry it needs for sensible logits.

Two structural pieces are still missing. Attention mixes information *across* positions, but within a position it's a fairly weak transformation: the value is a softmax-weighted *linear* combination of other positions' values, and the only nonlinearity in the whole attention sublayer is the softmax on the *weights* — there's no per-position nonlinear processing of the content itself. The model needs somewhere to take the mixed representation at position `i` and transform it nonlinearly. So between attention sublayers I put a feed-forward net applied identically and independently to every position. How wide and how deep? A single `d→d` linear adds nothing the value-projection couldn't already do, so the point is a genuine nonlinear map — and a ReLU MLP's expressive power scales with its hidden width: more hidden units means more linear regions, more distinct features it can carve out of the `d`-dimensional input before projecting back. So expand to a wide hidden layer, ReLU, project back: `FFN(x) = max(0, xW_1 + b_1) W_2 + b_2`, with `d_model = 512 → d_ff = 2048 → 512`. Why 4× specifically? Width is per-position capacity and this sublayer holds the majority of the layer's parameters; 4× is the capacity-vs-compute knee that gives real nonlinear headroom without letting the FFN dominate the cost. (Equivalently it's two kernel-1 convolutions — same weights at every position, which is what "position-wise" means.) And a sanity check that this belongs in the same family: `ReLU(xW_1)W_2` has the shape of `compat(x, K)V` with the rows of `W_1` as keys and rows of `W_2` as values and ReLU as the compatibility function — it's attention over a fixed set of *learned parameters* instead of over the sequence.

The other structural piece is what lets me actually stack these. I want `N` layers deep, and deep stacks of anything don't train naively — gradients vanish or explode through the depth. So I want each sublayer to leave a clean, undiminished path for the gradient. Wrap each sublayer as `x + Sublayer(x)` rather than `Sublayer(x)` (He et al. 2016): then `∂/∂x [x + F(x)] = I + F'(x)`, and the identity term `I` means the upstream gradient reaches `x` undiminished no matter how small `F'` is — a magnitude-1 path back through every sublayer, so the signal can't vanish through depth. It also reframes each sublayer as learning a *residual* correction to the identity, which is easier than learning a full transformation from scratch. Then normalize, because the residual sum's scale would otherwise drift across the stack: every sublayer becomes `LayerNorm(x + Sublayer(x))`.

Which normalizer, though? The batch dimension is the wrong axis to normalize over *here*. BatchNorm normalizes each feature across the batch, using the batch's mean and variance — but my sequences are variable-length and padded, so the set of real tokens at a given position varies across the batch and the padding injects spurious zeros, making the per-position batch statistics fluctuate wildly and unreliable as a normalizer. Worse, BatchNorm keeps running statistics for inference, and my decoder runs autoregressively, one token at a time, where there is no meaningful batch to normalize over — a built-in train/test mismatch. The axis that *is* stable is the feature axis of a single token: normalize each token's vector across its own `d` features, independently of every other token and of the batch. That is LayerNorm (Ba et al. 2016) — batch-size invariant (works with a batch of one), identical at training and at one-token-at-a-time decoding, immune to padding because each token is normalized alone. So: `LayerNorm(x + Sublayer(x))` around every attention and every FFN, all sublayers and embeddings at width `d = 512` so the residual sums line up, `N = 6` layers in the encoder (each: self-attention, FFN) and `N = 6` in the decoder (each: masked self-attention, cross-attention, FFN). That completes the `SequenceModel` slot — the whole thing is now attention, position-wise FFNs, residual-plus-LayerNorm glue, and positional codes, with no recurrence anywhere.

Now the optimization, which has its own wall. These deep attention stacks are touchy at the very start, and I should name why rather than just bolting on a schedule. There are two problems at initialization, and they compound. First, the attention itself: with random projections the dot-product logits can be large and the softmax can saturate early — the same Jacobian-collapse from before — so the first gradients are large and noisy. Second, Adam: it scales each update by `1/√v_t` where `v_t` is a running estimate of the squared-gradient magnitude, and early on `v_t` is computed from only a handful of samples, so it's a high-variance estimate — the adaptive step size is unreliable in exactly the first steps. And these two interact viciously: a big noisy early gradient gets recorded into Adam's second moment `v_t`, and that distorts the step scale for subsequent updates. A single bad early step can knock the run off course. The countermeasure is to keep the learning rate tiny while the statistics settle, then raise it — warm up, then decay. Concretely
`lrate = d_model^{−0.5} · min(step^{−0.5}, step · warmup^{−1.5})`,
with `warmup = 4000`, under Adam (`β_1=0.9, β_2=0.98, ε=1e-9`). The `β_2` value is lower than Adam's usual `0.999`, so the second-moment estimate can follow the rapidly changing gradient scale early in training; the warmup keeps updates small while that more responsive estimate settles. Read the two branches: for `step < warmup`, the `min` picks `step · warmup^{−1.5}`, which is *linear* in `step` — the warmup ramp, learning rate climbing from near zero so early updates are small and Adam's `v_t` can stabilize before any large step is taken. For `step > warmup`, the `min` picks `step^{−0.5}` — decay as the inverse square root of the step, annealing as training settles. The two branches cross exactly at `step = warmup`, which is where the peak rate sits. And the `d_model^{−0.5}` prefactor scales the whole rate down for wider models: a larger `d` means larger activations and gradients, so the peak learning rate should be smaller — `d_model^{−0.5}` is that automatic downscaling, so the schedule transfers across model sizes without retuning. The warmup isn't a superstition; it's the direct countermeasure to early softmax saturation feeding Adam's variance. This replaces the placeholder constant-rate Adam in the harness with a `LambdaLR` wrapping exactly this `rate`.

Two regularizers finish the recipe. The densest, most co-adaptation-prone points in the network are the sublayer outputs and the summed embedding+positional input, so that's where I put dropout (Srivastava et al. 2014) — on the output of each sublayer *before* the residual add (dropping the sublayer output, not the residual path, keeps the identity gradient highway intact), and on the embedding sum. Rate `0.1`. And the loss: a plain one-hot cross-entropy pushes the model to drive the gold logit to `+∞` and all others to `−∞`, which is both unattainable and overconfident — it makes the model brittle and miscalibrated, and translation has many acceptable outputs, so overconfidence on one of them hurts. Soften the target instead: put `1 − ε_ls` on the gold token and spread `ε_ls` over the rest of the vocabulary, and train against it with KL divergence. This is label smoothing (Szegedy et al. 2015) at `ε_ls = 0.1`; it may raise perplexity because the model is deliberately less peaked, but that is the trade I want for less brittle sequence choices. So the harness's plain cross-entropy becomes this label-smoothed KL.

Let me put it in real code, mirroring a clean implementation, each block tied to the step that forced it. One implementation detail has to stay explicit: the architecture I just derived normalizes after the residual sum, `LayerNorm(x + Sublayer(x))`, while the canonical clean PyTorch code puts the norm first, `x + Sublayer(LayerNorm(x))`, for code simplicity and then applies a final norm at the end of each stack. This fills the `SequenceModel` slot and the surrounding places the harness left adjustable: target masking, the tied embedding/generator weights, the loss, and the optimizer schedule.

```python
import math, copy, torch, torch.nn as nn
from torch.nn.functional import log_softmax

def clones(module, N):
    "N identical layers — the encoder/decoder are stacks of these."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# --- the primitive: scaled dot-product attention ---
# softmax(QK^T / sqrt(d_k)) V. The 1/sqrt(d_k) is the variance fix: raw q.k has
# variance d_k, which saturates the softmax (near one-hot) so its Jacobian -> 0
# and the attention weights stop receiving gradient.
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)   # future/pad -> -inf -> 0 weight
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# --- multi-head: h parallel attentions in d/h-dim subspaces so one average
#     doesn't blur distinct relations together; W^O mixes the heads back into one
#     d_model vector; total cost ~ one full-width attention ---
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h            # 64 when d_model=512, h=8
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)   # W^Q, W^K, W^V, W^O
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)       # same mask for all heads
        nb = query.size(0)
        # project, then split into h heads of width d_k
        query, key, value = [
            lin(x).view(nb, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))]
        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nb, -1, self.h * self.d_k)  # concat heads
        return self.linears[-1](x)         # final W^O mixes heads + maps to d_model

# --- per-position FFN: the only per-position nonlinear compute; 512->2048->512
#     so the ReLU has width to carve real features before projecting back ---
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)    # 512 -> 2048
        self.w_2 = nn.Linear(d_ff, d_model)    # 2048 -> 512
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

# --- positional encoding: inject order (attention is permutation-equivariant);
#     sinusoids so a shift by k is a fixed, position-independent rotation, and so
#     positions beyond training length still have a code (extrapolation) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # geometric freqs 2pi .. 10000*2pi
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)   # ADD, not concat
        return self.dropout(x)

class LayerNorm(nn.Module):
    # per-token over the d features: batch-size invariant, identical at one-token
    # decode, immune to padding — unlike BatchNorm's across-batch statistics.
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.a_2 * (x - mean) / torch.sqrt(var + self.eps) + self.b_2

# --- norm-first residual wrapper, matching the clean PyTorch implementation's
#     code-simplicity ordering; the +x path keeps gradients flowing, and dropout
#     touches the sublayer output rather than the identity path ---
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):           # two sublayers: self-attn, then FFN
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # Q=K=V=x
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):           # three sublayers: masked self-attn, cross-attn, FFN
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn, self.src_attn, self.feed_forward = self_attn, src_attn, feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.size = size
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # causal self-attn
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))   # cross: Q=dec, K,V=enc
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

def subsequent_mask(size):
    "Causal mask: position i may attend only to <= i — the extra target-side mask."
    return torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.uint8) == 0

# --- this is what fills the SequenceModel slot from the scaffold: it takes the
#     embedded source + target and emits per-target-position features ---
class SequenceModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, src, src_mask, tgt, tgt_mask):
        memory = self.encoder(src, src_mask)
        return self.decoder(tgt, memory, src_mask, tgt_mask)

class Embeddings(nn.Module):             # scale by sqrt(d_model) to match PE amplitude
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class Generator(nn.Module):
    # tied to the unscaled token matrix; Embeddings.forward applies sqrt(d_model)
    # only on lookup, not on the output scores.
    def __init__(self, tied_embedding):
        super().__init__()
        vocab, d_model = tied_embedding.lut.weight.shape
        self.proj = nn.Linear(d_model, vocab, bias=False)
        self.proj.weight = tied_embedding.lut.weight
    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

def tied_embeddings_and_generator(vocab, d_model):
    src_embed = Embeddings(d_model, vocab)
    tgt_embed = Embeddings(d_model, vocab)
    tgt_embed.lut.weight = src_embed.lut.weight
    generator = Generator(tgt_embed)
    return src_embed, tgt_embed, generator
```

And the two places the surrounding harness changes — the warmup-then-inverse-sqrt schedule the optimization wall forced, and the loss that becomes label-smoothed KL:

```python
def rate(step, model_size, factor, warmup):
    if step == 0: step = 1
    # step < warmup: linear ramp (step * warmup^-1.5) lets Adam's variance settle;
    # step > warmup: decay ~ 1/sqrt(step); model_size^-0.5 shrinks peak LR for wider d.
    return factor * (model_size ** -0.5 * min(step ** -0.5, step * warmup ** -1.5))

class LabelSmoothing(nn.Module):
    "Soft targets via KL — one-hot would push gold logit to +inf and over-confidence."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
    def forward(self, x, target):
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist.clone().detach())
```

So the causal chain, start to finish: the pain was the `O(n)` sequential depth of recurrence, which hardware can't parallelize away and which long sequences make worse. Attention had already shown that any-to-any routing in `O(1)` path length is possible, and dot-product scoring makes it one hardware-friendly matmul, but attention was always parasitic on an RNN; the convolutional alternatives killed the recurrence but reintroduced a path length that grows with distance. Demanding *both* `O(1)` sequential ops and `O(1)` path length points to one cell — self-attention over the sequence itself — cheaper than recurrence whenever `n < d`, which subword vocabularies guarantee. Committing to build the entire encoder and decoder from it surfaces a cascade of forced choices: the dot-product variance of `d_k` saturates the softmax and collapses its Jacobian, forcing the `1/√d_k` scaling; a single average blurs distinct relations, forcing `h` parallel heads in `d/h`-dim subspaces with a `W^O` to mix them, all at the cost of one full-width head; pure attention is order-blind, forcing positional codes, which I *add* (not concatenate, to avoid widening every matrix) and build from sinusoids (so a relative shift is a position-independent rotation and the code extends past training length), amplitude-matched to the embedding by `√d_model`. Wrap every attention and FFN — the FFN wide so it has real per-position nonlinear capacity — in residual-plus-LayerNorm glue, conceptually `LayerNorm(x + Sublayer(x))` and in the clean code `x + Sublayer(LayerNorm(x))` for code simplicity, stack six, mask the decoder's self-attention with `−∞` to preserve autoregression, tie the shared token matrix across source lookup, target lookup, and output scoring, and train under Adam with a warmup-then-inverse-sqrt schedule that defuses the early softmax-saturation-into-Adam-variance trap, plus dropout and label smoothing. And the recurrence is gone entirely, replaced by attention all the way down.
