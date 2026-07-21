I am starting here with a Transformer encoder over the 60-day window, and I should be honest about why I
reach for a sequence model at all rather than the default Ridge or a tree. The Alpha360 feature vector is
not really 360 unrelated numbers; it is six base ratios — open, close, high, low, volume, vwap, each
normalized to the latest close — unrolled across sixty trading days. A flat regressor or a tree sees
`[N, 360]` and has no idea that column 5 and column 65 are the *same* base ratio one day apart. The
temporal structure, which is the whole reason a 60-day window is the input, is invisible to it. So the
first thing worth trying is a model whose inductive bias is exactly "this is a length-60 sequence of
6-dimensional observations," and whose job is to summarize that window into one forward-return score per
stock-day. That is a sequence-to-one regression, and I want the sequence architecture that lets any day
in the window talk to any other day in one hop.

"Temporal bias" alone does not single out a transformer, so I walk the two other sequence-shaped options
first. The cheapest is a 1-D convolution along the 60-day axis: kernels sliding over time, each layer
mixing a small local neighborhood. Fast, not initialization-sensitive — but its reach is bounded by
receptive field, and I can size that on paper. A convolution with kernel size `k` and `L` stacked layers
(stride 1, no dilation) has receptive field `L·(k−1)+1`. To let day 1 influence day 60 I need
`L·(k−1)+1 ≥ 60`, so with `k=3` that is `L ≥ 30` layers — thirty convolutions deep just to connect the
ends of the window once. Dilation shortcuts this to `L ~ log₂(60) ≈ 6` layers, but every dilated layer
still only sees a *fixed* stride pattern; it cannot decide, per stock-day, that today's move should be
compared to a spike forty-one days ago rather than forty. The comparison structure is baked into the
architecture, not learned from content. Attention connects the two ends in a *single* layer with a
*content-dependent* weight — exactly the property I want when I do not know in advance which lag matters.

There is also the shallow feedforward net — an MLP straight on the flat 360-vector — and "add
nonlinearity" is not the same as "add the right bias." An MLP lifts the linear default to nonlinear
capacity, but it is still *flat*: its first layer is one big `Linear(360, h)`, learning an independent
weight for every one of the 360 columns with no constraint tying column 5 and column 65 (the same ratio
one day apart) to a shared transform. It can in principle rediscover the 60×6 structure, but only by
spending parameters and data to relearn a regularity I already know for free — a single hidden layer of
width 256 is already `360·256 ≈ 9.2·10⁴` first-layer weights spent purely on that. The sequence models
encode that regularity as a hard architectural fact (weight-shared over days), so they never spend a
parameter learning it. That is the difference between more capacity and better-placed capacity, and on a
low-signal problem the latter is what I want.

Why attention rather than a recurrence for the *first* attempt? In a recurrent encoder the state at day
`t` is `h_t = f(h_{t-1}, x_t)`, so information from day 1 has to survive sixty sequential transformations
before it can influence the read-out at day 60 — and each can attenuate it. An attention layer instead
forms, for each position, a weighted average over *all* positions at once: every day reaches every other
day in a single content-based lookup, with no distance-dependent attenuation in the routing. For a window
where a move sixty days ago might matter as much as yesterday's, that any-to-any reach is the appealing
property. So the architecture I drop into the editable `CustomModel` is an encoder built from
self-attention: lift each day's 6 features to a working width, run a couple of self-attention layers over
the 60 positions, and read out a single score.

The task fixes the shape of this, and several moves I would make in a from-scratch transduction model
simply do not apply. This is an *encoder-only*, *sequence-to-one* regressor — no decoder, no
autoregressive generation, no target sequence. So there is no causal mask (every day may legitimately
attend to every other day; nothing is "in the future" to be hidden — the label is strictly outside the
window), no cross-attention, no tied embeddings, no label smoothing, no token vocabulary at all. The
"tokens" are continuous 6-vectors, one per day. What I keep from the attention machinery is: a linear
that lifts each day's 6 features to the model width, positional information so the layer is not
order-blind, scaled-dot-product self-attention with multiple heads, the position-wise feed-forward, the
residual-plus-LayerNorm glue, and a final linear reading the last position down to one number. Everything
that belonged to *generation* is dropped because this task does not generate — it scores.

The load-bearing choices that survive are each forced. First, lifting the input. Each day arrives as a
6-vector, far too thin to host multi-head attention — I want to slice the representation into several
heads, each in its own subspace, and 6 dimensions cannot be meaningfully split. So `feature_layer =
Linear(6, d_model)` lifts each day to `d_model = 64`. That width is wide enough to host `nhead = 2` heads
of 32 dimensions each, while staying small enough that an `n²·d` attention over only 60 positions is
cheap and, more importantly, does not have so much capacity that it memorizes noise. Financial
cross-sectional return prediction has an IC ceiling in the low single-digit percents; the danger is never
underfitting, it is fitting noise. A narrow model is a regularizer.

But I should pin down how much capacity is actually in this "narrow" model, because the width alone is
misleading. The input lift is `6·64 + 64 = 448`. Each `TransformerEncoderLayer` carries an attention
block of about `16640` parameters (fused Q/K/V `3·64·64 + 3·64` plus output projection `64·64 + 64`) —
but the position-wise feed-forward defaults, in the PyTorch layer I am instantiating, to an inner width
of `2048`: `Linear(64, 2048)` then `Linear(2048, 64)`, i.e. `264256` parameters. One encoder layer is
about `281k`, two layers about `562k`, plus a `Linear(64, 1)` read-out — roughly `563k` total, and the
two feed-forwards alone are about `94%` of it. So the model I keep calling "narrow" is narrow only in its
*attention* width; the bulk of its capacity is a wide per-position MLP the `d_model = 64` framing hides.
That reframes what is actually regularizing: not the width — the FFN is wide — but the shallow depth (two
layers), the tiny learning rate, weight decay `1e-3`, and early stopping. With `dropout = 0` here I have
*removed* the one regularizer built into the block, so a 563k-parameter model with a 94%-FFN mass and no
dropout, fit on a faint signal, is over-capacitied unless the optimizer is kept on a very short leash.
The failure mode to brace for is variance, not bias.

Second, position. Pure self-attention is permutation-equivariant — `softmax(QKᵀ/√d_k)V` is all dot
products and weighted sums over the *set* of positions, with no term that knows which day is which. If I
shuffle the sixty days the output shuffles identically, so the layer cannot tell a recent observation
from an old one — fatal for a window where recency matters. I add a sinusoidal positional code to each
lifted day-vector, `PE(pos, 2i) = sin(pos/10000^{2i/d})`, `PE(pos, 2i+1) = cos(pos/10000^{2i/d})`. At
`d_model = 64` the wavelengths run from `2π ≈ 6.3` positions (the fastest pair, resolving nearby days)
up to about `47000` positions (the slowest pair, essentially a monotone ramp giving a coarse "how far
into the window" coordinate), so nothing in the frequency budget is wasted at this length. I add rather
than concatenate — a learned linear over `e + p` is `We + Wp`, so downstream projections can already
separate content from position — and use sinusoids rather than a learned table because a shift by a fixed
offset becomes a fixed rotation of each sine/cosine pair, letting a head learn a uniform relative rule
like "look a few days back"; at 60 fixed positions the learned table would work too, but the sinusoid is
the parameter-free canonical choice.

Third, the attention primitive and its one wart. I score query against key by a dot product, softmax it,
then mix the values: `softmax(QKᵀ/√d_k)V`. The `1/√d_k` is not decorative. If `q` and `k` components are
roughly unit-variance and independent, `q·k = Σ q_i k_i` has mean 0 and variance `d_k`, so its typical
magnitude grows like `√d_k`. Feed logits of size `√d_k` into a softmax and it saturates toward one-hot;
its Jacobian `p_i(δ_ij − p_j)` collapses toward zero, and the attention weights stop receiving gradient.
With `d_k = 32` that is a factor of `√32 ≈ 5.66` on the logit scale — the difference between a softmax
over logits of typical size `1` (responsive, full gradient) and one over logits of size `~5.7` (already
sharply peaked, most of its Jacobian near zero). Not marginal at this width, so the scaling stays.

Fourth, multiple heads. A single softmax distribution per day is a single averaged summary, and an
average blurs: if a day needs to track a volume spike at one lag *and* a price reversal at another, one
head smears them. So I run `nhead = 2` attentions in parallel over 32-dim projections, each free to
attend to a different pattern, and concatenate. Two heads is modest, but the regime rewards restraint —
more heads is more capacity to overfit on a faint signal.

Fifth, the per-position feed-forward, the residual-plus-LayerNorm wrapping, and the depth. Attention
mixes information *across* days but does almost no nonlinear processing *within* a day, so between
attention sublayers I put the position-wise feed-forward applied identically to every day (the
`64→2048→64` MLP whose parameter mass I just counted), giving the model somewhere to transform each day's
mixed representation nonlinearly. I wrap each sublayer as `x + Sublayer(x)` so the identity path keeps
gradients flowing through depth, and LayerNorm normalizes each day-vector across its own 64 features,
which is batch-size-invariant and immune to variable, padded batches. I stack `num_layers = 2` such
layers — shallow on purpose, because depth multiplies capacity and on this data capacity is liability.

The one hyperparameter I am most wary of is where I expect trouble. This runs `dropout = 0`, Adam at a
very small `lr = 1e-4`, weight decay `1e-3`, batch size 2048, MSE loss, early-stop patience 5. That
`lr = 1e-4` is an order of magnitude below what a less initialization-sensitive architecture on this data
tolerates, and patience 5 rather than the generous twenty a forgiving model would get — both signals that
this model is touchy to train. Transformers are brittle at initialization: random projections can make
the dot-product logits large, the softmax saturates early (the same Jacobian collapse, now at the
*start* of training), the first gradients are large and noisy, and Adam — dividing by a running estimate
of squared-gradient magnitude built from a few early samples — records those noisy gradients into its
second moment and distorts the step scale for many steps after. A from-scratch transduction transformer
defuses this with a warmup-then-inverse-sqrt schedule; but the editable `CustomModel` runs a *plain*
constant-rate Adam with no warmup. So I am training the most initialization-sensitive architecture on the
ladder with the schedule that does the least to protect its first steps, on the noisiest data, with the
smallest attention width but the largest raw parameter count. A constant small rate that never warms up
can either crawl too slowly to escape a bad initial basin or, on a few bad early batches, still take the
destabilizing step warmup was supposed to prevent.

How many optimizer steps does this leash actually allow? The csi300 training span is 2008–2014, roughly
seven years; with on the order of three hundred names and about two hundred fifty trading days a year, the
training segment is on the order of a few hundred thousand stock-days — call it `~5·10⁵`. At batch size
2048, with the loop *dropping* the final partial batch each epoch, that is on the order of `250` gradient
steps per epoch. Patience 5 means the run can terminate five epochs past the best validation score — so a
bad early trajectory has only a low-thousands of steps to right itself. On a warmup-less transformer that
is genuinely little room, and it is the concrete reason I think this is the *risky* opening rather than
the safe one.

Mechanically the read-out is the last position: after the stack I take the day-60 representation and a
`Linear(d_model, 1)` maps it to the score. Day 60 is the most recent observation and, having attended
over all sixty days, its representation is the encoder's summary of the whole window conditioned on
"now." One shape trace matters, to be sure the reshape recovers the sequence the attention needs. A batch
arrives as `[N, 360]`; `src.reshape(N, 6, 60).permute(0, 2, 1)` reads the flat 360 as feature-major (the
first sixty numbers one base ratio across sixty days, and so on), reshapes to `[N, 6, 60]` indexed
`[stock, feature, day]`, then permutes to `[N, 60, 6]` indexed `[stock, day, feature]` — so row `t` is
exactly the six ratios *on day t*. That is the "length-60 sequence of 6-vectors" the bias assumes,
contingent on the feature-major flat layout, which is what makes the permute right rather than a scramble.
From there `feature_layer` lifts to `[N, 60, 64]`, a transpose to `[60, N, 64]` puts time first for the
non-batch-first encoder, the positional code adds, two encoder layers preserve `[60, N, 64]`, a transpose
back and the `[:, -1, :]` slice picks the day-60 row `[N, 64]`, and the read-out maps it to `[N]` scores —
the `-1` selecting the most-recent day, not the first. Training is masked MSE over the finite labels (NaN
targets dropped), gradient value-clipping at 3.0 to cap the exploding side, and early stopping on the
validation score with the best parameters restored. The full module is in the answer.

So against the default Ridge, which fit one global linear map on the flat 360-vector with no notion of the
window, this treats the input as a 60-step sequence of 6-vectors, lets every day attend to every other
through scaled multi-head self-attention with sinusoidal position, and reads the most-recent day's summary
down to a score. The inductive bias is finally correct — the model *knows* it is looking at a time window.

My honest expectation, given the brittleness laid out above, is that this is the risky opening, not the
safe one. If the constant `1e-4` Adam threads the needle between too-slow and unstable, the attention bias
could pay off and the IC could reach the low-single-digit range a working temporal model on this data
gets. But the failure I am bracing for is that it *does not catch*: a warmup-less transformer at a tiny
constant rate with patience-5 early stopping on a faint signal can early-stop on a barely-trained model
whose predictions are near noise — IC near zero. The sharper, still a-priori tell is a *negative*
portfolio information ratio: a near-noise ranking is still forced through TopkDropout to hold fifty names
and churn five a day, paying transaction costs on noise, so its return can go negative and its IR with it.
If that is what lands, the diagnosis writes itself — I do not need *more* architecture, I need a learner
that is either robust to this data without delicate optimization (a tree on the flat features, no init
sensitivity at all) or a temporal model whose training is far less brittle than a warmup-less
transformer's. That is the fork the next two attempts take.
