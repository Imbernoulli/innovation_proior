I am starting the ladder here, with a Transformer encoder over the 60-day window, and I should be
honest about why I start with a sequence model at all rather than the scaffold's Ridge or a tree. The
Alpha360 feature vector is not really 360 unrelated numbers; it is six base ratios — open, close, high,
low, volume, vwap, each normalized to the latest close — unrolled across sixty trading days. A flat
regressor or a tree sees `[N, 360]` and has no idea that column 5 and column 65 are the *same* base
ratio one day apart. The temporal structure, which is the whole reason a 60-day window is the input, is
invisible to it. So the first thing worth trying is a model whose inductive bias is exactly "this is a
length-60 sequence of 6-dimensional observations," and whose job is to summarize that window into one
forward-return score per stock-day. That is a sequence-to-one regression, and I want to start from the
sequence architecture that, in principle, lets any day in the window talk to any other day in one hop.

Before I commit to attention I should walk the two other sequence-shaped options I could reach for
first, because "temporal bias" alone does not single out a transformer. The cheapest temporal model is a
1-D convolution along the 60-day axis: a stack of kernels sliding over time, each layer mixing a small
local neighborhood. It is fast and it is not initialization-sensitive. But its reach is bounded by
receptive field, and I can size that on paper. A convolution with kernel size `k` and `L` stacked layers
(stride 1, no dilation) has receptive field `L·(k−1)+1`. To let day 1 influence day 60 I need
`L·(k−1)+1 ≥ 60`, so with `k=3` that is `L ≥ 30` layers — thirty convolutions deep just to connect the
ends of the window once. Dilation shortcuts this to `L ~ log₂(60) ≈ 6` layers, but every dilated layer
still only sees a *fixed* stride pattern; it cannot decide, per stock-day, that today's move should be
compared to a spike forty-one days ago rather than forty. The comparison structure is baked into the
architecture, not learned from content. Attention, by contrast, connects the two ends of the window in a
*single* layer with a *content-dependent* weight, which is exactly the property I want when I do not know
in advance which lag matters. The second option is a recurrence, and it earns its own paragraph.

There is also the shallow feedforward net from the standard baselines — an MLP straight on the flat
360-vector — and it is worth being explicit about why "add nonlinearity" is not the same as "add the
right bias." An MLP does lift the linear scaffold to nonlinear capacity, but it is still *flat*: its
first layer is one big `Linear(360, h)`, which learns an independent weight for every one of the 360
input columns with no constraint tying column 5 and column 65 (the same ratio one day apart) to a shared
transform. It can in principle *rediscover* the 60×6 structure, but only by spending parameters and data
to learn from scratch a regularity I already know for free. A single hidden layer of width 256 on the
flat input is already `360·256 ≈ 9.2·10⁴` first-layer weights spent purely to relearn that the columns
come in six ratios across sixty lags — capacity I would rather not hand a noisy regression. The sequence
models below encode that regularity as a hard architectural fact (weight-shared over days), so they
never spend a parameter learning it. That is the difference between more capacity and better-placed
capacity, and on a low-signal problem the latter is what I want.

Let me be precise about what "one hop" buys here and why I reach for attention rather than a recurrence
for the *first* attempt. In a recurrent encoder the state at day `t` is `h_t = f(h_{t-1}, x_t)`, so the
information from day 1 has to survive sixty sequential transformations before it can influence the
read-out at day 60 — and each transformation can attenuate it. An attention layer instead forms, for
each position, a weighted average over *all* positions at once: every day reaches every other day in a
single content-based lookup, with no distance-dependent attenuation in the routing. For a window where a
move sixty days ago might matter as much as yesterday's, that any-to-any reach is the appealing
property. So the architecture I drop into the editable `CustomModel` is an encoder built from
self-attention: project each day's 6 features up to a working width, run a couple of self-attention
layers over the 60 positions, and read out a single score.

Now I have to be careful, because the harness fixes the shape of this and several of the moves I would
make in a from-scratch transduction model simply do not apply here. This is an *encoder-only*,
*sequence-to-one* regressor — there is no decoder, no autoregressive generation, no target sequence. So
there is no causal mask (every day may legitimately attend to every other day in the window; nothing is
"in the future" to be hidden — the label is strictly outside the window), no cross-attention, no tied
input/output embeddings, no label smoothing, and no token vocabulary at all. The "tokens" are
continuous 6-vectors, one per day. The only pieces of the full attention machinery I actually keep are:
a linear that lifts each day's 6 features to the model width, positional information so the layer is not
order-blind, the scaled-dot-product self-attention with multiple heads, the position-wise feed-forward,
the residual-plus-LayerNorm glue, and a final linear that reads the last position's representation down
to one number. Everything that belonged to *generation* is dropped because this task does not generate —
it scores.

Let me walk the load-bearing choices that do survive, because each one is forced. First, lifting the
input. Each day arrives as a 6-vector, which is far too thin to host multi-head attention — I want to
slice the representation into several heads, each in its own subspace, and 6 dimensions cannot be
meaningfully split. So a `feature_layer = Linear(6, d_model)` lifts each day to `d_model = 64`. That
width is the qlib Alpha360 benchmark setting and it is a deliberate choice: 64 is wide enough to host
`nhead = 2` heads of 32 dimensions each, while staying small enough that an `n²·d` attention over only
60 positions is cheap and, more importantly, does not have so much capacity that it memorizes the noise
in a notoriously low signal-to-noise regression. Financial cross-sectional return prediction has an IC
ceiling in the low single-digit percents; the danger is never underfitting, it is fitting noise. A
narrow model is a regularizer.

I want to pin down exactly how much capacity is actually in this "narrow" model, because the width alone
is misleading, and this is where I first hit something uncomfortable. Count the parameters. The input
lift is `6·64 + 64 = 448`. Each `TransformerEncoderLayer` carries a self-attention block whose fused
Q/K/V projection is `3·64·64 + 3·64 = 12288 + 192` and whose output projection is `64·64 + 64 = 4160`,
so about `16640` attention parameters — but the position-wise feed-forward defaults, in the PyTorch
layer I am instantiating, to an inner width of `2048`, so it is `Linear(64, 2048)` then
`Linear(2048, 64)`, i.e. `(64·2048 + 2048) + (2048·64 + 64) = 133120 + 131136 = 264256` parameters, plus
two LayerNorms of `128` each. One encoder layer is therefore about `281152` parameters, two layers about
`562304`, and the read-out `Linear(64, 1)` adds `65`, for roughly `563k` total. The striking part is
where those parameters live: the two feed-forwards alone are `528512` of them, about `94%`. So the model
I keep calling "narrow" is narrow only in its *attention* width; the bulk of its capacity is in a wide
per-position MLP that the `d_model = 64` framing hides entirely. That reframes what is actually doing the
regularizing here. It is not the width — the FFN is wide. The real restraints are the shallow depth (two
layers), the tiny learning rate, weight decay `1e-3`, and early stopping. With `dropout = 0` in this
benchmark, I have *removed* the one regularizer built into the block, which sharpens my worry: a 563k-
parameter model with a 94%-FFN mass and no dropout, fit on a faint signal, is over-capacitied unless the
optimizer is kept on a very short leash. That is consistent with the benchmark's choice of a minute
learning rate, and it tells me the failure mode to brace for is variance, not bias.

Second, position. Pure self-attention is permutation-equivariant — `softmax(QKᵀ/√d_k)V` is all dot
products and weighted sums over the *set* of positions, with no term that knows which day is which. If I
shuffle the sixty days the output shuffles identically and nothing else changes, so the layer literally
cannot tell a recent observation from an old one. For a window where recency obviously matters, that is
fatal, so I must inject order. I add a sinusoidal positional code to each lifted day-vector before the
attention stack: `PE(pos, 2i) = sin(pos/10000^{2i/d})`, `PE(pos, 2i+1) = cos(pos/10000^{2i/d})`. It is
worth checking that these frequencies actually resolve a 60-day window rather than wasting their range.
With `d_model = 64` the exponents `2i/d` run `i = 0 … 31`, so the wavelengths are
`2π·10000^{i/32}`. The fastest pair (`i = 0`) has wavelength `2π ≈ 6.3` positions, completing about nine
and a half cycles across sixty days — fine enough to separate nearby days. The slowest pair (`i = 31`)
has wavelength `2π·10000^{31/32} ≈ 2π·7500 ≈ 47000` positions, so over a mere sixty days it is
essentially a monotone linear ramp — a coarse absolute-position coordinate. So the code hands the
attention both a fine ruler and a coarse "how far into the window" signal, and nothing in the frequency
budget is wasted at this length. I add rather than concatenate because a learned linear over the sum
`e + p` is `We + Wp` — the downstream projections can already separate content from position into
different directions of the 64-dim space — whereas concatenation would widen every matrix in the stack
for marginal benefit. Sinusoids rather than a learned position table because each pair of dimensions is a
sine/cosine at its own frequency, and a shift by a fixed offset `k` becomes a fixed, position-independent
rotation of that pair; that lets a head learn a relative rule like "look a few days back" that applies
uniformly along the window. With only 60 fixed positions the learned-table option would also work, but
the sinusoid is the canonical, parameter-free choice and there is no extrapolation concern either way at
this fixed length.

Third, the attention primitive and its one wart. I score query against key by a dot product and softmax
the result, then mix the values: `softmax(QKᵀ/√d_k)V`. The `1/√d_k` is not decorative. If `q` and `k`
components are roughly unit-variance and independent, then `q·k = Σ_{i=1}^{d_k} q_i k_i` has mean 0 and
variance `d_k`, so its typical magnitude grows like `√d_k`. Feed logits of size `√d_k` into a softmax
and it saturates toward one-hot; its Jacobian `p_i(δ_ij − p_j)` then collapses toward zero, and the
attention weights stop receiving gradient — the model freezes in a near-argmax it cannot learn its way
out of. With `d_k = 32` here that is a factor of `√32 ≈ 5.66` on the logit scale — the difference between
a softmax over logits of typical size `1` (responsive, near-uniform, full gradient) and one over logits
of size `~5.7` (already sharply peaked, most of its Jacobian near zero). That is not a marginal effect at
this width, so the scaling stays.

Fourth, multiple heads. A single softmax distribution per day is a single averaged summary, and an
average blurs: if a day needs to track a volume spike at one lag *and* a price reversal at another, one
head smears them together. So I run `nhead = 2` attentions in parallel over 32-dim projections, each
free to attend to a different pattern, and concatenate. Two heads is modest, but again the regime
rewards restraint — more heads is more capacity to overfit on a signal this faint, and the qlib
benchmark settled on two.

Fifth, the per-position feed-forward, the residual-plus-LayerNorm wrapping, and the depth. Attention
mixes information *across* days but does almost no nonlinear processing *within* a day — the only
nonlinearity in the attention sublayer is the softmax on the weights. So between attention sublayers I
put a position-wise feed-forward applied identically to every day, giving the model somewhere to
transform each day's mixed representation nonlinearly (this is the `64→2048→64` MLP whose parameter mass
I just counted). I wrap each sublayer as `x + Sublayer(x)` so the identity path keeps gradients flowing
through depth, and LayerNorm — normalizing each day-vector across its own 64 features, which is
batch-size-invariant and immune to the variable, padded batches — to keep the residual scale from
drifting. I stack `num_layers = 2` such encoder layers. Two is shallow on purpose: depth multiplies
capacity, and on this data capacity is liability.

It is also worth noting where the *compute* goes, because it is not where the parameters go for once. The
attention score matrix is `n²·d_model` work per layer, `60·60·64 ≈ 2.3·10⁵` multiply-adds — trivial. The
FFN does `2·64·2048 ≈ 2.6·10⁵` multiply-adds *per position*, times sixty positions, about `1.6·10⁷` per
layer: roughly seventy times the attention cost. So even though I chose attention for its any-to-any
routing, at a window this short the routing is nearly free and the model spends its flops in the
per-position MLP. The `n²` term that makes attention expensive on long sequences simply does not bite at
`n = 60`; if anything, this architecture is *under*-using the one thing attention is expensive-and-good
at, which is another quiet argument that the choice is about inductive bias, not scale.

Now the one hyperparameter I am most wary of, and it is where I expect trouble. The qlib Alpha360
Transformer benchmark sets `dropout = 0` here, with Adam at a very small `lr = 1e-4` and weight decay
`1e-3`, batch size 2048, MSE loss, early-stop patience 5. That `lr = 1e-4` is an order of magnitude below
what a less initialization-sensitive architecture on this same data can tolerate, and the early-stop
patience is 5 rather than the generous twenty a forgiving model would get — both are signals that this
model is touchy to train. Transformers are known to be brittle at initialization: random projections can
make the dot-product logits large, the softmax saturates early (the same Jacobian collapse from the
`√d_k` discussion, but now at the *start* of training), the first gradients are large and noisy, and
Adam — which divides by a running estimate of the squared-gradient magnitude built from only a few early
samples — records those noisy gradients into its second moment and distorts the step scale for many
steps after. A from-scratch transduction transformer defuses this with a warmup-then-inverse-sqrt
learning-rate schedule; but the harness's `CustomModel` runs a *plain* constant-rate Adam with no
warmup. So I am training the most initialization-sensitive architecture on the ladder with the schedule
that does the least to protect its first steps, on the noisiest data, with the smallest attention width
but the largest raw parameter count. The tiny constant `lr` and patience-5 early stop are the only
guards, and they may not be enough — a constant small rate that never warms up can either crawl too
slowly to escape a bad initial basin, or, on a few bad early batches, still take the destabilizing step
the warmup was supposed to prevent.

Let me also get the data arithmetic straight so I know how many optimizer steps this leash actually
allows. The csi300 training span is 2008–2014, roughly seven years; with on the order of three hundred
names and about two hundred fifty trading days a year, the training segment is on the order of a few
hundred thousand stock-days — call it `~5·10⁵`. At batch size 2048, and with the loop *dropping* the
final partial batch each epoch, that is on the order of `250` gradient steps per epoch. Patience-5 means
the run can terminate after as few as five epochs past the best validation score — so a bad early
trajectory has only a low-thousands of steps to right itself before early stopping fires. On a
warmup-less transformer that is genuinely little room, and it is the concrete reason I think this rung is
the *risky* opening rather than the safe one.

Mechanically the read-out is the last position. After the stack I take the day-60 representation —
`output.transpose(1,0)[:, -1, :]` in the loop's `[T, N, F]` convention — and a `Linear(d_model, 1)`
maps it to the single score. Why the last day rather than a pooled summary? Because day 60 is the most
recent observation and, having attended over all sixty days, its representation is the encoder's summary
of the whole window conditioned on "now." Let me trace the shapes once end to end to be sure the reshape
recovers the sequence the attention needs. A batch arrives as `src` of shape `[N, 360]`. The code does
`src.reshape(N, 6, 60).permute(0, 2, 1)`, which reads the flat 360 as feature-major (the first sixty
numbers being one base ratio across sixty days, and so on), reshapes to `[N, 6, 60]` indexed
`[stock, feature, day]`, then permutes to `[N, 60, 6]` indexed `[stock, day, feature]` — so row `t` of
each stock's matrix is exactly the six base ratios *on day t*. That is precisely the "length-60 sequence
of 6-vectors" the inductive bias assumes, contingent only on that feature-major flat layout, which is
what makes the permute the right one rather than a scramble. From there `feature_layer` lifts to
`[N, 60, 64]`, a transpose to `[60, N, 64]` puts time first for the non-batch-first encoder, the
positional code broadcasts and adds, the two encoder layers preserve `[60, N, 64]`, a transpose back and
the `[:, -1, :]` slice picks the day-60 row `[N, 64]`, and the decoder maps it to `[N, 1]`, squeezed to
`[N]` scores. The dimensions close, and the `-1` slice is indeed selecting the most-recent day, not the
first. Training is masked MSE over the finite labels (NaN targets dropped), gradient value-clipping at
3.0 to cap the exploding side, and early stopping on the validation score with the best parameters
restored. The full scaffold module is in the answer.

So the delta from the scaffold default is: where Ridge fit one global linear map on the flat 360-vector
with no notion of the window, this rung treats the input as a 60-step sequence of 6-vectors, lets every
day attend to every other day through scaled multi-head self-attention with sinusoidal position, and
reads the most-recent day's summary down to a score. The inductive bias is finally correct — the model
*knows* it is looking at a time window.

What do I expect, and where will it fall down? My honest expectation, given the brittleness I just
laid out, is that this is the *risky* opening rather than the safe one. If training catches — if the
constant `1e-4` Adam threads the needle between too-slow and unstable — the attention bias could pay
off and the signal IC could land in the same low-single-digit range the other temporal models reach.
But the failure mode I am bracing for is that it *does not catch*: a transformer trained without warmup,
at a tiny constant rate, with patience-5 early stopping on a faint signal, can early-stop on a barely-
trained model whose predictions are close to noise. The falsifiable expectations against the rungs to
come are concrete. First, signal IC near zero — if the csi300 IC comes back near noise rather than
the low-single-digit-percent range a working temporal model on this data reaches, that is the
under-trained-transformer failure, not a property of attention. Second, and the sharper tell: a *negative* portfolio information ratio. A model
whose ranking is essentially noise will still be forced through TopkDropout to hold 50 names and churn
5 a day, paying transaction costs on noise, so its annualized return can go negative and its IR with it
— which would be the cleanest evidence that the signal never formed. If that is what I see, the
diagnosis for the next rung writes itself: I do not need *more* architecture, I need a learner that is
either robust to this data without delicate optimization (a tree on the flat features, no init
sensitivity at all) or a temporal model whose training is far less brittle than a warmup-less
transformer's. That is the fork the next two rungs take.
