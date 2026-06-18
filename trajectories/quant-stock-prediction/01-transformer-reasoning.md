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

Second, position. Pure self-attention is permutation-equivariant — `softmax(QKᵀ/√d_k)V` is all dot
products and weighted sums over the *set* of positions, with no term that knows which day is which. If I
shuffle the sixty days the output shuffles identically and nothing else changes, so the layer literally
cannot tell a recent observation from an old one. For a window where recency obviously matters, that is
fatal, so I must inject order. I add a sinusoidal positional code to each lifted day-vector before the
attention stack: `PE(pos, 2i) = sin(pos/10000^{2i/d})`, `PE(pos, 2i+1) = cos(pos/10000^{2i/d})`. I add
rather than concatenate because a learned linear over the sum `e + p` is `We + Wp` — the downstream
projections can already separate content from position into different directions of the 64-dim space —
whereas concatenation would widen every matrix in the stack for marginal benefit. Sinusoids rather than
a learned position table because each pair of dimensions is a sine/cosine at its own frequency, and a
shift by a fixed offset `k` becomes a fixed, position-independent rotation of that pair; that lets a head
learn a relative rule like "look a few days back" that applies uniformly along the window. With only 60
fixed positions the learned-table option would also work, but the sinusoid is the canonical, parameter-
free choice and there is no extrapolation concern either way at this fixed length.

Third, the attention primitive and its one wart. I score query against key by a dot product and softmax
the result, then mix the values: `softmax(QKᵀ/√d_k)V`. The `1/√d_k` is not decorative. If `q` and `k`
components are roughly unit-variance and independent, then `q·k = Σ_{i=1}^{d_k} q_i k_i` has mean 0 and
variance `d_k`, so its typical magnitude grows like `√d_k`. Feed logits of size `√d_k` into a softmax
and it saturates toward one-hot; its Jacobian `p_i(δ_ij − p_j)` then collapses toward zero, and the
attention weights stop receiving gradient — the model freezes in a near-argmax it cannot learn its way
out of. Dividing the scores by `√d_k` puts the logit variance back at 1 and keeps the softmax in its
responsive region. With `d_k = 32` here the effect is real, so the scaling stays.

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
transform each day's mixed representation nonlinearly. I wrap each sublayer as `x + Sublayer(x)` so the
identity path keeps gradients flowing through depth, and LayerNorm — normalizing each day-vector across
its own 64 features, which is batch-size-invariant and immune to the variable, padded batches — to keep
the residual scale from drifting. I stack `num_layers = 2` such encoder layers. Two is shallow on
purpose: depth multiplies capacity, and on this data capacity is liability.

Now the one hyperparameter I am most wary of, and it is where I expect trouble. The qlib Alpha360
Transformer benchmark sets `dropout = 0` here, with Adam at a very small `lr = 1e-4` and weight decay
`1e-3`, batch size 2048, MSE loss, early-stop patience 5. That `lr = 1e-4` is an order of magnitude
below the LSTM's `1e-3`, and the early-stop patience is 5 rather than 20 — both are signals that this
model is touchy to train. Transformers are known to be brittle at initialization: random projections can
make the dot-product logits large, the softmax saturates early (the same Jacobian collapse from the
`√d_k` discussion, but now at the *start* of training), the first gradients are large and noisy, and
Adam — which divides by a running estimate of the squared-gradient magnitude built from only a few early
samples — records those noisy gradients into its second moment and distorts the step scale for many
steps after. A from-scratch transduction transformer defuses this with a warmup-then-inverse-sqrt
learning-rate schedule; but the harness's `CustomModel` runs a *plain* constant-rate Adam with no
warmup. So I am training the most initialization-sensitive architecture on the ladder with the schedule
that does the least to protect its first steps, on the noisiest data, with the smallest model. The tiny
constant `lr` and patience-5 early stop are the only guards, and they may not be enough — a constant
small rate that never warms up can either crawl too slowly to escape a bad initial basin, or, on a few
bad early batches, still take the destabilizing step the warmup was supposed to prevent.

Mechanically the read-out is the last position. After the stack I take the day-60 representation —
`output.transpose(1,0)[:, -1, :]` in the loop's `[T, N, F]` convention — and a `Linear(d_model, 1)`
maps it to the single score. Why the last day rather than a pooled summary? Because day 60 is the most
recent observation and, having attended over all sixty days, its representation is the encoder's summary
of the whole window conditioned on "now." Training is masked MSE over the finite labels (NaN targets
dropped), gradient value-clipping at 3.0 to cap the exploding side, and early stopping on the validation
score with the best parameters restored. The reshape into the model is the loop's `src.reshape(N, 6,
60).permute(0, 2, 1) -> [N, 60, 6]`, so the time axis the attention needs is recovered from the flat
feature vector. The full scaffold module is in the answer.

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
come are concrete. First, signal IC near zero — if the csi300 IC comes back around 0.01 rather than the
0.04–0.05 a working temporal model reaches, that is the under-trained-transformer failure, not a
property of attention. Second, and the sharper tell: a *negative* portfolio information ratio. A model
whose ranking is essentially noise will still be forced through TopkDropout to hold 50 names and churn
5 a day, paying transaction costs on noise, so its annualized return can go negative and its IR with it
— which would be the cleanest evidence that the signal never formed. If that is what I see, the
diagnosis for the next rung writes itself: I do not need *more* architecture, I need a learner that is
either robust to this data without delicate optimization (a tree on the flat features, no init
sensitivity at all) or a temporal model whose training is far less brittle than a warmup-less
transformer's. That is the fork the next two rungs take.
