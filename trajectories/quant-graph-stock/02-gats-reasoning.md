The boosting floor landed about where I expected it to, and reading it carefully tells me exactly which
assumption to break next. On csi300 LightGBM gave IC 0.0399, ICIR 0.308, Rank IC 0.0492 — real,
positive, modest factor signal, the rank-normalized forest doing its honest job. But the portfolio side
is where the row-independence assumption shows: information ratio only 0.280 on csi300 and annualized
return 0.0203, and on the harder csi100 the portfolio actually loses money — IR −0.334, annualized
return −0.0172 — even though the *signal* IC there is still positive at 0.0363. That gap is the tell.
The model can rank stocks well enough on average to post a positive IC, but the TopkDropout backtest
lives or dies on getting the *top of the daily ranking* right day after day, and a model that scores
each stock from its own 360 features, blind to what the rest of the cross-section is doing that day,
produces a daily top-50 that is too noisy to compound — especially on csi100 where there are only a
hundred names, so the breadth that averages out idiosyncratic noise is gone and the negative IR is the
direct consequence. The deterministic GBDT also gives identical numbers across all three seeds, which
confirms the variance I am seeing is not training noise — it is structural: the floor has no mechanism
to use the one thing the research question is about, co-movement within the day's cross-section.

So the assumption to break is row-independence, and the minimal way to break it is to let the stocks
present on a given day *look at each other* before each one's prediction is read out. That is the move
this rung makes, and I want to be careful about what form it takes, because the obvious textbook form
is not what this harness exposes or rewards. The textbook graph-attention layer attends a node over a
*masked* neighborhood — you supply an adjacency, only edges in the graph get a score, structure enters
as the mask. Here I do have a candidate adjacency: the stock-concept membership matrix the harness loads.
I could build a stock-stock graph by connecting two stocks that share a concept and run masked attention
over it. But pause on what that buys and costs. The concept graph is curated, static, and incomplete —
the edges it has are frozen the same way every day, and the edges it lacks (an unlabeled emerging theme,
a freshly listed name with no tags) it can never propagate. Worse, on a single day's cross-section the
number of stocks is small — a few hundred at most for csi300, a hundred for csi100 — so the *cost* reason
for masking (avoiding all-pairs `O(N²)` on a graph with millions of nodes) simply does not bite. With at
most a few hundred stocks in a day, the full `N×N` attention matrix is tiny and trivially affordable. The
masking that is essential on a citation network is, here, an unnecessary restriction that throws away the
chance to discover relevance the curated graph never encoded.

So I drop the mask. This rung's attention is *all-to-all over the day's cross-section*: every stock
attends to every other stock present that day, with the attention weight learned from the stocks'
encoded features rather than read from any graph. The concept matrix is not used at all in this rung —
that is a deliberate diff from the curated-graph story, and it is the right call for two reasons. First,
the cost argument above: full attention is cheap at this scale. Second, the diagnostic argument: I want
to test whether *any* learned cross-stock interaction beats the row-independent floor before I commit to
the specific structure the curated concepts impose; if all-to-all learned attention already lifts the
portfolio numbers, that isolates "cross-section helps" from "the curated graph helps," and the next rung
can then ask whether structured, concept-aware aggregation does better still. So the design is: encode
each stock temporally, let the day's stocks attend to each other with one learned all-pairs attention,
add the attended information back to each stock, and read out a per-stock score.

Start with the encoder, because the attention needs a vector per stock and the raw input is a sequence.
Each stock-day is 360 Alpha360 features that reshape to a 60-day × 6-channel window. I run that through a
recurrent backbone and take the last hidden state as the stock's representation `h_i`. The harness's GATs
fill uses an LSTM backbone here, not a GRU — `hidden_size=64`, `num_layers=2`, `dropout=0.7` — and it
*loads a pretrained LSTM* (`model_lstm_csi300.pkl`) into the matching backbone weights before training,
so the recurrent encoder starts from a sensible single-stock predictor and the attention layer learns the
cross-stock correction on top of it. That pretraining-then-attend structure matters: it means the
attention is not trying to learn temporal encoding and cross-sectional mixing simultaneously from scratch;
the temporal part is warm-started, and gradient descent spends its capacity on the new thing, the
neighbor weighting. The high dropout (0.7) is the published Alpha360 setting and is there for the same
reason LightGBM needed enormous L1/L2 — the signal is faint and the model will overfit the 2008–2014
window without aggressive regularization.

Now the attention itself, and here I have to land the *exact* form the harness uses, which differs from
the canonical multi-head masked layer in several deliberate ways. After the backbone produces the last
hidden state `h_i ∈ R^{64}` for every stock in the day, the layer first passes each through a shared
linear transformation `t(h) = W_t h` — one transform, no per-head projections; this rung uses a *single*
attention head, not the eight-head concatenation of the citation-network setup, because at this data
scale and with heavy dropout a single head trained on top of a pretrained encoder is the stable choice
the benchmark settled on. Then it scores every ordered pair. The score is the additive form: for stocks
`i` and `j`, concatenate their transformed hidden states and project with a learned vector `a ∈ R^{128}`,
`e_ij = LeakyReLU(aᵀ[t(h_i) ‖ t(h_j)])`. Additive, not dot-product, for the same reason as in any
attention design — the concatenation lets `a` hold separate weights for the query half and the key half,
so the score is asymmetric (stock `i`'s relevance to `j` need not equal `j`'s to `i`, which is right: a
large-cap leading a sector is more relevant to its followers than the reverse) and the comparison has its
own trainable capacity decoupled from `W_t`. LeakyReLU rather than ReLU so that a low (negative) score —
the layer saying "this stock is *un*important to that one" — still carries gradient and can be learned
down rather than dead-zeroed.

The implementation detail that makes this affordable is that the layer materializes the full pairwise
score matrix directly: it expands the `[N, dim]` transformed hidden into an `[N, N, dim]` tensor of the
query side, transposes for the key side, concatenates along the feature axis to `[N, N, 2·dim]`, and
applies `a` to get the `[N, N]` score matrix in one shot. With `N` at most a few hundred that is a small
tensor. Then — and this is the crucial no-mask step — it applies `softmax` across the *entire* row, over
all `N` stocks, not over a masked neighborhood: `α_ij = softmax_j(e_ij) = exp(e_ij) / Σ_{k} exp(e_ik)`,
the sum running over every stock in the day. So each stock's attention is a full convex combination over
the whole cross-section, with the curated graph nowhere in sight. The softmax normalizes per stock so a
day with 300 names and a day with 100 names both produce comparable per-stock weights summing to one,
which is the same degree-invariance argument as the masked version but now over the full set.

Then the aggregation and read-out, and there is one more harness-specific choice that earns comment. The
attended information is `Σ_j α_ij t(h_j)` — the attention-weighted combination of the day's transformed
hidden states for stock `i`. But the layer does not *replace* `h_i` with this; it *adds* it back with a
residual: `hidden_i = (Σ_j α_ij t(h_j)) + h_i`. The residual is doing real work. Without it, a stock's
representation after one attention layer is purely a mixture of the cross-section, and if the attention
is noisy early in training the stock loses its own signal entirely. With the residual, the attended
cross-stock information is a *correction* added to the stock's own pretrained-encoder hidden state — the
floor signal is preserved and the cross-section only adjusts it, which is exactly the inductive bias I
want when I have warm-started the encoder. After the residual, a learned `fc` linear layer plus another
LeakyReLU mixes the combined representation, and a final `fc_out` reads out the scalar score per stock.

Training is per-day cross-sections — the whole attention computation is over the stocks present on a
single date, so a batch is one day's full set; mixing dates would mix unrelated cross-sections and the
attention would be meaningless across them. The loss is MSE against the rank-normalized next-day-return
label, masked to finite labels, optimized with Adam at `lr=1e-4`, with gradient-value clipping at 3.0,
early stopping on validation IC with patience 20, up to 200 epochs. The label normalization is the
`CSRankNorm` the handler applies, so the model is again learning to order within a day. The full scaffold
fill — the LSTM/GRU backbones, the `GATModel` with `cal_attention`, the daily-batched train/test/predict
loop — is in the answer.

Now the falsifiable expectations against the LightGBM floor. The whole thesis of this rung is that
letting the day's stocks attend to each other fixes the portfolio-noise problem the row-independent floor
had, so the place I expect the clearest win is *information ratio*, not necessarily signal IC. I expect
csi300 IR to jump well above LightGBM's 0.280 — a learned cross-sectional smoothing should produce a far
more stable daily top-50 — and annualized return to rise correspondingly above 0.0203. On the signal
metrics I expect IC to improve modestly over 0.0399 (the cross-section carries some extra predictability),
but the IR gain should dwarf the IC gain, because IR is precisely the metric that punishes the noisy
daily ranking the floor could not avoid. On csi100, where LightGBM posted a *negative* IR of −0.334, the
test is sharper: does cross-stock attention rescue the low-breadth universe, or does a hundred names give
the all-to-all attention too little to work with? I expect csi100 IR to improve markedly from the floor's
−0.334 — the attention should at least stop the bleeding — though it may well stay negative, because a
hundred-name universe is genuinely hard and the attention is learning over a small set. The one risk I am
explicitly leaving on the table for the next rung: all-to-all attention with no structure can over-smooth
— pull every stock toward the day's mean and wash out the idiosyncratic signal that, on csi100
especially, is much of what there is. If I see csi300 IR rise strongly while csi100 stays negative, that
is the signature of exactly that over-smoothing, and the fix is to stop attending to *everything* and
instead aggregate through concept structure that keeps a stock's own and its theme's signal separate —
which is the next rung.
