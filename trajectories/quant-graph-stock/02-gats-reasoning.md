The boosting floor landed about where I expected it to, and reading it carefully tells me exactly which
assumption to break next. On csi300 LightGBM gave IC 0.0399, ICIR 0.308, Rank IC 0.0492 — real,
positive, modest factor signal, the rank-normalized forest doing its honest job. But the portfolio side
is where the row-independence assumption shows: information ratio only 0.280 on csi300 and annualized
return 0.0203, and on the harder csi100 the portfolio actually loses money — IR −0.334, annualized
return −0.0172 — even though the *signal* IC there is still positive at 0.0363. The two universes carry
barely different signal quality: csi100 IC 0.0363 is 0.0363/0.0399 ≈ 0.91 of csi300's, nine-tenths of the
same predictive edge, yet the portfolio outcomes sit on opposite sides of zero, a swing of 0.61 in the
metric the backtest compounds. Nearly identical signal, opposite money. That is not a signal problem; it
is a portfolio-construction problem, and it is the one thing a per-row scorer cannot touch.

I can back out the mechanism rather than assert it. Information ratio is annualized return over
annualized volatility, so return/IR gives an implied portfolio volatility: csi300 0.0203/0.280 ≈ 0.072,
csi300_recent 0.0197/0.266 ≈ 0.074, csi100 −0.0172/−0.334 ≈ 0.052. The three top-50 books run at
comparable small volatilities — the IR differences are not coming from wildly different risk, they come
from the *sign and size of the return* the daily ranking earns. And the ICIR column says why the narrow
ranking is fragile: csi300 ICIR 0.308 versus csi100 0.220, so per unit of daily-IC volatility the csi100
signal is about a third less stable day to day. A less stable daily IC, on a universe with only a hundred
names to diversify across, is exactly the recipe for a top-50 whose realized return wanders below zero
even while the average correlation stays positive. The max-drawdown column agrees: csi100 bleeds to
−0.158 against csi300's −0.099 and recent's −0.076 — the deepest hole is dug precisely where the breadth
to climb out of it is thinnest.

One more reading isolates breadth as the lever rather than raw signal strength. csi300_recent has the
*weakest* signal of the three — IC only 0.0253, ICIR 0.182 — and yet its portfolio IR is 0.266,
essentially tied with csi300's 0.280 and firmly positive. A universe with two-thirds of csi300's IC but
the same 300-name breadth still compounds into a positive book, while csi100 with 90% of csi300's IC but
a third of the names goes negative. Breadth, not signal level, is deciding sign here. And the deterministic
GBDT gives *identical* numbers across all three seeds, so the variance between universes is structural,
not training noise: the floor has no mechanism to use the one thing the research question is about —
co-movement within the day's cross-section — so its weakness is the same every seed.

So the assumption to break is row-independence, and the minimal way to break it is to let the stocks
present on a given day *look at each other* before each one's prediction is read out. The form matters,
because the obvious textbook form is not what this scale rewards. The textbook form is a masked
graph-attention layer: attend each node over a neighborhood supplied by an adjacency, only edges in the
graph get a score. I do have a candidate adjacency — the stock-concept membership matrix the harness
loads — and could connect two stocks that share a concept and run masked multi-head attention over it,
the eight-head citation-network recipe. But masking exists at all because of cost: on a citation network
with millions of nodes the dense `N×N` score matrix is unaffordable, so you only score the edges the
graph gives you. Here `N` is one day's cross-section — a few hundred names for csi300, a hundred for
csi100 — so the dense score matrix is `300×300 ≈ 9·10⁴` floats, recomputed once per day; the cost
argument for masking simply does not bite. What masking *would* do here is throw away the ability to
discover relevance the curated graph never encoded — an unlabeled emerging theme, a freshly listed name
with no tags, two stocks that co-move for a reason no curator wrote down. The concept graph is static and
incomplete: the edges it has are frozen the same way every trading day, the edges it lacks it can never
propagate. And eight heads inflate parameters for no clear payoff at this data scale — with a faint signal
and a 2008–2014 training window, more heads is more to overfit. So the masked multi-head option costs me
discovery and regularization headroom to buy a compute saving I do not need.

Dot-product self-attention (`e_ij ∝ (W_q h_i)·(W_k h_j)/√d`) is cheap and standard, but it couples query
and key through a single bilinear form whose symmetry I don't want, and at init a dot-product over
64-dimensional hidden states produces sharply peaked softmaxes that would let one noisy stock dominate a
neighbor's representation before the encoder has adapted. The form I take is all-to-all *additive*
attention with a single head and a residual: every stock attends to every other stock present that day,
with the weight learned from the stocks' encoded features rather than read from any graph. The concept
matrix is not used at all this rung — a deliberate diff from the curated-graph story. If all-to-all
learned attention already lifts the portfolio numbers, that cleanly isolates "cross-section helps" from
"the curated graph helps," and a later rung can ask whether structured, concept-aware aggregation does
better still. So the design is: encode each stock temporally, let the day's stocks attend to each other
with one learned all-pairs attention, add the attended information back with a residual, and read out a
per-stock score.

Start with the encoder, because the attention needs a vector per stock and the raw input is a sequence.
Each stock-day is 360 Alpha360 features reshaped to a 60-day × 6-channel window; the harness hands me a
flat `[N, 360]` block, I reshape to `[N, 6, 60]` and permute to `[N, 60, 6]` so time is the sequence axis.
I run that through a recurrent backbone and take the last hidden state as `h_i ∈ R^{64}`. The harness's
GATs fill uses an LSTM backbone — `hidden_size=64`, `num_layers=2`, `dropout=0.7` — and *loads a
pretrained LSTM* (`model_lstm_csi300.pkl`) into the matching backbone weights before training. That
warm-start decides where gradient is spent. The two-layer LSTM is on the order of 5.2·10⁴ parameters, all
loaded from the pretrained single-stock predictor; the *new* machinery this rung adds is small — the
shared transform `W_t` (`64×64`, ≈4.2k), the attention vector `a ∈ R^{128}`, the post-attention `fc`
(`64×64`, ≈4.2k), call it ~8.4k new relational weights against ~52k warm-started ones. So about
six-sevenths of the model starts from a sensible temporal encoder and gradient descent spends its
capacity almost entirely on the one new thing, the neighbor weighting, rather than learning temporal
encoding and cross-sectional mixing from scratch at once. The high dropout (0.7) is the published Alpha360
setting, there for the same reason LightGBM needed enormous L1/L2 — the signal is faint and the model will
overfit the training window without aggressive regularization.

Now the attention itself, in the exact form the harness uses. After the backbone produces `h_i`, the
layer passes each through a shared linear transformation `t(h) = W_t h` — one transform, no per-head
projections, a *single* head, because at this data scale with heavy dropout on a pretrained encoder a
single head is the stable choice. Then it scores every ordered pair with the additive form: concatenate
the two transformed hidden states and project with a learned `a ∈ R^{128}`,
`e_ij = LeakyReLU(aᵀ[t(h_i) ‖ t(h_j)])`. Additive, not dot-product, so `a` holds separate weights for the
query half (first 64 entries multiply `t(h_i)`) and the key half (last 64 multiply `t(h_j)`), making the
score asymmetric — `e_ij ≠ e_ji` in general, which is right, because a large-cap leading a sector is more
relevant to its followers than the reverse — and giving the comparison its own trainable capacity
decoupled from `W_t`. LeakyReLU rather than ReLU so a low (negative) score — the layer saying "this stock
is *un*important to that one" — still carries gradient and can be learned down rather than dead-zeroed.
The layer materializes the full `[N, N]` pairwise score matrix in one matmul (a small tensor at this `N`),
then — the crucial no-mask step — applies `softmax` across the *entire* row over all `N` stocks
(`dim=1`), not over a masked neighborhood: `α_ij = exp(e_ij) / Σ_{k} exp(e_ik)`, the sum running over
every stock present that day. Each stock's attention is a full convex combination over the whole
cross-section, with the curated graph nowhere in sight, and the per-stock softmax normalizes so a 300-name
day and a 100-name day both produce weights summing to one — the same degree-invariance a masked layer
buys, now over the full set, which matters because the three universes hand me very different `N`.

Then the aggregation and read-out, with one more harness-specific choice worth comment. The attended
information is `Σ_j α_ij t(h_j)` — the attention-weighted combination of the day's transformed hiddens —
but the layer does not *replace* `h_i` with this; it *adds* it back with a residual:
`hidden_i = (Σ_j α_ij t(h_j)) + h_i`. The residual does real work, and a limiting trace shows it. Near
initialization, or in the failure mode where the learned attention collapses, if `a → 0` then
`e_ij = 0` for every pair, the softmax over equal scores is uniform, `α_ij = 1/N`, and the attended term
is exactly the *column mean* of the transformed hiddens — the day's average stock. Without the residual,
every stock's representation would then be identical, the day-mean, and all information about *which*
stock is which would be gone — the read-out would score every stock the same. With the residual that
degenerate case gives `hidden_i = mean + h_i`: each stock keeps its own warm-started signal and the
cross-section only adds a shared market-factor offset that shifts every stock equally and therefore does
not disturb the within-day ranking at all. So the residual guarantees a floor — in the worst case the
attention adds a harmless common term and the model degrades gracefully to the encoder's own ranking
rather than collapsing to the mean — which is precisely the inductive bias I want when I have warm-started
the encoder: the cross-section is a *correction* to a signal I already trust, not a replacement. After the
residual a learned `fc` plus another LeakyReLU mixes the representation, and `fc_out` reads out the scalar
score.

But the same trace names a malign case, and naming it now sets up the next rung. If the learned attention
becomes *near*-uniform-but-not-quite on a universe where the idiosyncratic spread between names is most of
the edge, the residual add pulls every stock toward the day's mean by an amount that no longer cancels,
and the within-day ranking flattens toward the market. On a low-breadth universe that flattening is
exactly how a model keeps a positive average correlation while producing a top-50 with no realized edge.
All-to-all attention with no structure has no way to *not* attend to the irrelevant stocks; it can only
down-weight them, and a soft down-weight spread over a hundred names still leaks the mean in. I am building
that risk in knowingly, because testing the pure cross-section hypothesis first is worth it — but I am
watching for its signature.

Training is per-day cross-sections — the whole attention computation is over the stocks present on a
single date, so a batch is one day's full set; mixing dates would mix unrelated cross-sections and the
attention would be meaningless across them. The daily batching is built by `groupby(level=0).size()` into
a cumulative index over dates, shuffled at the date level so epoch order varies without ever splitting a
day. The loss is MSE against the rank-normalized next-day-return label, masked to finite labels, Adam at
`lr=1e-4`, gradient-value clipping at 3.0 (the faint signal and the `N²` attention can spike gradients),
early stopping on validation with patience 20, up to 200 epochs, keeping best-validation parameters. The
full scaffold fill — the LSTM/GRU backbones, the `GATModel` with `cal_attention`, the daily-batched loop
and the pretrained warm-start — is in the answer.

One choice deserves a second look, because it sits in tension with what the task scores: early stopping
watches the validation *loss*, not validation IC. The scored objective is a ranking one, and MSE against a
`CSRankNorm`-ed label is only a surrogate — MSE penalizes getting the *magnitude* of each rank-normalized
target wrong, over-weighting the extreme names, whereas IC rewards getting the *ordering* right. They are
aligned (a predictor reproducing the normalized labels has IC 1 and loss 0) but along the path an epoch
that trims a few large-name errors can lower loss while barely moving day-averaged rank correlation. I
keep loss anyway, for two reasons I can weigh: the label is already rank-normalized so MSE operates on
ranks and the surrogate gap is small; and per-day validation IC is itself noisy on a hundred-name universe
(its ICIR of 0.22 on the floor's csi100 says daily IC swings by more than four times its mean), so a
loss-based gate can be the *steadier* early-stopping signal even though it is less direct. Moving the gate
onto validation IC is a free lever I am leaving untouched here.

Now the falsifiable expectations against the LightGBM floor, in the metrics the task reports. The thesis
is that letting the day's stocks attend to each other fixes the portfolio-noise problem, so the clearest
win should be in *information ratio*, not necessarily signal IC. I expect csi300 IR well above 0.280 and
annualized return above 0.0203, with max drawdown pulling in from −0.099 toward zero, and csi300_recent to
track it given the same breadth. On the signal metrics I expect IC to improve modestly over 0.0399 and
Rank IC over 0.0492, but the IR gain should dominate the IC gain in proportional terms, because IR is
precisely the metric that punishes the noisy daily ranking the floor could not avoid. csi100 is the sharp
test: does cross-stock attention rescue the low-breadth universe, or does a hundred names give the
all-to-all attention too little to work with? I expect its IR to improve off −0.334, but I would not be
surprised if it stays negative — a hundred-name universe is genuinely hard, and the malign near-uniform
failure mode I traced above should bite hardest exactly where the idiosyncratic spread is most of the
edge. If csi100's signal holds up while its portfolio still cannot cross zero, that would be the
over-smoothing fingerprint, and the remedy it points at is clear: stop attending to *everything*
indiscriminately, and stop the idiosyncratic signal from being averaged away in the first place. That is
the thread I would pull on next.
