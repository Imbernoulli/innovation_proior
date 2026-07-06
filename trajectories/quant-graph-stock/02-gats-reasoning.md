The boosting floor landed about where I expected it to, and reading it carefully tells me exactly which
assumption to break next. On csi300 LightGBM gave IC 0.0399, ICIR 0.308, Rank IC 0.0492 — real,
positive, modest factor signal, the rank-normalized forest doing its honest job. But the portfolio side
is where the row-independence assumption shows: information ratio only 0.280 on csi300 and annualized
return 0.0203, and on the harder csi100 the portfolio actually loses money — IR −0.334, annualized
return −0.0172 — even though the *signal* IC there is still positive at 0.0363. That gap is the tell,
and I want to make it precise before I trust it, because a story I can't put numbers to is a story I'll
over-fit to. So take the two universes side by side. The signal quality is barely different: csi300 IC
0.0399 versus csi100 IC 0.0363, the narrow universe carrying 0.0363/0.0399 ≈ 0.91 of the broad one's
correlation — nine-tenths of the same predictive edge. Yet the portfolio outcomes are on opposite sides
of zero: IR 0.280 against −0.334, a swing of 0.61 in the metric the backtest actually compounds. Nearly
identical signal, opposite money. That is not a signal problem; it is a portfolio-construction problem,
and it is the one thing a per-row scorer cannot touch.

I can push the arithmetic one notch further to see the mechanism rather than just assert it. Information
ratio is annualized return over annualized volatility, so the ratio return/IR backs out an implied
portfolio volatility: csi300 gives 0.0203/0.280 ≈ 0.072, csi300_recent 0.0197/0.266 ≈ 0.074, and csi100
−0.0172/−0.334 ≈ 0.052. The three top-50 books run at comparable, small volatilities — the differences
in IR are not coming from wildly different risk, they are coming from the *sign and size of the return*
the daily ranking earns. And the ICIR numbers say why the ranking is fragile on the narrow universe:
csi300 ICIR 0.308 versus csi100 ICIR 0.220, a ratio of 0.220/0.308 ≈ 0.71, so per unit of daily-IC
volatility the csi100 signal is about a third less stable day to day. A less stable daily IC, on a
universe with only a hundred names to diversify across, is exactly the recipe for a top-50 whose realized
return wanders below zero even while the average correlation stays positive. The max-drawdown column
agrees: csi100 bleeds to −0.158 against csi300's −0.099 and recent's −0.076 — the deepest hole is dug
precisely where the breadth to climb out of it is thinnest.

One more reading of the table, because it isolates breadth as the lever rather than raw signal strength.
csi300_recent has the *weakest* signal of the three — IC only 0.0253, ICIR 0.182, the lowest of every
universe — and yet its portfolio IR is 0.266, essentially tied with csi300's 0.280 and firmly positive.
A universe with two-thirds of csi300's IC but the same 300-name breadth still compounds into a positive
book, while csi100 with 90% of csi300's IC but a third of the names goes negative. Breadth, not signal
level, is deciding sign here. I also notice Rank IC exceeds IC on every universe — csi300 0.0492 vs
0.0399, csi100 0.0445 vs 0.0363, and most strikingly recent 0.0409 vs 0.0253, a 62% rank-over-linear gap
— which tells me the predictable part of the label is monotone-but-nonlinear, the tails carrying more
order information than a linear IC captures; the label is `CSRankNorm`-ed, so a model that thinks in
ranks has room the linear forest is leaving on the floor. The deterministic GBDT also gives *identical*
numbers across all three seeds {42, 123, 456}, which is not a curiosity but a control: it confirms the
variance I am seeing between universes is not training noise, it is structural. The floor has no mechanism
to use the one thing the research question is about — co-movement within the day's cross-section — so its
answer is the same every seed and its weakness is the same every seed.

So the assumption to break is row-independence, and the minimal way to break it is to let the stocks
present on a given day *look at each other* before each one's prediction is read out. That is the move
this rung makes, and I want to be careful about what form it takes, because the obvious textbook form
is not what this harness exposes or rewards. Before I commit, let me lay out the moves actually available
to me right now and walk the tempting ones far enough to reject them on arithmetic rather than taste.

The first candidate is the textbook graph-attention layer: attend each node over a *masked* neighborhood
supplied by an adjacency, only edges in the graph get a score, structure enters as the mask. I do have a
candidate adjacency — the stock-concept membership matrix the harness loads — and I could build a
stock-stock graph by connecting two stocks that share a concept and run masked, multi-head attention over
it, the eight-head citation-network recipe. Pause on what that buys and costs, concretely. The reason
masking exists at all is cost: on a citation network with millions of nodes the dense `N×N` score matrix
is unaffordable, so you only ever score the edges the graph gives you. Here `N` is one day's
cross-section — at most a few hundred names for csi300, a hundred for csi100. The dense score matrix is
then `300 × 300 = 9·10⁴` floats, about 360 KB at fp32, recomputed once per day; that is nothing. The cost
argument for masking simply does not bite at this scale. What masking *would* do here is throw away the
ability to discover relevance the curated graph never encoded — an unlabeled emerging theme, a freshly
listed name with no tags, two stocks that co-move for a reason no curator wrote down. The concept graph
is static and incomplete: the edges it has are frozen the same way every trading day, and the edges it
lacks it can never propagate. And the eight-head choice inflates parameters for no clear payoff at this
data scale — with a faint signal and a 2008–2014 training window, more heads is more to overfit. So the
masked multi-head option costs me discovery and regularization headroom while buying me a compute saving
I do not need. Rejected on the numbers, not on feel.

The second candidate is dot-product self-attention, the transformer form: score `e_ij ∝ (W_q h_i)·(W_k
h_j)/√d`. It is cheap and standard, but it couples query and key through a single bilinear form whose
symmetry I don't want and whose scale I'd have to manage; more to the point, at init a dot-product over
64-dimensional hidden states produces sharply peaked softmaxes that, before the encoder has adapted,
would let one noisy stock dominate a neighbor's representation. The third candidate — the one I take — is
all-to-all *additive* attention with a single head and a residual, which keeps the asymmetry I want, keeps
the attention gentle at init, and tests the cross-section hypothesis in its purest form.

So I drop the mask. This rung's attention is *all-to-all over the day's cross-section*: every stock
attends to every other stock present that day, with the attention weight learned from the stocks'
encoded features rather than read from any graph. The concept matrix is not used at all in this rung —
that is a deliberate diff from the curated-graph story, and it is the right call for two reasons. First,
the cost argument above: full attention is cheap at this scale. Second, the diagnostic argument: I want
to test whether *any* learned cross-stock interaction beats the row-independent floor before I commit to
the specific structure the curated concepts impose. If all-to-all learned attention already lifts the
portfolio numbers, that cleanly isolates "cross-section helps" from "the curated graph helps," and a
later rung can then ask whether structured, concept-aware aggregation does better still. So the design is:
encode each stock temporally, let the day's stocks attend to each other with one learned all-pairs
attention, add the attended information back to each stock, and read out a per-stock score.

Start with the encoder, because the attention needs a vector per stock and the raw input is a sequence.
Each stock-day is 360 Alpha360 features that reshape to a 60-day × 6-channel window; the harness hands me
a flat `[N, 360]` block, I reshape to `[N, 6, 60]` and permute to `[N, 60, 6]` so time is the sequence
axis and the six OHLC-style channels are the features. I run that through a recurrent backbone and take
the last hidden state as the stock's representation `h_i ∈ R^{64}`. The harness's GATs fill uses an LSTM
backbone here, not a GRU — `hidden_size=64`, `num_layers=2`, `dropout=0.7` — and it *loads a pretrained
LSTM* (`model_lstm_csi300.pkl`) into the matching backbone weights before training. That warm-start is
worth quantifying, because it tells me where gradient will actually be spent. The two-layer LSTM at these
sizes is roughly `4·(6·64 + 64·64)` weights in the first layer and `4·(64·64 + 64·64)` in the second,
plus biases — on the order of 5.2·10⁴ parameters — and every one of them is loaded from the pretrained
single-stock predictor. The *new* machinery this rung adds on top is small: the shared transform `W_t` at
`64×64` (≈4.2k), the attention vector `a ∈ R^{128}` (128), and the post-attention `fc` at `64×64` (≈4.2k)
— call it ~8.4k new relational weights against ~52k warm-started ones. So about six-sevenths of the model
starts from a sensible temporal encoder and gradient descent spends its capacity almost entirely on the
one new thing, the neighbor weighting — it is not trying to learn temporal encoding and cross-sectional
mixing simultaneously from scratch. The high dropout (0.7) is the published Alpha360 setting and is there
for the same reason LightGBM needed enormous L1/L2 — the signal is faint and the model will overfit the
2008–2014 window without aggressive regularization.

Now the attention itself, and here I have to land the *exact* form the harness uses, which differs from
the canonical multi-head masked layer in several deliberate ways. After the backbone produces `h_i` for
every stock in the day, the layer first passes each through a shared linear transformation `t(h) = W_t h`
— one transform, no per-head projections; this rung uses a *single* attention head, not the eight-head
concatenation of the citation-network setup, because at this data scale and with heavy dropout a single
head trained on top of a pretrained encoder is the stable choice. Then it scores every ordered pair. The
score is the additive form: for stocks `i` and `j`, concatenate their transformed hidden states and
project with a learned vector `a ∈ R^{128}`, `e_ij = LeakyReLU(aᵀ[t(h_i) ‖ t(h_j)])`. Additive, not
dot-product, for the reason I settled on above — the concatenation lets `a` hold separate weights for the
query half (the first 64 entries multiply `t(h_i)`) and the key half (the last 64 multiply `t(h_j)`), so
the score is asymmetric: `e_ij ≠ e_ji` in general, which is right, because a large-cap leading a sector
is more relevant to its followers than the reverse. And the comparison gets its own trainable capacity in
`a`, decoupled from `W_t`. LeakyReLU rather than ReLU so that a low (negative) score — the layer saying
"this stock is *un*important to that one" — still carries gradient and can be learned down rather than
dead-zeroed at the origin.

The implementation detail that makes this affordable is that the layer materializes the full pairwise
score matrix directly, and it is worth tracing the shapes once to be sure it is doing what I think. It
expands the `[N, 64]` transformed hidden into `e_x` of shape `[N, N, 64]` — the query side, constant
along the second axis — transposes the first two axes to get `e_y = [N, N, 64]` — the key side, constant
along the first — concatenates along the feature axis to `[N, N, 128]`, flattens to `[N², 128]`, and
applies `aᵀ` (shape `[1, 128]`) by `a_t.mm(attention_inᵀ)` to get `[1, N²]`, reshaped to the `[N, N]`
score matrix in one shot. With `N` at most a few hundred that is a small tensor and the whole thing is one
matmul. Then — and this is the crucial no-mask step — it applies `softmax` across the *entire* row, over
all `N` stocks, `dim=1`, not over a masked neighborhood: `α_ij = exp(e_ij) / Σ_{k=1}^{N} exp(e_ik)`, the
sum running over every stock present that day. So each stock's attention is a full convex combination over
the whole cross-section, with the curated graph nowhere in sight. The softmax normalizes per stock so a
day with 300 names and a day with 100 names both produce per-stock weights summing to one — the same
degree-invariance a masked layer buys, but now over the full set, which matters because the three
universes hand me very different `N` and I do not want the aggregate magnitude to scale with the count.

Then the aggregation and read-out, and there is one more harness-specific choice that earns comment. The
attended information is `Σ_j α_ij t(h_j)` — the attention-weighted combination of the day's transformed
hidden states. But the layer does not *replace* `h_i` with this; it *adds* it back with a residual:
`hidden_i = (Σ_j α_ij t(h_j)) + h_i`. The residual is doing real work, and I can show it with a limiting
trace. Consider the model near initialization, or in the failure mode where the learned attention has
collapsed to something uninformative: if `a → 0`, then `e_ij = LeakyReLU(0) = 0` for every pair, the
softmax over a row of equal scores is uniform, `α_ij = 1/N`, and the attended term becomes exactly the
*column mean* of the transformed hidden states — the day's average stock. Without the residual, every
stock's representation would then be identical, the day-mean, and all cross-sectional information about
*which* stock is which would be gone — the read-out would score every stock the same. With the residual,
that degenerate case gives `hidden_i = mean + h_i`: each stock keeps its own warm-started signal and the
cross-section only adds a shared market-factor offset that shifts every stock equally and therefore does
not disturb the within-day ranking at all. So the residual guarantees a floor: in the worst case the
attention adds a harmless common term and the model degrades gracefully to the encoder's own ranking,
rather than collapsing to the mean. That is precisely the inductive bias I want when I have warm-started
the encoder — the cross-section is a *correction* to a signal I already trust, not a replacement for it.
After the residual, a learned `fc` linear layer plus another LeakyReLU mixes the combined representation,
and a final `fc_out` reads out the scalar score per stock; a quick dimension check on the smallest case
`N = 1` confirms nothing breaks — `att_weight` is `[1, 1]`, softmax gives 1, `hidden = 1·h + h = 2h`, and
the read-out proceeds — so single-stock days do not crash the matmul.

I do note the risk this exact design leaves open, because naming it now is what sets up the next rung.
The trace above shows the *benign* degenerate case, an offset that cancels in the ranking. But there is a
malign one: if the learned attention becomes *near*-uniform-but-not-quite on a universe where the
idiosyncratic spread between names is most of the edge, the residual add pulls every stock toward the
day's mean by an amount that no longer cancels, and the within-day ranking flattens toward the market. On
a low-breadth universe that flattening is exactly how a model keeps a positive average correlation while
producing a top-50 with no realized edge. All-to-all attention with no structure has no way to *not*
attend to the irrelevant stocks; it can only down-weight them, and a soft down-weight spread over a
hundred names still leaks the mean in. I am building that risk in knowingly, because the diagnostic value
of testing the pure cross-section hypothesis first is worth it — but I am watching for its signature.

Training is per-day cross-sections — the whole attention computation is over the stocks present on a
single date, so a batch is one day's full set; mixing dates would mix unrelated cross-sections and the
attention would be meaningless across them. The daily batching is built by `groupby(level=0).size()` into
a cumulative index over dates, shuffled at the date level during training so epoch order varies without
ever splitting a day. The loss is MSE against the rank-normalized next-day-return label, masked to finite
labels, optimized with Adam at `lr=1e-4`, with gradient-value clipping at 3.0 (the faint signal and the
`N²` attention can spike gradients; clipping by value keeps a single explosive day from wrecking the
step), early stopping on the validation metric with patience 20, up to 200 epochs, keeping the
best-validation parameters. The label normalization is the `CSRankNorm` the handler applies, so the model
is again learning to order within a day. The full scaffold fill — the LSTM/GRU backbones, the `GATModel`
with `cal_attention`, the daily-batched train/test/predict loop and the pretrained warm-start — is in the
answer.

One choice in the fill deserves a second look before I accept it, because it sits in tension with what
the task actually scores: early stopping here watches the validation *loss*, not validation IC. The
scored objective is a ranking one — IC, Rank IC, and the IR that compounds them — and MSE against a
`CSRankNorm`-ed label is only a surrogate for it. The two are aligned but not identical: MSE penalizes
getting the *magnitude* of each rank-normalized target wrong, which over-weights the extreme names whose
normalized labels are largest in absolute value, whereas IC rewards getting the *ordering* right across
the whole cross-section. Minimizing MSE does pull the predictions toward the right order — a predictor
that perfectly reproduced the normalized labels would have IC 1 and loss 0, so the global optimum
coincides — but along the path they can diverge, and an epoch that trims a few large-name errors can
lower loss while barely moving, or even nudging down, the day-averaged rank correlation. Stopping on loss
is therefore a slightly indirect gate for this task. I keep it anyway, for two reasons that I can weigh
concretely: first, the label is already rank-normalized, so MSE is operating on ranks, not raw returns,
and the surrogate gap is small — the extreme-name over-weighting is bounded by the normalization's
compression of the tails; second, per-day validation IC is itself a noisy statistic on a hundred-name
universe (its own ICIR of 0.22 on the floor's csi100 says the daily IC swings by more than four times its
mean), so a loss-based gate can actually be the *steadier* early-stopping signal even though it is the
less direct one. That is a real trade rather than an oversight, and I am recording it because moving the
early-stopping gate off loss and onto validation IC directly, to match the ranking objective the task
scores, is one of the free levers I still have untouched.

Now the falsifiable expectations against the LightGBM floor, stated in the metrics this task actually
reports so I can be wrong cleanly. The whole thesis of this rung is that letting the day's stocks attend
to each other fixes the portfolio-noise problem the row-independent floor had, so the place I expect the
clearest win is *information ratio*, not necessarily signal IC. I expect csi300 IR to jump well above
LightGBM's 0.280 — a learned cross-sectional smoothing should produce a far more stable daily top-50 —
and annualized return to rise correspondingly above 0.0203, with max drawdown pulling in from −0.099
toward zero as the book stabilizes. On the signal metrics I expect IC to improve modestly over 0.0399 and
Rank IC over 0.0492 — the cross-section carries some extra predictability — but the IR gain should dwarf
the IC gain in proportional terms, because IR is precisely the metric that punishes the noisy daily
ranking the floor could not avoid, and the arithmetic above showed the floor's weakness was portfolio
construction, not correlation. csi300_recent, with the same breadth, should track csi300: IR up sharply
from 0.266, return up from 0.0197, even though its raw IC is the weakest to start. On csi100, where
LightGBM posted a *negative* IR of −0.334, the test is sharper: does cross-stock attention rescue the
low-breadth universe, or does a hundred names give the all-to-all attention too little to work with? I
expect csi100 IR to improve markedly from −0.334 — the attention should at least stop the bleeding — and
csi100 IC to rise off 0.0363 like the others, but I would not be surprised if the IR stays *negative*,
because a hundred-name universe is genuinely hard and the malign near-uniform failure mode I traced above
bites hardest exactly there. The signature I will be watching for is the diagnostic split: csi300 and
csi300_recent IR rising strongly while csi100 IR improves but stays under zero, and — the sharpest tell —
csi100's *signal* IC holding up or even leading while its portfolio IR lags. That precise combination,
strong signal on the narrow universe but no profitable book, would be the fingerprint of over-smoothing:
the attention washing the idiosyncratic spread that is most of csi100's edge toward the day's mean. If I
see it, the diagnosis points at a specific remedy to pursue next: stop attending to *everything*
indiscriminately, because a soft weighting over the whole cross-section cannot help leaking the mean —
the next thing to try is to make the borrowing selective and to stop the idiosyncratic signal from being
averaged away in the first place. That is the thread I would pull on from here.
