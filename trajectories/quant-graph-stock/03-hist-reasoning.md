All-to-all attention did exactly what I hoped on the broad universes and exactly what I feared on the
narrow one, and the split is the whole story. On csi300 the GATs rung lifted information ratio from
LightGBM's 0.280 to 1.362 and annualized return from 0.0203 to 0.106 — a more-than-fourfold IR jump —
with signal IC also up from 0.0399 to 0.0495. That confirms the diagnosis from the floor: the portfolio
problem was a cross-section problem, and letting the day's stocks attend to each other gives a far more
stable daily top-50. csi300_recent held by almost the same multiples — IR 0.266 → 1.324, return
0.0197 → 0.103 — reassuring precisely because recent had the *weakest* raw signal, so the gain is coming
from cross-sectional stabilization rather than the encoder finding more correlation. But csi100 is the
tell I was watching for. IR went from −0.334 to −0.252 and annualized return stayed negative at −0.0141:
the attention stopped some of the bleeding but did not cross zero. And here is the part that pins the
mechanism — csi100's *signal* IC did not merely hold, it improved the *most* of any universe,
0.0363 → 0.0465, a 28% lift, larger in proportional terms than csi300's 24% or recent's 14%. So the model
got *better* at ranking csi100 stocks and *still* could not build a profitable book from them. Best signal
improvement, worst portfolio — that is not a signal deficit, it is a portfolio-construction pathology, and
it is the exact fingerprint I said I would watch for.

The numbers say which knob to turn. After the GATs rung csi100 IC 0.0465 and csi300 IC 0.0495 are within
six percent of each other, yet IR −0.252 versus 1.362 sits on opposite sides of zero by more than 1.6 in
the compounding metric — the attention closed most of the IC gap while leaving the IR gap wide open. The
drawdown column is more damning: csi100's max drawdown went from the floor's −0.158 to −0.203, the
attention making the deepest hole *deeper*, while on csi300 it healed drawdown from −0.099 to −0.067. A
model that improves signal, deepens drawdown, and cannot cross zero on the narrow universe is
concentrating its bets in the wrong place: it shares information indiscriminately, every stock borrowing
from every other including the irrelevant ones, so the daily top-50 is pulled toward the day's mean and
the idiosyncratic spread — which on a hundred-name universe is most of what there is to trade — gets
averaged away. Rising IC, deepening drawdown, IR stuck below zero: the over-smoothing signature made
concrete. The softmax over the full cross-section has no way to *not* attend to a name; it can only
down-weight it, and a soft down-weight over a hundred names still leaks the mean in. And the curated
concept graph, which I deliberately ignored last rung to isolate "does cross-section help," is sitting
unused on disk, holding exactly the structure that would tell the model *which* stocks to borrow from.

So this rung brings the concept structure back, but not as a frozen adjacency mask — the curated graph is
static and incomplete, and I still believe that. The failure admits more than one repair, and the cheaper
ones fail on their mechanics. The cheapest is to sparsify the attention I already have: keep all-to-all
scoring but hard-threshold to each stock's top-k neighbors, so a stock borrows from ten names instead of a
hundred. But top-k still *averages* the k survivors into the stock's representation, so the idiosyncratic
channel is still overwritten, just by fewer contributors; and it introduces a discrete `k` with no
principled value, when the real problem is not *how many* neighbors but *that the stock's own signal has
nowhere to live separately from the borrowed one*. A plain residual skip of the raw embedding to the
output *does* preserve the individual signal, but discovers no new structure — it cannot catch the
unlabeled co-movement the curated concepts miss, the other half of what a hundred-name universe needs. The
move that does both is to make the concept a first-class object with its own date-specific representation,
so that "how much stock `i` relates to theme `k` today" is *computed from the current embeddings* rather
than read from a fixed table — for both the curated concepts *and* a set of concepts discovered from the
data — and to keep the leftover idiosyncratic part in its own separate channel. Three sources of signal
per stock, kept separate: what it shares with predefined themes, what it shares with discovered hidden
themes, and its own residual. The separation is what preserves the idiosyncratic part that all-to-all
attention washed out on csi100.

Start from the same backbone, because nothing about the encoder changed. Each stock's `[60, 6]` window
goes through a recurrent encoder and I take the last hidden state `x_i ∈ R^{64}`. The harness's HIST fill
uses an LSTM backbone, `hidden_size=64`, `num_layers=2`, `dropout=0.0`, and like the GATs rung loads a
pretrained LSTM into the backbone before training, so the relational modules learn on top of a sensible
single-stock encoder. This rung's relational part is far heavier than the last: the three modules use on
the order of ten `64×64` linear transforms — the shared-info projections `fc_es`/`fc_is`, their forecast
and backcast heads, the individual head — roughly 4.2·10⁴ new relational parameters, about five times the
GATs rung's ~8.4k. So I am spending an order of magnitude more capacity than a bare attention layer to buy
the structured, separated aggregation; that only pays if the separation earns its keep on the narrow
universe, which the falsifiable test at the end checks. The dropout is zero here rather than 0.7 — the
three-way residual decomposition is itself a strong structural prior that does some of the work heavy
dropout did last rung. The new machinery is what happens after `x_i`.

**Predefined concept module.** The first wall with the curated graph was that membership is binary and
frozen while a stock's relevance to a theme drifts day to day. The fix is to use membership only to
*initialize* a concept's representation, then recompute the edges from current embeddings. Take the binary
stock-concept matrix `M` (rows = the day's stocks, columns = concepts) and form each concept's vector as a
degree-normalized aggregate of its member stocks' hidden states. The `+1` smoothing in the normalization
is the detail that makes it robust, and the boundary cases show why. For a concept with a single member,
the masked column-sum is 1 and I divide by `(1 + 1) = 2`, so the concept vector is `0.5 · x_member` — a
singleton theme is deliberately damped, not allowed to masquerade as a strong concept. For a concept with
no members that day, the column sum is 0, the initialization is a row of zeros, and I drop it with the
`hidden.sum(1) != 0` filter before it contributes a spurious all-zero theme. For a twenty-member concept,
each contribution is divided by 21, so the aggregate is very nearly the plain mean with a mild shrink —
degree-invariant, big and small themes on a comparable scale. That gives a concept representation `e_k` per
surviving column. Then — the drift fix — send the concept information *back* to the stocks not through the
frozen membership but through cosine similarity on the *current* embeddings: for each stock, score cosine
similarity to every surviving concept vector, softmax over concepts (`softmax_t2s`, `dim=1`), and aggregate
the concept vectors by those weights into a per-stock shared vector through a learned `fc_es`. A stock can
now borrow from a known theme even when its original membership row did not attach it strongly, and the
weights are recomputed every date — the dynamic edge editing the frozen mask could never express. Instead
of attending to all 300 stocks, a stock attends to the *themes* it currently resembles, a far
lower-dimensional structured neighborhood of sector-coherent averages rather than individual noisy stocks.

**Residual hand-off (backcast/forecast).** The reason the three sources stay separate — and the reason
csi100's idiosyncratic spread is preserved — is the residual-decomposition discipline, and it is
load-bearing rather than decorative. The predefined module emits not only a forecast head `output_es` (a
LeakyReLU on a learned transform of the shared vector) but also a *backcast* `e_shared_back` (a learned
linear head, no nonlinearity) representing the part of `x_i` it has accounted for. The next module runs on
the residual `x¹_i = x_i − e_shared_back`. The concrete failure the subtraction prevents: if I did *not*
subtract and ran all three modules on the full `x_i`, a stock dominated by a strong sector move would have
that move present in its predefined-shared vector, again in whatever the hidden module extracts, and a
third time in the individual head — the shared trend counted three times in
`all_info = output_es + output_is + output_indi`, inflating confidence on exactly the co-moving names and
leaving the individual head re-encoding the sector trend instead of the stock's own peculiarity. That
triple-counting *is* the over-smoothing failure in another guise. Because `x_i = e_shared_back + (x_i −
e_shared_back)`, each downstream module only sees what the previous ones could not explain, and the
individual head at the end sees a residual with both shared trends already removed. That is what keeps the
three channels disjoint and the idiosyncratic signal alive.

**Hidden concept module.** The second wall was edges that *aren't there* — emerging themes the curators
never labeled. So discover concepts from the residual `x¹_i`, unlabeled, on the part of the signal the
predefined themes could not explain. The construction is essentially parameter-free: posit one hidden
concept *seeded by each stock* (seed `k` initialized as `x¹_k`), measure every stock's cosine similarity
to every seed, and connect each stock to its single most similar seed; seeds many stocks point at *are*
emergent groups, seeds nobody points at are spurious and deleted. There is a bug here that would silently
no-op the whole module. Every stock's cosine similarity to its own seed is exactly 1 — cosine of a vector
with itself, the maximum possible — so the row-argmax over the raw similarity matrix would *always* pick
the diagonal: every stock selects its own seed, every seed has exactly one member, no two stocks ever land
in the same group, and nothing is discovered. The fix is to zero out the diagonal before the row-max —
multiply the similarity matrix by `(1 − I)` so every self-similarity becomes 0 — then take each row's
argmax over the *remaining* columns as that stock's chosen concept, keep only that one connection per row,
and finally re-add a surviving seed's originating stock to its membership so a real group includes the
stock that seeded it. With three stocks, if 1 and 2 are most similar to seed 3 and stock 3 is most similar
to seed 1, seeds 1 and 3 survive with real members and seed 2 — pointed at by nobody — is dropped by the
`hidden.sum(1) != 0` filter. Aggregate the residual embeddings of each surviving concept's members
(similarity-weighted) into the hidden concept's representation, delete empty columns, and send the hidden
shared information back to stocks via the same cosine→softmax→weighted-sum→`fc_is` path. This stage has
essentially no concept-specific parameters — a similarity-driven clustering rerun fresh every date, so the
themes it finds are free to change as the market changes. This is the channel that should catch the
co-movement on csi100 that the curated concepts miss and that indiscriminate attention drowned.

**Individual module.** Run the residual discipline once more: the hidden module emits its own backcast
`i_shared_back`, and the individual information is `x²_i = x_i − e_shared_back − i_shared_back` — the part
explained by neither predefined nor hidden themes. A nonlinear head `fc_indi` on this residual is the
idiosyncratic forecast. *This* is the channel all-to-all attention destroyed on csi100: the stock's own
peculiar signal, kept separate and never averaged against the cross-section. If 94% of csi100's
post-attention IC quality was already there but produced negative IR, the missing ingredient was not more
correlation but a place for the idiosyncratic spread to survive portfolio construction — which is exactly
this channel.

**Read-out.** Because the backcasts subtract on the way down, `x_i = e_shared_back + i_shared_back + x²_i`
by construction, so no slice of `x_i` feeds two forecast heads and the three forecasts *sum* on the way
out without double-counting: `all_info = output_es + output_is + output_indi`, then one final `fc_out`
reads the scalar score. Predefined-shared trend plus hidden-shared trend plus individual trend — the
additive structure the backcast subtraction earns; drop the backcasts and each module would consume the
full `x_i` and the sum would over-count the shared move.

It is worth being explicit about why rescuing the narrow universe is the right thing to optimize even at a
cost elsewhere, because the scoring rule makes it lopsided in my favor. The task score is a geometric mean
across the three universe scores, and a gmean is dominated by its smallest factor. Take schematic scores
(0.60, 0.40, 0.60): gmean ≈ 0.525. Lift the worst from 0.40 to 0.50 → ≈ 0.564, a gain of 0.039; lift a
strong 0.60 to 0.70 instead → ≈ 0.552, a gain of only 0.027. Same 0.10 of metric improvement, but the one
spent on the worst universe moves the task score half again as much. So trading a little csi300 IR to pull
csi100 off the floor is not a reluctant compromise — it is the higher-leverage move, and exactly what the
three-channel separation is built to do.

The harness specifics I honor: early stopping here watches `ic`, not loss — the `metric_fn` computes the
per-day Pearson correlation between predictions and labels and early-stops on validation IC with patience
20. That is the free lever I flagged last rung: the scored objective is a ranking one, so gating model
selection on the ranking statistic directly aligns it with what the task rewards; the cost is that per-day
validation IC is noisier than loss, but with three separated channels the model is more stable than the
bare attention was, so I can afford the more direct gate. Training is per-day cross-sections (a batch that
mixed dates would mix unrelated graphs), MSE loss masked to finite labels, Adam at `lr=1e-4`, grad-value
clip 3.0, up to 200 epochs. The concept matrix per day is fetched from the stock2concept file via the
stock-index map, with index 733 as the padding fallback for unknown instruments. The full forward — the
membership-normalized predefined initialization, the masked-argmax hidden clustering, the three
forecast/backcast heads — is in the answer.

Now the falsifiable expectations against the GATs rung. The thesis is that structured, concept-aware,
residual-separated aggregation fixes the over-smoothing all-to-all attention suffered on the narrow
universe while holding the broad-universe gains. The sharpest test is csi100 information ratio, which GATs
left at −0.252 with return −0.0141 and a deepened drawdown of −0.203. I expect HIST to close most of that
IR gap toward zero — driven by the preserved individual channel and the hidden-concept channel catching
unlabeled co-movement — with the return coming up from −0.0141 and the drawdown healing back from −0.203
as the bets de-concentrate; a hundred names is genuinely hard, so I would not bank on it turning firmly
positive. On the signal side I expect IC to rise above GATs across all three universes, because the concept
structure adds predictability the bare attention could not, Rank IC likewise. The one place I will *not* be
surprised to see HIST trade a little is raw csi300 IR: GATs's 1.362 is strong, and the structured model
spends some of its cross-sectional sharing on the idiosyncratic channel rather than maximally smoothing the
broad universe, so csi300 IR could come in somewhat below 1.362 even as its IC is higher — an acceptable
cost of un-smoothing, since under a geometric mean rescuing the worst universe outweighs shaving the best.
If csi100 IR stays as negative as GATs's −0.252, the concept structure is not buying what I claimed and the
separation discipline is not preserving the idiosyncratic signal — that is the result that would falsify
this rung.
