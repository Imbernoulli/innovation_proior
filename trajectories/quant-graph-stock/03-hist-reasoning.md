All-to-all attention did exactly what I hoped on the broad universes and exactly what I feared on the
narrow one, and the split is the whole story. On csi300 the GATs rung lifted information ratio from
LightGBM's 0.280 to 1.362 and annualized return from 0.0203 to 0.106 — a more-than-fourfold IR jump,
1.362/0.280 ≈ 4.9×, and return 0.106/0.0203 ≈ 5.2× — with signal IC also up from 0.0399 to 0.0495, a
24% lift. That confirms the diagnosis from the floor: the portfolio problem was a cross-section problem,
and letting the day's stocks attend to each other gives a far more stable daily top-50. On csi300_recent
the same thing held and by almost the same multiples — IR 0.266 → 1.324 (≈5.0×), return 0.0197 → 0.103
(≈5.2×) — which is reassuring precisely because recent had the *weakest* raw signal of the three, so the
gain is clearly coming from cross-sectional stabilization rather than from the encoder finding more
correlation. But csi100 is the tell I was watching for, and it is worth reading in full because the shape
of the failure names the fix. IR went from −0.334 to −0.252 and annualized return stayed negative at
−0.0141. The attention stopped some of the bleeding — 0.082 of IR recovered — but did not cross zero. And
here is the part that pins the mechanism: csi100's *signal* IC did not merely hold, it improved the
*most* of any universe, 0.0363 → 0.0465, a 28% lift, larger in proportional terms than csi300's 24% or
recent's 14%. So the model got *better* at ranking csi100 stocks and *still* could not build a profitable
book from them. Best signal improvement, worst portfolio — that is not a signal deficit, it is a
portfolio-construction pathology, and it is the exact fingerprint I said I would watch for.

Let me make the pathology quantitative rather than leave it as a slogan, because the numbers say which
knob to turn. Put csi100 and csi300 side by side after the GATs rung: IC 0.0465 versus 0.0495, a ratio of
0.0465/0.0495 ≈ 0.94 — the two signals are now within six percent of each other in quality — yet IR
−0.252 versus 1.362, on opposite sides of zero by more than 1.6 in the compounding metric. Near-identical
signal, opposite money, the same paradox the floor showed but now *sharper*, because the attention closed
most of the IC gap while leaving the IR gap wide open. The drawdown column is even more damning: csi100's
max drawdown went from the floor's −0.158 to −0.203 — the attention made the deepest hole *deeper* by
0.045, while on csi300 it healed drawdown from −0.099 to −0.067. A model that improves signal, deepens
drawdown, and cannot cross zero on the narrow universe is a model that is concentrating its bets in the
wrong place: it is sharing information indiscriminately, every stock borrowing from every other stock
including the irrelevant ones, so the daily top-50 gets pulled toward the day's mean and the idiosyncratic
spread — which on a hundred-name universe is most of what there is to trade — gets averaged away. That is
the over-smoothing signature I flagged one rung ago made concrete in three numbers: rising IC, deepening
drawdown, IR stuck below zero. The softmax over the full cross-section has no way to *not* attend to a
name; it can only down-weight it, and a soft down-weight spread over a hundred names still leaks the mean
in. And the curated concept graph, which I deliberately ignored last rung to isolate "does cross-section
help," is sitting unused on disk, holding exactly the structure that would tell the model *which* stocks
to borrow from.

So this rung brings the concept structure back, but not as a frozen adjacency mask — I argued last time
that the curated graph is static and incomplete, and I still believe that. Before I commit to the design,
let me be honest about the alternatives, because the failure I just diagnosed admits more than one repair
and I want to reject the cheaper ones on their mechanics. The cheapest is to sparsify the attention I
already have: keep all-to-all scoring but hard-threshold to each stock's top-k neighbors, so a stock
borrows from ten names instead of a hundred. Walk it. Top-k prunes the softmax's tail, so it reduces the
mean-leak somewhat — but it still *averages* the k survivors into the stock's representation, so the
idiosyncratic channel is still overwritten, just by fewer contributors; and it introduces a
discrete `k` I would have to tune per universe, with no principled value, when the real problem is not
*how many* neighbors but *that the stock's own signal has nowhere to live separately from the borrowed
one*. Top-k does not give me a separate idiosyncratic channel, so on the metric that is failing — a
narrow universe whose edge is idiosyncratic — it treats the symptom, not the cause. Rejected. A second
alternative is to add a plain residual skip of the raw embedding to the output, which *does* preserve the
individual signal; but it discovers no new structure — it cannot catch the unlabeled co-movement the
curated concepts miss, which is the other half of what a hundred-name universe needs. The move that does
both is to make the concept a first-class object with its own date-specific representation, so that "how
much stock `i` relates to theme `k` today" is *computed from the current embeddings* rather than read
from a fixed table, and to do this for both the curated concepts *and* a set of concepts discovered from
the data, and to keep the leftover idiosyncratic part in its own separate channel. That gives me three
sources of signal per stock — what it shares with predefined themes, what it shares with discovered hidden
themes, and its own residual — and crucially keeps them separate, so the idiosyncratic part that
all-to-all attention washed out on csi100 is preserved rather than averaged. That is the structure that
should fix the narrow universe while holding the broad-universe gains.

Start from the same backbone, because nothing about the encoder changed. Each stock's `[60, 6]` window
goes through a recurrent encoder and I take the last hidden state `x_i ∈ R^{64}`. The harness's HIST fill
uses an LSTM backbone, `hidden_size=64`, `num_layers=2`, `dropout=0.0`, and like the GATs rung it loads a
pretrained LSTM into the backbone before training — the same warm-start logic, so the relational modules
learn on top of a sensible single-stock encoder. It is worth pricing the added machinery, because it says
how much this rung is really betting. Last rung the relational part added ~8.4k parameters on top of the
~52k warm-started LSTM. This rung adds far more: the three modules use on the order of ten `64×64` linear
transforms — the shared-info projections `fc_es`/`fc_is`, their forecast and backcast heads, the
individual head — plus four scalar read-outs, which totals roughly 4.2·10⁴ new relational parameters,
about five times the GATs rung's ~8.4k. So I am spending an order of magnitude more capacity than a bare
attention layer to buy the structured, separated aggregation; that is only justified if the separation
actually earns its keep on the narrow universe, which is exactly what the falsifiable test at the end will
check. The hidden size here is 64, matching the GATs rung, not the larger width sometimes used for this
family; I take the harness's setting, and the smaller width with zero dropout is what the relational
regularization below substitutes for — the three-way residual decomposition is itself a strong structural
prior that does some of the work heavy dropout did last rung. The new machinery is what happens after
`x_i`.

**Predefined concept module.** The first wall with the curated graph was that membership is binary and
frozen while a stock's relevance to a theme drifts day to day. The fix is to stop treating membership as
the final edge weight and instead use it only to *initialize* a concept's representation, then recompute
the edges from current embeddings. Concretely, take the binary stock-concept matrix `M` (rows = the day's
stocks, columns = concepts), and form each concept's vector as a degree-normalized aggregate of its
member stocks' hidden states. The normalization needs care, and the `+1` smoothing is the detail that
makes it robust, so let me trace it on the two boundary cases. For a concept with a single member stock,
the masked column-sum is 1, I divide that member's contribution by `(1 + 1) = 2`, so the concept vector
is `0.5 · x_member` rather than a full-strength copy of one stock — a singleton theme is deliberately
damped, not allowed to masquerade as a strong concept. For a concept with no members that day, the column
sum is 0, the initialization produces a row of zeros, and I drop it with the `hidden.sum(1) != 0` filter
before it can contribute a spurious all-zero theme. For a concept with, say, twenty members, each
member's contribution is divided by 21, so the aggregate is very nearly the plain mean of its members'
hiddens with a mild shrink toward zero — exactly the degree-invariant behavior I want, big and small
themes on a comparable scale. That gives a concept representation `e_k` per surviving column. Then — and
this is the drift fix — send the concept information *back* to the stocks not through the frozen
membership but through cosine similarity computed on the *current* embeddings: for each stock, score its
cosine similarity to every surviving concept vector, softmax over concepts (`softmax_t2s`, `dim=1`, each
stock choosing a mixture over concept columns), and aggregate the concept vectors by those weights into a
per-stock shared vector, passed through a learned `fc_es` transform. A stock can now borrow from a known
theme even when its original membership row did not attach it strongly, and the weights are recomputed
every date — exactly the dynamic edge editing the frozen mask could never express. This is the
concept-aware aggregation the all-to-all attention lacked: instead of attending to all 300 stocks, a
stock attends to the *themes* it currently resembles, a far lower-dimensional, structured neighborhood —
if there are, say, sixty active concepts on a day, the stock's mixture is over sixty structured columns
rather than three hundred raw names, and the columns are already sector-coherent averages rather than
individual noisy stocks.

**Residual hand-off (backcast/forecast).** The reason the three sources stay separate — and the reason
csi100's idiosyncratic spread is preserved — is the residual-decomposition discipline, and I want to be
precise about why it is load-bearing rather than decorative. The predefined module emits not only a
forecast head `output_es` (a LeakyReLU on a learned transform of the shared vector) but also a *backcast*
`e_shared_back` (a learned linear head, no nonlinearity) that represents the part of `x_i` it has
accounted for. The next module runs on the residual `x¹_i = x_i − e_shared_back`. Here is the concrete
failure the subtraction prevents. Suppose I did *not* subtract, and ran all three modules on the full
`x_i`. A stock dominated by a strong sector move would have that move present in its predefined-shared
vector, present again in whatever the hidden module extracts, and present a third time in the individual
head — the shared trend gets counted three times in `all_info = output_es + output_is + output_indi`,
inflating the model's confidence on exactly the co-moving names and, worse, leaving the individual head
re-encoding the sector trend instead of the stock's own peculiarity. That triple-counting of the shared
move at the expense of the idiosyncratic part *is* the over-smoothing failure in another guise. The
backcast forces a decomposition: because `x_i = e_shared_back + (x_i − e_shared_back)`, each downstream
module only sees what the previous ones could not explain, and the individual head at the end sees a
residual from which both shared trends have already been removed. That is what keeps the three channels
disjoint and what keeps the idiosyncratic signal alive.

**Hidden concept module.** The second wall was edges that *aren't there* — emerging themes the curators
never labeled. So discover concepts from the residual `x¹_i`, with no labels, on the part of the signal
the predefined themes could not explain. The construction is essentially parameter-free: posit one hidden
concept *seeded by each stock* (initialize seed `k` as `x¹_k`), measure every stock's cosine similarity
to every seed, and connect each stock to its single most similar seed. Seeds that many stocks point at
*are* emergent groups; seeds nobody points at are spurious and get deleted. There is an immediate bug to
handle, and it is the kind that would silently no-op the whole module if I missed it, so let me trace it
on a tiny case. Every stock's cosine similarity to its own seed is exactly 1 — cosine of a vector with
itself — and 1 is the maximum possible cosine, so the row-argmax over the raw similarity matrix would
*always* pick the diagonal: every stock would select its own seed, every seed would have exactly one
member (itself), no two stocks would ever land in the same group, and the module would discover nothing.
Take three stocks with residuals `x¹_1, x¹_2, x¹_3`: the similarity matrix has 1's on the diagonal and
whatever cross-similarities off it, and `argmax` of each row is column `i` — self, every time. The fix is
to zero out the diagonal before the row-max: multiply the similarity matrix by `(1 − I)` so every
self-similarity becomes 0, then take each row's argmax over the *remaining* columns as that stock's chosen
concept, keep only that one connection per row (set the winning entry to a sentinel, zero everything else,
restore the winning similarity value), and finally re-add a surviving seed's own originating stock to its
membership via the diagonal term so a real group includes the stock that seeded it. Now with three stocks,
if stocks 1 and 2 are most similar to seed 3 and stock 3 is most similar to seed 1, seeds 1 and 3 survive
with real members and seed 2 — pointed at by nobody — is dropped by the `hidden.sum(1) != 0` filter.
Aggregate the residual embeddings of each surviving concept's members (similarity-weighted) into the
hidden concept's representation, delete empty columns, and send the hidden shared information back to
stocks via the same cosine→softmax→weighted-sum→`fc_is` path. This whole stage has essentially no
concept-specific parameters — it is a similarity-driven clustering rerun fresh every date — which is what
I want, because the themes it finds should be free to change as the market changes. This is the channel
that should catch the co-movement on csi100 that the curated concepts miss and that indiscriminate
attention drowned.

**Individual module.** Run the residual discipline once more: the hidden module emits its own backcast
`i_shared_back`, and the individual information is `x²_i = x_i − e_shared_back − i_shared_back` — the part
explained by neither predefined nor hidden themes. A nonlinear head `fc_indi` on this residual is the
idiosyncratic forecast. *This* is the channel that all-to-all attention destroyed on csi100: the stock's
own peculiar signal, kept separate and never averaged against the cross-section. Preserving it is the
whole reason I expect csi100 to recover, and the arithmetic from the GATs failure says why — if 94% of
csi100's post-attention IC quality was already there but produced negative IR, the missing ingredient was
not more correlation, it was a place for the idiosyncratic spread to survive portfolio construction, which
is exactly what this channel is.

**Read-out.** Because the backcasts subtract on the way down, the three slices add back up to the original
embedding exactly — `x_i = e_shared_back + i_shared_back + x²_i` by construction — so the forecasts *sum*
on the way out without double-counting: `all_info = output_es + output_is + output_indi`, then one final
`fc_out` reads the scalar score. Predefined-shared trend plus hidden-shared trend plus individual trend —
the additive structure the residual stack guarantees, and the reason the sum is legitimate rather than a
triple-count is precisely the backcast subtraction traced above.

Let me verify the additive read-out is self-consistent before I trust it, with a one-line vector check.
Write `a = e_shared_back`, `b = i_shared_back`, `c = x²`. By construction `c = x − a − b`, so `a + b + c
= x` identically — the decomposition partitions the embedding with no remainder and no overlap. The three
forecast heads are separate learned functions of the three *shared/residual information* vectors, not of
`a, b, c` directly, so `all_info` is not literally a function of `x` alone; but the point the check
secures is the one that matters — because the backcasts subtract the exact quantities each module claimed,
no slice of `x` feeds two forecast heads, and summing the three outputs cannot double-count a trend. Drop
the backcasts and the identity breaks: each module would consume the full `x` and the sum would over-count
the shared move, which is the failure I traced above. The subtraction is what earns the sum. And it is
worth being explicit about why rescuing the narrow universe is the right thing to optimize even at a cost
elsewhere, because the scoring rule makes it lopsided in my favor. The task score is a geometric mean
across the three universe scores, and a gmean is dominated by its smallest factor. Take three schematic
universe scores of (0.60, 0.40, 0.60): the gmean is `(0.60·0.40·0.60)^{1/3} ≈ 0.525`. Lift the worst from
0.40 to 0.50 and it becomes `(0.60·0.50·0.60)^{1/3} ≈ 0.564`, a gain of 0.039; but lift one of the strong
0.60s to 0.70 instead and it becomes `(0.70·0.40·0.60)^{1/3} ≈ 0.552`, a gain of only 0.027. Same
0.10 of metric improvement, but the one spent on the worst universe moves the task score half again as
much. So trading a little csi300 IR to pull csi100 off the floor is not a reluctant compromise — under a
geometric mean it is the higher-leverage move, and it is exactly what the three-channel separation is
built to do.

The harness specifics I have to honor: the metric for early stopping is `ic` here, not loss — the
`metric_fn` computes the Pearson correlation between predictions and labels on each day's batch and
early-stops on validation IC with patience 20. That is a deliberate change from the GATs rung, and it is
the free lever I flagged last time: the scored objective is a ranking one, so gating early stopping on the
ranking statistic directly, rather than on the MSE surrogate, aligns the model-selection criterion with
what the task rewards. The cost is that per-day validation IC is noisier than loss, but with three
separated channels the model is more stable than the bare attention was, so I can afford the more direct
gate. Training is per-day cross-sections (the entire relational computation is over one date's stocks; a
batch that mixed dates would mix unrelated graphs), MSE loss masked to finite labels, Adam at `lr=1e-4`,
grad-value clip 3.0, up to 200 epochs. The concept matrix per day is fetched from the stock2concept file
via the stock-index map, with index 733 as the padding fallback for unknown instruments. The full forward
— the membership-normalized predefined initialization, the masked-argmax hidden clustering, the three
forecast/backcast heads — is in the answer.

Now the falsifiable expectations against the GATs rung. The thesis is that structured, concept-aware,
residual-separated aggregation fixes the over-smoothing that all-to-all attention suffered on the narrow
universe while holding the broad-universe gains. So the sharpest test is csi100 information ratio: GATs
left it at −0.252 with a negative return of −0.0141 and a deepened drawdown of −0.203. I expect HIST to
close most of that IR gap — much closer to zero, possibly still slightly negative because a hundred names
is genuinely hard, but the near-flat csi100 IR is the signature I am predicting, driven by the preserved
individual channel and the hidden-concept channel catching unlabeled co-movement; I would expect the
csi100 return to come up toward zero from −0.0141 and the drawdown to heal back from −0.203 as the bets
de-concentrate. On the signal side I expect IC to rise above GATs across all three universes (0.0495 →
higher on csi300, 0.0465 → higher on csi100, 0.0289 → higher on recent), because the concept structure
adds predictability the bare attention could not, and Rank IC likewise. The one place I will *not* be
surprised to see HIST trade a little is raw csi300 information ratio: GATs's 1.362 is a strong number, and
the structured model spends some of its cross-sectional sharing on the narrow universe and the
idiosyncratic channel rather than maximally smoothing the broad one, so HIST's csi300 IR could come in a
touch below 1.362 even as its IC is higher — that would be the expected cost of un-smoothing, and it is a
good trade if csi100 crosses toward zero and IC rises everywhere, because the task score is a geometric
mean across universes and rescuing the worst universe moves the gmean more than shaving the best one costs
it. If instead csi100 IR stays as negative as GATs's −0.252, the concept structure is not buying what I
claimed and the separation discipline is not preserving the idiosyncratic signal — that is the result that
would falsify this rung. The numbers I am betting on: HIST highest IC and Rank IC of the three baselines
on every universe, csi100 IR dramatically improved from −0.252 toward zero, and the broad-universe
portfolio gains held within reach of the GATs rung.
