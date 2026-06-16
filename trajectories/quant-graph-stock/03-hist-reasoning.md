All-to-all attention did exactly what I hoped on the broad universes and exactly what I feared on the
narrow one, and the split is the whole story. On csi300 the GATs rung lifted information ratio from
LightGBM's 0.280 to 1.362 and annualized return from 0.0203 to 0.106 — a more-than-fourfold IR jump,
with signal IC also up from 0.0399 to 0.0495. That confirms the diagnosis from the floor: the portfolio
problem was a cross-section problem, and letting the day's stocks attend to each other gives a far more
stable daily top-50. On csi300_recent the same thing held — IR 1.324, return 0.103. But csi100 is the
tell I was watching for: IR went from −0.334 to −0.252, and annualized return stayed negative at −0.0141.
The attention stopped some of the bleeding but did not cross zero. Signal IC on csi100 is actually fine —
0.0465, even slightly higher than csi300's IC in some respects — so the model *can* rank csi100 stocks;
it just cannot build a profitable top-50 from them. That is the over-smoothing signature I flagged: with
only a hundred names and one all-to-all attention with no structure, the softmax over the full
cross-section pulls every stock toward the day's mean, and on a low-breadth universe where most of the
edge lives in the *idiosyncratic* spread between names, washing out that spread is precisely how you keep
positive IC but negative IR. The attention is sharing information indiscriminately — every stock borrows
from every other stock, including the irrelevant ones — and the curated concept graph, which I
deliberately ignored last rung to isolate "does cross-section help," is sitting unused on disk, holding
exactly the structure that would tell the model *which* stocks to borrow from.

So this rung brings the concept structure back, but not as a frozen adjacency mask — I argued last time
that the curated graph is static and incomplete, and I still believe that. The move is to make the
concept a first-class object with its own date-specific representation, so that "how much stock `i`
relates to theme `k` today" is *computed from the current embeddings* rather than read from a fixed
table, and to do this for both the curated concepts *and* a set of concepts discovered from the data.
That gives me three sources of signal per stock — what it shares with predefined themes, what it shares
with discovered hidden themes, and its own idiosyncratic residual — and crucially keeps them separate, so
the idiosyncratic part that all-to-all attention washed out on csi100 is preserved in its own channel.
That is the structure that should fix the narrow universe while holding the broad-universe gains.

Start from the same backbone, because nothing about the encoder changed. Each stock's `[60, 6]` window
goes through a recurrent encoder and I take the last hidden state `x_i`. The harness's HIST fill uses an
LSTM backbone, `hidden_size=64`, `num_layers=2`, `dropout=0.0`, and like the GATs rung it loads a
pretrained LSTM into the backbone before training — the same warm-start logic, so the relational modules
learn on top of a sensible single-stock encoder. (The hidden size here is 64, matching the GATs rung, not
the larger width sometimes quoted for HIST; I take the harness's setting, and the smaller width with
zero dropout is what the relational regularization below substitutes for.) The new machinery is what
happens after `x_i`.

**Predefined concept module.** The first wall with the curated graph was that membership is binary and
frozen while a stock's relevance to a theme drifts day to day. The fix is to stop treating membership as
the final edge weight and instead use it only to *initialize* a concept's representation, then recompute
the edges from current embeddings. Concretely, take the binary stock-concept matrix `M` (rows = the day's
stocks, columns = concepts), and form each concept's vector as a degree-normalized aggregate of its
member stocks' hidden states. The normalization is a smoothed membership degree: divide each member's
contribution by `(membership-masked column-sum + 1)`, so a singleton concept does not explode into a
full-strength copy of one stock and an empty concept produces a zero row I can drop. That gives a concept
representation `e_k` per surviving column. Then — and this is the drift fix — send the concept information
*back* to the stocks not through the frozen membership but through cosine similarity computed on the
*current* embeddings: for each stock, score its cosine similarity to every concept vector, softmax over
concepts (`Softmax(dim=1)`, each stock choosing a mixture over concept columns), and aggregate the
concept vectors by those weights into a per-stock shared vector, passed through a learned `fc_es`
transform. A stock can now borrow from a known theme even when its original membership row did not attach
it strongly, and the weights are recomputed every date — exactly the dynamic edge editing the frozen
mask could never express. This is the concept-aware aggregation the all-to-all attention lacked: instead
of attending to all 300 stocks, a stock attends to the *themes* it currently resembles, which is a far
lower-dimensional, structured neighborhood.

**Residual hand-off (backcast/forecast).** The reason the three sources stay separate — and the reason
csi100's idiosyncratic spread is preserved — is the residual-decomposition discipline. The predefined
module emits not only a forecast head `output_es` (a LeakyReLU on a learned transform of the shared
vector) but also a *backcast* `e_shared_back` (a learned linear head, no nonlinearity) that represents
the part of `x_i` it has accounted for. The next module runs on the residual `x¹_i = x_i −
e_shared_back`. Without this subtraction, three parallel modules would all see the full embedding and
redundantly re-encode the same sector move three times, double-counting the shared trend and starving the
idiosyncratic part — which is exactly the over-smoothing failure in another guise. The backcast forces a
decomposition: each module only sees what the previous ones could not explain.

**Hidden concept module.** The second wall was edges that *aren't there* — emerging themes the curators
never labeled. So discover concepts from the residual `x¹_i`, with no labels, on the part of the signal
the predefined themes could not explain. The construction is parameter-free: posit one hidden concept
*seeded by each stock* (initialize seed `k` as `x¹_k`), measure every stock's cosine similarity to every
seed, and connect each stock to its single most similar seed. Seeds that many stocks point at *are*
emergent groups; seeds nobody points at are spurious and get deleted. There is an immediate bug to handle:
every stock's similarity to its own seed is exactly 1 (cosine of a vector with itself), so the row-argmax
would always pick itself and no grouping happens. The fix is to zero out the diagonal of the
similarity matrix before the row-max, take each row's argmax over the remaining columns as that stock's
chosen concept, keep only that one connection per row, then re-add a surviving seed's own originating
stock to its membership. Aggregate the residual embeddings of each surviving concept's members
(similarity-weighted) into the hidden concept's representation, delete empty columns, and send the hidden
shared information back to stocks via the same cosine→softmax→weighted-sum→`fc_is` path. This whole stage
has essentially no concept-specific parameters — it is a similarity-driven clustering rerun fresh every
date — which is what I want, because the themes it finds should be free to change as the market changes.
This is the channel that should catch the co-movement on csi100 that the curated concepts miss and that
indiscriminate attention drowned.

**Individual module.** Run the residual discipline once more: the hidden module emits its own backcast
`i_shared_back`, and the individual information is `x²_i = x_i − e_shared_back − i_shared_back` — the part
explained by neither predefined nor hidden themes. A nonlinear head `fc_indi` on this residual is the
idiosyncratic forecast. *This* is the channel that all-to-all attention destroyed on csi100: the
stock's own peculiar signal, kept separate and never averaged against the cross-section. Preserving it is
the whole reason I expect csi100 to recover.

**Read-out.** Because the backcasts subtract on the way down, the three slices add back up to the original
embedding exactly, so the forecasts *sum* on the way out: `all_info = output_es + output_is +
output_indi`, then one final `fc_out` reads the scalar score. Predefined-shared trend plus hidden-shared
trend plus individual trend — the additive structure the residual stack guarantees.

The harness specifics I have to honor: the metric for early stopping is `ic` here, not loss — the
`metric_fn` computes the Pearson correlation between predictions and labels on each day's batch and
early-stops on validation IC with patience 20, which is the right objective for a ranking task and a
change from the GATs rung's loss-based stopping. Training is per-day cross-sections (the entire
relational computation is over one date's stocks; a batch that mixed dates would mix unrelated graphs),
MSE loss masked to finite labels, Adam at `lr=1e-4`, grad-value clip 3.0, up to 200 epochs. The concept
matrix per day is fetched from the stock2concept file via the stock-index map, with index 733 as the
padding fallback for unknown instruments. The full forward — the membership-normalized predefined
initialization, the masked-argmax hidden clustering, the three forecast/backcast heads — is in the
answer.

Now the falsifiable expectations against the GATs rung. The thesis is that structured, concept-aware,
residual-separated aggregation fixes the over-smoothing that all-to-all attention suffered on the narrow
universe while holding the broad-universe gains. So the sharpest test is csi100 information ratio: GATs
left it at −0.252 with a negative return of −0.0141. I expect HIST to close most of that gap — IR much
closer to zero, possibly still slightly negative because a hundred names is genuinely hard, but the
near-flat csi100 IR is the signature I am predicting, driven by the preserved individual channel and the
hidden-concept channel catching unlabeled co-movement. On the signal side I expect IC to rise above GATs
across all three universes (0.0495 → higher on csi300, 0.0465 → higher on csi100), because the concept
structure adds predictability the bare attention could not, and Rank IC likewise. The one place I will
*not* be surprised to see HIST trade a little is raw csi300 information ratio: GATs's 1.362 is a strong
number, and the structured model spends some of its cross-sectional sharing on the narrow universe and
the idiosyncratic channel rather than maximally smoothing the broad one, so HIST's csi300 IR could come
in a touch below GATs's even as its IC is higher — that would be the expected cost of un-smoothing, and
it is a good trade if csi100 crosses toward zero and IC rises everywhere. If instead csi100 IR stays as
negative as GATs's, the concept structure is not buying what I claimed and the separation discipline is
not preserving the idiosyncratic signal — that is the result that would falsify this rung. The numbers I
am betting on: HIST highest IC and Rank IC of the three baselines on every universe, csi100 IR
dramatically improved from −0.252 toward zero, and the broad-universe portfolio gains held within reach
of the GATs rung.
