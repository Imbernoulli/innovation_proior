The four baselines have converged, and reading them together tells me what the ladder has and has not
solved. DLinear set the affine floor (13.39 / 10.50 / 14.36). PatchTST added a learned nonlinear
representation and instance normalization and beat it everywhere but only by tenths (12.97 / 10.22 /
13.68), because attention over two-to-four patch tokens on a `2·pred_len` window is starved. TimeMixer
added explicit multi-scale decomposition-mixing and won on every regime (12.80 / 10.21 / 13.38), its
gains concentrated where trend and season separate across scales. TimesNet added FFT period-discovery
and a 2D intra/interperiod layout, and edged ahead overall (12.803 / 10.089 / 13.442) — but the win is
*entirely Quarterly*: it took the cleanest-period regime from 10.21 to 10.089, essentially tied
TimeMixer on Monthly (12.803 vs 12.802), and actually came in *worse* on Yearly (13.442 vs 13.378),
exactly where there is no period to discover and trend is the whole story. So the strongest baseline,
at mean SMAPE ~12.11, is strong on periodic regimes and *no better than a multi-scale MLP on the
trend-dominated one* — and across the whole ladder the Yearly number has barely moved from DLinear's
14.36 down to ~13.38-13.68, a far smaller relative gain than Monthly or Quarterly saw. The unsolved
problem is sharp: every rung so far is a single architecture that must serve trend-only Yearly and
strong-seasonal Monthly with *one* fixed inductive structure, and whichever structure it picks (pooling
ladder, period FFT, patch attention) is mismatched to at least one regime. What I want is an
architecture whose inductive bias is not "one lens" but *sequential refinement* — peel off whatever
the previous part of the model could not explain — so it adapts to a pure-trend series and a strongly
seasonal one without a regime-specific lens, and ideally exposes a trend/season split as a by-product
rather than imposing one.

The structural idea is a deep stack of fully-connected **blocks** wired by a **double residual** loop,
each block constrained to emit a *basis expansion*. Let me derive it from the failure modes above.
First, the output strategy is settled by the same direct-multi-step argument that has held since
DLinear: emit the whole horizon at once, no roll-out, no error accumulation — and a feed-forward
network whose output spans the horizon does this natively. The novelty is what goes between input and
output. Instead of one network explaining the whole look-back at once (which spends capacity wherever
the loss is largest — the trend-starves-season pathology DLinear's decomposition was a crude fix for),
build a *stack* in which each block reads the *residual* look-back — what earlier blocks have not yet
explained — and outputs two things: a partial forecast (its contribution to the horizon) and a
backcast (its reconstruction of the part of the look-back it just used). The double-residual loop is
`r_b = r_{b-1} − x̂_b` (subtract this block's backcast from the running look-back residual, so the next
block faces a cleaner signal) and `y += ŷ_b` (accumulate this block's forecast). The backcast stream
flows backward, peeling the input apart component by component; the forecast stream flows forward as a
sum. This is the boosting/ResNet idea applied to the look-back: each block models only a small
correction, so a deep stack stays trainable, and — the part that matters for the Yearly/Monthly tension
— the decomposition is *emergent and per-series*, not a fixed pooling window or a fixed FFT lens. A
pure-trend Yearly series and a strong-seasonal Monthly series are decomposed by *whatever each block
learns to peel off*, so one architecture adapts to both.

Inside a block I want the output *constrained*, because the M4 series are short and an unconstrained
per-block horizon would overfit training noise — precisely the failure that sank naive MLPs on M4. So
a block maps its residual input through a small FC stack (a few `Linear`+`ReLU` layers of width `W`) to
a low-dimensional coefficient vector `θ`, and the forecast is `V·θ` for a fixed basis `V`. The
**generic** basis is the identity — two linear maps, `hidden → seq_len` for the backcast and
`hidden → pred_len` for the forecast — which is maximally flexible and is the raw-accuracy
configuration I will reach for here, because across three heterogeneous regimes I care more about
accuracy than about reading off the components. (The **interpretable** alternative constrains the basis
to a low-degree polynomial in trend blocks and a Fourier sine/cosine basis in seasonality blocks, so
stacking a trend stack then a seasonality stack yields an *explicit* learned trend/season split — the
same decomposition STL imposes by hand, here emergent. That variant trades a little accuracy for
transparency; the generic stack already gets the decomposition-by-refinement for free through the
residual loop, so it is the right default for a leaderboard.) Either way the backcast and forecast
share the block's hidden representation but use *different* basis sizes (`seq_len` vs `pred_len`). The
interpretable configuration additionally shares weights across blocks within a stack — a strong
regularizer that forces the stack to learn one residual operator rather than many independent ones; the
generic configuration I ship here keeps the blocks *distinct* (the canonical generic factory), relying
instead on the small theta dimension, the persistence seed, and the 10-epoch early-stopping cap for the
regularization the short M4 series demand and the naive-MLP entrants lacked.

Two details the loop needs. First, reverse the look-back before feeding it in, so the most recent point
sits in a stable leading position regardless of `seq_len` — recent values matter most and I want them
read consistently across the three regimes' different window lengths. Second, seed the running forecast
with the *last observed value* (`x[:, -1:]`) rather than zero, so the network only has to learn the
*deviation* from a persistence forecast — a naive-last baseline is already decent on short horizons,
and starting the residual sum there makes the blocks' job a correction on top of persistence, which is
both easier to learn and a sensible prior on the trend-dominated Yearly regime where the strongest
baselines stalled.

Now grounding in *this* harness, where I only get to fill `models/Custom.py` under the fixed Custom
protocol. The harness gives `seq_len = 2·pred_len` univariate windows, calls
`model(batch_x, None, dec_inp, None)` (no time marks — N-BEATS never wanted them, it reads only the raw
look-back), optimizes SMAPE, and passes `d_model = 512`, `e_layers = 2`, `batch_size = 16`,
`lr = 1e-3`, 10 epochs, patience 3. N-BEATS does not use `d_model`/`e_layers` as an embedding/attention
depth, so I repurpose what the harness offers sensibly: FC width `W = 512` (reusing the wide channels
the previous rungs left stranded — here they feed a fully-connected block that can actually use width),
a moderate number of generic blocks with weight sharing, 4 FC layers per block. I squeeze the channel
axis (`enc_in = 1`) to feed the `[batch, seq_len]` vector to the blocks and unsqueeze the
`[batch, pred_len]` forecast back. The input mask is all-ones (M4 windows are dense), so the backcast
subtraction is unmasked. No ensembling — the harness trains one model at one look-back for 10 epochs —
which is the honest limitation I have to forecast against: N-BEATS' published M4 result leans on
ensembling over look-back lengths, inits, and losses to average away short-series variance, and a
single un-ensembled fit under a 10-epoch cap will not reach that. So the bar I am clearing is not the
paper's ensembled number; it is *this ladder's strongest baseline under this exact protocol*.

What must this clear, and where do I expect it to win or lose against TimesNet's measured
12.803 / 10.089 / 13.442? The case for N-BEATS here is strongest exactly where the ladder stalled.
**Yearly** is the region of greatest hope: it is trend-dominated, the strongest baselines were *no
better than the multi-scale MLP* there (13.442 / 13.378), and N-BEATS' persistence-seeded residual
refinement plus its trend-friendly generic blocks are built for pure-trend short horizons — seeding the
forecast at the last value and correcting it is precisely the right prior for six trend steps, so I
expect to beat 13.44 here, the clearest opportunity of the three. **Quarterly** is the hardest to beat:
TimesNet's FFT lock on the sharp 4-step period drove it to 10.089, and a generic FC stack with no
explicit period basis must rediscover that periodicity from data under a 10-epoch cap — I expect to be
*competitive* (around or slightly above 10.09) but would not be surprised to fall just short there.
**Monthly** I expect to be close to the ~12.80 cluster: the residual stack can model the 12-step
seasonality through its generic basis, but again without the period prior TimesNet/TimeMixer encode, so
a tie-to-slight-win is the realistic target. So the finale's claim to the endpoint rests on a *mean*
win driven by Yearly: if the trend-regime gain (where every baseline plateaued) outweighs a possible
Quarterly shortfall and Monthly is a wash, the persistence-seeded refinement architecture takes the
ladder's mean SMAPE below TimesNet's ~12.11 — and it does so by being the first rung whose inductive
bias is *adaptive refinement* rather than a single fixed lens, which is the exact thing the per-regime
split in the baselines showed was missing. If instead Yearly does not move, that falsifies the bet that
the unsolved regime was a *refinement* problem rather than a capacity problem, and would point back to
ensembling — the one piece of N-BEATS' recipe this fixed protocol withholds — as the missing ingredient.
