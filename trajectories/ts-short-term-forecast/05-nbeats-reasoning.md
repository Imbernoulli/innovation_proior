The four baselines have converged, and reading them together tells me what the ladder has and has not
solved. DLinear set the affine floor (13.39 / 10.50 / 14.36). PatchTST added a learned nonlinear
representation and instance normalization and beat it everywhere but only by tenths (12.97 / 10.22 /
13.68), because attention over one-to-four patch tokens on a `2·pred_len` window is starved. TimeMixer
added explicit multi-scale decomposition-mixing and won on every regime (12.80 / 10.21 / 13.38), its
gains concentrated where trend and season separate across scales. TimesNet added FFT period-discovery
and a 2D intra/interperiod layout, and edged ahead overall (12.803 / 10.089 / 13.442) — but the win is
*entirely Quarterly*: it took the cleanest-period regime from 10.209 to 10.089, essentially tied
TimeMixer on Monthly (12.803 vs 12.802), and actually came in *worse* on Yearly (13.442 vs 13.378),
exactly where there is no period to discover and trend is the whole story. That last fact is the one I
want to sit with, because it is where the frontier has stalled. Let me trace the Yearly column down the
ladder: 14.358 → 13.679 → 13.379 → 13.442. The big drop came early (DLinear to PatchTST, RevIN removing
level drift), TimeMixer squeezed a little more with top-down trend mixing to the ladder-best 13.379, and
then the strongest baseline *regressed* by 0.064 — because its period machinery has nothing to find on a
trend-dominated six-step horizon and the FFT's coarse low-frequency bins are noise. So at the top of the
ladder the trend regime is not improving; the best Yearly number belongs to a multi-scale MLP, and the
period-aware model is slightly worse there. Contrast Quarterly, which kept falling to 10.089, and Monthly,
which sits in a tight 12.80 cluster. The strongest baseline, at mean SMAPE ~12.11, is strong on periodic
regimes and *no better than a multi-scale MLP on the trend-dominated one*.

The unsolved problem is therefore sharp, and it is not "add more capacity" — the ladder already tried
width (PatchTST's 512-wide encoder) and specialized structure (TimeMixer's ladder, TimesNet's FFT) and
each new lens helped its own regime while leaving Yearly's frontier where a simple mixer put it. Every
rung so far is a *single* architecture that must serve trend-only Yearly and strong-seasonal Monthly with
one fixed inductive structure, and whichever structure it picks — pooling ladder, period FFT, patch
attention — is mismatched to at least one regime. TimesNet's Quarterly win and Yearly regression is that
mismatch made visible in one model: the very lens that unlocked the sharp-period regime is dead weight on
the trend regime. What I want is an architecture whose inductive bias is not "one lens" but *sequential
refinement* — peel off whatever the previous part of the model could not explain — so it adapts to a
pure-trend series and a strongly seasonal one without a regime-specific lens, and ideally exposes a
trend/season split as a by-product rather than imposing one.

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
learns to peel off*, so one architecture adapts to both, which is precisely the "not one lens" property
the per-regime baseline split showed was missing.

Let me trace the loop concretely on Monthly so the shapes and the peeling are unambiguous, because a
double-residual stack is exactly the kind of thing that silently no-ops if the backcast and forecast
slices overlap or the residual is not actually subtracted. The look-back is `[B, 36]` after squeezing the
channel. I reverse it so the most recent point leads, `residuals = flip(x)`, and seed the running
forecast with the last observed value, `forecast = x[:, -1:]` broadcast to `[B, 18]`. Block 1 reads
`residuals ∈ [B, 36]`, runs it through four `Linear+ReLU` layers of width 512, and a final
`Linear(512 → theta_size)` with `theta_size = seq_len + pred_len = 54`; it slices `theta[:, :36]` as the
backcast and `theta[:, -18:]` as the forecast. I check the slices are disjoint: `[:36]` is indices 0..35
and `[-18:]` is indices 36..53, and `36 + 18 = 54`, so they tile the 54-dim theta exactly with no
overlap — the block cannot double-count a dimension between its backcast and forecast. Then
`residuals ← residuals − backcast` (now a cleaner `[B, 36]`) and `forecast ← forecast + block_forecast`.
Blocks 2 through 6 repeat on the progressively cleaned residual. After six blocks I unsqueeze the channel
back to `[B, 18, 1]`. The subtraction is the load-bearing line: if I forgot it, every block would see the
same input and the stack would collapse to six copies of one block; because it is there, block `b` only
ever sees `x` minus what blocks `1..b−1` already explained, which is what makes the decomposition
sequential.

Inside a block I want the output *constrained*, because the M4 series are short and an unconstrained
per-block horizon would overfit training noise — precisely the failure that sank naive MLPs on M4. So
a block maps its residual input through the small FC stack to a low-dimensional coefficient vector `θ`,
and the forecast is `V·θ` for a fixed basis `V`. There are two real configurations here and I have to
choose deliberately. The **interpretable** basis constrains `V` to a low-degree polynomial in trend
blocks and a Fourier sine/cosine basis in seasonality blocks, so stacking a trend stack then a
seasonality stack yields an *explicit* learned trend/season split — the same decomposition STL imposes by
hand, here emergent. It is genuinely attractive, and on a task where I wanted to *read off* the trend it
would be the choice. But I am optimizing a leaderboard across three heterogeneous regimes, and the
polynomial/Fourier basis trades a little raw accuracy for that transparency: it constrains each block's
forecast to a smooth low-order shape, which is exactly the flexibility I do not want to give up on
Monthly's sharp seasonal shape or Quarterly's period structure. The **generic** basis is the identity —
two linear maps, `hidden → seq_len` for the backcast and `hidden → pred_len` for the forecast — which is
maximally flexible and lets each block emit any horizon shape. The generic stack already gets the
decomposition-by-refinement for free through the residual loop, so it captures the adaptive-decomposition
property I came for *without* the accuracy cost of the constrained basis. So the generic identity basis
is the right default for a leaderboard, and I ship it.

That choice interacts with a second one: whether to share weights across blocks. The interpretable
configuration shares weights across the blocks within a stack — a strong regularizer that forces the
stack to learn one residual operator rather than many independent ones. The generic configuration I ship
keeps the blocks *distinct* (the canonical generic factory): a flat stack of `n_blocks = 6` separate
generic blocks, each with its own weights. Distinct blocks are more expressive — each can specialize on a
different residual component — but they are also more parameters, so I have to be sure the short M4 series
can afford them. Count it: each block is `36·512` for the first FC layer, `3·512·512` for the three
hidden layers, and `512·54` for the theta map, about `0.83M` parameters, so six distinct blocks is
roughly `5M` on Monthly (`~4.8M` on the shorter regimes). That is the largest trainable footprint on the
entire ladder — larger than PatchTST's ~3M encoder — and it is fitting series that are individually
thirty-six points long. The regularization I am counting on to make that safe is *not* weight sharing
(which I gave up) but three other things: the small theta dimension per block, the persistence seed that
means each block only has to learn a correction rather than the whole forecast, and the 10-epoch
early-stopping cap. Whether ~5M distinct-block parameters overfit under those constraints is the real risk
of this finale, and it is the price of the generic configuration's expressivity.

The stack depth is its own small budget calculation, not a free knob. Each block peels off one residual
component, so more blocks means finer sequential decomposition — but each block is ~0.8M parameters and
every added block is more to fit inside the same ten epochs, so depth trades peeling resolution against
trainability under the cap. Too few blocks and the stack cannot separate trend from the several seasonal
components a Monthly series carries; too many and the later blocks are fitting residuals that are mostly
noise on a thirty-six-point series, and the whole ~`n·0.8M` footprint has to converge in ten epochs. A
moderate `n_blocks = 6` with `layers = 4` FC layers each is the balance I choose: six peeling stages is
enough to strip a trend and a couple of seasonal harmonics on Monthly while keeping the total near 5M,
which the persistence seed and early stopping can plausibly train. I am not tuning this against held-out
SMAPE — the protocol gives me one shot — so I pick the moderate canonical depth rather than pushing
blocks up and gambling the fit budget.

Two details the loop needs, and both are deliberate. First, reverse the look-back before feeding it in,
so the most recent point sits in a stable leading position regardless of `seq_len` — recent values
matter most and I want them read consistently across the three regimes' different window lengths (36,
16, 12), so the block's first-layer weights always find "most recent" in the same coordinate. Second,
seed the running forecast with the *last observed value* (`x[:, -1:]`) rather than zero, so the network
only has to learn the *deviation* from a persistence forecast. Let me verify this seed does something
sensible at the limit: if every block learned to output zero forecast (the safe thing when it has nothing
to add), the total forecast would be exactly the last observed value repeated across the horizon — the
naive persistence baseline, which is already decent on short horizons. So the seed makes persistence the
*floor* the stack corrects downward from, rather than something the blocks must reconstruct from scratch;
starting the residual sum there is both easier to learn and a sensible prior on the trend-dominated
Yearly regime where the strongest baselines stalled, because six trend steps are close to persistence
plus a small drift.

It is worth checking that the emergent decomposition actually falls out on the two regimes at the ends of
the spectrum, because "the blocks peel off whatever each series needs" is the whole claim and I should be
able to trace it. On a pure-trend Yearly window the first block, reading the reversed twelve points, can
fit a backcast that reconstructs the near-linear level and a forecast that extends it — most of the
signal is gone after one peel, and the remaining blocks refine a small residual that is mostly noise, so
the stack behaves like "persistence-seed plus a one-block trend correction," which is the right model for
six trend steps. On a strong-seasonal Monthly window the early blocks can peel the 12-step periodic shape
(the generic basis is free to emit any horizon shape, so a block can output one period's worth of
seasonal pattern), and later blocks peel the weaker harmonics and the residual trend. The *same* six
distinct blocks produce a one-component decomposition on Yearly and a multi-component one on Monthly with
no regime switch — the stack allocates its peeling depth to whatever the series contains. That is the
adaptivity the fixed-lens baselines lacked, and tracing it on the two extreme regimes is what convinces
me the architecture is doing something different in kind from adding capacity.

Now grounding in *this* harness, where I only get to fill `models/Custom.py` under the fixed Custom
protocol. The harness gives `seq_len = 2·pred_len` univariate windows, calls
`model(batch_x, None, dec_inp, None)` (no time marks — N-BEATS never wanted them, it reads only the raw
look-back), optimizes SMAPE, and passes `d_model = 512`, `e_layers = 2`, `batch_size = 16`, `lr = 1e-3`,
10 epochs, patience 3. N-BEATS does not use `d_model`/`e_layers` as an embedding/attention depth, so I
repurpose what the harness offers sensibly: FC width `W = 512` (reusing the wide channels the previous
rungs left stranded — here they feed a fully-connected block that can *actually* use width, unlike
attention idling over two tokens or an inception convolving a `6 × 4` grid), a moderate stack of generic
blocks (`n_blocks = 6`), 4 FC layers per block. I squeeze the channel axis (`enc_in = 1`) to feed the
`[batch, seq_len]` vector to the blocks and unsqueeze the `[batch, pred_len]` forecast back. The input
mask is all-ones (M4 windows are dense), so the backcast subtraction is unmasked. I am tempted to smuggle the ensemble back in *inside* `Custom.py` — nothing stops me from instantiating,
say, three parallel stacks with different initializations and averaging their forecasts, a bagging
ensemble hidden inside one `Model`. It is the obvious way to reclaim the variance-averaging that the full
recipe gets from ensembling. But I walk it and reject it, for two reasons that are specific to this
protocol. First, the footprint: one stack is already ~5M parameters, so three parallel stacks is ~15M
fitting thirty-six-point series under a 10-epoch cap — each sub-model would be *more* under-trained than a
single stack, because they share the same ten epochs and the same batch budget, so I would be averaging
three worse fits rather than three good ones. The ensemble's variance reduction assumes each member is
itself well-trained, which the 10-epoch cap does not deliver. Second, it would blur exactly the reading I
want from the finale: I am testing whether *adaptive refinement* is the missing inductive bias, and a
hidden ensemble would confound "refinement helped" with "averaging helped." So I ship the honest single
flat stack and name ensembling as the withheld ingredient rather than sneaking a degraded version of it
in. No ensembling — the harness trains one model at one look-back for 10 epochs — which is the honest
limitation I have to forecast against: N-BEATS' full recipe leans on ensembling over look-back lengths, inits, and losses to
average away short-series variance, and a single un-ensembled fit under a 10-epoch cap will not reach
that. So the bar I am clearing is not an ensembled number; it is *this ladder's strongest baseline under
this exact protocol*.

What must this clear, and where do I expect it to win or lose against TimesNet's measured
12.803 / 10.089 / 13.442? The case for N-BEATS here is strongest exactly where the ladder stalled.
**Yearly** is the region of greatest hope: it is trend-dominated, the frontier there stopped improving at
the top of the ladder (TimeMixer's 13.378 was best and the strongest baseline regressed to 13.442), and
N-BEATS' persistence-seeded residual refinement plus its trend-friendly generic blocks are built for
pure-trend short horizons — seeding the forecast at the last value and correcting it is precisely the
right prior for six trend steps, so I expect to beat 13.442, and ideally to push under TimeMixer's 13.378
frontier, the clearest opportunity of the three. **Quarterly** is the hardest to beat: TimesNet's FFT
lock on the sharp 4-step period drove it to 10.089, and a generic FC stack with no explicit period basis
must rediscover that periodicity from data under a 10-epoch cap — I expect to be *competitive* (around or
slightly above 10.089) but would not be surprised to fall just short there, because the identity basis
has no periodic prior the way TimesNet's 2D reshape does. **Monthly** I expect to be close to the ~12.80
cluster: the residual stack can model the 12-step seasonality through its generic basis, but again
without the period prior TimesNet/TimeMixer encode, so a tie-to-slight-win is the realistic target. So
the finale's claim to the endpoint rests on a *mean* win driven by Yearly: if the trend-regime gain
(where every baseline plateaued) outweighs a possible Quarterly shortfall and Monthly is a wash, the
persistence-seeded refinement architecture takes the ladder's mean SMAPE below TimesNet's ~12.11 — and it
does so by being the first rung whose inductive bias is *adaptive refinement* rather than a single fixed
lens, which is the exact thing the per-regime split in the baselines showed was missing. If instead
Yearly does not move, that falsifies the bet that the unsolved regime was a *refinement* problem rather
than a capacity problem, and would point back to ensembling — the one piece of N-BEATS' recipe this
fixed protocol withholds — as the missing ingredient.
