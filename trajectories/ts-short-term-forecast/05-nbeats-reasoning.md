The four baselines have converged, and reading them together tells me what the ladder has and has not
solved. DLinear set the affine floor (13.39 / 10.50 / 14.36). PatchTST added a learned nonlinear
representation and instance normalization and beat it everywhere but only by tenths (12.97 / 10.22 /
13.68), because attention over one-to-four patch tokens is starved. TimeMixer added explicit
multi-scale decomposition-mixing and won on every regime (12.80 / 10.21 / 13.38), its gains where trend
and season separate across scales. TimesNet added FFT period-discovery and a 2D intra/interperiod
layout and edged ahead overall (12.803 / 10.089 / 13.442) — but the win is *entirely Quarterly*: it took
the cleanest-period regime from 10.209 to 10.089, tied TimeMixer on Monthly (12.803 vs 12.802), and came
in *worse* on Yearly (13.442 vs 13.378), exactly where there is no period to discover. That last fact is
where the frontier has stalled. Trace the Yearly column down the ladder: 14.358 → 13.679 → 13.379 →
13.442. The big drop came early (RevIN removing level drift), TimeMixer squeezed a little more with
top-down trend mixing to the ladder-best 13.379, and then the strongest baseline *regressed* by 0.064,
because its period machinery has nothing to find on a trend-dominated six-step horizon. So at the top of
the ladder the trend regime is not improving; the best Yearly number belongs to a multi-scale MLP, and
the period-aware model is slightly worse there. Quarterly kept falling to 10.089, Monthly sits in a
tight 12.80 cluster. The strongest baseline is strong on periodic regimes and *no better than a
multi-scale MLP on the trend-dominated one*.

The unsolved problem is sharp, and it is not "add more capacity" — the ladder already tried width
(PatchTST's 512-wide encoder) and specialized structure (TimeMixer's ladder, TimesNet's FFT), and each
new lens helped its own regime while leaving Yearly's frontier where a simple mixer put it. Every model
so far is a *single* architecture that must serve trend-only Yearly and strong-seasonal Monthly with one
fixed inductive structure, and whichever it picks — pooling ladder, period FFT, patch attention — is
mismatched to at least one regime. TimesNet's Quarterly win and Yearly regression is that mismatch made
visible in one model: the very lens that unlocked the sharp-period regime is dead weight on the trend
regime. What I want is an architecture whose bias is not "one lens" but *sequential refinement* — peel
off whatever the previous part could not explain — so it adapts to a pure-trend series and a strongly
seasonal one without a regime-specific lens, and exposes a trend/season split as a by-product rather
than imposing one.

The structural idea is a deep stack of fully-connected **blocks** wired by a **double residual** loop,
each constrained to emit a *basis expansion*. The output strategy is settled by the direct-multi-step
argument that has held since DLinear: emit the whole horizon at once, no roll-out, no error
accumulation. The novelty is what goes between input and output. Instead of one network explaining the
whole look-back at once — which spends capacity wherever the loss is largest, the trend-starves-season
pathology DLinear's decomposition crudely fixed — build a *stack* in which each block reads the
*residual* look-back (what earlier blocks have not yet explained) and outputs two things: a partial
forecast (its contribution to the horizon) and a backcast (its reconstruction of the part of the
look-back it just used). The double-residual loop is `r_b = r_{b-1} − x̂_b` (subtract this block's
backcast from the running look-back residual, so the next block faces a cleaner signal) and `y += ŷ_b`
(accumulate the forecast). The backcast stream flows backward, peeling the input apart component by
component; the forecast stream flows forward as a sum. This is the boosting/ResNet idea applied to the
look-back: each block models only a small correction, so a deep stack stays trainable, and — the part
that matters for the Yearly/Monthly tension — the decomposition is *emergent and per-series*, not a
fixed pooling window or FFT lens. A pure-trend Yearly series and a strong-seasonal Monthly series are
decomposed by whatever each block learns to peel off, so one architecture adapts to both.

The subtraction is the load-bearing line, so let me trace the loop on Monthly. The look-back is
`[B, 36]` after squeezing the channel. I reverse it so the most recent point leads
(`residuals = flip(x)`) and seed the running forecast with the last observed value
(`forecast = x[:, -1:]` broadcast to `[B, 18]`). Block 1 reads `residuals ∈ [B, 36]`, runs it through
four `Linear+ReLU` layers of width 512 and a final `Linear(512 → theta_size)` with
`theta_size = seq_len + pred_len = 54`, slicing `theta[:, :36]` as the backcast and `theta[:, -18:]` as
the forecast. The slices are disjoint — `[:36]` is indices 0..35, `[-18:]` is 36..53, and `36 + 18 = 54`
— so they tile the 54-dim theta exactly with no overlap, and the block cannot double-count a dimension
between backcast and forecast. Then `residuals ← residuals − backcast` (a cleaner `[B, 36]`) and
`forecast ← forecast + block_forecast`. Blocks 2 through 6 repeat on the progressively cleaned residual;
after six I unsqueeze the channel back to `[B, 18, 1]`. Without the subtraction every block would see the
same input and the stack would collapse to six copies of one block; with it, block `b` only ever sees
`x` minus what blocks `1..b−1` already explained, which is what makes the decomposition sequential.

Inside a block I want the output *constrained*, because the M4 series are short and an unconstrained
per-block horizon would overfit training noise — the failure that sank naive MLPs on M4. So a block maps
its residual input through the FC stack to a low-dimensional coefficient vector `θ`, and the forecast is
`V·θ` for a fixed basis `V`. Two real configurations, and I choose deliberately. The **interpretable**
basis constrains `V` to a low-degree polynomial in trend blocks and a Fourier basis in seasonality
blocks, so stacking a trend stack then a seasonality stack yields an explicit learned trend/season split
— the same decomposition STL imposes by hand, here emergent. On a task where I wanted to read off the
trend it would be the choice. But I am optimizing a leaderboard across three heterogeneous regimes, and
the polynomial/Fourier basis trades a little raw accuracy for that transparency: it constrains each
block's forecast to a smooth low-order shape, exactly the flexibility I do not want to give up on
Monthly's sharp seasonal shape or Quarterly's period structure. The **generic** basis is the identity —
two linear maps, `hidden → seq_len` for the backcast and `hidden → pred_len` for the forecast — maximally
flexible, and it already gets the decomposition-by-refinement for free through the residual loop. So the
generic identity basis is the right default for a leaderboard, and I ship it.

That interacts with whether to share weights across blocks. The interpretable configuration shares
weights within a stack, a strong regularizer forcing one residual operator. The generic configuration I
ship keeps the blocks *distinct* (the canonical generic factory): a flat stack of `n_blocks = 6`
separate generic blocks, each with its own weights. Distinct blocks are more expressive — each can
specialize on a different residual component — but also more parameters, so I have to be sure the short
series can afford them. Count it: each block is `36·512` for the first FC layer, `3·512·512` for the
three hidden layers, and `512·54` for the theta map, about `0.83M`, so six distinct blocks is roughly
`5M` on Monthly (`~4.8M` on the shorter regimes). That is the largest footprint on the entire ladder —
larger than PatchTST's ~3M encoder — fitting series that are individually thirty-six points long. The
regularization I count on is not weight sharing (given up) but three other things: the small theta
dimension per block, the persistence seed so each block learns only a correction rather than the whole
forecast, and the 10-epoch early-stopping cap. Whether ~5M distinct-block parameters overfit under those
constraints is the real risk of this finale, and it is the price of the generic configuration.

Stack depth is its own small budget. Each block peels one residual component, so more blocks means finer
decomposition — but each is ~0.8M parameters and every added block is more to fit inside ten epochs, so
depth trades peeling resolution against trainability. Too few and the stack cannot separate trend from
the several seasonal components a Monthly series carries; too many and the later blocks fit residuals
that are mostly noise on a thirty-six-point series. A moderate `n_blocks = 6` with `layers = 4` FC layers
each is the balance: enough to strip a trend and a couple of seasonal harmonics on Monthly while keeping
the total near 5M. I am not tuning this against held-out SMAPE — the protocol gives one shot — so I pick
the moderate canonical depth rather than gambling the fit budget.

Two details the loop needs, both deliberate. Reverse the look-back before feeding it in, so the most
recent point sits in a stable leading position regardless of `seq_len` (36, 16, 12) — recent values
matter most and I want the block's first-layer weights to find "most recent" in the same coordinate
across regimes. And seed the running forecast with the last observed value rather than zero, so the
network only learns the *deviation* from a persistence forecast: if every block output zero (the safe
thing when it has nothing to add), the total forecast would be the last value repeated across the
horizon — the naive persistence baseline, already decent on short horizons. So the seed makes
persistence the floor the stack corrects from, both easier to learn and a sensible prior on the
trend-dominated Yearly regime where six trend steps are close to persistence plus a small drift.

It is worth checking the emergent decomposition falls out on the two extreme regimes, since "the blocks
peel off whatever each series needs" is the whole claim. On a pure-trend Yearly window the first block,
reading the reversed twelve points, can fit a backcast reconstructing the near-linear level and a
forecast extending it — most of the signal is gone after one peel, and the remaining blocks refine a
small mostly-noise residual, so the stack behaves like "persistence-seed plus a one-block trend
correction." On a strong-seasonal Monthly window the early blocks peel the 12-step periodic shape (the
generic basis is free to emit any horizon shape), and later blocks peel the weaker harmonics and the
residual trend. The same six distinct blocks produce a one-component decomposition on Yearly and a
multi-component one on Monthly with no regime switch — the adaptivity the fixed-lens baselines lacked.

Now grounding in this harness, where I only fill `models/Custom.py`. The harness gives
`seq_len = 2·pred_len` univariate windows, calls `model(batch_x, None, dec_inp, None)` (no time marks —
this model reads only the raw look-back), optimizes SMAPE, and passes `d_model = 512`, `e_layers = 2`,
batch 16, `lr = 1e-3`, 10 epochs, patience 3. Since there is no embedding or attention depth to spend
`d_model`/`e_layers` on, I repurpose the width as FC width `W = 512` — reusing the wide channels the
previous rungs left stranded, here feeding a fully-connected block that can *actually* use width, unlike
attention idling over two tokens or an inception convolving a `6 × 4` grid. I squeeze the channel axis to
feed `[batch, seq_len]` and unsqueeze the `[batch, pred_len]` forecast back; the input mask is all-ones
(M4 windows are dense), so the backcast subtraction is unmasked. I am tempted to smuggle the ensemble
back in inside `Custom.py` — three parallel stacks with different initializations, averaged, a bagging
ensemble hidden in one `Model`. I reject it for two protocol-specific reasons. First, footprint: one
stack is already ~5M, so three is ~15M fitting thirty-six-point series under a 10-epoch cap, and since
the three share the same ten epochs and batch budget each would be *more* under-trained — I would be
averaging three worse fits, and the ensemble's variance reduction assumes each member is itself
well-trained. Second, it would confound the reading I want: I am testing whether *adaptive refinement*
is the missing inductive bias, and a hidden ensemble would blur "refinement helped" with "averaging
helped." So I ship the honest single flat stack. This is the finale's known limitation to forecast
against: the full recipe leans on ensembling over look-back lengths, inits, and losses to average away
short-series variance, and a single un-ensembled fit under a 10-epoch cap will not reach that — but the
bar I am clearing is this ladder's strongest baseline under this exact protocol, not an ensembled number.

What must this clear, against TimesNet's 12.803 / 10.089 / 13.442? The case is strongest exactly where
the ladder stalled. **Yearly** is the region of greatest hope: trend-dominated, the frontier there
stopped improving at the top of the ladder (TimeMixer's 13.378 was best, the strongest baseline
regressed to 13.442), and persistence-seeded residual refinement plus trend-friendly generic blocks are
built for pure-trend short horizons — seeding at the last value and correcting it is the right prior for
six trend steps, so I expect to beat 13.442 and ideally push under the 13.378 frontier, the clearest
opportunity of the three. **Quarterly** is the hardest: TimesNet's FFT lock on the sharp 4-step period
drove it to 10.089, and a generic FC stack with no explicit period basis must rediscover that
periodicity from data under a 10-epoch cap — I expect to be competitive (around or slightly above
10.089) but would not be surprised to fall just short. **Monthly** I expect close to the 12.80 cluster:
the residual stack can model the seasonality through its generic basis, but without the period prior
TimesNet/TimeMixer encode, a tie-to-slight-win is the realistic target. So the finale's claim to the
endpoint rests on a *mean* win driven by Yearly: if the trend-regime gain outweighs a possible Quarterly
shortfall and Monthly is a wash, the persistence-seeded refinement architecture takes the mean SMAPE
below TimesNet's — as the first model whose inductive bias is *adaptive refinement* rather than a single
fixed lens. If Yearly does not move, that falsifies the bet that the unsolved regime was a refinement
problem rather than a capacity one, and points back to ensembling, the one piece of the recipe this
protocol withholds, as the missing ingredient.
