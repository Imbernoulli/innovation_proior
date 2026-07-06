The multi-scale mixer paid off where I bet it would, and reading exactly *where* tells me what is still
unexploited. TimeMixer came in at SMAPE 12.80 on Monthly, 10.21 on Quarterly, 13.38 on Yearly — beating
PatchTST on all three (12.97 / 10.22 / 13.68), and the biggest gains landed on Monthly and especially
Yearly, the regimes where trend and season live at different resolutions and the top-down trend mixing
had real work to do. So the multi-scale thesis held: explicit cross-scale routing of a decomposed
season/trend representation is more structure than a single-resolution patch encoder can express on
short windows, and it put the harness's wide channels to use. But the number that actually tells me what
to build next is the one that *did not* move. Let me read the three deltas against PatchTST as
percentages. Monthly improved 0.166 (1.28%), Yearly 0.300 (2.20%), and Quarterly 0.010 — one
hundredth of a SMAPE point, 0.10%, statistically indistinguishable from no change. That is a sharp,
localized failure hiding inside an across-the-board win, and it is exactly diagnostic. Quarterly is the
regime with the *cleanest* single dominant period (the 4-quarter cycle) and the shortest window after
Yearly (16 steps, two cycles), so one average-pooling step to a coarse 8-step view does not separate
much: there is no trend/detail split across scales to exploit when the structure is *one sharp period*.
The fixed pooling ladder is simply the wrong lens for that regime. That is the gap. TimeMixer exposes
scale by *blind* pooling — it never asks which periodicity a given series actually carries; it just
halves the length and hopes trend and season fall out. On a series whose structure is a single strong
period, the informative thing is not "coarse vs fine," it is "the value one period back," and a pooling
ladder represents that only incidentally.

So the move is to discover the period structure *inside the window* and represent the series around it,
rather than relying on a fixed downsampling lens to expose it. Let me reason about what kinds of
dependency a forecast point actually has, because that is what should drive the layout. A point depends
on its immediate neighbors — the local shape within the current cycle, short-range, *intraperiod*. But
it also depends on the same phase one cycle ago and one cycle ahead — the same quarter last year, the
same month in adjacent years — which is how the corresponding phase *changes from cycle to cycle*,
long-range, *interperiod*. Those are two genuinely different dependencies, and real M4 series carry both
(plus, often, more than one period at once). Now the structural problem: the data is a 1D sequence
indexed by `t`. Adjacency along that axis gives the intraperiod neighbors for free (`t−1`, `t+1`), but
the interperiod neighbor at `t−p` is `p` steps away with an entire cycle crammed in between. A 1D
convolution with any sane kernel never sees `t` and `t−p` together; an MLP over absolute window position
(DLinear, TimeMixer's per-scale predictors) has no explicit handle on *which* period a series carries
and cannot pull the within-cycle and across-cycle parts apart; attention relates point pairs but, over
a window that is mostly ordinary points, the similarity is dominated by them. So every tool so far,
including the pooling ladder, inherits the same bottleneck: the 1D layout can present intraperiod
variation as locality but effectively hides interperiod variation. That is exactly why Quarterly
stalled — the one-period-back relationship that *is* the Quarterly signal is never made local.

The fix is to change the layout so both kinds of locality appear at once. For a period `p`, chop the
series into consecutive blocks of length `p` and stack them as the rows of a 2D array: walking along a
row traverses one cycle (intraperiod shape), walking down a column traverses the same phase across
successive cycles (interperiod change). The point that was `p` apart in 1D is now one step apart along
the cross-period axis. So reshape the 1D series, for a given `p`, into a 2D tensor whose two axes are
precisely intraperiod and interperiod — and then *both* dependencies are adjacencies a 2D convolution
can read simultaneously, which is the thing a 1D layout and a pooling ladder structurally cannot do.

Which `p`? Discover it, do not assume it — and let me be explicit about the alternative, because
assuming it is tempting when I *know* Quarterly's period is 4 and Monthly's is 12. I could hard-code the
seasonal period per regime and skip the FFT entirely. Two reasons I reject that. First, the harness runs
*one* `Custom.py` across all three regimes under fixed hyperparameters; a hard-coded period would need to
branch on `pred_len` to pick 4/12/1, and on Yearly there is no period to pick at all. Second and deeper,
real M4 series carry more than one period and carry it imperfectly — a monthly series has a 12-month
cycle but often a weaker 6-month harmonic — and a data-driven read adapts per series where a hard-coded
constant cannot. So I discover it: take the FFT of the window, the amplitude spectrum tells me how much
energy sits at each frequency, and the peaks are the dominant periodicities. Average the amplitude over
channels (one channel here), zero the DC term (it is the window mean, not a period), and take the top-`k`
frequencies by amplitude; each frequency `f` gives a period `p = T/f` and an amplitude that is its
*confidence*. Top-`k` (not all frequencies) because the spectrum of a short real series is sparse and the
small-amplitude bins are noise. This is the direct period read that TimeMixer's pooling only
approximated: on Quarterly the FFT should put a sharp peak at the 4-step period and reshape the window
into a 2D grid whose columns are within-quarter shape and whose rows are year-over-year change — making
the one-period-back relationship local for the first time on the ladder.

There is a subtlety in the period read I should not gloss over, because it interacts with the harness's
short windows, and I can put numbers on it. The FFT here runs over the *extended* sequence
`seq_len + pred_len` — I will justify the extension below — so the lengths are Monthly `36 + 18 = 54`,
Quarterly `16 + 8 = 24`, Yearly `12 + 6 = 18`. The real FFT of a length-`T` window returns
`floor(T/2) + 1` non-negative-frequency bins: 28 for Monthly, 13 for Quarterly, 10 for Yearly. So the
Quarterly spectrum has thirteen bins to locate a peak in, and a clean 4-step period sits at frequency
`f = 24/4 = 6`, a bin well inside range with plenty of resolution — the reshape then gives
`24 / 4 = 6` rows of 4 columns, six full cycles stacked, a genuinely informative grid. Yearly is the
opposite: ten coarse bins over a trend-dominated 18-step window, where the top-`k` peaks will be
low-frequency bins giving *long* periods like `p = 18/1 = 18` or `p = 18/2 = 9` (a 2×9 or near-degenerate
grid) — there is simply no real short period to find. And periods come out as `p = floor(T/f)`, which on
a short window can collapse several distinct frequencies onto the same integer period or onto periods so
short (2 or 3 steps) that the 2D reshape has many rows of two columns — a degenerate grid. That is fine:
a degenerate short-period grid simply lets the inception block act almost like a 1D convolution there, so
the model gracefully falls back to local smoothing on regimes where there is no real long period to
find. The genuine win only materializes when a real period — Quarterly's 4, Monthly's 12 — has enough
cycles in the window to fill several rows, which is exactly the regime split I will forecast against.

Once the series is a 2D grid I have a choice of what operator reads it, and the obvious rival to a 2D
convolution is 2D attention over the grid cells. I walk it and reject it on this data. Attention over
the `(rows × cols)` cells would relate every phase to every other phase globally, but I already know the
dependencies I want are *local* on this layout by construction — the intraperiod neighbor is one column
over, the interperiod neighbor is one row down — so a small 2D convolution reads exactly the adjacencies
that carry the signal, while attention would spend a quadratic budget rediscovering that locality from a
handful of cells (Quarterly's grid is only `6 × 4 = 24` cells). The reshape did the hard part; a
convolution is the operator that matches it. So for each discovered period I reshape (zero-padding the
time axis up to a multiple of `p` first), then process the 2D tensor with a parameter-efficient
inception block — several 2D kernels of increasing size in parallel, averaged — so the block reads intra-
and interperiod variation at multiple 2D scales at once. Multiple kernel sizes matter because a single
`3×3` would fix one intra/inter receptive field, whereas `num_kernels = 6` parallel kernels of sizes
`1,3,5,7,9,11` let the block see a single phase, a short intra-cycle run, and a multi-cycle interperiod
span all at once and average their responses — the 2D analogue of reading several resolutions, but within
one period's grid rather than across a pooling ladder. Let me trace the reshape and its padding on a real case so the axes are unambiguous. On Monthly
the extended length is 54; if the top period is 12, then `54 % 12 = 6 ≠ 0`, so I pad the time axis up to
`⌈54/12⌉·12 = 60` with zeros, reshape `[B, 60, N]` to `[B, 60/12, 12, N] = [B, 5, 12, N]`, and permute to
`[B, N, 5, 12]` — five stacked 12-step cycles, columns intraperiod, rows interperiod — run the shared
inception over the `(5, 12)` grid, permute and flatten back to `[B, 60, N]`, and truncate to the first 54.
On Quarterly with period 4 and length 24, `24 % 4 = 0`, no padding, reshape to `[B, 6, 4, N]`. The
padding-then-truncate keeps every period's output the same length so I can stack them. Share one
inception across all `k` periods, so model size is independent of `k`: the same convolution weights are
reused for every one of the top-5 periods' grids, which is what lets me take `top_k = 5` on a short
window without the parameter count scaling with `k` — the cost of an extra period is one more reshape and
one more pass through the shared conv, not more weights. Reshape back to 1D and truncate. Fuse the `k`
period-specific representations by a softmax over their amplitudes — amplitude is the confidence of each
period, so a convex combination weighted by confidence is the principled aggregation (the same
confidence-softmax idea the auto-correlation forecasters used, now over genuinely separate 2D
representations rather than 1D rolls). Let me sanity-check that the softmax does the right thing at the
extremes: on Quarterly, where one 4-step period dominates the amplitude spectrum, the softmax collapses
toward a one-hot weight on that period's 2D representation, so the fused output is essentially the clean
period-4 grid's convolution — exactly what I want when there is a single sharp period. On Yearly, where
five weak low-frequency bins have comparable amplitudes, the softmax spreads roughly uniformly and the
fusion averages several near-degenerate grids, which is a mild smoothing — harmless, and the graceful
fallback I argued for. So the amplitude-softmax adapts from "trust the one real period" to "average the
noise" without any regime-specific switch. Stack a couple of these blocks residually with LayerNorm, and wrap
the backbone in the same reversible per-window instance normalization that has helped since PatchTST.
The normalization choice here is worth one beat, because I deliberately picked LayerNorm over the
BatchNorm I insisted on for PatchTST — and the reason is that the two rungs normalize different things.
In PatchTST the tokens were *raw* patches whose within-token statistics an outlier step could corrupt,
which is what made BatchNorm's across-batch dilution matter. Here every position is already a learned
`d_model`-dimensional embedding of a window that was instance-normalized *before* embedding, so the
raw-outlier problem is handled upstream, and normalizing across the learned feature axis with LayerNorm
is the simpler, standard-for-this-backbone stable choice. Different token, different normalization — not
a default I carried over unexamined.

One thing to be honest about on cost, since every rung so far has had a protocol-mismatch tax: the
inception block is the heaviest backbone on the whole ladder. Six parallel 2D kernels of sizes up to
`11×11` at width `512`, and two such inceptions per TimesBlock (`d_model → d_ff`, `d_ff → d_model`), make
this far more parameter-heavy than the mixer's channel FFN or the patch encoder — the shared-across-
periods design is the *only* thing keeping the count from also multiplying by `top_k = 5`. On M4's short
windows that is a lot of convolutional capacity aimed at grids as small as `6 × 4`, so the same
"capacity the data may not use" caution that dogged PatchTST applies here too; reversible instance
normalization, the residual LayerNorm structure, and 10-epoch early stopping are again the regularizers I
am relying on to keep the wide inception from overfitting the handful of cells a short-window grid
provides.

For forecasting specifically — and this is the part that differs from a reconstruction backbone — the
horizon has to be *created* before the period machinery runs, because the future steps do not exist in
the input window. So after instance-normalizing and embedding the length-`seq_len` window to `d_model`
(a value embedding plus positional embedding; the harness passes no time marks, so the temporal-feature
branch contributes nothing and the embedding must accept `x_mark=None`), apply one linear map along time
that extends the sequence from `seq_len` to `seq_len + pred_len`. Let me trace that shape, because it is
the load-bearing move: `enc_out` is `[B, seq_len, d_model]`; permute to `[B, d_model, seq_len]`; apply
`predict_linear = Linear(seq_len → seq_len + pred_len)` to get `[B, d_model, seq_len + pred_len]`;
permute back to `[B, seq_len + pred_len, d_model]`. Now the TimesBlocks operate on the full extended
sequence, discovering periods over `seq_len + pred_len` and convolving across it, so the forecast region
is filled by the same period-aware 2D convolution that models the observed region — the future steps are
not left blank for the period machinery, they are *predicted* by it. A final linear projection maps
`d_model` back to the single output channel, and the output is denormalized and truncated to the last
`pred_len` steps. This is why the FFT lengths above are 54/24/18 rather than the raw 36/16/12: the period
discovery sees the horizon it must fill.

It is worth pausing on why I extend *first* rather than the more familiar alternative, because the order
is the whole difference between using the period machinery for forecasting and wasting it. The
alternative is to run the TimesBlocks on the length-`seq_len` observed window alone — a reconstruction
pass — and then attach a flatten head `seq_len·d_model → pred_len` to read off the horizon, exactly the
PatchTST-style head. But then the forecast region is produced by a generic linear head that never touches
the 2D period convolution; the interperiod structure I went to all this trouble to expose would model
only the *observed* cycles and contribute nothing to the future ones. By extending the temporal axis to
`seq_len + pred_len` before the blocks run, the reshape stacks the horizon's steps into the same
period-grid as the history, so the interperiod convolution literally places "the same phase, next cycle"
into the forecast region — the next quarter is predicted from the same-quarter column, which is the
entire point of making interperiod structure local. So the extend-then-convolve order is not incidental;
it is what routes the method's strength into the horizon rather than into a reconstruction the head then
ignores.

I should be clear-eyed about the harness protocol and where it strains this rung, because it has caught
every rung so far. The fixed Custom settings give `d_model = 512`, `d_ff = 512`, `e_layers = 2`, and
the unset `top_k`/`num_kernels` default to 5 and 6 — a much wider model than TimesNet's own short-term
script (`d_model = 32`). On the very short M4 windows the FFT has few frequency bins to work with:
Yearly's window is 12 steps (extended to 18, ten bins), so the spectrum is coarse and the top-5 periods
include near-meaningless short ones; the 2D reshape into a few-row grid is thin. Taking `top_k = 5` on a
ten-bin Yearly spectrum means I am keeping half the available frequencies, most of them noise, and only
the amplitude-softmax's down-weighting of the low-amplitude ones keeps them from doing harm. So I expect
the period machinery to deliver most where the window holds *several clean cycles of a real period* and
least where there is essentially no period to find.

That makes the falsifiable expectation against TimeMixer's numbers pointed and *non-uniform* — this is
not a rung I expect to win everywhere. Where I am most confident: **Quarterly**, the regime that stalled
for TimeMixer (10.22 → 10.21, essentially flat), because the FFT should lock onto the sharp 4-step period
and make the year-over-year (interperiod) relationship local for the first time — the six-row-by-four-
column grid is genuinely informative where the pooling ladder was not, so I expect to clear TimeMixer's
10.21 there with the clearest margin of the three. **Monthly** I expect to be roughly a tie: the 12-step
seasonality is real and the 36-step window holds three cycles, so the 2D reshape is informative, but
TimeMixer's decomposition-mixing already handles the trend+season superposition well, so the two strong
period-aware models should land close to each other near 12.80. **Yearly** is the regime where I expect
*not* to beat TimeMixer: there is no meaningful period in a trend-dominated six-step horizon, the FFT has
ten coarse bins and nothing real to find, and the top-down trend mixing TimeMixer was built for is the
better tool — I would not be surprised to come in slightly *above* TimeMixer's 13.38 on Yearly. So the
bar this rung must clear to be the strongest baseline is a *mean* win driven by Quarterly: if the
Quarterly gain outweighs a possible Yearly loss and Monthly is a wash, the period-aware 2D model edges
ahead overall — and that would confirm that the binding constraint on this task was making interperiod
structure local, exactly the thing the pooling ladder could not do.
