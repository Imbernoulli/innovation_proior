The multi-scale mixer paid off where I bet it would, and reading exactly *where* tells me what is still
unexploited. TimeMixer came in at SMAPE 12.80 on Monthly, 10.21 on Quarterly, 13.38 on Yearly —
beating PatchTST on all three (12.97 / 10.22 / 13.68), with the biggest gains on Monthly (0.166,
1.28%) and Yearly (0.300, 2.20%), the regimes where trend and season live at different resolutions and
top-down trend mixing had real work to do. So the multi-scale thesis held. But the number that tells
me what to build next is the one that *did not* move: Quarterly improved 0.010 — one hundredth of a
SMAPE point, statistically indistinguishable from no change. That is a sharp, localized failure hiding
inside an across-the-board win, and it is diagnostic. Quarterly is the regime with the cleanest single
dominant period (the 4-quarter cycle) and the shortest window after Yearly (16 steps, two cycles), so
one average-pooling step to a coarse 8-step view does not separate much: there is no trend/detail split
across scales when the structure is *one sharp period*. TimeMixer exposes scale by *blind* pooling — it
never asks which periodicity a series carries; it halves the length and hopes trend and season fall
out. On a series whose structure is a single strong period, the informative thing is not "coarse vs
fine," it is "the value one period back," and a pooling ladder represents that only incidentally.

So the move is to discover the period structure *inside the window* and represent the series around it.
Reason about what dependency a forecast point actually has. It depends on its immediate neighbors — the
local shape within the current cycle, short-range, *intraperiod*. It also depends on the same phase one
cycle ago and one cycle ahead — the same quarter last year, the same month in adjacent years — how the
corresponding phase changes from cycle to cycle, long-range, *interperiod*. Those are two genuinely
different dependencies, and real M4 series carry both (plus often more than one period at once). The
structural problem: the data is a 1D sequence indexed by `t`. Adjacency along that axis gives the
intraperiod neighbors for free (`t−1`, `t+1`), but the interperiod neighbor at `t−p` is `p` steps away
with an entire cycle crammed in between. A 1D convolution never sees `t` and `t−p` together; an MLP
over absolute window position (DLinear, TimeMixer's per-scale predictors) has no explicit handle on
*which* period a series carries; attention relates point pairs but over a window that is mostly
ordinary points the similarity is dominated by them. So every tool so far, the pooling ladder included,
inherits the same bottleneck: the 1D layout can present intraperiod variation as locality but hides
interperiod variation. That is exactly why Quarterly stalled — the one-period-back relationship that
*is* the Quarterly signal is never made local.

The fix is to change the layout so both localities appear at once. For a period `p`, chop the series
into consecutive blocks of length `p` and stack them as rows of a 2D array: walking along a row
traverses one cycle (intraperiod shape), walking down a column traverses the same phase across
successive cycles (interperiod change). The point that was `p` apart in 1D is now one step apart along
the cross-period axis, so both dependencies become adjacencies a 2D convolution can read
simultaneously, which a 1D layout and a pooling ladder structurally cannot.

Which `p`? Discover it, do not assume it — even though I know Quarterly's period is 4 and Monthly's 12.
Two reasons against hard-coding. First, the harness runs one `Custom.py` across all three regimes under
fixed hyperparameters; a hard-coded period would need to branch on `pred_len` to pick 4/12/1, and on
Yearly there is no period to pick at all. Second and deeper, real M4 series carry more than one period
and carry it imperfectly — a monthly series has a 12-month cycle but often a weaker 6-month harmonic —
and a data-driven read adapts per series where a constant cannot. So I take the FFT of the window: the
amplitude spectrum says how much energy sits at each frequency, and the peaks are the dominant
periodicities. Average the amplitude over channels, zero the DC term (the window mean, not a period),
and take the top-`k` frequencies by amplitude; each `f` gives a period `p = T/f` and an amplitude that
is its *confidence*. Top-`k`, not all frequencies, because the spectrum of a short real series is
sparse and the small-amplitude bins are noise.

There is a subtlety that interacts with the short windows. The FFT runs over the *extended* sequence
`seq_len + pred_len` (justified below), so the lengths are Monthly 54, Quarterly 24, Yearly 18, and a
real FFT returns `floor(T/2) + 1` non-negative-frequency bins: 28, 13, 10. Quarterly's clean 4-step
period sits at frequency `f = 24/4 = 6`, a bin well inside range, and the reshape gives `24/4 = 6` rows
of 4 columns — six full cycles stacked, a genuinely informative grid. Yearly is the opposite: ten coarse
bins over a trend-dominated 18-step window, where the top peaks are low-frequency bins giving long
periods like 18 or 9 (a 2×9 or near-degenerate grid) — there is simply no real short period to find.
And `p = floor(T/f)` on a short window can collapse distinct frequencies onto the same integer period,
or onto periods so short the 2D grid has many rows of two columns. That is fine: a degenerate
short-period grid lets the convolution act almost like a 1D convolution, so the model gracefully falls
back to local smoothing where there is no real long period. The genuine win only materializes when a
real period — Quarterly's 4, Monthly's 12 — fills several rows.

What operator reads the grid? The obvious rival to a 2D convolution is 2D attention over the cells, and
I reject it: the dependencies I want are *local* on this layout by construction — the intraperiod
neighbor one column over, the interperiod neighbor one row down — so a small 2D convolution reads
exactly the adjacencies that carry the signal, while attention would spend a quadratic budget
rediscovering that locality from a handful of cells (Quarterly's grid is only `6 × 4 = 24`). The
reshape did the hard part; convolution matches it. So for each discovered period I reshape
(zero-padding the time axis up to a multiple of `p` first) and process the 2D tensor with a
parameter-efficient inception block — `num_kernels = 6` parallel 2D kernels of sizes `1,3,5,7,9,11`,
averaged — so the block sees a single phase, a short intra-cycle run, and a multi-cycle interperiod
span all at once, the 2D analogue of reading several resolutions but within one period's grid. On
Monthly with top period 12 and length 54, `54 % 12 = 6`, so I pad up to 60, reshape `[B, 60, N]` to
`[B, 5, 12, N]` and permute to `[B, N, 5, 12]` — five stacked 12-step cycles — convolve, flatten back,
and truncate to 54. On Quarterly with period 4 and length 24, no padding, reshape to `[B, 6, 4, N]`.
The same inception weights are shared across all `k` periods, so model size is independent of `k` and I
can take `top_k = 5` on a short window without the parameter count scaling — an extra period costs one
more reshape and pass, not more weights. Fuse the `k` representations by a softmax over their
amplitudes: amplitude is each period's confidence, so a confidence-weighted convex combination is the
principled aggregation. At the extremes this adapts on its own — on Quarterly the dominant 4-step
period collapses the softmax toward one-hot on that grid's convolution (exactly what I want for a
single sharp period), while on Yearly five weak comparable bins spread it roughly uniformly and the
fusion averages several near-degenerate grids, a harmless mild smoothing. Stack a couple of these
blocks residually with LayerNorm, wrapped in the reversible per-window instance normalization that has
helped since PatchTST. The normalization choice is worth one beat: I pick LayerNorm here, not the
BatchNorm I insisted on for PatchTST, because the two rungs normalize different things. There the
tokens were *raw* patches an outlier step could corrupt, which is what made BatchNorm's across-batch
dilution matter. Here every position is already a learned `d_model`-dimensional embedding of a window
that was instance-normalized *before* embedding, so the raw-outlier problem is handled upstream, and
LayerNorm over the learned feature axis is the simpler stable choice.

On cost: the inception block is the heaviest backbone on the whole ladder. Six parallel 2D kernels up
to `11×11` at width 512, two inceptions per block (`d_model → d_ff`, `d_ff → d_model`), make it far more
parameter-heavy than the mixer's channel FFN or the patch encoder — the shared-across-periods design is
the only thing keeping the count from also multiplying by `top_k = 5`. On M4's short windows that is a
lot of convolutional capacity aimed at grids as small as `6 × 4`, so the same "capacity the data may not
use" caution applies; reversible instance normalization, the residual LayerNorm structure, and 10-epoch
early stopping are again the regularizers I rely on.

For forecasting specifically — and this differs from a reconstruction backbone — the horizon has to be
*created* before the period machinery runs, because the future steps do not exist in the input window.
So after instance-normalizing and embedding the window to `d_model` (value plus positional embedding;
the harness passes no time marks, so the embedding must accept `x_mark=None`), apply one linear map
along time extending the sequence `seq_len → seq_len + pred_len`: `enc_out` `[B, seq_len, d_model]`,
permute, `predict_linear = Linear(seq_len → seq_len + pred_len)`, permute back. The reason for extending
*first* rather than running the blocks on the observed window and attaching a flatten head is the whole
difference between using the period machinery for forecasting and wasting it. A flatten head would
produce the forecast region with a generic linear map that never touches the 2D period convolution, so
the interperiod structure I went to all this trouble to expose would model only the observed cycles.
Extending first stacks the horizon's steps into the same period grid as the history, so the interperiod
convolution literally places "the same phase, next cycle" into the forecast region — the next quarter
predicted from the same-quarter column, which is the entire point of making interperiod structure local.
A final linear projection maps `d_model → 1`, denormalize, truncate to the last `pred_len` steps. (This
is why the FFT lengths above are 54/24/18 rather than 36/16/12: period discovery sees the horizon it
must fill.)

Where does the protocol strain this model? The fixed settings give `d_model = 512`, `e_layers = 2`, and
the unset `top_k`/`num_kernels` default to 5 and 6 — much wider than TimesNet's own short-term script
(`d_model = 32`). On the very short windows the FFT has few bins: Yearly (extended to 18, ten bins) has
a coarse spectrum whose top-5 periods include near-meaningless short ones, and the 2D reshape into a
few-row grid is thin. Taking `top_k = 5` on a ten-bin Yearly spectrum keeps half the available
frequencies, most of them noise, and only the amplitude-softmax's down-weighting keeps them from doing
harm. So the period machinery should deliver most where the window holds several clean cycles of a real
period and least where there is essentially none.

That makes the expectation against TimeMixer's numbers pointed and *non-uniform* — not a model I expect
to win everywhere. Most confident on **Quarterly**, the regime that stalled for TimeMixer: the FFT
should lock onto the sharp 4-step period and make the year-over-year relationship local for the first
time, the six-by-four grid genuinely informative where the pooling ladder was not, so I expect to clear
10.21 with the clearest margin of the three. **Monthly** I expect roughly a tie near TimeMixer's 12.80:
the 12-step seasonality is real and the 36-step window holds three cycles, but TimeMixer's
decomposition-mixing already handles the trend+season superposition well, so two strong period-aware
models should land close. **Yearly** is where I expect *not* to win: there is no meaningful period in a
trend-dominated six-step horizon, the FFT has ten coarse bins and nothing real to find, and TimeMixer's
top-down trend mixing is the better tool — I would not be surprised to come in slightly above its 13.38.
So the bar to become the strongest baseline is a *mean* win driven by Quarterly: if the Quarterly gain
outweighs a possible Yearly loss and Monthly is a wash, the period-aware model edges ahead overall,
confirming that the binding constraint was making interperiod structure local — the thing the pooling
ladder could not do.
