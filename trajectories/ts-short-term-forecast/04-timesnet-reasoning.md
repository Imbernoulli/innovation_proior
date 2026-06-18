The multi-scale mixer paid off where I bet it would, and reading exactly *where* tells me what is still
unexploited. TimeMixer came in at SMAPE 12.80 on Monthly, 10.21 on Quarterly, 13.38 on Yearly — beating
PatchTST on all three (12.97 / 10.22 / 13.68), and the biggest gains landed on Monthly and especially
Yearly, the regimes where trend and season live at different resolutions and the top-down trend mixing
had real work to do. So the multi-scale thesis held: explicit cross-scale routing of a decomposed
season/trend representation is more structure than a single-resolution patch encoder can express on
short windows, and it put the harness's wide channels to use. But look at Quarterly: TimeMixer's
10.21 was barely below PatchTST's 10.22 — essentially flat. That is the regime with the *cleanest*
single dominant period (the 4-quarter cycle) and the shortest window after Yearly (16 steps, two
cycles), so one average-pooling step to a coarse 8-step view does not separate much; the fixed pooling
ladder is the wrong lens when the structure is *one sharp period* rather than a trend/detail split
across scales. That is the gap. TimeMixer exposes scale by *blind* pooling — it never asks which
periodicity a given series actually carries; it just halves the length and hopes trend and season fall
out. On a series whose structure is a single strong period, the informative thing is not "coarse vs
fine," it is "the value one period back," and a pooling ladder represents that only incidentally.

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

Which `p`? Discover it, do not assume it. Take the FFT of the window, the amplitude spectrum tells me
how much energy sits at each frequency, and the peaks are the dominant periodicities. Average the
amplitude over channels (one channel here), zero the DC term (it is the window mean, not a period),
and take the top-`k` frequencies by amplitude; each frequency `f` gives a period `p = T/f` and an
amplitude that is its *confidence*. Top-`k` (not all frequencies) because the spectrum of a short real
series is sparse and the small-amplitude bins are noise. This is the direct period read that TimeMixer's
pooling only approximated: on Quarterly the FFT should put a sharp peak at the 4-step period and reshape
the window into a 2D grid whose columns are within-quarter shape and whose rows are year-over-year
change — making the one-period-back relationship local for the first time on the ladder.

There is a subtlety in the period read I should not gloss over, because it interacts with the harness's
short windows. The amplitude spectrum of a real series is conjugate-symmetric, so I only look at the
non-negative frequencies up to `T/2`; and the periods come out as `p = floor(T/f)`, which on a short
window can collapse several distinct frequencies onto the same integer period or onto periods so short
(2 or 3 steps) that the 2D reshape has many rows of two columns — a degenerate grid. That is fine: a
degenerate short-period grid simply lets the inception block act almost like a 1D convolution there, so
the model gracefully falls back to local smoothing on regimes where there is no real long period to
find. The genuine win only materializes when a real period — Quarterly's 4, Monthly's 12 — has enough
cycles in the window to fill several rows, which is exactly the regime split I will forecast against.

For each discovered period, reshape (zero-padding the time axis up to a multiple of `p` first), then
process the 2D tensor with a parameter-efficient inception block — several 2D kernels of increasing
size in parallel, averaged — so the block reads intra- and interperiod variation at multiple 2D scales
at once. Share one inception across all `k` periods, so model size is independent of `k`; reshape back
to 1D and truncate. Fuse the `k` period-specific representations by a softmax over their amplitudes —
amplitude is the confidence of each period, so a convex combination weighted by confidence is the
principled aggregation (the same confidence-softmax idea the auto-correlation forecasters used, now over
genuinely separate 2D representations rather than 1D rolls). Stack a couple of these blocks residually
with LayerNorm, and wrap the backbone in the same reversible per-window instance normalization that has
helped since PatchTST.

For forecasting specifically — and this is the part that differs from a reconstruction backbone — the
horizon has to be *created* before the period machinery runs, because the future steps do not exist in
the input window. So after instance-normalizing and embedding the length-`seq_len` window to `d_model`
(a value embedding plus positional embedding; the harness passes no time marks, so the temporal-feature
branch contributes nothing and the embedding must accept `x_mark=None`), apply one linear map along time
that extends the sequence from `seq_len` to `seq_len + pred_len`. The TimesBlocks then operate on the
full extended sequence, discovering periods over `seq_len + pred_len` and convolving across it, so the
forecast region is filled by the same period-aware 2D convolution that models the observed region; a
final linear projection maps `d_model` back to the single output channel, and the output is
denormalized and truncated to the last `pred_len` steps.

I should be clear-eyed about the harness protocol and where it strains this rung, because it has caught
every rung so far. The fixed Custom settings give `d_model = 512`, `d_ff = 512`, `e_layers = 2`, and
the unset `top_k`/`num_kernels` default to 5 and 6 — a much wider model than TimesNet's own short-term
script (`d_model = 32`). On the very short M4 windows the FFT has few frequency bins to work with:
Yearly's window is 12 steps (extended to 18), so the spectrum is coarse and the top-5 periods include
near-meaningless short ones; the 2D reshape into a few-row grid is thin. So I expect the period
machinery to deliver most where the window holds *several clean cycles of a real period* and least where
there is essentially no period to find.

That makes the falsifiable expectation against TimeMixer's numbers pointed and *non-uniform* — this is
not a rung I expect to win everywhere. Where I am most confident: **Quarterly**, the regime that stalled
for TimeMixer, because the FFT should lock onto the sharp 4-step period and make the year-over-year
(interperiod) relationship local for the first time — I expect to clear TimeMixer's 10.21 there with the
clearest margin of the three. **Monthly** I expect to be roughly a tie: the 12-step seasonality is real
and the 36-step window holds three cycles, so the 2D reshape is informative, but TimeMixer's
decomposition-mixing already handles the trend+season superposition well, so the two strong period-aware
models should land close to each other near 12.80. **Yearly** is the regime where I expect *not* to beat
TimeMixer: there is no meaningful period in a trend-dominated six-step horizon, so FFT period-discovery
has little to discover and the top-down trend mixing TimeMixer was built for is the better tool — I
would not be surprised to come in slightly *above* TimeMixer's 13.38 on Yearly. So the bar this rung
must clear to be the strongest baseline is a *mean* win driven by Quarterly: if the Quarterly gain
outweighs a possible Yearly loss and Monthly is a wash, the period-aware 2D model edges ahead overall —
and that would confirm that the binding constraint on this task was making interperiod structure local,
exactly the thing the pooling ladder could not do.
