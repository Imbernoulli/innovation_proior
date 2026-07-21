DLinear's numbers confirmed the diagnosis and exposed the next one. Its mean F1 (0.8194) cleared the patch
backbone (0.8135), exactly where I predicted: MSL jumped from 0.7904 to 0.8187, recall from 0.7130 to
0.7530 — the under-flexible linear map left the MSL anomalies as cleaner residual. PSM ticked up too
(0.9663 vs 0.9617). But SMAP went the *other* way — F1 fell from 0.6883 to 0.6733, recall from 0.5557 to
0.5383, the worst on the board, and its precision barely moved (`0.9040→0.8987`), so the regression is
entirely recall: the linear map reconstructed *more* of SMAP's anomalies, not fewer. The linear
reconstructor helped where the normal structure is a steady trend-plus-periodicity and *hurt* where the
periodicity is least stable. That sign flip is the signal. A single linear map is one flat,
position-indexed function: weight `W[i,j]` for "reconstruct step `i` from step `j`" is fixed regardless of
*which* period the current window carries. On PSM and MSL the dominant period is steady enough that one
fixed map works; SMAP's satellite telemetry shifts rhythm window to window, so a single fixed lag pattern
is wrong for most windows, the reconstruction smears the periodic structure, and anomalies stop standing
out. If I could merely restore SMAP to its patch-backbone 0.6883 while keeping the MSL and PSM gains the
mean would already be `(0.9663 + 0.8187 + 0.6883)/3 ≈ 0.824`, so SMAP is precisely where the next
backbone's budget must go. The wall to break: the next backbone must discover and adapt to the periods
present in each window, not bake one in.

Be precise about what dependency a point in a normal window actually has, because vague talk of
"periodicity" is what stuck the linear map. A point depends on its immediate neighbors — the step before
and after, the short-term movement inside the current cycle. But it also depends on the same phase one
cycle ago and one cycle ahead — the same hour yesterday and tomorrow. Those are two genuinely different
dependencies: the local shape *within* a period, and how the corresponding phase *changes from period to
period*. Call them intraperiod (within a cycle, short-range) and interperiod (across cycles, same phase,
long-range) variation. Real telemetry has both, and several periods at once — daily and weekly on a
server, several rhythms in satellite telemetry. The linear map folds all of this into one position-indexed
matrix and cannot pull the within-cycle and across-cycle parts apart, let alone adapt the period per
series. The patch Transformer at least let attention relate patches, but it too operated on the 1D layout
and never separated the two variations.

Here is the structural problem, the same wall that stopped both prior rungs. The data is a 1D sequence
indexed by `t`. Adjacency along that axis gives intraperiod neighbors for free — `t−1` and `t+1` sit next
to `t`. But the interperiod neighbor, the same phase one period back, is at `t−p` with the entire previous
cycle crammed in between. A linear map can place weight at lag `p`, but only a *fixed* `p`; a convolution
along time with any sane kernel never sees `t` and `t−p` together; attention can reach `t−p` but treats it
as just another of `L` points, with no notion that it is the *same phase*. So the 1D layout presents one
variation cleanly (intra) and effectively hides the other (inter), and whatever I build on the raw 1D
window inherits this. The linear map's SMAP failure is exactly this bottleneck made visible: when `p`
shifts window to window, a fixed-`p` map is wrong, and no 1D backbone recovers the interperiod relation
cleanly while adapting `p`.

So the question sharpens: two kinds of locality — within a period and across periods — and a 1D layout that
expresses only one as locality. What if I change the *layout* so both become locality? Chop the window into
consecutive blocks of length `p` and stack them as the rows of a matrix: walking along a row walks through
one cycle (intraperiod), walking down a column walks through the same phase across successive cycles
(interperiod). Both dependencies are now *adjacencies* in a 2D array — the thing that was `p` apart in 1D
is now one step apart along the cross-period axis. That is the move: reshape the 1D window, for a given
period `p`, into a 2D tensor whose two axes are exactly intraperiod and interperiod.

The instant the window is a 2D grid with two meaningful local axes, the mature toolbox of 2D convolution is
free. A 2D kernel sees, in one receptive field, a few adjacent steps within the cycle *and* the same window
of phases in a couple of neighboring cycles — modeling intra- and interperiod variation simultaneously,
which the 1D layout could never do with one operator. The bottleneck was never that the variation is hard;
the 1D layout was the wrong space to look at it in. This directly addresses the DLinear failure: instead of
a single fixed lag pattern, the 2D conv reads the periodic structure *as a grid*, correctly phase-aligned
when `p` is right.

But which `p`? Real windows have several periods at once, unknown a priori, differing by dataset and window
— precisely the variability that broke the fixed map on SMAP. So *discover* the dominant periods from each
window via the spectrum. The FFT amplitude at frequency `j` measures how strongly a periodic basis function
of period `T/j` is present; a strong period is a tall amplitude peak. Take the real FFT along time, its
amplitude, average over batch and channels to one length-`T/2+1` profile — I want the *same* reshape
applied to every channel so they stay aligned in the grid, so a single set of periods for the whole window
is right, which matters more here because an anomaly may show in one channel while the periods are best
estimated by pooling spectral energy across all 25–55. Details that must be right: the DC term (frequency
0) is the window's mean energy, no real period, and usually huge — zero it before picking peaks. The real
FFT already gives only frequencies up to `T/2`. And the spectrum is sparse with a noisy high tail, so take
only the top-`k` amplitudes, getting periods `p_i = T // f_i` and keeping the `k` amplitudes as importance
weights for later. One function, `FFT_for_Period`, does the per-window period discovery the linear map
lacked.

Now the reshape. For period `p_i` I want a grid with `T//p_i` rows and `p_i` columns. `T` is generally not
a multiple of `p_i`, so pad with zeros along time up to the next multiple, reshape to `(length//p_i) ×
p_i`, process, then truncate back to `T` — the padding is harmless filler the truncation discards.
Concretely, with `p=25` on `L=100`, `t=30` lands at row `1`, column `5`, and the same phase one cycle
earlier, `t=5`, at row `0`, column `5` — the cell directly above it. What was 25 steps apart in 1D is one
step apart along the grid's row axis, so a `3×3` kernel centered on `t=30` spans its intraperiod neighbors
(columns 4 and 6) and its interperiod neighbor (row 0, column 5) in one receptive field — the same phase a
1D convolution would have to reach across 25 steps to see.

What runs over each grid? A 2D conv — but at what kernel size? The reshape fixes one period, yet the
variation has structure at several scales: a fine wiggle over two or three steps, a broader hump over a
third of the cycle, a slow tilt across several cycles. A single kernel commits to one scale, so I want
several at once — exactly the Inception block, which runs several kernel sizes in parallel and combines
them, multi-scale by construction and parameter-efficient against one giant kernel. The harness exposes
`Inception_Block_V1`; I make the TimesBlock a two-layer affair — inception expanding `d_model → d_ff`,
GELU, inception contracting back — a 2D channel-bottleneck feedforward with capacity to transform the
features, not just smear them. I have `k` reshaped tensors, one per period; giving each its own inception
would grow model size with `k`, a knob I want free, so share one inception across all `k` — each period
gets a different *reshape*, the same conv weights. That is conceptually right: the inception learns "how to
read 2D temporal variation," which should not depend on which period produced the grid.

The per-dataset capacity settings turn the diagnosis into a budget. MSL gets the thinnest stack (`d_model =
d_ff = 8`, a single layer): its period is steady enough that DLinear's near-zero-capacity map already
reached 0.8187, so I want only a period-aware model that can refine, deliberately avoiding the
over-flexibility that sank the patch backbone's MSL recall to 0.7130. SMAP, whose shifting rhythm the fixed
map could not track and whose recall bottomed at 0.5383, gets the deepest, widest stack (`d_model = 128`,
three layers) to actually model per-window period structure. PSM sits between (`d_model = 64`, two layers).
The knob that broke the earlier rungs — too much flexibility on steady data, too little adaptivity on
shifting data — is here set per dataset in the direction each needs, which is only possible because period
discovery makes the flexibility *adaptive* rather than merely large.

After the shared inception transforms each grid, reshape back to 1D and truncate to `T`, giving `k`
candidate representations of the window. Fuse them by their amplitudes — an amplitude *is* how strongly a
period is expressed — pushed through a softmax over the `k` periods to convex weights, then a weighted sum.
A window dominated by one cycle puts most weight on that period's representation; a window with two strong
rhythms splits it; a SMAP window whose rhythm has shifted since the last window produces a different
amplitude profile and therefore a different convex combination — the per-window adaptivity a single fixed
`W` could never express, realized as data-dependent mixing weights recomputed from each window's own
spectrum. That is the exact degree of freedom whose absence dropped SMAP's recall. A direct sum would
ignore importance and raw amplitudes are unnormalized and scale-sensitive, so the softmax normalization is
the principled choice.

One block does: discover periods → reshape to `k` grids → shared inception → reshape back → amplitude-
softmax aggregate. Call it a TimesBlock, and stack several with a residual `X^l = TimesBlock(X^{l−1}) +
X^{l−1}` and a LayerNorm, so each block learns only the *correction* to the variation representation and
composes across depth (early blocks catch coarse structure, later ones refine). The architecture is an
embedding lifting the raw window to a `d_model` feature sequence, a stack of residual TimesBlocks, and a
projection back out.

The reconstruction path matches the two prior rungs: I reconstruct the window, not a horizon. The
TimesBlock maps length-`T` to length-`T`, so `pred_len` is 0, the internal pad/reshape uses just `seq_len`,
and there is no decoder — I embed the window, run the residual blocks over its own length, and the final
`Linear(d_model, c_out)` projects every position back to the input channels, giving a same-length
reconstruction the harness scores at all 100 steps. Setting `pred_len = 0` is what keeps the internal pad
length equal to `seq_len` alone, so nothing about a forecast leaks into the reconstruction.

One thing I do that DLinear deliberately did not: wrap the backbone in per-window instance normalization —
subtract the window's mean, divide by its std (biased variance plus `1e-5`), detach both, run, then
de-normalize with the same statistics. DLinear skipped this because its trend linear absorbed the level
directly; a deep conv backbone cannot, so the reversible normalization the patch backbone used returns here
for the same reason. The level-shift subtlety is the one from the first rung — per-window centering only
partly absorbs a pure level-shift anomaly, and the anomalies that matter are variation violations that
survive centering because they are deviations from the periodic structure the model reconstructs — so it
carries no new risk. The embedding is value plus fixed positional, no time marks for anomaly detection.

The delta from DLinear is aimed at its SMAP failure: instead of one fixed, position-indexed lag pattern, I
discover each window's periods by FFT, reshape so both variations are 2D-local, and fuse the periods by
amplitude so the representation adapts window to window. SMAP is the test — period adaptivity is the whole
point of this backbone, so I expect it to recover past DLinear's 0.6733 regression, and if it does not the
period-discovery story is wrong and SMAP's anomalies have some other cause. PSM, near saturation (0.9663),
should lift a little from the multi-scale conv catching structure the flat map smears; MSL, where a steady
period already suited the linear map, should roughly match. Clearing DLinear's 0.8194 mean primarily
through SMAP and PSM, dataset by dataset, is what would confirm period-adaptivity over a generic capacity
win.
