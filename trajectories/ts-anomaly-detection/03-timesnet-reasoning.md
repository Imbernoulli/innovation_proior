DLinear's numbers confirmed the diagnosis I had and exposed the next one. Its mean F1 (0.8194) did clear
the patch backbone (0.8135), and it did so exactly where I predicted: MSL jumped from 0.7904 to 0.8187,
recall from 0.7130 to 0.7530 — the under-flexible linear reconstructor left the MSL anomalies as cleaner
residual instead of partly reconstructing them. PSM ticked up too (0.9663 vs 0.9617). But SMAP went the
*other* way — F1 fell from 0.6883 to 0.6733, recall from 0.5557 to 0.5383, the worst recall on the
board. So the linear reconstructor helped on the datasets whose normal structure is a steady
trend-plus-periodicity and *hurt* on the one whose periodicity is least stable. That is the tell. A
single linear map is one flat, position-indexed function of the window: weight `W[i,j]` is fixed for
"reconstruct step `i` from step `j`" regardless of *which* period the current window actually carries.
On PSM and MSL the dominant period is steady enough that one fixed map works; on SMAP the satellite
telemetry shifts its rhythm window to window, so a single fixed lag pattern is wrong for most windows,
the reconstruction smears the periodic structure, and anomalies stop standing out. The linear map has no
handle on *which* periodicities a given window carries. That is the wall I have to break, and it means
the next backbone must discover and adapt to the periods present in each window, not bake one in.

Let me quantify the split in DLinear's results, because the direction of each move is the whole
diagnosis. MSL rose `0.8187 − 0.7904 = +0.0283` in F1 and `0.7530 − 0.7130 = +0.0400` in recall, PSM
rose `+0.0046`, but SMAP fell `0.6733 − 0.6883 = −0.0150` in F1 and `0.5383 − 0.5557 = −0.0174` in
recall. Crucially, SMAP's precision barely moved (`0.9040 → 0.8987`), so the regression is entirely
recall — the linear map reconstructed *more* of SMAP's anomalies than the patch backbone did, not fewer.
The under-flexibility that helped MSL hurt SMAP, and that sign flip is the signal: the two datasets
differ in whether one fixed linear readout suits every window. The mean moved from 0.8135 to 0.8194,
`+0.0059`, carried by MSL and dragged by SMAP; if I could merely restore SMAP to its patch-backbone
0.6883 while keeping MSL's and PSM's gains the mean would already be `(0.9663 + 0.8187 + 0.6883)/3 ≈
0.824`, so SMAP is precisely where the next backbone's budget must go.

Let me be precise about what kind of dependency a single point in a normal window actually has, because
vague talk of "periodicity" is what got the linear map stuck. Take a point now. It depends on its
immediate neighbors — the step before and after — which is the short-term movement inside the current
cycle. But it also depends on the same phase one cycle ago and one cycle ahead: the same hour yesterday
and tomorrow, the same phase of adjacent periods. Those are two genuinely different dependencies. One is
the local shape *within* a period; the other is how the corresponding phase *changes from period to
period*. Call them intraperiod variation (within a cycle) and interperiod variation (across cycles, same
phase). The first is short-range, the second long-range. Real telemetry has both, and worse, several
periods at once — daily and weekly on a server, several rhythms in satellite telemetry. The linear map
folds all of this into one position-indexed matrix and cannot pull the within-cycle and across-cycle
parts apart, let alone adapt the period series by series. The patch Transformer at least let attention
relate patches, but it too operated on the 1D layout and never separated the two variations.

Here is the structural problem, and I want to feel exactly where it bites, because it is the same wall
that stopped both prior rungs. The data is a 1D sequence indexed by `t`. Adjacency along that axis gives
me intraperiod neighbors for free — `t−1` and `t+1` sit right next to `t`. But the interperiod neighbor,
the same phase one period back, is at `t−p` where `p` is the period length, and in the 1D layout that
point is `p` steps away with the entire previous cycle crammed in between. A linear map can in principle
place weight at lag `p`, but only a *fixed* `p`; a convolution along time with any sane kernel never sees
`t` and `t−p` together; attention can reach `t−p` but treats it as just another of `L` points, with no
notion that it is the *same phase*. So the 1D representation presents one of the two variations cleanly
(intra) and effectively hides the other (inter). Whatever I build on the raw 1D window inherits this. The
linear map's SMAP failure is exactly this bottleneck made visible: when `p` shifts window to window, a
fixed-`p` map is wrong, and there is no 1D backbone that recovers the interperiod relation cleanly while
adapting `p`.

So the question sharpens: I have two kinds of locality — within a period and across periods — and a 1D
layout that can only express one of them as locality. What if I change the *layout* so both become
locality? Stare at the 1D window and a period `p`. If I chop the window into consecutive blocks of length
`p` and stack those blocks as the rows of a matrix, then walking along a row is walking through one cycle
(intraperiod, the within-period shape), and walking down a column is walking through the same phase across
successive cycles (interperiod). Both dependencies are now *adjacencies* in a 2D array — one along
columns within a period, one along rows across periods. The thing that was `p` apart in 1D is now one
step apart along the cross-period axis. That is the move: reshape the 1D window, for a given period `p`,
into a 2D tensor whose two axes are exactly intraperiod and interperiod, and both variations I care about
become local in 2D.

And the instant the window is a 2D grid with two meaningful local axes, I get the entire mature toolbox
of 2D convolution for free. A 2D kernel sliding over this grid sees, in one receptive field, a few
adjacent steps within the cycle *and* the same window of phases in a couple of neighboring cycles — it
models intra- and interperiod variation simultaneously, which the 1D layout could never do with one
operator. The bottleneck was never that the variation is hard; it is that the 1D layout was the wrong
space to look at it in. Lift it to 2D and ordinary vision backbones can chew on temporal variation. This
also directly addresses the dlinear failure mode: instead of a single fixed lag pattern, the 2D conv
reads the periodic structure *as a grid*, and if I pick the right `p` per window the grid is correctly
phase-aligned.

But which `p`? Real windows have several periods at once, unknown a priori, differing by dataset and even
by window — precisely the variability that broke the fixed linear map on SMAP. I need to *discover* the
dominant periods from each window, which is where the spectrum comes in. The amplitude of the Fourier
transform at frequency `j` measures how strongly a periodic basis function of period `T/j` is present in
the window; a strong period shows up as a tall amplitude peak. So compute the FFT of the window, take
amplitudes, and the tallest peaks give the dominant frequencies, hence the dominant periods `p = T/f`.
For a window of length `T` with `C` channels, take the real FFT along time, its amplitude, and average
the amplitude over batch and channels to get one length-`T/2+1` profile — I want the *same* reshape
applied to every channel so they stay aligned in the 2D grid, so a single set of periods for the whole
window is exactly right. This matters more here than univariately: the streams have 25–55 channels and an
anomaly may show in one channel while the periods are best estimated by pooling spectral energy across
all of them.

A few details I must get right. The DC term (frequency 0) is just the window's mean energy, corresponds
to no real period, and is usually huge — zero it out before picking peaks or it dominates everything. The
spectrum of a real signal is conjugate-symmetric so I only consider frequencies up to `T/2`, which the
real FFT gives directly. And I do not want all peaks — the spectrum is sparse and its high-frequency tail
is mostly noise — so take only the top-`k` amplitudes, getting the `k` most significant frequencies and
their periods `p_i = T // f_i`, keeping the `k` amplitudes around because I have a hunch they will serve
as importance weights. One function, `FFT_for_Period`, maps the window to its `k` dominant periods and
their amplitudes — this is exactly the per-window period discovery the linear map lacked.

Now the reshape, carefully, because the dimensions must come out exactly. For period `p_i` and frequency
`f_i` I want a grid with `f_i` rows and `p_i` columns — roughly `f_i` cycles of length `p_i`. But `T` is
generally not `f_i × p_i`, so pad the window with zeros along time up to the next multiple of `p_i`,
reshape the padded length into `(length // p_i) × p_i`, process, then truncate back to `T`. The padding
is harmless filler the truncation discards. So for each of the `k` periods I get a 2D tensor of shape
(num-periods) × (period-length) per channel — `k` different 2D views of the same window, each exposing a
different periodicity's intra/inter structure.

Let me make the reshape concrete with plausible periods so the padding math is nailed down and the
adjacency claim is checked, not just asserted. Say the top-3 frequencies of a window come out as `f = 4,
7, 10`, giving periods `p = 100 // 4, 100 // 7, 100 // 10 = 25, 14, 10`. For `p = 25`, `100` is already
`4 · 25`, so the grid is `4 × 25` with no padding. For `p = 14`, `100 % 14 = 2 ≠ 0`, so I pad to `(100
// 14 + 1) · 14 = 8 · 14 = 112`, reshape to `8 × 14`, and truncate the filler back off after the conv.
For `p = 10`, the grid is `10 × 10`. And the interperiod-adjacency claim is checkable: with `p = 25`,
the point at `t = 30` lands at row `30 // 25 = 1`, column `30 % 25 = 5`; the same phase one cycle
earlier, `t = 5`, lands at row `0`, column `5` — the cell directly above it. What was 25 steps apart in
the 1D window is one step apart along the grid's row axis, so a `3 × 3` conv kernel centered on `t = 30`
now spans its intraperiod neighbors (columns 4 and 6, same row) *and* its interperiod neighbor (column
5, row 0) in one receptive field. The lift-to-2D does exactly what it claimed, and it is the same phase
`t = 5` that a 1D convolution over the raw window would have to reach across 25 intervening steps to
see.

What runs over each 2D tensor? A 2D conv, obviously — but at what kernel size? The reshape fixes one
period, yet the actual variation has structure at several scales: a fine wiggle over two or three steps,
a broader hump over a third of the cycle, a slow tilt across several cycles. A single fixed kernel
commits to one scale. I want several at once — exactly what the Inception block was built for: run
several kernel sizes in parallel inside one block and combine them, multi-scale by construction and
parameter-efficient compared to one giant kernel. The harness exposes `Inception_Block_V1` (a set of 2D
convs with kernel sizes `2i+1`, each padded to preserve spatial size, run in parallel and mean-combined).
I make the block a small two-layer affair with a nonlinearity — inception expanding `d_model → d_ff`, a
GELU, inception contracting `d_ff → d_model` — a channel-bottleneck feedforward in 2D with the capacity
to transform the 2D features rather than just smear them.

One decision that matters for efficiency and cleanliness: I have `k` reshaped tensors, one per period. Do
I give each its own inception weights? If so, model size grows with `k`, and `k` is a knob I want to tune
freely. So share one inception block across all `k` periods — each period gets a different *reshape*
(different grid geometry), but the same conv weights process all of them. Model size is then invariant
to `k`. That is conceptually right too: the inception learns "how to read 2D temporal variation," which
should not depend on which period produced the grid.

The per-dataset capacity settings are not arbitrary, and reading them against the earlier failures is
where the whole ladder pays off. An inception block with `num_kernels = 6` holds six parallel 2D convs
of kernel sizes `1, 3, 5, 7, 9, 11`, so a `d → d` inception costs `d^2 · (1 + 9 + 25 + 49 + 81 + 121) =
286 d^2` weights, and a TimesBlock stacks two of them (expand `d_model → d_ff`, contract back). On PSM
with `d_model = d_ff = 64` that is `2 · 286 · 64^2 ≈ 2.34M` per block over `e_layers = 2`; on SMAP with
`d = 128` it is `2 · 286 · 128^2 ≈ 9.4M` per block over `e_layers = 3` — the most capacity of the three;
on MSL it drops all the way to `d = 8`, `2 · 286 · 64 ≈ 37K` per block over a *single* layer. That
ordering is the diagnosis turned into a budget. MSL's period is steady enough that DLinear's near-zero
capacity linear map already reached 0.8187, so I give it the thinnest period-aware model that can only
refine, deliberately avoiding the over-flexibility that sank the patch backbone's MSL recall to 0.7130.
SMAP, whose shifting rhythm the fixed map could not track and where recall bottomed at 0.5383, gets the
deepest, widest stack to actually model per-window period structure. PSM sits in between. The knob that
broke the earlier rungs — too much flexibility on steady data, too little adaptivity on shifting data —
is here set per dataset in the direction each one needs, which is only possible because the period
discovery makes the flexibility *adaptive* rather than merely large.

After the shared inception transforms each 2D tensor, reshape each back to 1D and truncate to `T`, giving
`k` candidate 1D representations of the window. Now fuse them. A plain sum throws away that some periods
are far more present in this window than others — and "more present" is exactly the per-window
adaptivity the fixed linear map lacked. I kept the amplitudes, and an amplitude *is* how strongly a
period is expressed. So weight the `k` representations by their amplitudes, pushed through a softmax over
the `k` periods to get convex weights, then take the weighted sum. A window dominated by one cycle puts
most weight on that period's representation; a window with two strong rhythms splits the weight; a SMAP
window whose rhythm shifts gets a *different* convex combination than the previous window — the adaptivity
that a single fixed map could never give. The alternatives are worse: a direct sum ignores importance,
and raw amplitudes are unnormalized and scale-sensitive, so the softmax's normalization is the principled
choice. I should check the softmax does what I want at its extremes rather than trust it. If one period
dominates — amplitudes like `[10, 1, 1]` on the three peaks — then softmax puts weight `e^{10} / (e^{10}
+ 2e^{1}) ≈ 22026 / 22031 ≈ 0.9998` on that period and near-zero elsewhere, so the block essentially
reconstructs through the single correct grid: the sharp, confident case. If two rhythms are comparably
present, say `[5, 5, 1]`, the weights split roughly `0.49, 0.49, 0.02`, blending the two grids. And a
SMAP window whose dominant rhythm has shifted since the last window produces a *different* amplitude
profile and therefore a different convex combination — precisely the per-window adaptivity a single
fixed `W` could never express, now realized as data-dependent mixing weights recomputed from each
window's own spectrum. That is the exact degree of freedom whose absence dropped SMAP's recall to
0.5383.

So one block does: discover periods → reshape to `k` 2D tensors → shared inception on each → reshape back
→ amplitude-softmax aggregate. Call it a TimesBlock. I stack several so the representation refines with
depth, with a residual `X^l = TimesBlock(X^{l-1}) + X^{l-1}` and a layer norm to keep activations
well-scaled — the residual also means each block learns only the *correction* to the variation
representation, which composes across layers (early blocks catch coarse structure, later ones refine).
The architecture is an embedding lifting the raw window into a `d_model` feature sequence, a stack of
residual TimesBlocks, and a projection back out.

Now the reconstruction path, and here anomaly detection is cleaner than forecasting in one way that
exactly matches the two prior rungs: I am not predicting any future steps, I am reconstructing the window
I was given. The TimesBlock maps a length-`T` feature sequence to a length-`T` feature sequence — same
length in, same length out — which is exactly a reconstruction map. So `pred_len` is 0, the pad/reshape
inside the block uses just `seq_len`, there is no horizon to invent and no decoder bolted on. I embed the
window, run the residual TimesBlocks over its own length, and project every position back to the input
channels; the harness reads the score off the squared difference between this reconstruction and the
input. Trace it once to be sure the reconstruction is same-length. A PSM batch `[32, 100, 25]` is
instance-normalized, the value-plus-position embedding lifts it to `[32, 100, 64]`, each residual
TimesBlock maps `[32, 100, 64] → [32, 100, 64]` (the FFT reshape pads internally and truncates back to
100, so length is conserved across the block), the LayerNorm over the 64-wide feature leaves it `[32,
100, 64]`, and the final `Linear(64, 25)` projects every one of the 100 positions back to the 25
channels, giving `[32, 100, 25]` — identical to the input, so the per-point squared error is defined at
all 100 steps. `pred_len = 0` is exactly what keeps the internal pad length equal to `seq_len` alone;
there is no horizon appended, so nothing about a forecast leaks into the reconstruction and every
position the harness scores is a genuine reconstruction of an observed step.

One real problem I must address — and here I do something DLinear deliberately did not. The channels and
datasets span very different magnitudes and the level wanders, so I wrap the backbone in per-window
instance normalization: subtract the window's temporal mean, divide by its std (biased variance plus
`1e-5`), detach both as constants, run the model, then de-normalize the output with the same mean and std.
This is what lets one shared backbone reconstruct streams spanning orders of magnitude. (DLinear skipped
this because its trend linear absorbed the level directly; a deep conv backbone cannot, so the explicit
normalization comes back — the same reversible-instance-norm the patch backbone used, returning here for
the same reason.) There is a subtlety for anomaly detection: per-window centering could partly absorb a
pure level-shift anomaly — but the data is already globally Z-scored, windows are short, and the anomalies
that matter are violations of the *variation* (the cycle breaks, the shape goes wrong), which survive
per-window centering precisely because they are deviations from the periodic structure the model
reconstructs. The embedding is the value embedding (a 1D conv lifting channels to `d_model`) plus a fixed
positional embedding, no time marks for anomaly detection — generic harness, not the contribution.

So the delta from DLinear is exactly aimed at its SMAP failure: where the linear map applied one fixed,
position-indexed lag pattern to every window — fine on steady-period PSM/MSL, wrong on shifting-period
SMAP — I now discover each window's dominant periods by FFT, reshape so both intra- and interperiod
variation are 2D-local, read them with a shared multi-scale inception, and fuse the `k` periods by their
amplitude confidence so the representation *adapts* its period structure window to window.

Here is what I expect against DLinear's measured numbers, falsifiably. On SMAP — DLinear's regression
(0.6733, recall 0.5383) — the per-window period adaptivity is the whole point, so I expect TimesNet to
*recover and exceed* it; if SMAP does not move up, the period-discovery story is wrong and I would look
for a different cause of its un-separable anomalies. On PSM, already near saturation (0.9663), I expect a
small lift from the multi-scale 2D conv catching structure the flat linear map smears, into the
mid-0.97s. On MSL, where DLinear already did well (0.8187), I expect TimesNet to roughly match or
slightly edge it — the gain there is smaller because a steady period already suited the linear map. The
bar to clear is DLinear's mean F1 of 0.8194; I expect TimesNet to clear it primarily by fixing SMAP and
nudging PSM, which is the dataset-by-dataset shape that would confirm the period-adaptivity diagnosis
rather than a generic capacity win.
