PatchTST's three numbers confirm the diagnosis I built it on, and one of them confirms it by *failing* in
the exact way I flagged. Handwriting moved from the linear floor's 0.2306 to 0.2541 — a real gain, and the
sharpest test I named: local-shape patch tokens plus a nonlinear encoder do help on the 26-class gesture
problem, so local temporal shape was indeed a bottleneck. FaceDetection went 0.6822 → 0.6853, parity within
noise — exactly as predicted, because channel-independence forbids in-encoder cross-channel modeling on the
one dataset whose entire signal is cross-channel covariance, and the richer per-channel features bought
essentially nothing there. EthanolConcentration went 0.2890 → 0.2852, a hair *below* the floor — also a
prediction I made explicitly: on smooth spectra, patching with overlapping windows and per-instance
normalization washes out the slow global trend that the decomposition-linear captured directly, and the
encoder's local-shape bias does not recover it. So PatchTST is a sideways-to-up move: a clear win where local
shape matters, flat where cross-channel matters, a small loss where the global trend matters. Let me put the
deltas on paper so "about even" is a number, not a vibe. Handwriting +0.0235 (0.2306 → 0.2541), FaceDetection
+0.0031 (0.6822 → 0.6853), EthanolConcentration −0.0038 (0.2890 → 0.2852). The floor's mean over the three is
(0.2890 + 0.6822 + 0.2306)/3 = 0.4006; PatchTST's is (0.2852 + 0.6853 + 0.2541)/3 = 0.4082, a mean gain of
just +0.0076. Read that carefully: essentially the entire mean improvement is the single +0.0235 on
Handwriting, diluted by a third because the other two datasets barely moved (one up 0.003, one down 0.004,
nearly cancelling). So this is not a broad lift — it is one dataset moving and two standing still. That is
why I cannot stop here: I have traded one weakness for another rather than removing a weakness, and the
mechanism that moved Handwriting (local-shape tokens) did nothing for the two datasets whose bottleneck is
elsewhere.

Reading across the two rungs, the common structural defect is now visible. Both models present the time axis
as a *single* axis and ask the model to recover everything from adjacency along it (the floor) or from
attention over local patches along it (PatchTST). But the discriminative structure in these series is
multi-scale and, crucially, *cross-period*: in Handwriting a stroke recurs phase-aligned across the gesture;
in EthanolConcentration the spectral curve has structure at several wavelength scales at once; in
FaceDetection the MEG rhythm has a characteristic period and the signal is how successive cycles relate.
A patch token sees a local stretch, but a patch never relates the same phase one period earlier to the same
phase one period later — those two points are `period` steps apart in the 1-D layout, with a whole cycle
crammed between them, and no single patch or kernel spans them. That is the bottleneck PatchTST inherits from
the 1-D representation: it can read *within*-stretch shape but not *across*-period shape. And both rungs share
a second, simpler defect I deferred at each step: neither consults the padding mask, so on the
variable-length datasets the right-padded tail is embedded, attended, and flattened as if it were signal.

So the two things to fix are: (1) make cross-period structure as reachable as within-period structure, and
(2) stop feeding padding into the classifier. The second is a one-line fix once I have a per-timestep feature
representation; the first is the real design problem, and it is the doubt I have been circling since the
floor — the input *representation* is the leverage, not the kernel. PatchTST changed the token; now I change
the *layout*.

Stare at a 1-D series and a period `p`. Within the layout, the within-period neighbour (`t−1`, `t+1`) is
adjacent, but the cross-period neighbour (`t−p`, the same phase one cycle back) is `p` steps away. What if I
chop the series into consecutive blocks of length `p` and stack them as the rows of a matrix? Then walking
along a row is walking through one cycle — the intraperiod, within-cycle shape — and walking down a column is
walking through the same phase across successive cycles — the interperiod, cross-cycle variation. Both
dependencies are now *adjacencies* in a 2-D array: the thing that was `p` apart in 1-D is one step apart along
the column axis. That single reshape turns the cross-period structure PatchTST could not reach into ordinary
2-D locality. And the instant the data is a 2-D grid with two meaningful local axes, I inherit the entire
mature toolbox of 2-D convolution: a 2-D kernel sliding over the grid sees, in one receptive field, a few
adjacent timesteps within the cycle *and* the same window of phases in neighbouring cycles — modelling
intra- and inter-period variation simultaneously, which neither the linear floor nor patch attention could do
with one operator. The bottleneck was never that the variation is hard; it is that the 1-D layout was the
wrong space to look at it in.

Let me trace the reshape on a tiny concrete case to be sure the two axes mean what I claim. Take `T = 12` and
a period `p = 4`, so the grid is `12 // 4 = 3` rows by 4 columns. Lay the timesteps in row-major order:
row 0 is `[x0, x1, x2, x3]`, row 1 is `[x4, x5, x6, x7]`, row 2 is `[x8, x9, x10, x11]`. Now walking down
column 1 visits `x1, x5, x9` — timesteps 1, 5, 9, which are exactly `p = 4` apart, the *same phase* in three
successive cycles. In the 1-D layout `x1` and `x5` were four steps apart with a whole cycle between them; in
the grid they are vertically adjacent, one step apart. So a 2-D kernel of height 2 that sits on rows 0–1 of
column 1 sees `x1` and `x5` together — the cross-period relation PatchTST could never span without a kernel as
wide as the period. And walking along a row visits within-cycle neighbours as before. The reshape genuinely
converts period-distance into grid-adjacency; the trace confirms it rather than my just asserting it. On the
real datasets the grids are large — if EthanolConcentration's dominant period comes out around 175, the grid
is roughly 10 rows × 175 columns; if FaceDetection's is around 15, roughly 4 rows × 15 — but the geometry is
the same: columns are phase, rows are cycles.

Which period `p`, though? The tempting shortcut is to hand-set one — but the frozen one-file constraint means
a single fixed `p` would have to serve a 1,750-step spectrum, a 62-step MEG window, and a 152-step gesture at
once, and there is no single period that is meaningful across all three; a `p` that tiles EthanolConcentration
into sensible cycles would exceed the entire length of a FaceDetection window. So a fixed period is a non-starter
under this substrate, which is the concrete reason I must *discover* it. These series have several periodicities
at once and I do not know them a priori — they differ by dataset and even by window. So discover them from the data. The amplitude of the Fourier
transform at frequency `j` measures how strongly a periodic component of period `T/j` is present; a strong
period is a tall amplitude peak. So take the real FFT of the window along time, take amplitudes, average over
batch and channels to get one length-`T/2+1` amplitude profile, zero out the DC term (frequency 0 is the
window mean, not a period, and it is usually huge — it would dominate peak selection), and pick the top-`k`
peaks. Each peak gives a dominant frequency `f_i` and a period `p_i = T // f_i`; keep the `k` amplitudes as
well, because an amplitude is a measure of how strongly that period is expressed, and I will want it as an
importance weight. I take only the top `k` (the spectrum is sparse and its high-frequency tail is mostly
noise), with `k = 3` here.

The DC-drop step is not cosmetic, and a number shows why. The frequency-0 bin of the real FFT is the sum
(hence, up to scale, the mean) of the window. On EthanolConcentration the absorbance level is large and
positive at every timestep, so that mean is huge — its amplitude can dwarf every genuine periodic peak by
one or two orders of magnitude. If I left it in, `topk` would pick frequency 0 as the "strongest period,"
and `p = T // 0` is undefined (division by zero) besides being meaningless — the mean is a level, not a
rhythm. Zeroing `frequency_list[0]` before `topk` removes that trap and lets the selection see the actual
periodic structure. Note this is a *selection*-time removal only: I zero the DC bin in the amplitude profile
used to rank periods, but the raw window (level intact) is what the TimesBlocks and the reshape actually
process, so I have not thrown the level away — I have only stopped it from hijacking period discovery. That
distinction matters for the EthanolConcentration argument below, where keeping the level is the point.

Now the block, carefully. For each period `p_i` I want to lay the `T` timesteps into a grid of
`(T // p_i)` rows by `p_i` columns. `T` is generally not a multiple of `p_i`, so I pad the series with zeros
along time up to the next multiple of `p_i`, reshape to `[B, length // p_i, p_i, channels]`, permute to put
channels first so it reads as an image `[B, channels, num_periods, period]`, and after processing reshape
back to 1-D and truncate the padding away. What runs over each 2-D view is a 2-D convolution — but the
variation has structure at several spatial scales (a fine wiggle over two steps, a broad hump over a third of
the cycle, a slow tilt across several cycles), so a single fixed kernel commits to one scale. That is exactly
the problem the Inception block solves in vision: run several kernel sizes in parallel inside one block and
combine them, multi-scale by construction and parameter-efficient. So I use a parameter-efficient inception
block — parallel 2-D convolutions with increasing kernel sizes, mean-combined — as a two-layer bottleneck
`d_model → d_ff → d_model` with a GELU between, with `num_kernels = 6` scales. One decision that matters for
efficiency and cleanliness: I share *one* inception block across all `k` periods. Each period gets a
different *reshape* (different grid geometry), but the same conv weights process all of them, so the model
size is invariant to `k` — I can dial `k` purely as a width-of-search knob — and conceptually the inception is
learning "how to read 2-D temporal variation," which should not depend on which period produced the grid.

The sharing decision is not just tidy — it is what keeps this rung trainable on the tiny UEA splits, and the
parameter count shows why. Each Inception block holds `num_kernels = 6` parallel 2-D convs with kernel sizes
1, 3, 5, 7, 9, 11, so a `Conv2d(in, out)` at those sizes costs `in · out · Σ(size²) = in · out · (1 + 9 + 25 +
49 + 81 + 121) = in · out · 286` weights. The block's first inception is `128 → 256`: `128 · 256 · 286 ≈ 9.37M`.
The second is `256 → 128`: another `≈ 9.37M`. So one TimesBlock's conv is ~18.7M weights, and with `e_layers =
3` blocks the stack is ~56M weights. Against EthanolConcentration's ~260 training series that is already a
216,000:1 ratio — this rung leans on early stopping even harder than the linear floor did. Now the payoff of
sharing one inception across the `k = 3` periods: if instead I gave each period its own conv, every TimesBlock
would triple to ~56M and the stack to ~168M, tripling an already-enormous model for no conceptual gain, since
"read a 2-D temporal grid" is the same operation whichever period reshaped it. Sharing makes model size
independent of `k`, so `k` becomes a pure width-of-search knob I can raise without paying parameters. Given
the 56M figure I keep `k = 3` — the amplitude spectrum of these windows is sparse, its top three peaks capture
the dominant rhythms, and a larger `k` would only add noisier high-frequency views to aggregate over.

After the shared inception transforms each of the `k` 2-D views, I reshape each back to 1-D, truncate to `T`,
and fuse them. A plain sum throws away the fact that some periods are far more present in this window than
others; I already kept the amplitudes, so I push the `k` amplitudes through a softmax (turning raw,
scale-sensitive amplitudes into convex weights) and take the amplitude-weighted sum — a window dominated by
one rhythm puts most weight on that period's view, a window with two rhythms splits the weight. Trace it with
three amplitudes: if the top-3 periods come back with raw amplitudes proportional to, say, `[6, 2, 1]`, a
softmax gives roughly `[0.94, 0.02, 0.03]`... let me actually compute — `e^6, e^2, e^1 = 403, 7.4, 2.7`, sum
413, so weights `[0.976, 0.018, 0.007]`: the dominant period's 2-D view carries essentially all the weight,
which is the desired behaviour for a strongly single-rhythm window. But that also exposes a subtlety I should
name: raw FFT amplitudes are large numbers and softmax is scale-sensitive, so on a window with one towering
peak the aggregation collapses to almost-hard selection of one view, while on a window with three comparable
peaks (amplitudes like `[2, 1.9, 1.8]` → weights ≈ `[0.37, 0.34, 0.30]`) it genuinely blends all three. That
is acceptable and arguably right — it means the fusion is confident when the spectrum is peaky and hedges when
it is flat — but it is a consequence of feeding raw amplitudes into softmax that I am choosing deliberately
rather than stumbling into. That is one
TimesBlock: discover periods → reshape to `k` 2-D views → shared inception on each → reshape back →
amplitude-softmax aggregate, with a residual connection so a block only learns the correction to the
representation and the stack stays stable. I stack `e_layers = 3` residual TimesBlocks with a LayerNorm
between them, refining the representation with depth.

In front of the stack I need an embedding to lift the raw `enc_in` channels into a `d_model = 128` feature
sequence. For classification there are no useful calendar marks (the windows are bare series), so the
embedding is the value embedding (a 1-D conv over the channel axis) plus a fixed positional embedding,
dropped out — I pass no time marks. Note one important contrast with PatchTST and the floor: TimesNet does
*not* do per-instance subtract/divide normalization for classification. The reason is that the discriminative
cue for classification can live in the absolute level and scale (a spectral curve's overall absorbance, a
gesture's acceleration magnitude), and the embedding's conv plus LayerNorm inside the blocks already handle
scale heterogeneity without erasing the level — whereas PatchTST's per-window normalization, which I argued
washed out EthanolConcentration's global trend, is exactly the move I want to *not* make here. That is a
concrete reason to expect TimesNet to recover the EthanolConcentration ground PatchTST lost. Make the
mechanism explicit: PatchTST subtracted each window's temporal mean and divided by its std before the encoder,
which on a spectral trace removes exactly the overall absorbance level that separates concentrations — I traced
at the previous rung that this is a plausible cause of its −0.0038 dip on EthanolConcentration. TimesNet skips
that subtraction entirely; the value-embedding conv reads the level-intact window, and the only normalizations
inside are the LayerNorm between blocks (over the feature dimension, not erasing the temporal level) and
BatchNorm-free convs. So the level cue survives all the way to the head. If EthanolConcentration climbs back
above the floor's 0.2890 here, the not-normalizing choice is the most likely reason, and the multi-scale 2-D
convolution reading the smooth curve at several kernel widths is the second.

Now the head, and here is where the classification path finally does the thing both earlier rungs skipped.
After the TimesBlock stack I have a per-timestep feature representation `[B, seq_len, d_model]`. I apply a
GELU and dropout, and then — the load-bearing line — I multiply by the padding mask:
`output = output * x_mark_enc.unsqueeze(-1)`, zeroing out the feature vectors at the padded positions before
I flatten. This is the mask-awareness both the linear floor and PatchTST lacked: the padded tail contributes
exactly zero to the flattened `seq_len · d_model` vector that the final `Linear(d_model · seq_len, num_class)`
projects to logits, so the classifier cannot learn spurious weights on padding or have its decision drifted
by where a window happens to end. On the variable-length datasets this should be a clean, free gain over the
two mask-blind rungs. The final projection is `Linear(d_model · seq_len, num_class)`, and its width is worth
noting because it is the same flatten-and-project shape as the floor's head, just over learned features:
EthanolConcentration `128 · 1750 · 4 ≈ 896k` weights, Handwriting `128 · 152 · 26 ≈ 506k`, FaceDetection
`128 · 62 · 2 ≈ 16k`. So on the long-window datasets the head is again large, and on FaceDetection it is tiny —
which matters because FaceDetection's cross-channel decision must be squeezed through that 16k-weight join, the
only place channels meet. The channels are mixed in the value embedding and in the flatten-and-project head, as
before; what is new is the cross-period 2-D modelling in between and the mask-aware pooling.

There is one subtlety about the mask multiply I should check rather than wave past: does zeroing the padded
features actually remove padding's influence, given that the TimesBlocks ran *before* the mask is applied? The
FFT period-discovery and the 2-D reshape operate on the full padded window, so padding does leak into the
period estimate and into the conv receptive fields near the tail. But the load-bearing claim is narrower: the
*final flattened vector* the classifier reads has exact zeros at padded positions, so the linear head cannot
place weight on "what value sits at padded position t" — the spurious cue the floor and PatchTST both learned.
That is a real, if partial, fix: it cleans the decision layer even though the intermediate representation still
saw the tail. On Handwriting, whose windows vary in length and get right-padded to the dataset maximum, that
cleaned decision layer is where I expect the mask to pay off, stacked on top of the cross-period gain.

The falsifiable expectations against PatchTST's numbers. On Handwriting (PatchTST 0.2541, floor 0.2306): the
gestures are strongly phase-recurrent, which is exactly what the 2-D cross-period view captures and patch
attention could not, *and* Handwriting has variable-length windows so the mask-aware pooling should help on
top — I expect the largest gain of the three, a clear jump above 0.25, plausibly into the low-to-mid 0.3s. On
EthanolConcentration (PatchTST 0.2852, floor 0.2890): I expect TimesNet to recover and exceed both, because
it does not per-window-normalize away the global trend and its multi-scale 2-D convolution reads the spectral
shape at several scales — I expect it to clear 0.29, into the low 0.3s. On FaceDetection (PatchTST 0.6853,
floor 0.6822): this is the dataset I am least confident about, because TimesNet still mixes channels only in
the embedding and head, not with cross-channel attention — so I expect at best parity, around 0.67–0.69, and
a small loss here (giving up a little cross-channel-friendly per-step attention for cross-period 2-D structure
that FaceDetection needs less) would be entirely consistent. The single sharpest test is whether TimesNet
beats PatchTST on *both* Handwriting and EthanolConcentration: if it does, then cross-period 2-D modelling
plus mask-aware pooling is the right leverage for these heterogeneous series and TimesNet is the strongest
rung; if it only matches PatchTST, then the 2-D reshape is not buying what I claim and the task is bounded by
something else entirely. The distilled module and the literal `Custom.py` fill are in the answer.
