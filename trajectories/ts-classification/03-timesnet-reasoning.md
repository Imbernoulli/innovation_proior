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
shape matters, flat where cross-channel matters, a small loss where the global trend matters. Averaged it is
about even with the floor, which is why I cannot stop here — I have traded one weakness for another rather
than removing a weakness.

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

Which period `p`, though? These series have several periodicities at once and I do not know them a priori —
they differ by dataset and even by window. So discover them from the data. The amplitude of the Fourier
transform at frequency `j` measures how strongly a periodic component of period `T/j` is present; a strong
period is a tall amplitude peak. So take the real FFT of the window along time, take amplitudes, average over
batch and channels to get one length-`T/2+1` amplitude profile, zero out the DC term (frequency 0 is the
window mean, not a period, and it is usually huge — it would dominate peak selection), and pick the top-`k`
peaks. Each peak gives a dominant frequency `f_i` and a period `p_i = T // f_i`; keep the `k` amplitudes as
well, because an amplitude is a measure of how strongly that period is expressed, and I will want it as an
importance weight. I take only the top `k` (the spectrum is sparse and its high-frequency tail is mostly
noise), with `k = 3` here.

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

After the shared inception transforms each of the `k` 2-D views, I reshape each back to 1-D, truncate to `T`,
and fuse them. A plain sum throws away the fact that some periods are far more present in this window than
others; I already kept the amplitudes, so I push the `k` amplitudes through a softmax (turning raw,
scale-sensitive amplitudes into convex weights) and take the amplitude-weighted sum — a window dominated by
one rhythm puts most weight on that period's view, a window with two rhythms splits the weight. That is one
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
concrete reason to expect TimesNet to recover the EthanolConcentration ground PatchTST lost.

Now the head, and here is where the classification path finally does the thing both earlier rungs skipped.
After the TimesBlock stack I have a per-timestep feature representation `[B, seq_len, d_model]`. I apply a
GELU and dropout, and then — the load-bearing line — I multiply by the padding mask:
`output = output * x_mark_enc.unsqueeze(-1)`, zeroing out the feature vectors at the padded positions before
I flatten. This is the mask-awareness both the linear floor and PatchTST lacked: the padded tail contributes
exactly zero to the flattened `seq_len · d_model` vector that the final `Linear(d_model · seq_len, num_class)`
projects to logits, so the classifier cannot learn spurious weights on padding or have its decision drifted
by where a window happens to end. On the variable-length datasets this should be a clean, free gain over the
two mask-blind rungs. The channels are mixed in the value embedding and in the flatten-and-project head, as
before; what is new is the cross-period 2-D modelling in between and the mask-aware pooling.

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
something else. The distilled module and the literal `Custom.py` fill are in the answer.
