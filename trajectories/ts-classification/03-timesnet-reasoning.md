PatchTST's three numbers confirm the diagnosis I built it on, and one confirms it by *failing* in the exact
way I flagged. Handwriting moved 0.2306 → 0.2541 — a real gain, and the sharpest test I named: local-shape
patch tokens plus a nonlinear encoder do help on the 26-class gesture problem, so local temporal shape was
indeed a bottleneck. FaceDetection went 0.6822 → 0.6853, parity within noise — as predicted, because
channel-independence forbids in-encoder cross-channel modeling on the one dataset whose signal is
cross-channel covariance. EthanolConcentration went 0.2890 → 0.2852, a hair *below* the floor — also
predicted: on smooth spectra, patching with overlapping windows and per-instance normalization washes out
the slow global trend the decomposition captured directly. The deltas make "about even" a number:
Handwriting +0.0235, FaceDetection +0.0031, EthanolConcentration −0.0038; the floor's three-dataset mean
0.4006, PatchTST's 0.4082, a mean gain of just +0.0076. Essentially the entire mean improvement is the
single +0.0235 on Handwriting, diluted by the two datasets that barely moved. So this is not a broad lift —
one dataset moving and two standing still. I traded one weakness for another rather than removing one.

Reading across the two rungs, the common structural defect is now visible. Both present the time axis as a
*single* axis and recover everything from adjacency along it (the floor) or attention over local patches
along it (PatchTST). But the discriminative structure here is multi-scale and, crucially, *cross-period*: in
Handwriting a stroke recurs phase-aligned across the gesture; in EthanolConcentration the spectral curve has
structure at several wavelength scales; in FaceDetection the MEG rhythm has a characteristic period and the
signal is how successive cycles relate. A patch sees a local stretch but never relates the same phase one
period earlier to the same phase one period later — those points are `period` steps apart in the 1-D layout,
a whole cycle between them, and no single patch or kernel spans them. That is the bottleneck PatchTST
inherits from the 1-D representation: within-stretch shape, but not across-period shape. And both rungs share
a simpler defect I deferred each step: neither consults the padding mask, so the right-padded tail is
embedded, attended, and flattened as if it were signal.

So two things to fix: make cross-period structure as reachable as within-period structure, and stop feeding
padding into the classifier. The second is a one-line fix once I have a per-timestep feature representation;
the first is the real design problem, and it is the doubt I have circled since the floor — the input
*representation* is the leverage. PatchTST changed the token; now I change the *layout*.

Stare at a 1-D series and a period `p`. The within-period neighbour (`t±1`) is adjacent, but the
cross-period neighbour (`t−p`, the same phase one cycle back) is `p` steps away. Chop the series into
consecutive blocks of length `p` and stack them as rows of a matrix: walking along a row is walking through
one cycle (intraperiod shape), walking down a column is walking through the same phase across successive
cycles (interperiod variation). Both dependencies become *adjacencies* in a 2-D array — the thing that was
`p` apart in 1-D is one step apart along the column. And the instant the data is a 2-D grid with two
meaningful local axes, I inherit 2-D convolution: a kernel sliding over the grid sees, in one receptive
field, a few adjacent timesteps within the cycle *and* the same window of phases in neighbouring cycles —
modelling intra- and inter-period variation simultaneously, which neither the linear floor nor patch
attention could do with one operator. The bottleneck was never that the variation is hard; it is that the
1-D layout was the wrong space to look at it in. Concretely with `T = 12, p = 4` the grid is 3 rows × 4
columns and column 1 holds timesteps 1, 5, 9 — same phase, three cycles, now vertically adjacent where in
1-D they were four steps apart. On the real datasets the grids are large (EthanolConcentration maybe ~10 ×
175, FaceDetection maybe ~4 × 15) but the geometry is the same: columns are phase, rows are cycles.

Which period `p`? A single fixed `p` under the one-file constraint would have to serve a 1,750-step
spectrum, a 62-step MEG window, and a 152-step gesture at once — a `p` that tiles EthanolConcentration into
sensible cycles would exceed an entire FaceDetection window. So I must *discover* it. These series have
several periodicities at once, differing by dataset and even by window, so discover them from the data. The
amplitude of the Fourier transform at frequency `j` measures how strongly a component of period `T/j` is
present; a strong period is a tall amplitude peak. Take the real FFT of the window along time, take
amplitudes, average over batch and channels to one length-`T/2+1` profile, zero the DC term, and pick the
top-`k` peaks. Each gives a frequency `f_i` and period `p_i = T // f_i`; keep the `k` amplitudes too, as
importance weights. I take `k = 3` — the spectrum is sparse and its high-frequency tail is mostly noise.

The DC-drop is not cosmetic. The frequency-0 bin is (up to scale) the window mean, and on
EthanolConcentration the absorbance level is large and positive everywhere, so that mean can dwarf every
genuine periodic peak by one or two orders of magnitude. Left in, `topk` would pick frequency 0 as the
"strongest period," and `p = T // 0` is undefined besides being meaningless — a level, not a rhythm.
Crucially this is a *selection*-time removal only: I zero the DC bin in the ranking profile, but the raw
window (level intact) is what the reshape and convolutions process, so I have not thrown the level away — I
have only stopped it from hijacking period discovery. That distinction matters for the EthanolConcentration
argument below.

For each period `p_i` I lay the `T` timesteps into a `(T // p_i) × p_i` grid — since `T` is generally not a
multiple, I zero-pad along time to the next multiple, reshape to `[B, length // p_i, p_i, channels]`, permute
to `[B, channels, num_periods, period]` so it reads as an image, and truncate the padding after processing.
What runs over each view is a 2-D convolution — but the variation has structure at several spatial scales (a
fine wiggle over two steps, a broad hump over a third of the cycle, a slow tilt across cycles), so a single
fixed kernel commits to one scale. That is the problem the Inception block solves: several kernel sizes in
parallel inside one block, mean-combined, multi-scale by construction. So the operator is a
parameter-efficient inception block — parallel 2-D convs of increasing size — as a `d_model → d_ff →
d_model` bottleneck with a GELU between, `num_kernels = 6` scales. And I share *one* inception across all `k`
periods: each period gets a different reshape, but the same conv weights process all of them, so model size
is invariant to `k` and I can dial `k` purely as a width-of-search knob — conceptually the inception learns
"how to read 2-D temporal variation," which should not depend on which period produced the grid.

The sharing is what keeps this trainable on the tiny UEA splits. Each inception holds 6 parallel convs of
sizes 1,3,5,7,9,11, so a `Conv2d(in, out)` costs `in · out · (1+9+25+49+81+121) = in · out · 286`. The `128
→ 256` and `256 → 128` pair is ~9.37M each, ~18.7M per TimesBlock, ~56M over `e_layers = 3` — against
EthanolConcentration's ~260 series a 216,000:1 ratio, so this rung leans on early stopping even harder than
the floor. If instead each period had its own conv, every block would triple to ~56M and the stack to ~168M,
tripling an already-enormous model for no conceptual gain. Sharing makes size independent of `k`; given 56M
I keep `k = 3`, since the top three peaks capture the dominant rhythms and a larger `k` only adds noisier
high-frequency views.

After the shared inception, I reshape each of the `k` views back to 1-D, truncate to `T`, and fuse them. A
plain sum ignores that some periods are far more present in this window than others; I already kept the
amplitudes, so I softmax them into convex weights and take the amplitude-weighted sum. Because raw FFT
amplitudes are large and softmax is scale-sensitive, a window with one towering peak collapses to
almost-hard selection of that period's view, while a window with three comparable peaks genuinely blends all
three — the fusion is confident when the spectrum is peaky and hedges when it is flat. That is a consequence
of feeding raw amplitudes into softmax I choose deliberately rather than stumble into. One TimesBlock is
then: discover periods → reshape to `k` 2-D views → shared inception on each → reshape back →
amplitude-softmax aggregate, with a residual so a block learns only the correction. I stack `e_layers = 3`
residual TimesBlocks with a LayerNorm between them.

In front of the stack an embedding lifts the raw `enc_in` channels into a `d_model = 128` feature sequence:
for classification the windows are bare series with no calendar marks, so it is the value embedding (a 1-D
conv over the channel axis) plus a fixed positional embedding, dropped out, no time marks. One deliberate
contrast with PatchTST and the floor: TimesNet does *not* subtract/divide per-instance for classification.
The discriminative cue can live in absolute level and scale (a spectral curve's overall absorbance, a
gesture's acceleration magnitude), and PatchTST's per-window normalization is exactly what I argued washed
out EthanolConcentration's trend and cost it −0.0038. Here the value-embedding conv reads the level-intact
window, and the only normalizations inside are the LayerNorm between blocks (over the feature dimension, not
the temporal level). So the level cue survives to the head. If EthanolConcentration climbs back above the
floor here, this not-normalizing choice is the most likely reason, with the multi-scale 2-D convolution
reading the smooth curve at several widths second.

Now the head does the thing both earlier rungs skipped. After the stack I have a per-timestep
representation `[B, seq_len, d_model]`; I apply GELU and dropout, then the load-bearing line: multiply by the
padding mask, `output = output * x_mark_enc.unsqueeze(-1)`, zeroing the feature vectors at padded positions
before flattening. The padded tail then contributes exactly zero to the flattened `seq_len · d_model` vector
the final `Linear(d_model · seq_len, num_class)` reads, so the classifier cannot learn spurious weights on
padding or drift with where a window ends. There is a subtlety worth naming: the FFT period-discovery and
the 2-D reshape run on the full padded window, so padding does leak into the period estimate and the conv
receptive fields near the tail — the mask multiply does *not* undo that. But the narrower claim holds: the
final flattened vector has exact zeros at padded positions, so the linear head cannot place weight on "what
value sits at padded position t," the spurious cue the floor and PatchTST both learned. A real, if partial,
fix — it cleans the decision layer even though the intermediate representation saw the tail. The head width
is the floor's flatten-and-project shape over learned features: EthanolConcentration `128 · 1750 · 4 ≈
896k`, Handwriting `128 · 152 · 26 ≈ 506k`, FaceDetection `128 · 62 · 2 ≈ 16k` — tiny on FaceDetection,
which matters because its cross-channel decision must squeeze through that 16k-weight join, the only place
channels meet. What is new between embedding and head is the cross-period 2-D modelling and the mask-aware
pooling.

The expectations against PatchTST. On Handwriting (0.2541, floor 0.2306) I expect the largest gain of the
three, a clear jump above 0.25: the gestures are strongly phase-recurrent, exactly what the 2-D cross-period
view captures and patch attention could not, and the variable-length windows mean mask-aware pooling helps
on top. On EthanolConcentration (0.2852, floor 0.2890) I expect TimesNet to recover and clear the floor,
because it does not normalize away the global trend and reads the spectral shape at several scales. On
FaceDetection (0.6853, floor 0.6822) I am least confident — TimesNet still mixes channels only in embedding
and head, not with cross-channel attention — so I expect at best parity, and a small loss (trading a little
cross-channel-friendly per-step attention for cross-period structure this dataset needs less) would be
entirely consistent. The sharpest test is whether TimesNet beats PatchTST on *both* Handwriting and
EthanolConcentration: if it does, cross-period 2-D modelling plus mask-aware pooling is the right leverage;
if it only matches, the 2-D reshape is not buying what I claim and the task is bounded elsewhere.
