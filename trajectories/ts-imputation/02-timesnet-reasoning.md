The linear interpolator did exactly what I built it to do, and its numbers tell me precisely where it
runs out. On seed 42 it lands MSE 0.1498 / MAE 0.2693 on ETTh1, 0.0503 / 0.1143 on Weather, and 0.1132
/ 0.2428 on ECL — a decisive win over mean-guessing everywhere, confirming that a length-one linear map
along time already captures most of the interpolable structure. But the pattern across the three datasets
is the interesting part, and it falls out along the axis I flagged. Weather is its best result by a wide
margin; ETTh1 and ECL are roughly four-to-five times worse in MSE. ETTh1 I can read as the trend-plus-
seasonality strain I worried about: hourly transformer temperature has sharp local transients that an
affine map of the window smears, and the decomposition only half-fixes it. ECL is the one that indicts
the design directly. With 321 strongly co-moving electricity clients, the channel-blind map throws away
the single most informative cue at a masked entry — the simultaneous values of the hundreds of correlated
channels — and it pays for it: 0.1132 MSE is the worst of the three, and it is *not* a hard interpolation
problem in the temporal sense (the clients are smooth and periodic), so the only explanation for that
error is the cross-channel information the model cannot see. So the diagnosis is sharp and it is twofold:
first, an affine map of a window cannot represent genuinely nonlinear temporal variation (the ETTh1
transients); second, and more costly, the model is channel-blind exactly where channels are most
correlated (ECL). The next rung has to fix both — it needs nonlinearity and it needs to look across
channels.

Let me think about what *kind* of nonlinearity, because I do not want to reach blindly for a Transformer
and re-import the tokenisation problem that made the linear map competitive in the first place. The thing
the linear map got right is that a time series is about *shape over stretches*, not isolated points. The
thing it got wrong is that it modelled each channel's window as a flat 1-D vector, so all temporal
structure — the daily cycle, the weekly cycle, the within-day shape — got mashed into one axis where it
overlaps and interferes. On a real channel several variations live in the same 1-D record at once: a slow
trend, a daily oscillation, a weekly one, and short fluctuations, all superimposed. A 1-D model, linear or
not, has to disentangle all of them along a single axis, and that is genuinely hard. So the question I want
to ask is: is there a representation in which these overlapping variations stop overlapping?

Here is the observation that cracks it. A periodicity is a statement that values one period apart are
related. If I knew a channel's dominant period *p*, I could fold its length-96 window into a 2-D array of
shape `(96/p, p)` — successive rows are successive periods, each column is a fixed phase of the cycle. Then
two kinds of variation become two *orthogonal axes*: moving down a column traverses the same phase across
successive cycles (the slow, between-period variation — trend and the cycle's evolution), and moving along
a row traverses one full cycle (the fast, within-period shape). What was one tangled 1-D axis becomes two
clean 2-D axes, and crucially a 2-D structure is exactly what a 2-D convolution is built to model — local
neighbourhoods in both the within-period and between-period directions at once. The interference is gone
because the two scales are now on different axes. That reframing — fold the 1-D window into 2-D by period,
then use 2-D convolution — is the move I want, because it gives me real nonlinearity (stacked convs with
GELU) that is *aligned with the structure of the data* rather than fighting it, and 2-D convs over a folded
window naturally mix neighbouring phases and cycles in a way a flat linear map cannot.

But which period? Real series do not have one clean period; they have several, and I do not know them in
advance. The principled way to find the dominant periodicities of a window is the frequency domain: take
the FFT of the window along time, look at the amplitude spectrum, and the frequencies with the largest
amplitude are the dominant periodicities; their reciprocals (scaled by the window length) are the periods.
So I compute the rFFT of the window, average the amplitude over batch and channels to get one spectrum per
window-set, zero out the zero-frequency (DC) bin so the trend does not masquerade as a period, and take the
top-*k* frequencies by amplitude. Each gives a candidate period *p_i = 96 / freq_i*. I fold the window by
each of the top-*k* periods separately, run the 2-D conv stack on each folded view, unfold back to 1-D, and
then I need to combine the *k* reconstructions. The natural weighting is the amplitudes themselves: a
period that carried more spectral energy should count for more, so I softmax the top-*k* amplitudes into
weights and take the weighted sum of the *k* unfolded outputs. That adaptive aggregation means the model
discovers, per window, which periodicities matter and leans on them proportionally — no hand-set period,
no single-period assumption.

That is one block; I stack a few of them with a residual connection so the representation can be refined,
and a LayerNorm between blocks to keep it stable. Each block: FFT to find the top-*k* periods and their
amplitudes, fold-conv-unfold for each period, amplitude-weighted sum, add the input back. Now I have to
wire this into the imputation contract, and there are several task-specific details that are *not* the same
as the forecasting or anomaly-detection versions of this idea, so I derive them rather than transplant
them.

First, the input space. A 2-D conv over a folded window needs a richer per-timestep representation than a
raw scalar — the conv channels are what carry the learned features. So I embed the masked window into a
`d_model`-dimensional space first with the library's `DataEmbedding`: a value projection plus a positional
code plus the time-feature (`x_mark_enc`) embedding. The time features matter here in a way they did not
for the linear map: the calendar stamp tells the model the phase of the daily and weekly cycle directly,
which is exactly the periodic structure the FFT folding exploits. After the embedding the FFT for period
discovery runs in the embedded space, and the folding reshapes `(B, 96, d_model)` into `(B, 96/p, p,
d_model)`, permutes `d_model` to the conv-channel axis, and the 2-D conv mixes within-period and
between-period neighbourhoods.

Second — and this is the part I must get right for *imputation specifically* — normalisation under a mask.
The Non-stationary-Transformer normalisation that wraps the model (centre and scale per window, undo
after) cannot use the standard mean/variance over all timesteps, because a quarter of the entries are
fake zeros and would corrupt the statistics. So I compute the statistics over the *observed entries only*:
the mean is the sum of `x_enc` over time divided by the count of observed entries per channel
(`sum(mask == 1)`), then I subtract it and re-zero the masked positions (`masked_fill(mask == 0, 0)`) so
the holes stay holes after centring, and the standard deviation is the root-mean-square of those centred,
re-masked values, again normalised by the observed count. Both statistics are detached so the
normalisation is treated as a fixed transform, not something to backprop through. After the TimesBlocks
and the linear projection back to `c_out` channels, I de-normalise by multiplying by the stored std and
adding the stored mean. This masked normalisation is the single most important task-specific change: it is
what lets the periodic 2-D modelling see the window's *shape* on a common scale without the punched-out
zeros biasing the centre and width. Note that, unlike the forecasting path, there is no `predict_linear`
to stretch the temporal axis — for imputation `pred_len = seq_len`, the sequence length is preserved end to
end, so the embedded window goes straight into the TimesBlocks at its native length and the projection maps
`d_model → c_out` at every one of the 96 positions.

Does this answer both halves of the DLinear diagnosis? The nonlinearity is real now — stacked 2-D convs
with GELU over a folded window are a genuine nonlinear map, so the ETTh1 transients that an affine map
smeared can be represented. And the channel-blindness is addressed, though indirectly and this is the
honest caveat: TimesNet is not channel-independent like the linear map was. The `DataEmbedding`'s value
projection mixes all `enc_in` channels into the `d_model` features at each timestep, and the 2-D convs and
the final projection operate on those mixed features, so cross-channel information *is* available to the
reconstruction — a masked client on ECL can be filled using the embedded representation that saw all 321
clients at that timestep. It is not an explicit channel-attention, but it is not blind either, and on ECL
that should be the decisive difference versus rung one.

So rung two is the periodic 2-D model: masked-statistics normalisation, embed with time features, stack
`e_layers` TimesBlocks that FFT-discover the top-*k* periods, fold each into 2-D, run an Inception-style
2-D conv stack, amplitude-weight and sum, residual and LayerNorm, then project and de-normalise — the full
scaffold module is in the answer. Now the falsifiable expectations against the DLinear numbers. I expect a
clear improvement on all three datasets, because every one of them has the periodic structure this model is
built to exploit. The two places I expect the *largest* gains are exactly DLinear's two failure points:
ETTh1, where the nonlinearity should cut the transient-smearing error well below 0.1498 MSE; and ECL,
where the cross-channel mixing in the embedding should beat the channel-blind 0.1132 MSE substantially. On
Weather, already DLinear's strong suit, I expect improvement but a smaller margin. If I am wrong — if ECL
does *not* improve much — that would say the embedding's implicit channel mixing is too weak to exploit the
correlation, and the next rung would need explicit cross-channel structure. If ETTh1 does not improve, the
nonlinearity is not the bottleneck and the problem is elsewhere. Concretely, the bar this rung must clear
is DLinear's seed-42 line on every dataset, and I expect it to clear it most decisively on ETTh1 and ECL —
the two the linear interpolator handled worst.
