The inversion delivered exactly the split I predicted, and the split is the whole lesson. ECL — the
321-channel laggard — fell from 0.1819 to 0.1482, the largest single-dataset MSE drop on the ladder so
far, with MAE crashing 0.2743 → 0.2398. That is the cross-variate lever paying off precisely where I
said it would. But ETTh1 went the other way: 0.3794 → 0.3950, *worse* than PatchTST, landing essentially
back at the linear control's 0.3962. That is the over-parameterization I called: 7 variate tokens of
dimension 512 with weak channel coupling is all capacity and no structure to fit, and the inversion
threw away the fine intra-series temporal detail that PatchTST's patches preserved. Weather barely moved
(0.1738 → 0.1753, a hair worse). So the mean MSE improved (0.2451 → 0.2395) but the improvement is
entirely ECL; on the small, less-coupled datasets the inversion actively regressed. The bookkeeping makes
that literal: the mean fell 0.0056, and ECL alone contributed 0.0337/3 ≈ 0.0112 of downward push while
ETTh1's +0.0156 regression added 0.0052 back up and Weather's +0.0015 added 0.0005 — net
0.0112 − 0.0052 − 0.0005 ≈ 0.0055, essentially the whole headline drop. In other words the single number
that improved is a tug-of-war the cross-variate win happened to win by a nose, not a broad advance; strip
ECL out and the inversion is a net loss. That is not a model I can build on by pushing further in the same
direction — the next rung cannot simply be "more cross-variate." This is the clearest
possible statement of the trade I have been circling: PatchTST nails intra-series temporal detail but
cannot see channels; iTransformer sees channels but crushes the per-series temporal detail into one
coarse vector. Each rung's virtue is the other's missing piece, and the ladder has been ping-ponging
between them. The next rung has to stop choosing.

So I step back from "temporal attention vs cross-variate attention" entirely and ask what actually
defeats these series, channels aside. I have a length-P window of a C-variate series and I want the
length-F future. The regression is easy; the killer is that the past window is a *tangle*. In one window
the signal is climbing, falling, oscillating at a couple of rhythms, and slowly drifting all at once, and
these are not on separate channels — they are superimposed in the same scalar stream. Whatever I build
has to *disentangle* before it predicts, because predicting a sum of a stationary wiggle and a
non-stationary drift with one map is asking a single head to be two functions at once. That is the same
tension the linear control hit and that decomposition fixed — and decomposition is the thread I want to
pull, because both stronger rungs quietly dropped it. PatchTST and iTransformer both abandoned the
trend/seasonal split that made the linear model competitive in the first place.

The field has two ways to disentangle and I should be honest about where each stops. The first is
season-trend decomposition: a moving average isolates the slowly varying part (trend), the residual is
the fast repeating part (season), because the two halves *behave differently* — season is short-term and
roughly stationary, trend is long-term and non-stationary. The scaffold's `series_decomp` is exactly
this differentiable block, and the linear control showed decompose-then-linear beats far heavier
Transformers. So decomposition is load-bearing and cheap MLP heads suffice. But every one of these does
the decomposition at a *single resolution* — one moving-average kernel, on the series at its native
sampling rate. The split it produces is whatever that one resolution happens to reveal. The second way is
multiperiodicity — find the dominant periods (FFT, fold into a 2D period-by-cycle tensor, 2D-conv). Also
single-axis, and heavy. It is worth pricing the multiperiod route before I set it aside, because it is
tempting: an FFT over a 96-window is cheap, and folding the top-k periods into 2D grids for a 2D-conv does
capture that a signal has both a daily and a weekly cycle. But it commits me to convolutions over
per-period 2D tensors — several of them, at d_model channels — which is a heavier compute and parameter
load than a stack of small linear mixers, and more to the point it still cuts along a single axis: it
separates *period from period* at one resolution, never *scale from scale*. Two cycles of different
length are not the same thing as two views of the same process at different sampling rates — the weekly
and daily periods both live in the fine signal, whereas the fine-vs-coarse distinction is about what
survives low-passing. So the multiperiod route answers a different question than the one the ping-pong
posed, and at higher cost; I set it aside. Both paradigms cut the signal along *one* axis and call it
done.

Let me sit with whether that is the only axis, because one observation keeps nagging. Take a road
sampled every five minutes: a sharp commute spike twice a day. Average that same road to a daily series
and the commute spike is *gone* — what is left is weekday-versus-weekend and holiday structure. Average
to monthly and even that washes out; only a slow seasonal drift remains. It is one physical process, but
it presents a completely different pattern at each sampling scale. That is not a quirk of traffic; it is
what averaging does — an average pool is a low-pass filter, so climbing a downsampling ladder slides the
dominant content monotonically from microscopic detail to macroscopic structure. I can put the filtering
in exact terms. An average-pool of width w replaces w samples by their mean, which annihilates any
component whose period is w samples (it averages a full cycle to zero) and attenuates everything faster
than that, so pooling is a low-pass whose cutoff falls as 1/w. Take the five-minute road: a commute spike
recurs with a roughly twelve-hour period, or 144 samples; pool to a daily series — width 288 — and the
144-sample cycle sits well inside the killed band, so it vanishes exactly as observed, while the
seven-day weekly structure at 2016 samples survives because it is far below the daily cutoff. This is not
hand-waving about "different patterns at different scales"; it is a filter bank, and each rung of the
downsampling ladder is a different pass-band of the same signal. Decomposition cuts
season from trend; periodicity cuts one period from another; but nobody is cutting along the *scale axis
itself*. Fine and coarse views are not redundant copies — they are complementary descriptions, and every
prior rung kept exactly one and threw the rest away. This reframes the ETTh1/ECL ping-pong: PatchTST's
patches are a *fine*-scale view (local shapes), iTransformer's whole-series token is a *coarse*-scale
view (the entire history crushed to one vector). Each rung literally picked one point on the scale axis.
No wonder one wins where the other loses.

A second thing, specific to forecasting, sharpens this. The future is jointly determined by variations
at several scales: the next hour depends both on the immediate five-minute momentum and on which day of
the week it is. So a predictor off the fine view and a predictor off the coarse view do not just see
different inputs — they have different *skills*: the fine one is good at the next few high-frequency
wiggles, the coarse one at the slow drift. If that is true, collapsing everything to one representation
before the prediction head is exactly wrong, because it merges the skills before I can use them. And I
should check the "even multiscale models" objection: Pyraformer and SCINet do build multiple resolutions
internally, but they build the pyramid to *extract* a richer single representation, then merge and
predict once. The forecast never draws on the scales simultaneously and separately. So multiscale
*extraction* is on the table; multiscale *prediction* — a separate predictor per scale, combined at the
end — is the opening.

That gives me the two-stage design. Stage one, build the scale ladder: downsample the input window with
average pooling, window/2, window/4, window/8, producing a list of progressively coarser views of the
same series. The shapes are concrete under the scaffold's L = 96 and a pooling window of 2: an
`AvgPool1d(2)` applied three times gives lengths 96 → 48 → 24 → 12, four scales in all. That last scale,
twelve steps, is coarse enough that a single step spans eight of the original samples yet still long
enough to carry the slow drift — a sensible floor, which is why three downsampling layers rather than
five: pool twice more and I would be at three steps, too short to hold any shape at all. Each scale is its
own pass-band of the window, and the whole ladder is four filtered copies I will now mix and predict from
separately. Stage two has two distinct mixings. First, *Past-Decomposable-Mixing* (the encoder): at
each scale, decompose into season and trend with `series_decomp`, then mix *across scales separately for
each component*. Season mixes bottom-up (fine→coarse): the fine-scale seasonal detail informs the
coarser seasonal patterns, because a clean micro-period constrains what the macro-period can be. Trend
mixes top-down (coarse→fine): the coarse-scale trend (the reliable slow drift) informs the finer trends,
because the macro-drift sets the baseline the micro-trends ride on. The asymmetry is the point — the two
components carry information in *opposite directions* across scales, so mixing them with the same
direction (or merging them) would smear exactly the structure I am trying to keep. Concretely the season
mixing is a stack of down-projections 96 → 48 → 24 → 12 (each a small linear-GELU-linear that pushes the
fine seasonal residual down onto the next-coarser length and adds it in), while the trend mixing is the
mirror stack of up-projections 12 → 24 → 48 → 96 running the other way; the two use the same scale ladder
but traverse it in opposite orders, which is the asymmetry made into shapes.

I should test the direction choice rather than assert it, because getting it backwards would be silent
poison. Why must season flow fine→coarse? A clean sub-daily period measured at the finest scale pins down
the phase and shape that the coarser views, having low-passed most of it away, can no longer resolve on
their own — so the fine season is the *authority* on periodicity and should inform upward. Run it the
other way and the coarse view, which has already blurred the period, would overwrite the one scale that
still sees it: I would be sanding off the seasonal detail with a coarser estimate of the same thing, the
exact loss iTransformer suffered. Why must trend flow coarse→fine? The slow drift is best estimated where
the noise has been averaged out — the coarsest scale — and the finer trends are noisier glimpses of that
same baseline, so the coarse trend is the authority and should inform downward. Flip it and a noisy
fine-scale trend would perturb the reliable slow baseline. Each component's authority sits at the opposite
end of the ladder, which is precisely why one mixing goes up and the other goes down. This is the
disentanglement the single-head linear map could not do, now done jointly along *both* the
season/trend axis and the scale axis. Second, *Future-Multipredictor-Mixing* (the decoder): one linear
predictor per scale maps that scale's encoded representation straight to the full horizon F, and I *sum*
the per-scale forecasts. Concretely that is four linear heads — 96 → 96, 48 → 96, 24 → 96, 12 → 96 — each
reading its own scale's length and emitting the whole pred_len = 96 horizon, then added. This is the
multiscale-prediction move — each scale's predictor contributes its own skill (fine→high-frequency,
coarse→drift) and the ensemble combines them, instead of one head over a merged representation. The sum is
what makes it an ensemble rather than a concatenation: if the coarse head learns the weekly baseline and
the fine head learns the intraday correction, adding their horizons superposes a slow prediction and a
fast one, which is exactly the additive structure decomposition assumed in the first place — the future,
like the past, is trend plus season, so summing per-scale forecasts reconstructs it the same way summing
season and trend reconstructed the window.

Now the channel decision, informed by the whole ladder. iTransformer's cross-variate attention won ECL
but cost ETTh1; the trade was not worth it on the small datasets. TimeMixer's default is
*channel-independence*: treat each channel as its own series through a shared backbone, the same setting
that made the linear control and PatchTST competitive on ETTh1. I keep it. The bet here is different from
iTransformer's — it is that *multiscale disentanglement* generalizes across datasets better than
*cross-variate correlation* does, because every series has structure at multiple scales, while only some
datasets have strong channel coupling. If that bet is right, TimeMixer should recover the ETTh1 ground
iTransformer lost (channel-independent again, plus richer temporal disentangling than even PatchTST)
*and* hold most of the ECL ground (multiscale is still strong there, even without explicit channels).

The reason it can hold ECL ground while ignoring channels is worth stating, because on its face
channel-independence sounds like it should surrender ECL entirely. But ECL's 321 clients each still have
their own multi-scale temporal structure — daily, weekly, seasonal load rhythms — and the linear control
already showed that channel-blind temporal modeling captured most of ECL's forecastable content (it was
PatchTST-competitive there before the inversion). What the channel-blind models left on ECL was the
*cross-client* correlation, and I am not recovering that; what I am adding is a much richer per-client
temporal model than either the linear map or single-scale patches. So the prediction is asymmetric by
construction: on ECL I improve over PatchTST because the temporal modeling per channel got better, but I
cannot reach iTransformer because I still model no channel coupling. The channel-independent path here
runs through `DataEmbedding_wo_pos`, which embeds each scalar series to d_model with the channel folded
into the batch dimension — one shared temporal backbone seeing B·C sequences, the same regularizing reuse
that helped PatchTST on the wide data.

Two smaller structural choices follow from the multiscale layout and I want them deliberate. First, each
scale gets its own reversible instance normalization rather than one shared over the raw window: a
coarse-pooled view has a different variance than the fine view — averaging shrinks the spread — so
normalizing per scale keeps every rung of the ladder on a comparable footing before the mixers see it,
and the denormalization at the end restores the original window's statistics. Second, after the
season/trend streams are recombined at each scale I add them back to that scale's input through a small
feed-forward residual rather than replacing it — the cross-scale mixing should *refine* each scale's
representation, not overwrite it, so a residual keeps the scale's own content and lets the mixing
contribute a correction. Both are the same principle the decomposition taught: preserve the parts that are
already right and let the new machinery adjust rather than clobber.

The edit surface bites hardest at this rung, and I have to be explicit. TimeMixer's whole architecture
depends on the downsampling configuration — `down_sampling_layers`, `down_sampling_window`,
`down_sampling_method` — and on its own training recipe: the script uses `down_sampling_layers=3`,
`window=2`, `avg`, with `d_model=16`, `d_ff=32`, `lr=0.01`, `batch_size=128`, and up to 20 epochs. The
Custom edit gets *none* of that from `configs`: the run.py defaults are `down_sampling_layers=0`,
`down_sampling_window=1`, `down_sampling_method=None`, `d_model=512`, and the loop trains at `lr=1e-4`,
`batch_size=32`, 10 epochs. With `down_sampling_layers=0` the scale ladder collapses to a single scale —
the entire multiscale idea would be inert. So the model *must hardcode its own scale configuration*
internally (3 downsampling layers, window 2, avg pooling) rather than read it from the config; that is
the only way the architecture this rung is about actually exists here. And I can check the collapse the
harness default would cause as a limit, which doubles as a sanity test that my design degenerates
sensibly. Set down_sampling_layers to 0: the ladder has one scale, the season/trend cross-scale mixings
have nothing to mix and become identities, and the multipredictor reduces to a single linear head over one
decomposed representation — which is essentially the decompose-then-linear of the first rung, the linear
control. So the architecture's own degenerate limit is DLinear, exactly the model it is meant to
generalize; that is reassuring rather than alarming, but it means that under the harness default the whole
apparatus would silently reduce to the weakest rung on the ladder, which is why hardcoding three scales is
not a tuning nicety but the difference between this rung existing and not. I also keep `d_model` modest
internally where the method needs it small, but I read the scaffold's value and accept that the fixed
`lr=1e-4` / `batch_size=32` is a real handicap versus the method's `lr=0.01` / `batch_size=128` recipe —
TimeMixer is tuned to train fast at high learning rate, and the loop will not let it. This is the
sharpest "same-named baseline is not the original method" case on the ladder: the algorithm is faithful only
because the model supplies the multiscale config the harness omits, and it runs in a regime its own
recipe would not choose.

Falsifiable expectations against the iTransformer numbers. The headline test is ETTh1: if multiscale
disentanglement generalizes better than cross-variate attention, TimeMixer should *recover* the ETTh1
ground iTransformer lost — I expect it below iTransformer's 0.3950 and below PatchTST's 0.3794, into the
high 0.37s, because it is channel-independent again *and* disentangles temporally more richly than
patches alone. On Weather I expect a clear improvement over both prior rungs (0.1738 / 0.1753), into the
low 0.16s — Weather has genuine multi-scale structure (sub-daily, daily, synoptic) and no strong channel
coupling, the ideal case for this design. ECL is the risk: without explicit cross-variate attention I
expect TimeMixer to give back some of iTransformer's 0.1482 ECL win, landing in the mid-0.15s — still
far better than PatchTST's 0.1819, but not matching iTransformer there. If that pattern holds — ETTh1 and
Weather best-on-ladder, ECL second to iTransformer — it confirms that multiscale disentanglement is the
broader-generalizing lever, and the mean MSE should fall below iTransformer's 0.2395. The arithmetic of
that claim is the reverse of the tug-of-war I just diagnosed: where iTransformer bought its mean drop from
ECL alone and paid it back on ETTh1, I expect this rung to *gain on two of three* — recover most of the
0.0156 ETTh1 regression and improve Weather by a couple of hundredths — while conceding only part of the
ECL win, perhaps 0.007 of it. If ETTh1 comes back to the high 0.37s and Weather reaches the low 0.16s,
those two gains together are worth roughly 0.01 off the mean, more than enough to absorb a small ECL
give-back and still clear 0.2395. So this time the headline should move because the *broad* base improved,
not because one dataset carried it — which is exactly the difference between a lever that generalizes and
one that trades.
