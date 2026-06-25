The inversion delivered exactly the split I predicted, and the split is the whole lesson. ECL — the
321-channel laggard — fell from 0.1819 to 0.1482, the largest single-dataset MSE drop on the ladder so
far, with MAE crashing 0.2743 → 0.2398. That is the cross-variate lever paying off precisely where I
said it would. But ETTh1 went the other way: 0.3794 → 0.3950, *worse* than PatchTST, landing essentially
back at the linear control's 0.3962. That is the over-parameterization I called: 7 variate tokens of
dimension 512 with weak channel coupling is all capacity and no structure to fit, and the inversion
threw away the fine intra-series temporal detail that PatchTST's patches preserved. Weather barely moved
(0.1738 → 0.1753, a hair worse). So the mean MSE improved (0.2451 → 0.2395) but the improvement is
entirely ECL; on the small, less-coupled datasets the inversion actively regressed. This is the clearest
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
single-axis, and heavy. Both paradigms cut the signal along *one* axis and call it done.

Let me sit with whether that is the only axis, because one observation keeps nagging. Take a road
sampled every five minutes: a sharp commute spike twice a day. Average that same road to a daily series
and the commute spike is *gone* — what is left is weekday-versus-weekend and holiday structure. Average
to monthly and even that washes out; only a slow seasonal drift remains. It is one physical process, but
it presents a completely different pattern at each sampling scale. That is not a quirk of traffic; it is
what averaging does — an average pool is a low-pass filter, so climbing a downsampling ladder slides the
dominant content monotonically from microscopic detail to macroscopic structure. Decomposition cuts
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
same series. Stage two has two distinct mixings. First, *Past-Decomposable-Mixing* (the encoder): at
each scale, decompose into season and trend with `series_decomp`, then mix *across scales separately for
each component*. Season mixes bottom-up (fine→coarse): the fine-scale seasonal detail informs the
coarser seasonal patterns, because a clean micro-period constrains what the macro-period can be. Trend
mixes top-down (coarse→fine): the coarse-scale trend (the reliable slow drift) informs the finer trends,
because the macro-drift sets the baseline the micro-trends ride on. The asymmetry is the point — the two
components carry information in *opposite directions* across scales, so mixing them with the same
direction (or merging them) would smear exactly the structure I am trying to keep. This is the
disentanglement the single-head linear map could not do, now done jointly along *both* the
season/trend axis and the scale axis. Second, *Future-Multipredictor-Mixing* (the decoder): one linear
predictor per scale maps that scale's encoded representation straight to the full horizon F, and I *sum*
the per-scale forecasts. This is the multiscale-prediction move — each scale's predictor contributes its
own skill (fine→high-frequency, coarse→drift) and the ensemble combines them, instead of one head over a
merged representation.

Now the channel decision, informed by the whole ladder. iTransformer's cross-variate attention won ECL
but cost ETTh1; the trade was not worth it on the small datasets. TimeMixer's default is
*channel-independence*: treat each channel as its own series through a shared backbone, the same setting
that made the linear control and PatchTST competitive on ETTh1. I keep it. The bet here is different from
iTransformer's — it is that *multiscale disentanglement* generalizes across datasets better than
*cross-variate correlation* does, because every series has structure at multiple scales, while only some
datasets have strong channel coupling. If that bet is right, TimeMixer should recover the ETTh1 ground
iTransformer lost (channel-independent again, plus richer temporal disentangling than even PatchTST)
*and* hold most of the ECL ground (multiscale is still strong there, even without explicit channels).

The edit surface bites hardest at this rung, and I have to be explicit. TimeMixer's whole architecture
depends on the downsampling configuration — `down_sampling_layers`, `down_sampling_window`,
`down_sampling_method` — and on its own training recipe: the script uses `down_sampling_layers=3`,
`window=2`, `avg`, with `d_model=16`, `d_ff=32`, `lr=0.01`, `batch_size=128`, and up to 20 epochs. The
Custom edit gets *none* of that from `configs`: the run.py defaults are `down_sampling_layers=0`,
`down_sampling_window=1`, `down_sampling_method=None`, `d_model=512`, and the loop trains at `lr=1e-4`,
`batch_size=32`, 10 epochs. With `down_sampling_layers=0` the scale ladder collapses to a single scale —
the entire multiscale idea would be inert. So the model *must hardcode its own scale configuration*
internally (3 downsampling layers, window 2, avg pooling) rather than read it from the config; that is
the only way the architecture this rung is about actually exists here. I also keep `d_model` modest
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
broader-generalizing lever, and the mean MSE should fall below iTransformer's 0.2395.
