I want to denoise a single-cell count matrix, but before I write anything clever I have to know
what the floor looks like — what score I get for doing *nothing*. The temptation is to skip this,
because "return the input unchanged" feels too trivial to be a method. But the scoring here is
relative: every metric is normalized so that the raw counts sit at exactly zero and the true rate
sits at exactly one. If I don't actually run the raw baseline through the harness, I won't know
whether the normalization is behaving, whether the two anchors bracket the methods the way they
should, or whether my metric code has a sign error that makes "better" come out negative. So the
first rung is not really a denoiser at all — it is a calibration of the evaluator, and the honest
thing is to treat it as such.

Let me think about what returning `X_train` unchanged actually means under each metric. For the
MSE-in-log-normalized-space term, I am comparing the log-normalized training half against the
log-normalized test half. These are two independent Poisson draws of the same latent rate, each at
half depth, so they disagree on every entry by exactly the sampling noise — a gene that the rate
says should fire at `λ/2` per half shows up as two different small integers in the two halves, and
in log space those small integers are far apart. The raw MSE is therefore large and is a pure
measurement of how noisy the data is. That should be the zero point: any real denoiser has to shrink
that disagreement by borrowing strength across cells, and the normalization will report how much of
the gap from raw to the rate it closes.

For the Poisson term it is the same story from the likelihood side. I rescale the training half to
the test molecule budget and treat it as a rate, then ask how well that rate explains the held-out
counts. Predicting the noisy training half as the rate is the maximally over-dispersed prediction —
it carries every bit of the training sampling noise straight into the rate — so its NLL should be
poor, and again that is the honest floor. The perfect anchor, the true `Λ` scaled to depth, should
have a much better NLL because it is the smooth thing the counts were actually drawn from.

That is the reasoning I expect, but "should" is doing a lot of work in the last two paragraphs, and
the whole point of this rung is to not have to trust it. The strong claim I am leaning on is that
both normalized terms land at *exactly* zero — not approximately — because the numerator
`raw − method` is `raw − raw`. Before I believe that, I want to watch it happen on a matrix small
enough to read. So let me build a six-cell, five-gene toy: draw a true rate `Λ` from a gamma, draw
`X ~ Poisson(Λ)`, binomially thin it into `X_train` and `X_test`, and push everything through the
two metrics by hand. The raw anchor is `X_train` fed through the metric; the perfect anchor is `Λ`
rescaled to the train depth fed through the same metric; the method is the identity, so it is also
`X_train`.

Running it, the raw MSE comes out `0.398389` and the perfect-rate MSE `0.288246`; the raw Poisson
term is `4.940559` and the perfect-rate Poisson term `0.101424`. Two things to read off these. First,
`raw > perfect` on both — `0.398 > 0.288` for MSE and `4.94 > 0.10` for NLL — so the anchors really
do bracket in the direction the normalization assumes, and the denominator `raw − perfect` is
positive; a sign error would have flipped one of these and I would have seen a negative or inverted
bracket here instead of three rungs up. Second, the method MSE is `0.398389` and the method NLL is
`4.940559` — bit-for-bit the raw anchor — so `(raw − method)/(raw − perfect)` evaluates to `0.0`
and `0.0` on the nose for both terms. So the exact-zero claim is not wishful: it is what the
arithmetic does, and it does it for the dull reason that the method matrix *is* the matrix the anchor
is built from.

The one thing that could spoil "exact" is silly and worth ruling out before I trust it: the method
returns `X.astype(np.float64).copy()`, while the harness might build its raw anchor straight from the
integer counts. If `log1p(int_counts/sf)` and `log1p(float_counts/sf)` disagreed in the last bit, the
numerator would be a tiny nonzero number instead of a clean zero, and I would spend an afternoon
chasing a "denoiser" that scores `1e-16`. Checking it directly: the log-normalized integer matrix
and the log-normalized float64 matrix come back `np.array_equal` — identical, max abs difference
`0.0` — so the cast is harmless and the zero stays clean. Good; the dtype in the return is a contract
nicety, not a correctness hazard.

So the floor is `0.0000` on both terms, and now I know it is genuinely the floor and not a near-miss
hiding a bug. If the full harness ever hands back anything other than a clean zero here, the place to
look is the harness — a stray rescale, a copy that isn't, an anchor built from a different split —
because the method has no freedom to produce anything else.

What this rung cannot do is obvious, and naming it is the point. It pools no information across
cells whatsoever. The entire premise of denoising scRNA-seq is that cells in the same biological
state are independent noisy measurements of the same rate, so averaging similar cells beats down the
Poisson noise — and the identity does none of that. The whole distance from zero to one has to be
bought by borrowing strength between cells. The simplest way to borrow strength is to find, for each
cell, the handful of cells most like it and average them together, which is the next rung.
