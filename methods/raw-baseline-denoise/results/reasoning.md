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
measurement of how noisy the data is. That is the right zero point: any real denoiser has to shrink
that disagreement by borrowing strength across cells, and the normalization will report how much of
the gap from raw to the rate it closes.

For the Poisson term it is the same story from the likelihood side. I rescale the training half to
the test molecule budget and treat it as a rate, then ask how well that rate explains the held-out
counts. Predicting the noisy training half as the rate is the maximally over-dispersed prediction —
it carries every bit of the training sampling noise straight into the rate — so its NLL is poor,
and again that is the honest floor. The perfect anchor, the true `Λ` scaled to depth, will have a
much better NLL because it is the smooth thing the counts were actually drawn from.

There is nothing to tune. The denoiser is the identity. I do expect, and want to confirm in the
feedback, that both normalized terms come out at exactly zero on every dataset — not approximately,
exactly — because the "method" matrix is literally the raw matrix the anchor is built from, so the
numerator `raw − method` is identically zero. If it comes back as anything other than a clean zero,
that is a bug in the harness, not in the method, and I would much rather find it now than discover
it three rungs up when I am trying to read a small improvement off a miscalibrated scale.

What this rung cannot do is obvious, and naming it is the point. It pools no information across
cells whatsoever. The entire premise of denoising scRNA-seq is that cells in the same biological
state are independent noisy measurements of the same rate, so averaging similar cells beats down the
Poisson noise — and the identity does none of that. The whole distance from zero to one has to be
bought by borrowing strength between cells. The simplest way to borrow strength is to find, for each
cell, the handful of cells most like it and average them together, which is the next rung.
