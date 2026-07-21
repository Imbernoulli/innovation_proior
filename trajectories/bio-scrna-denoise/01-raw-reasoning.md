I want to denoise a single-cell count matrix, but before I write anything clever I have to know what
the floor looks like — what score I get for doing nothing. "Return the input unchanged" feels too
trivial to be a method, but the scoring here makes it load-bearing: every metric is normalized so raw
counts sit at exactly zero and the true rate at exactly one, `normalized = (raw − method) / (raw −
perfect)`, one number per metric, headline their mean. If I never run the raw baseline through the
harness I have no independent check that this fraction behaves — that the two anchors bracket the
methods, that `raw − perfect` has the sign I think it does, that a genuine improvement comes out
positive rather than negative because of a flipped comparison somewhere in the metric code. So the
first rung is not a denoiser at all; it is a calibration of the evaluator, and I read the feedback as
a wiring diagram rather than a score.

The evaluator simulates a smooth true rate `Λ`, draws `X ~ Poisson(Λ)`, then splits each count by a
binomial thinning: `X_train ~ Binomial(X, 0.5)`, `X_test = X − X_train`. Thinning is what manufactures
a ground truth out of data that has none. If `X ~ Poisson(λ)` and I keep each molecule independently
with probability `p`, the kept count is `Poisson(pλ)`, the discarded count `Poisson((1−p)λ)`, and — the
load-bearing part — the two are *independent*. So at `p = 0.5` the split hands me two independent
half-depth draws of the same latent rate, each `Poisson(Λ/2)`. That independence is what turns
`X_test` into an honest held-out target: a denoiser that learned the shared rate from `X_train`
predicts `X_test`, but nothing it did to fit `X_train`'s own sampling noise transfers. Holding out
whole cells instead would test generalization across cells, not removal of within-cell noise — a
denoiser could ace it while passing every count's noise through untouched — and a deeply-sequenced
"truth" version of these cells does not exist here. Thinning splits the molecules, so both halves share
the same cells and the same latent rate, and the only way to predict one from the other is to recover
the shared rate and discard the split noise. That also tells me the identity is the exact zero of the
natural scale: it recovers none of the shared rate and keeps all of the split noise.

The two metrics fail the identity in different ways. The log-normalized MSE library-normalizes each
cell, applies `log1p`, and averages the squared difference between the log-normalized test half and
the denoised output. When the output *is* `X_train`, I am comparing two independent `Poisson(Λ/2)`
draws through the same nonlinearity, and `log1p` is not gentle at the low counts that dominate here:
on an ordinary entry where the rate is a fraction of a molecule, one half reads `0` and the other `1`,
and after the rescale to a `1e4` budget those two land far apart. The squared disagreement, averaged
over the matrix, is a pure measurement of the split's sampling noise — no signal about denoising,
because nothing was denoised — and it is the largest this term can be for any sane predictor, since
every later method replaces at least one noisy draw with something smoother and pulls the two
log-normalized profiles together. The identity refuses to move off the raw integers, so it eats the
full penalty; the smooth truth sits between the two noisy reads and is scored against by both.

The Poisson term fails from the likelihood side. The metric rescales the denoised matrix to the test
budget, treats it as a rate `λ`, and scores `mean(λ − X_test·log λ)`. Predicting `X_train` is the
maximally over-dispersed thing I can do — every bit of the training half's noise handed straight into
`λ`. Holding one entry at true half-rate `λ` with both halves `Poisson(λ)`, the expected NLL of
predicting the noisy half is `λ − λ·E[log X_train]` versus `λ − λ·log λ` for predicting the rate, and
since `log` is concave, Jensen gives `E[log X_train] < log λ`: the raw predictor loses on this term
for a structural reason, not by accident. The gap is worst on dropout entries: when `X_train = 0`,
common in a ~59%-zero matrix, `λ` collapses to the `eps = 1e-8` floor, and if the independent test
half fires there, `−X_test·log(eps)` is a large penalty — exactly the holes a real denoiser fills. The
rescale by `test.sum()/train.sum()` is a single global factor putting the prediction on the test
budget; the two totals are random and never exactly equal, so it stops a trivial budget mismatch from
dominating the shape term.

I could calibrate with a global-mean or per-gene-mean baseline instead, but their scores would be some
non-zero number that entangles two questions I need kept separate — is the scale wired right, and is
this particular pooling any good. The identity is the one baseline whose correct answer I know exactly
in advance, because it is the anchor the normalization is built from: `raw − method` is identically
zero on every entry, so both normalized terms must return a clean `0.0000` on the tune seed and all
three held-out seeds. Any deviation is unambiguously a harness bug — a stray transform applied to the
method but not the anchor, an off-by-one in which matrix feeds which metric — and I would rather find
it here, where I know the answer must be zero, than three rungs up trying to read a hundredth of a
point off an unverified scale.

One subtlety fixes what "perfect" means: the `1` is not raw `Λ` but `Λ` scaled to each cell's observed
test depth. A denoiser only sees the training half and cannot know the test half's realized molecule
count, so the fairest possible competitor is the smooth truth rescaled to that same budget. Even this
competitor does not reach noiseless perfection — the held-out counts still scatter Poisson-wise, and
the multiplicative biological over-dispersion means `Λ` is not itself the noiseless mean a smoother
could recover — so the perfect anchor's metric values are strictly better than the identity's but
strictly worse than zero-error. That is the structural reason every rung lands below `1`, and a method
scoring above `1` would signal a mis-scaled anchor, not a triumph. The fixed MCV split seed of `42`
freezes the partition, so any difference I read between two rungs is the method talking, not a lucky
split.

In bias-variance terms the two anchors are the two extremes. The identity is pure variance — zero
bias, since its expectation over the split noise is the true half-rate itself, but it carries the full
Poisson variance of a single draw straight into its prediction. The perfect anchor is the opposite:
smooth `Λ` scaled to depth, zero estimation variance, paying only the irreducible over-dispersion plus
the held-out draw's own Poisson spread. Every method on the ladder sits between them by accepting a
little bias — a little smoothing that pulls a cell toward its neighbours and away from its own draw —
for a large cut in variance, and the normalized score reads how favourably it made that trade. So I
expect the first crude pooling to capture most of the variance and jump most of the way up, with later
rungs fighting over the diminishing remainder.

What this rung cannot do is the point of naming it: it pools no information across cells, treating each
noisy profile as its own final answer and inheriting every Poisson fluctuation. Yet the premise of
denoising scRNA-seq is that cells in the same biological state are independent noisy measurements of one
underlying rate, so averaging similar cells beats the sampling noise down while the shared signal
survives. The crudest honest way to start is to find, for each cell, the handful of cells most like it
and average them together — a nearest-neighbour smooth — which is the next rung.
