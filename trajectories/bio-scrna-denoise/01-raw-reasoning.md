I want to denoise a single-cell count matrix, but before I write anything clever I have to know
what the floor looks like — what score I get for doing *nothing*. The temptation is to skip this,
because "return the input unchanged" feels too trivial to be a method. But the scoring here is
relative in a way that makes the trivial thing load-bearing: every metric is normalized so that the
raw counts sit at exactly zero and the true rate sits at exactly one. The exact statement is
`normalized = (raw − method) / (raw − perfect)`, one such number per metric, and the headline is
their mean. If I never run the raw baseline through the harness, I have no independent check that
this fraction is behaving — that the two anchors actually bracket the methods, that the denominator
`raw − perfect` has the sign I think it does, that a genuine improvement comes out positive rather
than negative because of a flipped comparison somewhere in the metric code. So the first rung is not
really a denoiser at all. It is a calibration of the evaluator, and the honest thing is to build it
as such and read the feedback as a wiring diagram rather than as a score.

Let me be precise about what the evaluator is doing, because the whole ladder inherits its meaning
from this step. It simulates a smooth true rate matrix `Λ`, draws integer counts `X ~ Poisson(Λ)`,
and then performs a molecular cross-validation split: it thins each count into two halves by a
binomial draw, `X_train ~ Binomial(X, 0.5)` and `X_test = X − X_train`. The reason this split is the
right way to manufacture a ground truth out of data that has none is the thinning property of the
Poisson: if `X ~ Poisson(λ)` and I keep each of its molecules independently with probability `p`,
the kept count is `Poisson(pλ)`, the discarded count is `Poisson((1−p)λ)`, and — the part that makes
it usable — the two are *independent*. So at `p = 0.5` the split hands me two statistically
independent half-depth Poisson draws of the very same latent rate, `X_train` and `X_test` each
distributed `Poisson(Λ/2)`. That independence is what turns `X_test` into an honest held-out target:
whatever a denoiser learned from `X_train` about the shared rate, it could not have seen `X_test`,
and yet both are measurements of the same `Λ/2`. A good denoiser is one whose reconstruction from the
training half predicts the test half; the only way to do that is to recover the smooth rate the two
halves have in common rather than the sampling noise that distinguishes them.

It is worth pausing on why the evaluation is built this way at all, because the protocol is doing
something clever that constrains what a good denoiser even means. On real tissue there is no clean
reference — the whole difficulty of the problem is that I never observe `Λ`, only one noisy `X`. The
obvious alternatives to molecular cross-validation each fail for a concrete reason. I could hold out a
subset of *cells* and ask a method to predict them, but that tests whether the method generalizes
across cells, not whether it removes within-cell sampling noise, and a denoiser can ace cell
hold-out by memorizing the manifold while still passing every count's noise through untouched. I
could compare against a deeply-sequenced "truth" version of the same cells, but no such thing exists
in this setup and manufacturing one would just push the ground-truth problem back a step. Thinning
sidesteps both: it splits the *molecules* of each cell, so `X_train` and `X_test` share the exact same
cells and the exact same latent rate, differing only by which of the shared Poisson mean's molecules
fell into which half. That means the only way to predict `X_test` from `X_train` is to recover the
shared rate and discard the split noise — which is precisely the denoising objective, made
self-supervised without ever naming `Λ`. Understanding that is what tells me the identity is not just
weak but is the exact zero of the natural scale: it recovers none of the shared rate and keeps all of
the split noise, which is the definition of doing nothing.

That framing already tells me why the identity is the floor, and it is worth walking the two metrics
one at a time because the two failure modes of "predicting the noisy half as if it were the rate"
are different in the two terms. Take the log-normalized MSE first. The metric library-size-normalizes
each cell to a common budget, applies `log1p`, and averages the squared difference between the
log-normalized test half and the log-normalized denoised output. When the denoised output *is*
`X_train`, I am comparing `log1p` of the normalized training half against `log1p` of the normalized
test half — two independent `Poisson(Λ/2)` draws pushed through the same nonlinearity. On any entry
where the rate is small, which is most entries here, the two halves land on different small integers:
a gene the rate says should fire around half a molecule per half-cell shows up as, say, a 0 in one
half and a 1 in the other, and in log-normalized space those are not close. The squared disagreement,
averaged over the whole matrix, is therefore a *pure measurement of the sampling noise* — it contains
no signal about how well anything was denoised, because nothing was. It is the largest this term can
be for any sane predictor, because every later method will replace at least one of those two noisy
draws with something smoother and pull the two log-normalized profiles closer together. I expect this
raw MSE to land somewhere around one and a half in the harness's units; the exact number is the
evaluator's to report, and its size is precisely the height of the wall the ladder has to climb.

It helps to make that concrete on a single entry, because the log-normalization is not a gentle
transform at the low counts that dominate here. Take a half-depth cell whose molecules sum to about
450, so the metric's rescaling to a target of `1e4` multiplies every count by roughly `22`. A raw
count of `0` maps to `log1p(0) = 0`; a count of `1` maps to `log1p(22.2) ≈ 3.14`; a count of `2`
maps to `log1p(44.4) ≈ 3.82`. So on an entry where the training half read a `1` and the independent
test half read a `0` — an utterly ordinary Poisson fluctuation when the underlying half-rate is a
fraction of a molecule — the identity pays a squared penalty of about `3.14² ≈ 9.9` in this term,
and it pays something like it on a large share of the matrix. Now put the smooth truth in the middle:
if the true half-rate on that entry is around `0.5` molecules, it log-normalizes to `≈ 2.49`, sitting
between the two noisy reads. Scoring the test `0` against that smooth `2.49` costs `≈ 6.2`, and
scoring the training `1` against it costs only `≈ 0.42`. That is the whole game in one entry: the two
noisy halves are far apart because each carries its own draw's noise, and any predictor that moves off
the raw integers toward the shared smooth rate collapses the distance. The identity refuses to move,
so it eats the full `9.9`; the ladder's job is to earn back that difference across the matrix.

The Poisson term fails for a related but distinct reason, and this one I want to reason about
carefully because it is the more subtle of the two. The metric rescales the denoised matrix to the
test molecule budget, treats it as a rate `λ`, and scores the negative Poisson log-likelihood of the
held-out counts under that rate, `mean(λ − X_test·log λ)`. Predicting `X_train` as the rate is the
maximally over-dispersed thing I can do: I am handing every bit of the training half's sampling noise
straight into `λ` and asking it to explain a second, independent noisy draw. The likelihood punishes
this through the `−X_test·log λ` term. To see the direction cleanly, hold one entry with true
half-rate `λ` and let both halves be `Poisson(λ)`; the expected NLL of predicting the noisy half is
`E[X_train] − E[X_test]·E[log X_train] = λ − λ·E[log X_train]`, whereas predicting the true rate gives
`λ − λ·log λ`. Because `log` is concave, Jensen's inequality makes `E[log X_train] < log λ`, so the
raw predictor's expected NLL is strictly larger than the perfect predictor's — the identity loses on
this term for a structural reason, not by accident. And the gap is worst exactly on the dropout
entries: when `X_train = 0`, which happens on a large fraction of a matrix that is ~59% zeros, the
predicted rate collapses to the `eps` floor, and if the independent test half happens to fire there,
`−X_test·log(eps)` is a large penalty. Those entries are where a real denoiser earns its keep, by
filling a dropout hole with what the manifold says should be there rather than with a hard zero. Two
details in this term matter for reading it honestly. The metric does not treat the denoised matrix as
a rate on its own scale; it first rescales by `test.sum() / train.sum()`, a single global factor that
puts the prediction on the test half's molecule budget. At `p = 0.5` that factor is close to one in
expectation because both halves inherit the same depth, but the two totals are themselves random and
never exactly equal, so the rescale is what keeps the NLL from being dominated by a trivial budget
mismatch rather than by how well the shape of the rate matches. And the `eps = 1e-8` floor under `λ`
is the only thing standing between the identity and a catastrophic penalty on every entry where
`X_train = 0` but `X_test > 0`; because those dropout-versus-fire entries are common in a matrix that
is more than half zeros, the raw NLL is in large part a sum over how badly the identity handles
exactly the holes a real denoiser is supposed to fill. When
I checked the sign of this argument on a scratch Poisson toy it held up robustly — the raw predictor's
NLL sat well above the true-rate predictor's at every half-rate I tried, and the gap grew as the rate
fell into the dropout regime — but I am not going to pretend the toy predicts the harness's magnitude,
because the real `Λ` is heavily skewed and the aggregate depends on that skew. I expect the raw NLL to
come out somewhere near one and a half to two in the harness's units, sitting above a perfect anchor
that is markedly lower; the feedback table is what settles the actual number.

Before I commit to the identity as the calibration, I should ask whether some other cheap baseline
would calibrate the scale better, because the choice is not as forced as it looks. Three candidates
present themselves. I could return the global mean profile — every cell replaced by the average cell
— which is a legitimate denoiser of a sort and would give me a non-trivial score to look at. I could
return the per-gene mean, holding each gene at its matrix-wide average count. Or I could return the
input unchanged. The first two are tempting because they *do* something, and a rung that does
something feels more informative than one that does not. But that is exactly why they are the wrong
calibration: their scores would be some non-zero number that entangles two questions I need to keep
separate — "is the evaluator's scale wired correctly" and "is this particular pooling any good." If
the global-mean baseline came back at, say, `0.3`, I would have no way to know whether `0.3` means the
scale is honest and mean-pooling is mediocre, or the scale is subtly broken and the two errors
happen to land near `0.3`. The identity is the one baseline whose *correct* answer I know in advance
with certainty, precisely because it is the anchor the normalization is built from: it must be exactly
zero, so any deviation is unambiguously a harness bug rather than a property of the method. Calibration
wants a predictor whose output I can check against a known-exact value, and only the identity gives me
that. The global-mean and per-gene-mean baselines are real denoisers I could rank later if I wanted;
here they would just be noise in the instrument I am trying to trust. So the identity it is, chosen
not because it is the least effort but because it is the only choice that isolates the scale.

Now the calibration itself, which is the entire deliverable of this rung. The normalized score is
`(raw − method) / (raw − perfect)`. The `perfect` anchor is the true rate `Λ` scaled to each cell's
observed depth — the smooth thing the counts were actually drawn from, which is the best any denoiser
aiming at the rate could possibly produce. When my method matrix *is* the raw training matrix, the
numerator `raw − method` is identically zero on every entry of every dataset, not approximately but
exactly, because I am literally subtracting the anchor from itself. So both normalized terms must come
back as a clean `0.0000`, and their mean must be `0.0000`, on the tune seed and on all three held-out
seeds. This is the degenerate check that makes the rung worth running: if any entry of that column is
not exactly zero, the bug is in the harness — a stray transform applied to the method but not the
anchor, an off-by-one in which matrix feeds which metric — and I would vastly rather find it here,
where I know the answer must be zero, than three rungs up where I am trying to read a hundredth of a
point of real improvement off a scale I have not verified.

One subtlety in the top anchor is worth stating because it fixes what "perfect" even means on this
axis. The `1` is not the raw true rate `Λ`; it is `Λ` scaled to each cell's *observed test depth*. That
choice is what makes the ceiling a fair target rather than an unreachable one. The test half was drawn
at some realized budget, and a denoiser only ever sees the training half — it cannot know the exact
number of molecules the test half happened to capture. So the fairest possible competitor is the
smooth truth rescaled to that same realized budget, and the metric anchors `1` there. The consequence
I care about is that even this competitor does not reach a hypothetical noiseless perfection: the
held-out counts still scatter Poisson-wise around their mean, and the over-dispersion means `Λ`
carries multiplicative biological spread that no rescaled-rate predictor can absorb, so the perfect
anchor's raw metric values are strictly better than the identity's but strictly worse than zero-error.
That is the structural reason every rung will land below `1`, and it is also why I want the fixed MCV
split seed of `42`: because the partition is frozen, every method on the ladder is scored against the
identical train/test division and the identical realized depths, so any difference I read between two
rungs is the method talking, not a lucky split. Calibrating now means locking in that the `0` and the
`1` are both real, both fair, and both stable across the seeds I will report on.

There are two more properties of the bracket I want the feedback to confirm while I am at it, because
they fix the geometry of the whole `0`-to-`1` axis. First, the denominator `raw − perfect` has to be
strictly positive on both metrics — the raw anchor must score *worse* than the perfect anchor — or the
normalization would be dividing by something with the wrong sign and every later method's score would
be meaningless or inverted. Confirming the perfect anchor sits below the raw anchor on both terms is
what tells me a *lower* raw score really is worse and the fraction points the way I think it does.
Second, the two anchors have to actually bound the interesting region: the perfect anchor is the
`1`, and I expect no denoiser to exceed it, because the true rate carries a multiplicative biological
over-dispersion on top of its low-rank structure, so `Λ` itself is not the noiseless mean of anything
a smoother could recover — there is irreducible spread that even the perfect predictor pays for in
both metrics. That is why I expect every method on this ladder to live strictly between `0` and `1`
rather than punching through the top. A method scoring above `1` would not be a triumph; it would be a
sign that the perfect anchor is mis-scaled. So the three things I am really testing are: exact zero on
the numerator, correct sign on the denominator, and a perfect anchor that genuinely upper-bounds the
achievable. If all three hold, the scale is trustworthy and I can spend the rest of the ladder reading
small differences against it with confidence.

There is a cleaner way to see what the two anchors are and why the methods must live between them, in
the language of bias and variance. The identity is a pure-variance estimator: it has zero bias, since
its expectation over the split noise is the true half-rate itself, but it carries the full Poisson
variance of a single draw straight into its prediction, and both metrics are dominated by that
variance. The perfect anchor is the opposite extreme: it is the smooth `Λ` scaled to depth, so it has
zero *estimation* variance, and the only error it pays is the irreducible one — the multiplicative
biological over-dispersion that means `Λ` is not itself the noiseless generator the counts collapse
to, plus the pure Poisson spread of the held-out draw around its mean. Every method on this ladder is
going to sit between these two by making the same trade: accept a little bias — a little smoothing that
pulls a cell toward its neighbours and away from its own exact draw — in exchange for a large cut in
variance. The normalized score is essentially a readout of how favourably a method made that trade,
with `0` marking "took no variance reduction at all" and `1` marking "drove estimation variance to
zero." Reading the ladder this way tells me what to expect qualitatively before any of it runs: the
first method that pools even crudely should capture a large chunk of the variance and jump most of the
way up, and the later rungs will be fighting over the diminishing remainder — the bias they introduce
while chasing the last of the variance, and the parts of the two metrics that respond to different
kinds of smoothing. That is a hypothesis about the shape of the climb, not a result; the feedback
tables are what will confirm or break it.

What this rung cannot do is the whole point of naming it. The identity pools no information across
cells whatsoever — it treats each cell's noisy profile as its own final answer and inherits every
Poisson fluctuation. Yet the entire premise of denoising scRNA-seq is that cells in the same
biological state are independent noisy measurements of one underlying rate, so that averaging similar
cells beats the sampling noise down by a factor that grows with how many genuine neighbours I can pool,
while the shared signal survives the averaging untouched. The identity does none of that pooling, which
is exactly why the whole distance from zero to one is still open in front of me and every bit of it has
to be bought by borrowing strength between cells. The crudest honest way to start borrowing is to find,
for each cell, the handful of cells most like it and average them together — a nearest-neighbour smooth
— and that is the next rung.
