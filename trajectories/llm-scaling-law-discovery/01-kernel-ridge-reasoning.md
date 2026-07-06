The scaffold default predicts the train-target mean for every test point, and before I do anything I want
to know precisely how bad that floor is, because "predict the mean" is the number every later rung has to
beat. Write out its held-out $R^2$. With $\hat y_i = \bar y_{\text{train}}$ for every test point,
$R^2 = 1 - \sum_i (y_i - \bar y_{\text{train}})^2 / \sum_i (y_i - \bar y_{\text{test}})^2$, where both
sums run over the test set. Split the numerator by inserting $\pm\bar y_{\text{test}}$:
$\sum_i (y_i - \bar y_{\text{train}})^2 = \sum_i (y_i - \bar y_{\text{test}})^2 +
2(\bar y_{\text{test}} - \bar y_{\text{train}})\sum_i (y_i - \bar y_{\text{test}}) +
n(\bar y_{\text{test}} - \bar y_{\text{train}})^2$. The middle term vanishes because deviations about the
test mean sum to zero, so $R^2 = 1 - [\,SS_{\text{tot}} + n\,\Delta^2\,]/SS_{\text{tot}} =
-\,n\Delta^2/SS_{\text{tot}}$ with $\Delta = \bar y_{\text{test}} - \bar y_{\text{train}}$. The floor is
not "$R^2 = 0$" — it is strictly negative by exactly the squared train/test mean gap measured in units of
test variance, and it can only be zero if the two means coincide. But the held-out split is deliberately
the *larger and denser* configurations, where loss is systematically lower than in the cheap training
runs, so $\Delta \neq 0$ by construction and the trivial baseline is already off-distribution-penalized.
That sharpens the question: I am not trying to beat zero, I am trying to explain the test variance
$SS_{\text{tot}}$ itself, and any method that does not track the systematic drift of the mean across the
split starts below the floor.

So the question for the first real rung is not "what is the cleverest scaling law," it is "what is the
cheapest thing that uses the descriptors at all and that I trust not to have secretly baked in the
answer." I have three families of runs in front of me, and I have not yet committed to believing any
particular functional form. Before I hand-craft a Chinchilla-style power law I want to know what a model
with *no* law in it — a pure flexible regressor on the engineered features — can do, because that tells me
how much of the variance is just smooth structure in the inputs versus how much genuinely requires the
right asymptotic form. The first rung should be the black box: a model-free regressor that bends to the
data without my imposing any prior about floors or power-law tails.

So let me derive what the cheapest trustworthy flexible regressor is, from the bottom. I have training
pairs $(x_t, y_t)$ with the inputs in $\mathbb{R}^n$ and real targets, and I want a function that predicts
well under squared error. The cheapest thing I know is a straight line through the cloud: choose $w$
minimizing $\sum_t (y_t - w\cdot x_t)^2$, predict $w\cdot x$, closed form $w = (X'X)^{-1}X'y$. Two things
bite immediately on these data. The loss surfaces I am fitting are *curved* — loss versus $\log N$ or
versus $\log D$ is a decaying power, not a line — so a function linear in $x$ cannot bend the way I need.
And even the linear fit is fragile: writing $X = UDV'$, the estimator's variance scales like
$\sum_j d_j^{-2} v_j v_j'$, so any direction with a tiny singular value gets enormous, sign-unstable
coefficients, and if two descriptors are collinear — which they are here — some $d_j$ collapses, $X'X$ is
singular, and the estimator does not even exist. This is not hypothetical for these column sets. On
`sld-lrbsz` the numeric columns are `[lr, bsz, data_size, non_embedding_param_size]`, and in any
compute-scaled grid the token budget and the parameter count are swept together, so $\log D$ and $\log N$
are nearly proportional across the training runs and their columns are near-duplicates. On
`sld-dataconstrained` the columns are `[unique_tokens, params, tokens]` with $D \ge U$ by definition, and
in the fresh-data region $D \approx U$, so `tokens` and `unique_tokens` collapse onto one direction. A
rigidity problem and a stability problem at once, and the stability one is already fatal for a raw
least-squares fit.

It helps to make the collinearity concrete so I know the cure is real and not cosmetic. Suppose two
standardized log-columns, $\log N$ and $\log D$, have empirical correlation $r$ across the training runs;
then $X'X$ restricted to that pair is proportional to the $2\times 2$ matrix with unit diagonal and
off-diagonal $r$, whose eigenvalues are $1+r$ and $1-r$, so the small singular value is $d_{\min}^2 \propto 1-r$ and the
least-squares variance in that direction blows up like $1/(1-r)$. On these compute-scaled grids $r$ is not
$0.5$, it is more like $0.98$, so $1-r = 0.02$ and the variance in the $\log N - \log D$ contrast is fifty
times what it would be for an orthogonal design — the coefficients on $N$ and $D$ become huge and
oppositely signed, cancelling on the training points and then diverging on any test point that breaks the
grid's $N$–$D$ proportionality. That is exactly the extrapolation region. So the instability is not a
rounding artifact I can ignore; it is the dominant failure mode of a raw linear fit on this data, and it is
what the penalty has to tame.

The stability problem has a clean cure I should take first. Add a penalty on the size of $w$: minimize
$a\lVert w\rVert^2 + \sum_t (y_t - w\cdot x_t)^2$ for a fixed positive $a$. Setting the gradient to zero
gives $(X'X + aI)w = X'y$, so $w = (X'X + aI)^{-1}X'y$. Now $X'X$ is positive semidefinite and $aI$ is
strictly positive definite, so their sum is positive definite and always invertible — the singular case is
gone — and in the eigenbasis each coordinate is scaled by $d_j^2/(d_j^2 + a)$, near 1 for the big
directions and near 0 for the unstable tiny ones, trading a little bias for a large drop in variance.
That is ridge, and it fixes stability. But the rigidity is untouched — this is still linear in $x$.

The textbook route to nonlinearity is to stop feeding the raw $x$ and feed a richer set of derived
features $\phi(x)$ — products, powers, basis bumps — and run ridge on $\phi$ instead:
$w = (\Phi'\Phi + aI)^{-1}\Phi'y$, predict $w\cdot\phi(x)$. And here is the wall. If $\phi$ maps an
$n$-dimensional input to degree-$d$ monomials its dimension is on the order of $n^d$ — thousands to
millions of coordinates — and the matrix $\Phi'\Phi + aI$ lives in feature-dimension by feature-dimension;
forming and inverting it is hopeless, and the Gaussian bump $\exp(-\gamma\lVert x-y\rVert^2)$ I really want
has an *infinite* feature map, so the primal solution is not even a finite object. I need the same
regularized fit expressed so its cost depends on the number of examples $T$, not the dimension of $\phi$.
With only a few hundred runs per group, $T$ is tiny.

Stare at the ridge solution for the identity that moves the inverse. For rectangular $P$, $Q$ and a
positive scalar $a$, $(PQ + aI)P = PQP + aP = P(QP + aI)$; if both shifted matrices are invertible,
left-multiply by $(PQ + aI)^{-1}$ and right-multiply by $(QP + aI)^{-1}$ to get $(PQ + aI)^{-1}P =
P(QP + aI)^{-1}$. Set $P = \Phi'$, $Q = \Phi$: then $(\Phi'\Phi + aI)^{-1}\Phi' = \Phi'(\Phi\Phi' +
aI)^{-1}$, so $w = \Phi'(\Phi\Phi' + aI)^{-1}y$. The inverse is now $T\times T$ — its $(s,t)$ entry is
$\phi(x_s)\cdot\phi(x_t)$, one number per pair of training examples — no matter how huge or infinite the
feature space. And $w = \sum_t \alpha_t \phi(x_t)$ with $\alpha = (\Phi\Phi' + aI)^{-1}y$ is automatically
a combination of the training feature vectors: any component of $w$ orthogonal to every $\phi(x_t)$
contributes nothing to any training prediction but adds to $\lVert w\rVert^2$, so the penalty kills it. The
fit is finite-dimensional in disguise. Predicting at a new $x$, $w\cdot\phi(x) = \sum_t \alpha_t
(\phi(x_t)\cdot\phi(x))$ — $\phi$ appears *only* as an inner product. So I never need $\phi$ itself, only a
function $K(u,v) = \phi(u)\cdot\phi(v)$. With the $T\times T$ kernel matrix $K_{s,t} = K(x_s, x_t)$ and the
vector $k(x)_t = K(x_t, x)$, the coefficients solve $(K + aI)c = y$ and the prediction is $\sum_t c_t
K(x_t, x)$. That is the whole method: one $T\times T$ kernel matrix and one regularized solve, the feature
dimension never entering the cost.

Why squared loss specifically? Because squaring is exactly what makes the optimality condition *linear* in
the unknowns, so the dual condition is the linear system $(K + aI)c = y$ with a one-shot closed form; an
$\varepsilon$-insensitive or margin loss would turn this into a quadratic program with no closed form and
an iterative solver. For a few hundred examples I would much rather pay a dense coefficient vector and get
an exact, instant linear solve. The regularizer $a$ is the interpolation-to-smoothing dial: $a\to 0$
demands $f(x_t) = y_t$ exactly, pure interpolation through noisy points with coefficients exploding
wherever $K$ is near-singular; lifting every eigenvalue of $K$ by $a$ makes $K + aI$ strictly positive
definite (a stable Cholesky exists) and shrinks the fit toward the prior-mean function. I can put a number
on how much conditioning that buys. The RBF kernel has $K(x,x) = 1$ on the diagonal, so $\operatorname{tr}
K = T$ and every eigenvalue lies in $(0, T]$; near-duplicate runs — which the collinear grids guarantee —
produce near-identical rows and drive the smallest eigenvalues toward zero, so the raw $K$ is badly
conditioned. Adding $a = 0.05$ floors every eigenvalue at $0.05$, capping the condition number at
$T/0.05$; for $T$ in the low hundreds that is a few thousand, a safe Cholesky, and $0.05$ is small enough
to sit well below the bulk eigenvalues so it damps noise without erasing signal. Read the Bayesian way,
$a$ is the noise-to-prior ratio — how noisy I believe the data is relative to how large I believe the
function is — and the closed form is exactly the Gaussian-process posterior mean.

Now which kernel. I want genuine nonlinearity, a guaranteed valid inner product (symmetric, positive
semidefinite, so $K + aI$ is positive definite and every step above holds), and as few shape knobs as
possible. The Gaussian radial basis function $K(x,y) = \exp(-\gamma\lVert x-y\rVert^2)$ is positive
definite for any $\gamma > 0$, is universal (its infinite feature map can approximate any smooth
function), and has exactly one knob $\gamma$ — the bandwidth: large $\gamma$ narrows each bump so a
training point influences only its neighborhood (wiggly, can overfit), small $\gamma$ broadens it so every
point influences everywhere (smooth, can underfit). Its exponent argument $\gamma\lVert x-y\rVert^2$ grows
like the number of features for standardized inputs, so to keep it order 1 regardless of dimension I
default $\gamma = 1/n_{\text{features}}$. The arithmetic behind that default is worth doing: with each
feature standardized to unit variance and roughly independent, $\mathbb{E}\lVert x - x_t\rVert^2 =
\sum_j \mathbb{E}(x_j - x_{t,j})^2 \approx 2\,n_{\text{features}}$ (each coordinate difference has variance
about 2), so $\gamma\,\mathbb{E}\lVert x - x_t\rVert^2 \approx 2$ with $\gamma = 1/n_{\text{features}}$ —
order 1, exactly what I want. Otherwise, with $\gamma$ fixed as the dimension grows, the exponent scales
with $n_{\text{features}}$, the kernel saturates to the identity (every off-diagonal $\to 0$), and the fit
learns nothing but the diagonal. I can count the dimensions per family to see what $\gamma$ actually is:
`sld-vocab` has three numerics, so raw plus $\log(1+\cdot)$ gives six numeric columns, plus a one-hot of
the `group`; `sld-lrbsz` has four numerics, so eight numeric columns plus the one-hot; `sld-dataconstrained`
three, so six plus the one-hot. With a handful of groups that puts $\gamma$ near $1/10$ to $1/15$ — a
genuinely order-1 exponent against typical inter-point distances, not a saturated kernel.

Because the RBF uses Euclidean distance it is scale-sensitive: a descriptor measured in billions of
parameters and one measured in a learning rate of $10^{-3}$ would let the big-magnitude axis dictate every
kernel value, so before kernelizing I have to put the numeric features on comparable footing. That last
point is the one place this rung's *task* implementation diverges from the bare kernel-ridge recipe, and it
is worth stating exactly because it is what I actually fill into the scaffold. These descriptors span
enormous dynamic ranges — `num_characters` in the billions, `vocab_size` in the thousands, `lr` near
$10^{-3}$ — and the relationships are power-law, so the natural representation is *logarithmic*. So I build
a mixed feature map: standardized *raw* numerics and standardized $\log(1+\cdot)$ numerics (the log
captures the power-law geometry, since a power law is a straight line in log-space and equal ratios become
equal distances, and the raw captures any residual linear trend), concatenated with a one-hot encoding of
the categorical `group`. The one-hot is what lets the single shared regressor still distinguish families
inside the kernel — two runs from different groups differ by $\sqrt{2}$ in the one-hot block, an additive
constant in $\lVert x - x_t\rVert^2$ that pushes their RBF similarity down by a fixed factor, so
cross-group pairs are systematically less similar than within-group pairs and the kernel respects group
membership without my fitting a separate model per group. I take a small $a = 0.05$ and the
dimension-normalized $\gamma$. Fit is one solve $(K + aI)c = y$; predict is the cross-kernel against the
stored training inputs times $c$. Crucially the target is used *raw* — no log transform on $y$ — because
the vocab target is a unigram-normalised loss that can be negative, and I refuse to special-case it; a
single black-box path that works whether the target is signed or not.

The encoder has to survive real data, so a few of its mechanics are load-bearing rather than decorative. I
use $\log(1+\cdot)$ rather than $\log(\cdot)$ so a descriptor value of zero maps to zero instead of
$-\infty$, and I clip negatives to zero before the log so a stray non-physical entry cannot produce a NaN
that poisons the whole kernel row; missing numeric entries are filled with the column median before either
transform, because a single NaN in $\lVert x - x_t\rVert^2$ would make every kernel value against that
point undefined. I standardize each column by its own mean and standard deviation and floor that standard
deviation at a tiny constant, so a descriptor that happens to be constant within a family (its std zero)
divides by one instead of blowing up. The point of all this is that the RBF is a function of a single
scalar distance, and one pathological coordinate contaminates that distance for every pair — so the
encoder's defensive steps are what let the "one distance, one solve" simplicity actually hold on the raw
files.

Before I commit, it is worth being explicit about why kernel ridge and not one of the neighboring
model-free choices, because the point of this rung is a *clean* probe of flexibility, and the wrong probe
would muddy the diagnosis. Nearest-neighbor regression is even more purely local, but it is discontinuous
and its extrapolation is just "copy the nearest training label," which needs the very same standardized
metric I am building for the kernel and buys nothing extra. A full Gaussian process would give me the same
posterior-mean point estimate plus a calibrated variance, but I do not need calibrated uncertainty here, I
need one number per test point, and kernel ridge *is* that posterior mean without the extra machinery. A
small neural network would extrapolate differently — a ReLU net continues its boundary slope rather than
collapsing — but it is non-deterministic across initializations, has many knobs, and is unstable to train
on a few hundred points, whereas the task wants a deterministic fit at seed 42. What I want from this first
rung is the *cleanest* possible reading of what flexibility alone buys off the training hull, and for that
the right instrument is the learner whose off-hull behavior is the most transparent and the most forced by
its construction — the pure-locality one, whose kernel values I can literally watch decay to zero as the
query leaves the fit, so that whatever the metric shows can be attributed to locality and nothing else. So
the honest first probe is that one: kernel ridge on the RBF, whose failure mode is unambiguous and whose
reading will tell me, without confound, whether the gap that remains is flexibility or form.

So what do I expect this floor to do, and why is it the *weakest* real rung rather than a contender? Reason
about the one thing a smooth interpolator structurally cannot do: extrapolate past the convex hull of its
training inputs. Trace it concretely. A held-out configuration that is larger and denser than anything in
the fit sits, after standardization, at something like $+2$ to $+3$ standard deviations along the log-scale
axes it grows on. Its squared distance to the bulk of the training points (which sit near the origin) picks
up roughly $(2.5)^2$ per shifted axis on top of the ambient $\approx 2\,n_{\text{features}}$, so with
$\gamma \approx 1/12$ the exponent to a typical training point is order 2 to 3 and $K$ is already down to
$e^{-2}\approx 0.13$ or less; to the few boundary training points that share the shift, $K$ is larger, so
the prediction $\sum_t c_t K(x_t, x)$ is a shrunken blend of a handful of boundary coefficients. Push the
query further out and *every* $k(x)_t \to 0$; since kernel ridge carries no intercept, the prediction
$\sum_t c_t K(x_t, x) \to 0$. The model collapses toward zero, with no notion that loss should approach an
irreducible floor along a power-law tail. It has learned the *interior* smoothly and has nothing to say
about the *boundary*, which is the only thing being tested.

That collapse-toward-zero is a specific, falsifiable behavior, and it lets me predict the *ordering* of
the three families even before I see a metric. Where the test region is nearest the training hull and the
target itself lives near zero, collapse-to-zero is a small absolute error; where the test region is far
outside the hull and the target is order 1 to 3, it is a large one. The vocab target is a
unigram-normalised loss that hovers near zero and can be negative, and vocab is the most saturated,
most nearly-additive family whose test points sit closest to the training hull — so predicting near zero
there is the *least* wrong. The lrbsz and dataconstrained targets are ordinary positive losses of order
one or more, and their test regions push further out, so collapse-to-zero should be badly wrong there,
possibly catastrophically on whichever family's test points sit furthest outside the fitted hull. I cannot
attach numbers to that yet — the metric table will tell me the magnitudes, and I especially cannot
predict how a family with a *tight* target spread (an lm-loss basin, where near-optimal runs cluster) will
punish even a small absolute error through the variance-normalized metrics — but the *ranking* (vocab
least bad, the other two worse) is a genuine prediction of the collapse mechanism, and the mechanism, not
the ranking, is the payload.

There is a second, quieter prediction I can make about the shape of the evidence, and it matters because it
tells me the metric I should be reading is the *gap* rather than either number alone. With $a = 0.05$ small
relative to the bulk kernel eigenvalues, the in-sample fit will be close to interpolation: on the training
runs $K + aI$ nearly reproduces $y$, so the train-split $R^2$ should sit high, near one. If I only looked at
that I would conclude the black box had "learned the law." The held-out split is where the story inverts —
the same near-interpolating fit collapses off the hull — so a large train-to-test $R^2$ gap is the exact
fingerprint of "memorized the interior, no asymptotics for the boundary," as opposed to a genuinely
underpowered model that is mediocre everywhere. That distinction is what licenses the conclusion I am after:
if the test failure came with a *low* train $R^2$ I would suspect I simply had not given the regressor
enough flexibility, and the fix would be a richer kernel; but a high-train, negative-test signature says the
flexibility was ample and the missing thing is structure that carries into the extrapolation region. So I
will read the two splits together, and I expect the gap, not the raw test number, to be the cleanest
diagnostic.

A power law extrapolates because its asymptotics are baked into the form; an RBF kernel has no asymptotics,
only locality. So I expect this rung to look reasonable in-sample and to fail hard out-of-sample exactly
where the test points sit. That failure is the whole point of running it: it will tell me that the missing
ingredient is not flexibility, it is the *right asymptotic form*, and that the next rung has to stop being
model-free and start imposing the power-law/floor structure the literature laws carry. The full scaffold
module is in the answer; here the conclusion is that the cheapest trustworthy regressor is kernel ridge on
log+raw+one-hot features, and that its predicted failure mode — collapse off the training hull — is exactly
the gap a symbolic law is built to close.
