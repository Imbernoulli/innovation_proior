The scaffold default predicts the train-target mean for every test point, and that number is the floor
every later step has to beat, so I want its held-out $R^2$ exactly. With $\hat y_i = \bar y_{\text{train}}$,
$R^2 = 1 - \sum_i (y_i - \bar y_{\text{train}})^2 / \sum_i (y_i - \bar y_{\text{test}})^2$, both sums over
the test set. Insert $\pm\bar y_{\text{test}}$ in the numerator; the cross term vanishes because deviations
about the test mean sum to zero, leaving $R^2 = -\,n\Delta^2/SS_{\text{tot}}$ with
$\Delta = \bar y_{\text{test}} - \bar y_{\text{train}}$. So the floor is not "$R^2 = 0$" but strictly
negative by the squared train/test mean gap in units of test variance. And the held-out split is
deliberately the *larger and denser* configurations, where loss is systematically lower than in the cheap
training runs, so $\Delta \neq 0$ by construction. I am not trying to beat zero; I am trying to explain the
test variance $SS_{\text{tot}}$ itself, and any method that does not track the systematic drift of the mean
across the split starts below the floor.

So the first real question is not "what is the cleverest scaling law," it is "what is the cheapest thing
that uses the descriptors at all and that I trust not to have secretly baked in the answer." Before I
hand-craft a Chinchilla-style power law I want to know what a model with *no* law in it — a pure flexible
regressor on the engineered features — can do, because that separates how much of the variance is smooth
structure in the inputs from how much genuinely requires the right asymptotic form. So the first probe is
the black box: a model-free regressor that bends to the data without imposing any prior about floors or
power-law tails.

Start from the bottom. Training pairs $(x_t, y_t)$, inputs in $\mathbb{R}^n$, real targets, squared error.
The cheapest predictor is a line: $w$ minimizing $\sum_t (y_t - w\cdot x_t)^2$, closed form
$w = (X'X)^{-1}X'y$. Two things bite. The loss surfaces are *curved* — loss versus $\log N$ or $\log D$ is
a decaying power, not a line — so a function linear in $x$ cannot bend the way I need. And the linear fit
is fragile: writing $X = UDV'$, the estimator's variance scales like $\sum_j d_j^{-2} v_j v_j'$, so any
direction with a tiny singular value gets enormous, sign-unstable coefficients, and if two descriptors are
collinear some $d_j$ collapses, $X'X$ is singular, and the estimator does not even exist. That is not
hypothetical here. On `sld-lrbsz` the columns are `[lr, bsz, data_size, non_embedding_param_size]`, and in
any compute-scaled grid the token budget and parameter count are swept together, so $\log D$ and $\log N$
are near-proportional and their columns near-duplicate; on `sld-dataconstrained` the columns are
`[unique_tokens, params, tokens]` with $D \ge U$, and in the fresh-data region $D \approx U$, so `tokens`
and `unique_tokens` collapse onto one direction. With correlation near $0.98$ the least-squares variance in
their contrast blows up like $1/(1-r)$, coefficients coming out huge and oppositely signed, cancelling on
the training grid and diverging on any test point that breaks its $N$–$D$ proportionality — exactly the
extrapolation region. So instability, not rounding, is the dominant failure of a raw linear fit here.

The stability cure comes first: penalize the size of $w$, minimizing $a\lVert w\rVert^2 + \sum_t (y_t -
w\cdot x_t)^2$. The gradient gives $w = (X'X + aI)^{-1}X'y$; $X'X + aI$ is positive definite and always
invertible — the singular case is gone — and in the eigenbasis each coordinate is scaled by
$d_j^2/(d_j^2 + a)$, near 1 for the big directions and near 0 for the unstable tiny ones, trading a little
bias for a large drop in variance. That is ridge, and it fixes stability. But it is still linear in $x$, so
the rigidity is untouched.

The route to nonlinearity is to feed a richer set of derived features $\phi(x)$ — products, powers, basis
bumps — and run ridge on $\phi$: $w = (\Phi'\Phi + aI)^{-1}\Phi'y$. Here is the wall: degree-$d$ monomials
of an $n$-dimensional input give on the order of $n^d$ coordinates — thousands to millions — and the
Gaussian bump $\exp(-\gamma\lVert x-y\rVert^2)$ I really want has an *infinite* feature map, so the primal
solution is not even a finite object. I need the same regularized fit expressed so its cost depends on the
number of examples $T$, not the dimension of $\phi$ — and with a few hundred runs per group, $T$ is tiny.
The push-through identity moves the inverse: for rectangular $P, Q$ and positive scalar $a$,
$(PQ + aI)P = PQP + aP = P(QP + aI)$, so $(PQ + aI)^{-1}P = P(QP + aI)^{-1}$. With $P = \Phi'$, $Q = \Phi$,
$w = \Phi'(\Phi\Phi' + aI)^{-1}y$: the inverse is now $T\times T$, its $(s,t)$ entry $\phi(x_s)\cdot\phi(x_t)$,
one number per pair of training examples, no matter how huge the feature space. And $w = \sum_t \alpha_t
\phi(x_t)$ is automatically a combination of training feature vectors — any component orthogonal to every
$\phi(x_t)$ adds to $\lVert w\rVert^2$ without changing a training prediction, so the penalty kills it.
Predicting at a new $x$, $w\cdot\phi(x) = \sum_t \alpha_t (\phi(x_t)\cdot\phi(x))$ — $\phi$ appears *only*
as an inner product, so I never need $\phi$ itself, only $K(u,v) = \phi(u)\cdot\phi(v)$. With the kernel
matrix $K_{s,t} = K(x_s, x_t)$ the coefficients solve $(K + aI)c = y$ and the prediction is $\sum_t c_t
K(x_t, x)$: one $T\times T$ solve, the feature dimension never entering the cost.

Squared loss is what makes the dual condition *linear* — $(K + aI)c = y$ with a one-shot closed form; an
$\varepsilon$-insensitive or margin loss would turn this into a quadratic program with an iterative solver,
and for a few hundred examples I would much rather pay a dense coefficient vector for an exact instant
solve. The regularizer $a$ is the interpolation-to-smoothing dial: $a\to 0$ demands $f(x_t) = y_t$ exactly,
coefficients exploding wherever $K$ is near-singular; lifting every eigenvalue of $K$ by $a$ makes
$K + aI$ strictly positive definite and shrinks the fit toward the prior-mean function. The RBF kernel has
$K(x,x) = 1$, so $\operatorname{tr} K = T$ and every eigenvalue lies in $(0, T]$; near-duplicate runs drive
the smallest toward zero, so raw $K$ is badly conditioned, and $a = 0.05$ floors every eigenvalue there,
capping the condition number at $T/0.05$ — a few thousand for $T$ in the low hundreds, a safe Cholesky, and
small enough to sit below the bulk eigenvalues so it damps noise without erasing signal. Read the Bayesian
way, $a$ is the noise-to-prior ratio, and the closed form is exactly the Gaussian-process posterior mean.

Which kernel: I want genuine nonlinearity, a valid (symmetric, positive semidefinite) inner product, and as
few shape knobs as possible. The Gaussian RBF $K(x,y) = \exp(-\gamma\lVert x-y\rVert^2)$ is positive
definite for any $\gamma > 0$, is universal, and has one knob — the bandwidth: large $\gamma$ narrows each
bump so a point influences only its neighborhood (wiggly, overfits), small $\gamma$ broadens it (smooth,
underfits). Its exponent $\gamma\lVert x-y\rVert^2$ grows like the number of features for standardized
inputs, so to keep it order 1 regardless of dimension I set $\gamma = 1/n_{\text{features}}$: with each
feature standardized to unit variance and roughly independent, $\mathbb{E}\lVert x - x_t\rVert^2 \approx
2\,n_{\text{features}}$ (each coordinate difference has variance about 2), so $\gamma\,\mathbb{E}\lVert x -
x_t\rVert^2 \approx 2$. With $\gamma$ fixed as dimension grows the exponent scales up, the kernel saturates
to the identity (every off-diagonal $\to 0$), and the fit learns nothing but the diagonal. With raw plus
$\log(1+\cdot)$ numerics and a one-hot of `group`, each family lands around ten to fifteen features, so
$\gamma$ near $1/10$–$1/15$.

Because the RBF uses Euclidean distance it is scale-sensitive — a descriptor in billions of parameters and
one near $10^{-3}$ would let the big-magnitude axis dictate every kernel value — so I standardize before
kernelizing, and because these descriptors span enormous dynamic ranges and the relationships are
power-law, the natural representation is *logarithmic*. So the feature map is mixed: standardized *raw*
numerics and standardized $\log(1+\cdot)$ numerics (the log captures the power-law geometry, since equal
ratios become equal distances; the raw catches any residual linear trend), concatenated with a one-hot of
`group`. The one-hot lets the single shared regressor still distinguish families inside the kernel: two
runs from different groups differ by $\sqrt{2}$ in the one-hot block, an additive constant in
$\lVert x - x_t\rVert^2$ that pushes their RBF similarity down by a fixed factor, so cross-group pairs are
systematically less similar than within-group pairs without my fitting a separate model per group. I take
$a = 0.05$ and the dimension-normalized $\gamma$. The target is used *raw* — no log on $y$ — because the
vocab target is a unigram-normalised loss that can go negative, and one black-box path should work whether
the target is signed or not.

A few encoder mechanics are load-bearing because the RBF is a function of a single scalar distance and one
pathological coordinate contaminates that distance for every pair. I use $\log(1+\cdot)$ so a zero maps to
zero instead of $-\infty$, and clip negatives to zero before the log so a stray non-physical entry cannot
produce a NaN that poisons the whole kernel row; missing numerics are filled with the column median before
either transform; each column is standardized by its own mean and std with the std floored at a tiny
constant, so a descriptor that is constant within a family divides by one instead of blowing up.

Kernel ridge is the right instrument for this first probe over nearest-neighbours, a full Gaussian process,
or a small net: deterministic at seed 42, no uncertainty machinery I will not use, and the most transparent
off-hull behaviour of the lot — I can watch its kernel values decay to zero as the query leaves the fit, so
whatever the metric shows is attributable to locality and nothing else. That makes it the cleanest reading
of what flexibility alone buys off the training hull.

So what do I expect, and why is this the *weakest* real probe rather than a contender? A smooth interpolator
structurally cannot extrapolate past the convex hull of its training inputs. A held-out configuration larger
and denser than anything in the fit sits, after standardization, at $+2$ to $+3$ standard deviations along
the log axes it grows on; its squared distance to the training bulk picks up roughly $(2.5)^2$ per shifted
axis on top of the ambient $\approx 2\,n_{\text{features}}$, so with $\gamma \approx 1/12$ the exponent to a
typical training point is order 2 to 3 and $K$ is already down to $e^{-2}\approx 0.13$ or less. Push the
query further out and *every* $k(x)_t \to 0$; since kernel ridge carries no intercept, $\sum_t c_t K(x_t, x)
\to 0$ — the prediction collapses toward zero, with no notion that loss should approach an irreducible floor
along a power-law tail. It has learned the *interior* smoothly and has nothing to say about the *boundary*,
which is the only thing tested.

That collapse lets me call the *ordering* of the three families directionally, though not the numbers. Where
the test region is near the hull and the target lives near zero, collapse-to-zero is a small absolute error;
where the test region is far outside the hull and the target is order one or more, it is a large one. The
vocab target is a unigram-normalised loss hovering near zero on the most saturated family, whose test points
sit closest to the hull — predicting near zero there is *least* wrong. The lrbsz and dataconstrained targets
are ordinary positive losses of order one or more with test regions further out, so collapse-to-zero should
be badly wrong there. I cannot attach magnitudes — they depend on each family's target spread, and a family
whose near-optimal runs cluster tightly could punish even a small absolute error through the
variance-normalized metrics — but the direction (vocab least bad, the other two worse) follows from the
mechanism.

The signal to read is the *gap*, not either split alone. With $a = 0.05$ small relative to the bulk kernel
eigenvalues the in-sample fit is near interpolation, so train $R^2$ sits high near one while the same fit
collapses off the hull on test. A high-train, negative-test signature is the fingerprint of "memorized the
interior, no asymptotics for the boundary" — distinct from a genuinely underpowered model mediocre
everywhere. That distinction is what licenses the conclusion: a *low* train $R^2$ would say the regressor
lacked flexibility and wanted a richer kernel; high-train/negative-test says flexibility was ample and the
missing thing is structure that carries into the extrapolation region. So this probe should look reasonable
in-sample and fail hard out-of-sample exactly where the test points sit — and that failure is the point,
because it says the missing ingredient is not flexibility but the *right asymptotic form*, and the next step
has to impose the power-law/floor structure the literature laws carry. The full scaffold module is in the
answer.
