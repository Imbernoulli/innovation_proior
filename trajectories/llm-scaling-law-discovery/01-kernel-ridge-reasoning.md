The scaffold default predicts the train-target mean for every test point, which is the floor by
construction — it ignores every descriptor and every group, so on the held-out split it lands at the
trivial-mean baseline, an $R^2$ at or below zero on a distribution shift, with `MAE`/`RMSE` equal to the
test target's own spread. So the question for the first real rung is not "what is the cleverest scaling
law," it is "what is the cheapest thing that uses the descriptors at all and that I trust not to have
secretly baked in the answer." I have three families of runs in front of me, and I have not yet
committed to believing any particular functional form. Before I hand-craft a Chinchilla-style power law I
want to know what a model with *no* law in it — a pure flexible regressor on the engineered features —
can do, because that tells me how much of the variance is just smooth structure in the inputs versus how
much genuinely requires the right asymptotic form. The first rung should be the black box: a
model-free regressor that bends to the data without my imposing any prior about floors or power-law
tails.

So let me derive what the cheapest trustworthy flexible regressor is, from the bottom. I have training
pairs $(x_t, y_t)$ with the inputs in $\mathbb{R}^n$ and real targets, and I want a function that
predicts well under squared error. The cheapest thing I know is a straight line through the cloud:
choose $w$ minimizing $\sum_t (y_t - w\cdot x_t)^2$, predict $w\cdot x$, closed form
$w = (X'X)^{-1}X'y$. Two things bite immediately on these data. The loss surfaces I am fitting are
*curved* — loss versus $\log N$ or versus $\log D$ is a decaying power, not a line — so a function
linear in $x$ cannot bend the way I need. And even the linear fit is fragile: writing $X = UDV'$, the
estimator's variance scales like $\sum_j d_j^{-2} v_j v_j'$, so any direction with a tiny singular value
gets enormous, sign-unstable coefficients, and if two descriptors are collinear — which they are here,
because in these grids `params` and `tokens` move together — some $d_j$ collapses, $X'X$ is singular,
and the estimator does not even exist. A rigidity problem and a stability problem at once.

The stability problem has a clean cure I should take first. Add a penalty on the size of $w$: minimize
$a\lVert w\rVert^2 + \sum_t (y_t - w\cdot x_t)^2$ for a fixed positive $a$. Setting the gradient to zero
gives $(X'X + aI)w = X'y$, so $w = (X'X + aI)^{-1}X'y$. Now $X'X$ is positive semidefinite and $aI$ is
strictly positive definite, so their sum is positive definite and always invertible — the singular case
is gone — and in the eigenbasis each coordinate is scaled by $d_j^2/(d_j^2 + a)$, near 1 for the big
directions and near 0 for the unstable tiny ones, trading a little bias for a large drop in variance.
That is ridge, and it fixes stability. But the rigidity is untouched — this is still linear in $x$.

The textbook route to nonlinearity is to stop feeding the raw $x$ and feed a richer set of derived
features $\phi(x)$ — products, powers, basis bumps — and run ridge on $\phi$ instead: $w = (\Phi'\Phi +
aI)^{-1}\Phi'y$, predict $w\cdot\phi(x)$. And here is the wall. If $\phi$ maps an $n$-dimensional input
to degree-$d$ monomials its dimension is on the order of $n^d$ — thousands to millions of coordinates —
and the matrix $\Phi'\Phi + aI$ lives in feature-dimension by feature-dimension; forming and inverting
it is hopeless, and the Gaussian bump $\exp(-\gamma\lVert x-y\rVert^2)$ I really want has an *infinite*
feature map, so the primal solution is not even a finite object. I need the same regularized fit
expressed so its cost depends on the number of examples $T$, not the dimension of $\phi$. With only a few
hundred runs per group, $T$ is tiny.

Stare at the ridge solution for the identity that moves the inverse. For rectangular $P$, $Q$ and a
positive scalar $a$, $(PQ + aI)P = PQP + aP = P(QP + aI)$; if both shifted matrices are invertible,
left-multiply by $(PQ + aI)^{-1}$ and right-multiply by $(QP + aI)^{-1}$ to get
$(PQ + aI)^{-1}P = P(QP + aI)^{-1}$. Set $P = \Phi'$, $Q = \Phi$: then $(\Phi'\Phi + aI)^{-1}\Phi' =
\Phi'(\Phi\Phi' + aI)^{-1}$, so $w = \Phi'(\Phi\Phi' + aI)^{-1}y$. The inverse is now $T\times T$ — its
$(s,t)$ entry is $\phi(x_s)\cdot\phi(x_t)$, one number per pair of training examples — no matter how huge
or infinite the feature space. And $w = \sum_t \alpha_t \phi(x_t)$ with $\alpha = (\Phi\Phi' + aI)^{-1}y$
is automatically a combination of the training feature vectors: any component of $w$ orthogonal to every
$\phi(x_t)$ contributes nothing to any training prediction but adds to $\lVert w\rVert^2$, so the penalty
kills it. The fit is finite-dimensional in disguise. Predicting at a new $x$, $w\cdot\phi(x) = \sum_t
\alpha_t (\phi(x_t)\cdot\phi(x))$ — $\phi$ appears *only* as an inner product. So I never need $\phi$
itself, only a function $K(u,v) = \phi(u)\cdot\phi(v)$. With the $T\times T$ kernel matrix $K_{s,t} =
K(x_s, x_t)$ and the vector $k(x)_t = K(x_t, x)$, the coefficients solve $(K + aI)c = y$ and the
prediction is $\sum_t c_t K(x_t, x)$. That is the whole method: one $T\times T$ kernel matrix and one
regularized solve, the feature dimension never entering the cost.

Why squared loss specifically? Because squaring is exactly what makes the optimality condition *linear*
in the unknowns, so the dual condition is the linear system $(K + aI)c = y$ with a one-shot closed form;
an $\varepsilon$-insensitive or margin loss would turn this into a quadratic program with no closed form
and an iterative solver. For a few hundred examples I would much rather pay a dense coefficient vector
and get an exact, instant linear solve. The regularizer $a$ is the interpolation-to-smoothing dial:
$a\to 0$ demands $f(x_t) = y_t$ exactly, pure interpolation through noisy points with coefficients
exploding wherever $K$ is near-singular; lifting every eigenvalue of $K$ by $a$ makes $K + aI$ strictly
positive definite (a stable Cholesky exists) and shrinks the fit toward the prior-mean function. Read
the Bayesian way, $a$ is the noise-to-prior ratio — how noisy I believe the data is relative to how
large I believe the function is — and the closed form is exactly the Gaussian-process posterior mean.

Now which kernel. I want genuine nonlinearity, a guaranteed valid inner product (symmetric, positive
semidefinite, so $K + aI$ is positive definite and every step above holds), and as few shape knobs as
possible. The Gaussian radial basis function $K(x,y) = \exp(-\gamma\lVert x-y\rVert^2)$ is positive
definite for any $\gamma > 0$, is universal (its infinite feature map can approximate any smooth
function), and has exactly one knob $\gamma$ — the bandwidth: large $\gamma$ narrows each bump so a
training point influences only its neighborhood (wiggly, can overfit), small $\gamma$ broadens it so
every point influences everywhere (smooth, can underfit). Its exponent argument $\gamma\lVert
x-y\rVert^2$ grows like the number of features for standardized inputs, so to keep it order 1 regardless
of dimension I default $\gamma = 1/n_{\text{features}}$ — otherwise in high dimension the kernel
saturates to the identity and the fit learns nothing. And because the RBF uses Euclidean distance it is
scale-sensitive: a descriptor measured in billions of parameters and one measured in a learning rate of
$10^{-3}$ would let the big-magnitude axis dictate every kernel value, so before kernelizing I have to
put the numeric features on comparable footing.

That last point is the one place this rung's *task* implementation diverges from the bare kernel-ridge
recipe, and it is worth stating exactly because it is what I actually fill into the scaffold. These
descriptors span enormous dynamic ranges — `num_characters` in the billions, `vocab_size` in the
thousands, `lr` near $10^{-3}$ — and the relationships are power-law, so the natural representation is
*logarithmic*. So I build a mixed feature map: standardized *raw* numerics and standardized
$\log(1+\cdot)$ numerics (the log captures the power-law geometry, the raw captures any residual linear
trend), concatenated with a one-hot encoding of the categorical `group`. The one-hot is what lets the
single shared regressor still distinguish families inside the kernel — two runs from different groups are
pushed apart in feature space, so the RBF similarity respects group membership without my fitting a
separate model per group. I take a small $a = 0.05$ (enough to keep $K + aI$ well-conditioned and damp
noise, not so large it erases the signal) and the dimension-normalized $\gamma$. Fit is one solve
$(K + aI)c = y$; predict is the cross-kernel against the stored training inputs times $c$. Crucially the
target is used *raw* — no log transform on $y$ — because the vocab target is a unigram-normalised loss
that can be negative, and I refuse to special-case it; a single black-box path that works whether the
target is signed or not.

So what do I expect this floor to do, and why is it the *weakest* real rung rather than a contender?
Reason about the one thing a smooth interpolator structurally cannot do: extrapolate past the convex
hull of its training inputs. The RBF kernel value $\exp(-\gamma\lVert x - x_t\rVert^2)$ decays to zero as
the query $x$ moves far from every training point, so for a held-out configuration that is *larger and
denser* than anything in the fit — which is precisely the extrapolation region these tasks are built
from — every $k(x)_t \to 0$, the prediction $\sum_t c_t K(x_t, x) \to 0$, and the model collapses toward
zero (or toward whatever the kernel sum leaves), with no notion that loss should approach an irreducible
floor along a power-law tail. It has learned the *interior* smoothly and has nothing to say about the
*boundary*, which is the only thing being tested. A power law extrapolates because its asymptotics are
baked into the form; an RBF kernel has no asymptotics, only locality. So I expect this rung to look
reasonable in-sample and to fail hard out-of-sample exactly where the test points sit — possibly
catastrophically on a family where the test region is far outside the training hull. That failure is the
whole point of running it: it will tell me that the missing ingredient is not flexibility, it is the
*right asymptotic form*, and that the next rung has to stop being model-free and start imposing the
power-law/floor structure the literature laws carry. If even the vocab family — the most saturated, the
one closest to a clean additive power law — comes out only middling under a flexible black box, that
confirms the descriptors alone are not enough and the law has to be designed. The full scaffold module
is in the answer; here the conclusion is that the cheapest trustworthy regressor is kernel ridge on
log+raw+one-hot features, and that its predicted failure mode — collapse off the training hull — is
exactly the gap a symbolic law is built to close.
