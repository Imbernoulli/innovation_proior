The strategy is the whole point, but it fills a harness, and before I reach for anything clever I want
the simplest fill that could possibly work as the floor to climb from. The pain to start from is just
this: I am handed a tiny random subset of the entries of a matrix and asked for the rest. The first
honest thing to say is that fitting the data is trivial and useless. The driver gives me
`observed_values` ($M^\*$ masked to zero off the observation set), a boolean `observed_mask`, and asks
`recover(...)` for a full $[n,n]$ estimate scored on the *unobserved* entries. The set of matrices that
agree with the observations is enormous — far more unknowns than constraints — and the masked squared
error hits zero on every one of them. In the canonical rank5-100 environment I see $20\%$ of
$100^2 = 10{,}000$ entries, i.e. $2000$ numbers, and I have to produce the other $8000$; I could set
every unobserved entry to zero and have a perfectly valid zero-training-loss "solution" that recovers
nothing. So the question was never "can I fit"; it is "which fitting matrix do I end up at," and that is
entirely a property of the algorithm I write into `recover`, not of the objective.

What makes the answer well-posed is the one assumption the task hands me: $M^\*$ is genuinely low rank.
The driver samples $M^\* = (UV^\top)/\sqrt r$ with $U, V$ Gaussian $[n, r]$ and rescales to
$\|M^\*\|_F = n$, and it even passes me `rank_hint = r`. So I know the good completion is the low-rank
one. The classical move is to *put that in by hand*: relax rank to its convex envelope, the nuclear
norm $\|X\|_\* = \sum_i \sigma_i(X)$, and minimize that subject to fitting the observations. That is a
good thing to do and I will do it on a later rung. But I want to understand the cheaper, stranger
phenomenon first, because the reason this benchmark is interesting is the over-parameterized-networks
puzzle: those generalize *without* any explicit regularizer, and I would like to see that mechanism on
a case I can reason about. So for the floor I deliberately reach for the method that uses *no* explicit
penalty and *no* rank cap. The third option the context lists — explicit low-rank factorization
$X = UV^\top$ with the inner dimension hard-capped at $d = r$ — is tempting, but capping $d = r$ builds
the rank constraint in by hand (the very thing I want to *emerge*) and leans on `rank_hint` being
exactly right, brittle in a way the full-dimensional factorization is not. So the honest floor is the
full-dimensional depth-2 factorization, which imposes no rank cap and lets the dynamics alone recover
whatever low-rank structure they can.

First the boring case, gradient descent directly on the estimate $X$. The objective
$F(X) = \|(X - M^\*)\odot\text{mask}\|_F^2$ is convex with gradient supported only on the observed
entries, so from $X = 0$ every iterate stays in the $m$-dimensional span of the observation-indicator
directions, and the limit is the smallest-Frobenius matrix fitting the data — which is exactly the
impute-zeros matrix, since any nonzero off-support entry only adds to $\|X\|_F$. That is the one
interpolant I least want. Frobenius norm is the wrong complexity measure here, so whatever biases the
search toward low rank must change *which* norm gets implicitly minimized.

The lever is the parameterization. Instead of optimizing $X$, write $X = W_2 W_1^\top$ — a depth-2
product of two square bias-free linear layers — and descend the masked squared error on the factors. I
take the hidden dimension to be the full $n$, so the factorization imposes *no* explicit rank
constraint: $W_2 W_1^\top$ can still represent any $[n,n]$ matrix. And yet the dynamics differ.
Gradient flow on the factors makes the product obey not $\dot X = -\nabla F(X)$ (the flat slide that
gives min-Frobenius) but $\dot X = -\nabla F(X)\,X - X\,\nabla F(X)$ — the same residual direction, now
multiplied by the current $X$ on both sides. That multiplicative factor suppresses the velocity
wherever $X$ is small: start near zero and the flow can barely move in directions $X$ has not already
grown into. This is a self-reinforcing, rich-get-richer growth that prefers a few dominant directions —
low rank — and it is a property of the *flow*, not of a penalty I added.

Which low-rank point? In the case I can solve — commuting measurements, init $\alpha I$, $\alpha\to 0$ —
the flow has the form $X_t = \exp(s_t A)\,X_0\,\exp(s_t A)$, a two-sided power-iteration amplifier; to
reach a finite-scale fit from a vanishing start it must drive $|s_\infty|\to\infty$, at which point
$\exp(s_\infty A)$ collapses onto the top eigenspace of $A$. Reading the limit eigenvalue by eigenvalue
reproduces exactly the complementary-slackness conditions of the minimum-nuclear-norm program — on the
support the rescaled dual has eigenvalue $1$, off it below $1$ — so the limit minimizes the nuclear
norm. The single-entry indicators of matrix completion do *not* commute, so this is a theorem in the
commuting case and a well-motivated conjecture in general; but the mechanism — small init drives an
amplifier that collapses onto the top spectrum — is exactly what I am betting on.

That fixes every ingredient, and the harness exposes a knob for each. Factorize at all, because
descending on $X$ gives impute-zeros; factorizing swaps the implicit norm from Frobenius to nuclear.
Full hidden dimension $d = n$, so the low-rank preference comes entirely from the dynamics rather than a
rank cap I do not want to rely on knowing. Initialization close to zero, because reaching a finite-scale fit
from a vanishing start is what forces the amplifier to collapse onto the top eigenspace — start large
and the depth-2 flow behaves like plain descent on $X$. Small step, because the selecting manifold is
curved (its tangent space is a different subspace at every point), so only near-infinitesimal steps stay
on it.

It is the *end-to-end product* that has to start small, not the factors, and the fan-in scaling is what
arranges that. With `init_scale = 1e-3` and depth $2$, each layer's weight is Gaussian with std
$\texttt{init\_scale}^{1/2}\cdot n^{-1/2}$, so $\text{std}^2 = \texttt{init\_scale}/n$. The end-to-end
entry $(W_2 W_1^\top)_{ij} = \sum_{k=1}^n (W_2)_{ik}(W_1)_{jk}$ is a sum of $n$ independent products of
two mean-zero Gaussians, variance $n\cdot\text{std}^4 = \texttt{init\_scale}^2/n$, so
$\|W_2 W_1^\top\|_F \approx \texttt{init\_scale}\sqrt n$. At $n = 100$ that is $10^{-3}\cdot 10 = 0.01$
against a target $\|M^\*\|_F = 100$ — four orders of magnitude below the answer in relative terms. The
$n^{-1/2}$ fan-in factor cancels the $n$ picked up by the sum, so the product's magnitude is set by
`init_scale` and not inflated by the matrix size, and the vanishing start the amplifier needs is
quantitatively in place.

Mapping onto the scaffold edit: `nn.Sequential` of two `nn.Linear(n, n, bias=False)` layers, Gaussian
init with per-layer std `(init_scale ** 0.5) * (n ** -0.5)` — the `** 0.5` is the $1/\text{depth}$ power
spreading the small overall scale across the two multiplied factors, the `n ** -0.5` the fan-in
normalization — the end-to-end product formed by `_e2e` folding the layers left to right into
$W_2 W_1^\top$, and a plain Adam masked-MSE loop (loss = squared residual on $\Omega$ over the number of
observations) with an early stop at `train_thres = 1e-7`. The full module is in the answer.

Two harness specifics deviate from the textbook flow, and both point the same way. The optimizer is
**Adam** at `lr = 5e-3`, not vanilla small-step gradient descent — Adam's per-coordinate adaptivity is
*not* the gradient flow the implicit-bias theory is built on, and a not-tiny step is a coarse
approximation to the curved-manifold trajectory. And depth-2 is exactly the case the theory says gives
only the *nuclear-norm* surrogate: the per-mode growth rate carries a factor $(\sigma_r^2)^{1-1/N}$,
which at $N=2$ is just $|\sigma_r|$ — a power-law gap between leading and trailing singular values, some
low-rank bias but no saturating cap on the small modes. Eliminating the shared time-factor between a
strong mode and a weak one gives $\sigma_2 = c\,\sigma_1^{\alpha}$ with $0 < \alpha < 1$: as the top
mode grows to fit the data the trailing mode does not stay pinned, it keeps growing, just polynomially
slower, so the reconstruction carries a smear of spurious small-singular-value content rather than a
sharp shoulder. Adam and the $5\times$-larger step only coarsen even that power-law gap. So I *expect*
this floor to under-recover — most diagnostically on rank3-50, where the data cannot be the excuse.

"Where the data cannot be the excuse" I can rank across the three environments: the ratio of
observations to information predicts where the floor holds and where it breaks. A rank-$r$ $n\times n$
matrix has $r(2n - r)$ free parameters, but uniform sampling needs a
$\log$ factor to hit every row and column, so the real completion threshold is $O(nr\log n)$, not $nr$.
rank3-50: $0.30\cdot 2500 = 750$ observed against $nr\log n \approx 3\cdot 50\cdot\ln 50 \approx 587$ —
comfortably oversampled. rank5-100: $2000$ observed against $\approx 2303$ — already sitting right at
the edge. rank10-200: $4000$ observed against $\approx 10600$ — well below threshold, the regime where
*no* method, implicit or convex, can be expected to pin down the subspace. So a poor number on
rank10-200 is the data's fault, but a poor number on rank3-50 is the *method's* fault, and separating
those two is the entire diagnostic value of running the floor.

One thing to keep straight before reading whatever comes back: this is interpolation. With $n^2$
parameters and $m \le 0.3\,n^2$ observations the model has far more capacity than constraints, so the
masked training MSE will reach the `1e-7` early-stop threshold and the `train_mse` column will be
uninformative — a tiny training loss is guaranteed and says nothing about the unobserved entries. The
entire signal is in `test_rel_fro`, governed only by *which* zero-training-loss interpolant the dynamics
selected. So the reading is fixed in advance: if `train_mse` is at the floor but `test_rel_fro` is
large, that is not an optimization failure to fix with more iterations or a different step — it is the
implicit bias delivering the wrong interpolant, and the only cures are a stronger bias or an explicit
penalty — which is where I will have to go if the floor under-recovers even where the samples are
generous.
