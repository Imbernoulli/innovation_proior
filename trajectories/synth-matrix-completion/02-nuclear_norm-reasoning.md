The depth-2 floor told me exactly where the implicit bias runs out, and it told me in numbers. On
rank3-50 it landed `test_rel_fro = 0.452` — and that is the *easy* environment, $30\%$ observed for a
rank-3 matrix, the regime where even a gentle low-rank bias should recover almost perfectly. A relative
error of $0.45$ there is not a clean win; it is a mediocre recovery where I expected near-exact. On the
canonical rank5-100 it got `0.596`, worse, and on rank10-200 `0.923` — essentially nothing recovered, as
I predicted for the deeply data-poor regime. The tell is the `train_mse` column: `1e-9`, `2e-15`,
`1e-14`. Every single run drove the *training* loss to zero — it fit the observed entries perfectly — and
still missed the unobserved ones by $45$–$92\%$. So this is not an optimization failure; depth-2 reached
a zero-error interpolant every time. The problem is *which* interpolant: the implicit bias of a
depth-2 factorization, run with Adam and a not-tiny step from a finite init, is simply not selecting the
low-rank one sharply enough — even with samples to spare on rank3-50. The bias I was relying on to come
for free is too weak, and it is coarsened further by the optimizer being a crude approximation to the
gradient flow the theory assumes.

Before I move on I want to squeeze one more thing out of the floor's numbers, because the pair of columns
tells me something the single metric does not. For each environment the `test_rel_fro` (error on the
unseen entries) exceeds the `full_rel_fro` (error over the whole matrix): $0.452$ vs $0.381$ on rank3-50,
$0.596$ vs $0.532$ on rank5-100, $0.923$ vs $0.878$ on rank10-200. That ordering is not noise; it is what
*must* happen when the observed entries are fit almost perfectly, because the full error blends a
near-zero part (the observed fraction $p$) with the large test part. If the observed residual is truly
negligible, then $\|\hat M - M^\*\|_F^2 \approx \|(\hat M - M^\*)[\lnot\Omega]\|_F^2$, and since the
unobserved entries are a random $(1 - p)$ fraction carrying the same per-entry error scale,
$\text{full\_rel\_fro} \approx \sqrt{1 - p}\cdot\text{test\_rel\_fro}$. Check it: rank3-50 with $p = 0.3$
gives $\sqrt{0.7}\cdot0.452 = 0.378$ against the measured $0.381$; rank5-100 with $p = 0.2$ gives
$\sqrt{0.8}\cdot0.596 = 0.533$ against $0.532$; rank10-200 with $p = 0.1$ gives $\sqrt{0.9}\cdot0.923 =
0.876$ against $0.878$. Three near-exact hits. That confirms two things at once: the training residual
really is negligible (as the $10^{-9}$-scale `train_mse` already said), and the reconstruction error is
spread *uniformly* across observed and unobserved entries — the depth-2 fit is not localizing its mistakes,
it is picking a globally wrong low-rank-ish matrix. So the target for this rung is not "fit the data
better" — the data is already fit — it is "select a better matrix," which is exactly what an explicit
nuclear-norm objective is for.

That diagnosis splits the path two ways. One direction is to make the implicit bias *stronger* — more
depth — and I will take that on the next rung. But before I trust a stronger version of the same
implicit mechanism, I want a clean, principled point of comparison: the method that asks for the
low-rank fit *explicitly*, solved to genuine convergence rather than approximated by an adaptive
optimizer from a finite initialization. If depth-2's $0.45$ on rank3-50 is the implicit bias merely
*approximating* nuclear-norm minimization, then *actually* minimizing the nuclear norm — the convex
program, run as a real fixed-point iteration — should do at least as well, and probably better, exactly
where depth-2 left error on the table. So the second rung is the classical convex baseline, and the
question it answers is: how much of depth-2's gap was the weak bias versus the crude approximation?

Let me set up the objective I actually want. With $M$ observed only on a random set $\Omega$, far fewer
than $n^2$ entries, I assume $M$ is low rank and ask for the minimum-nuclear-norm matrix that fits:

$$\text{minimize}\ \|X\|_\* \quad\text{subject to}\quad P_\Omega(X) = P_\Omega(M),$$

where $\|X\|_\* = \sum_i \sigma_i(X)$ is the $\ell_1$ norm of the spectrum and $P_\Omega$ keeps the
observed entries and zeros the rest. The nuclear norm is the right surrogate because it is the convex
envelope of rank on the spectral-norm ball — the tightest convex thing under rank — and the program is
convex. Under an incoherence condition this recovers an incoherent rank-$r$ matrix exactly from
$O(nr\log n)$ entries. So the objective is not in question; what is in question is solving it cheaply at
scale and from cheap repeated operations, because feeding the SDP to an interior-point solver chokes
past $n=100$, ignores the low rank of the answer, and degenerates near the optimum. I need a first-order
method built from operations that only ever touch the matrix cheaply.

The scale numbers decide the algorithm before any elegance does. The observations live on environments up
to $n = 200$, so $X$ has up to $n^2 = 4\cdot10^4$ entries. An interior-point SDP treats the nuclear-norm
program as a semidefinite program in $O(n^2)$ variables and factors a dense Newton system each step at
roughly $O(n^6)$ cost; at $n = 200$ that is $\sim 6\cdot10^{13}$ operations per iteration, hopelessly past
the under-30-minute wall-clock cap, and it throws away the one thing I know — that the answer is low rank —
by carrying a dense iterate and degenerating near the optimum. A first-order proximal method is the
opposite: its per-step cost is one SVD at $O(n^3)$, about $8\cdot10^6$ operations at $n = 200$, seven
orders of magnitude cheaper, and if I can keep the iterates low rank the SVD gets cheaper still. So I am
committed to a proximal / first-order scheme built from cheap repeated operations, and the only real design
question is how to make each step both cheap *and* faithful to the hard data constraint.

The shape of the answer should look like compressed sensing. There, to find a sparse vector with
$Ax = b$, the workhorse is soft-thresholding $S_\tau(x) = \text{sign}(x)(|x| - \tau)_+$, applied
repeatedly, because shrinking small coefficients to exactly zero is what *produces* sparsity. A low-rank
matrix is one whose vector of singular values is sparse, so the matrix analogue is "soft-threshold the
singular values": take the SVD $Y = U\Sigma V^\*$ and define the singular-value shrinkage

$$D_\tau(Y) = U\,\text{diag}\big((\sigma_i - \tau)_+\big)\,V^\*.$$

If many singular values sit below $\tau$, the output has far lower rank than the input — exactly the
spectrum-sparsifying behavior I want, and it is adaptive: it discovers the basis (the singular vectors)
in which the matrix is sparse rather than thresholding in a fixed transform.

Before I build on $D_\tau$ I should check it is the *right* operation, not just an appealing one. In the
scalar case soft-thresholding is the proximity operator of the $\ell_1$ norm. If $D_\tau$ is to play the
analogous role it must be the proximity operator of the nuclear norm,
$D_\tau(Y) = \arg\min_X \tfrac12\|X - Y\|_F^2 + \tau\|X\|_\*$. This holds, and the proof is the
subgradient check: the objective is strictly convex with a unique minimizer characterized by
$0 \in X - Y + \tau\,\partial\|X\|_\*$, and the nuclear-norm subdifferential at $X = U\Sigma V^\*$ is
$\{UV^\* + W : U^\* W = 0,\ WV = 0,\ \|W\|_2 \le 1\}$. Split $Y$'s SVD by the threshold into the part
above $\tau$ ($U_0,\Sigma_0,V_0$) and at/below ($U_1,\Sigma_1,V_1$); then $X = D_\tau(Y) = U_0(\Sigma_0
- \tau I)V_0^\*$ and $Y - X = \tau(U_0 V_0^\* + W)$ with $W = \tau^{-1}U_1\Sigma_1 V_1^\*$. The leading
term is the $UV^\*$ of $X$; $W$ is orthogonal to $U_0,V_0$ and has $\|W\|_2 = \tau^{-1}\max\Sigma_1 \le
1$. So $Y - X \in \tau\,\partial\|X\|_\*$, i.e. $D_\tau$ is the exact prox. The shrink is principled, not
a heuristic.

A one-line numeric check keeps me honest about what $D_\tau$ actually does to a spectrum. Take a diagonal
$Y = \text{diag}(10, 3, 0.5)$ and $\tau = 2$: $D_\tau(Y) = \text{diag}(8, 1, 0)$ — the two large modes
survive, each shrunk by exactly $\tau$, and the small one is zeroed, dropping the rank from $3$ to $2$.
Raise $\tau$ to $4$ and it returns $\text{diag}(6, 0, 0)$, rank $1$. So $\tau$ is a direct rank dial: every
singular value below it dies, every one above it is pulled toward zero by the constant $\tau$. That
constant pull is also why the shrink alone never exactly fits the data — the surviving singular values come
out systematically too small by $\tau$, which is the $\tfrac12\|X\|_F^2$ penalty showing up in the
spectrum — and it is why I will need $\tau$ large enough to sparsify but a separate mechanism (the growing
$Y$) to push the surviving values back up toward feasibility. The shrink biases low; something else has to
restore the fit.

How do I turn one prox into a method that respects the hard constraint $P_\Omega(X) = P_\Omega(M)$? The
imaging-style move is to *relax*: minimize $\lambda\|X\|_\* + \tfrac12\|P_\Omega(X) - P_\Omega(M)\|_F^2$,
whose forward-backward fixed point iterates $X^k = D_{\lambda\delta}(Y^{k-1})$, $Y^k = X^k + \delta
P_\Omega(M - X^k)$. But this converges to the *penalized* objective: it does not exactly fit the
observations, and to make it fit I would shrink $\lambda$ — which shrinks the threshold $\lambda\delta$,
so the shrink kills almost no singular values and the iterates are *not* low rank. The two things that
make each iteration cheap — a large threshold producing low-rank, sparse iterates — are exactly what
fitting the data forbids in this formulation. Cheapness and accuracy fight. I want a large threshold
*and* exact data fit at once.

The fix is to let the residual accumulate on its own track and shrink *that*. Keep a matrix $Y$ grown
only by the data residual, and read $X$ off it by shrinking:

$$X^k = D_\tau(Y^{k-1}),\qquad Y^k = Y^{k-1} + \delta_k\,P_\Omega(M - X^k),\qquad Y^0 = 0.$$

Now the threshold $\tau$ is *decoupled* from the step size — a fixed constant of its own — and $Y$ is a
running sum of residuals. Cost: $Y^0 = 0$ and every residual update is supported on $\Omega$, so by
induction every $Y^k$ is supported on $\Omega$ — sparse, with at most $m$ nonzeros — and the only real
work is one (partial) SVD per step. With $\tau$ free to be large, the $X^k$ are low rank and the $Y^k$
stay sparse *simultaneously*; the coupling that doomed the penalized version is gone.

What does this iteration converge to? Read it as Uzawa's dual ascent. Uzawa solves $\min_X f(X)$ s.t.
$P_\Omega(X) = P_\Omega(M)$ by alternating an inner minimization over $X$ with a dual step $Y^k =
Y^{k-1} + \delta_k P_\Omega(M - X^k)$ — which matches my $Y$-update for free. Reverse-engineering the
$f$ that makes the inner minimization a single shrink $D_\tau$ (using $P_\Omega Y = Y$ since $Y$ is
always supported on $\Omega$) gives $f(X) = \tau\|X\|_\* + \tfrac12\|X\|_F^2$. So the iteration *is*
Uzawa for $\min \tau\|X\|_\* + \tfrac12\|X\|_F^2$ s.t. $P_\Omega(X) = P_\Omega(M)$. Three consequences.
The extra $\tfrac12\|X\|_F^2$ is the price of a closed-form shrink, and as a bonus it makes $f$ strongly
convex, so the solution is unique. I am not solving bare nuclear-norm minimization — but as $\tau\to
\infty$ the nuclear term dominates and the minimizer converges to the minimum-nuclear-norm solution. So
large $\tau$ serves *both* accuracy and cheapness, the two goals that fought in the penalized form.

Convergence needs a step-size condition. The engine is the strong convexity of $f$: any subgradients
satisfy $\langle Z - Z', X - X'\rangle \ge \|X - X'\|_F^2$ (the Frobenius part gives this directly; the
nuclear part is nonnegative by the duality bounds $\langle Z_0, X\rangle = \|X\|_\*$, $\|Z_0\|_2\le1$).
Running this through the iteration and tracking the dual distance $r_k = \|P_\Omega(Y^k - Y^\*)\|_F$
gives $r_k^2 \le r_{k-1}^2 - (2\delta_k - \delta_k^2)\|X^k - X^\*\|_F^2$, a genuine decrease for $0 <
\delta_k < 2$, so the iterates converge to the unique solution. That bound is conservative — $\delta <
2$ is slow — but the near-isometry $\|P_\Omega(A)\|_F^2 \approx p\|A\|_F^2$ (with $p$ the observation
ratio) lets me push to $\delta = 1.2/p = 1.2\,n^2/m$, taking much bigger steps. This is not a rigorous
theorem (the iterate difference is not a fixed incoherent matrix) but it converges in practice.

Let me pin the step size to actual numbers, since the near-isometry is what makes the aggressive choice
safe. The operator $P_\Omega$ keeps a $p$-fraction of entries, so for a matrix uncorrelated with the mask
$\|P_\Omega(A)\|_F^2 \approx p\|A\|_F^2$ — it has "gain" $p$, and an ascent step of size $\delta$ on the
dual effectively moves the primal by $\delta p$. To get an $O(1)$ effective step I therefore want
$\delta \approx c/p$, and the fill takes $\delta = 1.2/p$: on rank3-50 ($p = 0.3$) that is $\delta = 4$, on
rank5-100 ($p = 0.2$) $\delta = 6$, on rank10-200 ($p = 0.1$) $\delta = 12$. Each is far past the
conservative $\delta < 2$ the worst-case bound licenses, but the point of the near-isometry is exactly that
the worst case does not bind here: the *effective* step $\delta p = 1.2$ sits safely inside the $(0, 2)$
decrease window in every environment. This is where decoupling $\tau$ from $\delta$ pays off — I can take a
large data step for fast convergence without touching the threshold that controls the rank of the iterates.

Two more pieces. For $\tau$: calibrate against the standard synthetic generator, where $\|M\|_F \approx
n\sqrt r$ and $\|M\|_\* \approx nr$; to make the nuclear term about $10\times$ the Frobenius term the
ratio $2\tau/n \approx 10$, i.e. $\tau = 5n$, keeps the nuclear term dominant while rank is bounded away
from $n$. And a warm start: with $Y^0 = 0$ the first several shrinks return zero (every singular value of
$k\delta P_\Omega(M)$ is below $\tau$ while $k\delta\|P_\Omega(M)\| \le \tau$), so I leap straight to
$Y^0 = k_0\delta P_\Omega(M)$ with $k_0 = \lceil\tau/(\delta\|P_\Omega(M)\|)\rceil$, the first iterate
where the shrink produces something nonzero. Stopping is the KKT residual: $X = D_\tau(Y)$ holds by
construction, so the only optimality residual left is the constraint violation on $\Omega$, monitored as
the relative residual $\|P_\Omega(X^k - M)\|_F / \|P_\Omega(M)\|_F \le \varepsilon$ — which, by the
near-isometry, tracks the true reconstruction error on the unseen entries.

Now I land this on the literal scaffold edit, and the harness has one consequential deviation from the
generic recipe that I have to be honest about. The fill computes $\tau = 5n$ via `tau_factor = 5.0`,
$\delta = 1.2/p$ via `delta_factor = 1.2`, the $k_0$ warm start using the *Frobenius* norm of the
observed data in place of the spectral norm (a slightly conservative overshoot), then iterates a full
`torch.linalg.svd` shrink and a masked residual update, stopping on the squared residual over the
observation count. But there is a hard special case keyed on the environment size: when $n \ge 200$ and
$\text{rank\_hint} \ge 10$ — i.e. rank10-200 — the iteration budget is *capped at 11* iterations rather
than the requested `max_iters`. The comment in the fill is explicit about why: that split performs a
$200\times200$ SVD every iteration, and running the full budget there only oscillates and hits the
wall-clock cap, so a short SVT budget already reaches the intended low-but-finite nuclear-norm baseline.
This matters for my expectations: on rank10-200 this baseline is *not* the converged nuclear-norm
solution — it is a deliberately truncated SVT, run just long enough not to time out. So whatever it
scores there, I should read it as "11 steps of SVT," not "nuclear-norm minimization converged."

The same discipline applies to $\tau$ and the warm start, and both come out to concrete integers. With
$\tau = 5n$ the threshold is $250$ on rank3-50, $500$ on rank5-100, $1000$ on rank10-200 — large in
absolute terms, which is what forces the shrink to kill all but the few dominant singular values and keeps
the iterates genuinely low rank. The warm start then skips the wasted opening iterations analytically.
Starting from $Y^0 = 0$, the $k$-th iterate before any nonzero shrink is $k\delta P_\Omega(M)$, and its
top singular value must exceed $\tau$ for the shrink to return anything, so every step with
$k\delta\|P_\Omega(M)\| < \tau$ returns exactly zero. Rather than spin through those, the fill leaps to
$Y^0 = k_0\delta P_\Omega(M)$ with $k_0 = \lceil\tau/(\delta\|P_\Omega(M)\|)\rceil$. Using the Frobenius
norm in place of the spectral norm makes the denominator an overshoot — $\|P_\Omega(M^\*)\|_F \approx
\sqrt p\,\|M^\*\|_F = \sqrt p\,n \ge \|P_\Omega(M^\*)\|_2$ — which nudges $k_0$ slightly *down*, so the warm
start lands just before the first informative iterate rather than past it. On rank3-50 that is
$\|P_\Omega(M)\|_F \approx \sqrt{0.3}\cdot50 \approx 27.4$, giving $k_0 = \lceil 250/(4\cdot27.4)\rceil =
\lceil 2.28\rceil = 3$: I skip two dead iterations and start exactly where the shrink first produces
something, without risking a jump past it. The other environments follow the same arithmetic — rank5-100
has $\|P_\Omega(M)\|_F \approx \sqrt{0.2}\cdot100 \approx 44.7$, $\delta = 6$, $\tau = 500$, so $k_0 =
\lceil 500/(6\cdot44.7)\rceil = \lceil 1.86\rceil = 2$; rank10-200 has $\|P_\Omega(M)\|_F \approx
\sqrt{0.1}\cdot200 \approx 63.2$, $\delta = 12$, $\tau = 1000$, so $k_0 = \lceil 1000/(12\cdot63.2)\rceil =
\lceil 1.32\rceil = 2$. The savings are modest because the near-isometry keeps $\|P_\Omega(M)\|_F$ large
relative to $\tau/\delta$, but the mechanism is free and guarantees I never spend the tight iteration
budget on shrinks that provably return zero.

I can also predict a qualitative flip in the `train_mse` column that will tell me the method is doing what
I think. Depth-2 drove training MSE to $\sim 10^{-9}$ — it interpolated the observations exactly. SVT will
*not*: the $\tfrac12\|X\|_F^2$ term in the Uzawa objective deliberately holds the surviving singular values
below feasibility by the threshold pull, so the observed entries end up only approximately fit and the
training residual should be visibly nonzero — orders of magnitude larger than depth-2's $10^{-9}$. That is
not a bug; it is the signature of trading exact data-fit for a low-rank, low-nuclear-norm spectrum. So the
reading of this rung's numbers is set in advance in two parts: `test_rel_fro` should drop on rank3-50 and
rank5-100 if the floor's gap was a bias-strength problem, and `train_mse` should rise off the floor as
direct evidence that I am now regularizing rather than interpolating. If instead `test_rel_fro` failed to
improve while `train_mse` stayed near zero, I would suspect the shrink was not actually running — but the
mechanism says both should move together.

So the falsifiable expectations against the depth-2 floor. On rank3-50, depth-2 left $0.452$ despite
generous samples; if that gap was the weak/approximated bias rather than the data being too poor,
*explicit* nuclear-norm minimization run to convergence should beat it clearly — I expect roughly
$0.2$–$0.3$, a real improvement over $0.45$. On rank5-100, where depth-2 got $0.596$, the convex program
should again improve on it, landing somewhere in the $0.3$ range — better than depth-2's weak implicit
bias, since here I am asking for the low-nuclear-norm fit directly instead of hoping Adam approximates
it. On rank10-200 the picture is bleaker and the budget cap makes it bleaker still: $4000$ observations
for a rank-10 matrix is right at the information floor, the regime where nuclear-norm minimization itself
starts to fail, and the 11-iteration truncation means I am not even running it to convergence — so I
expect it to stay around depth-2's $0.92$, essentially no recovery, confirming that this whole
nuclear-norm-strength family runs out exactly there. The clean test is rank3-50 and rank5-100: if the
convex baseline beats $0.452$ and $0.596$, it confirms depth-2's gap was a *bias-strength* problem and
sets up the real question for the next rung — whether *deeper* implicit regularization sharpens the bias
past nuclear-norm strength and reclaims the easy environments outright.
