The strategy is the whole point, but it fills a harness, and before I reach for anything clever I want
the simplest fill that could possibly work as the floor to climb from. The pain to start from is just
this: I am handed a tiny random subset of the entries of a matrix and asked for the rest. The first
honest thing to say is that fitting the data is trivial and useless. The driver gives me
`observed_values` ($M^\*$ masked to zero off the observation set), a boolean `observed_mask`, and asks
`recover(...)` for a full $[n,n]$ estimate scored on the *unobserved* entries. The set of matrices that
agree with the observations is enormous — $n^2$ unknowns, far fewer than $n^2$ constraints — and the
masked squared error hits zero on every one of them. In the canonical rank5-100 environment I see
$20\%$ of $100^2 = 10{,}000$ entries, i.e. $2000$ numbers, and I have to produce the other $8000$. I
could set every unobserved entry to zero and have a perfectly valid zero-training-loss "solution" that
recovers nothing. So the question was never "can I fit"; it is "which fitting matrix do I end up at,"
and that is entirely a property of the algorithm I write into `recover`, not of the objective.

What makes the answer well-posed is the one assumption the task hands me: $M^\*$ is genuinely low rank.
The driver samples $M^\* = (UV^\top)/\sqrt r$ with $U, V$ Gaussian $[n, r]$ and rescales to
$\|M^\*\|_F = n$, and it even passes me `rank_hint = r$. So I know the good completion is the low-rank
one. The classical move is to *put that in by hand*: relax rank to its convex envelope, the nuclear
norm $\|X\|_\* = \sum_i \sigma_i(X)$, and minimize that subject to fitting the observations — the
Recht-Fazel-Parrilo / Candès-Recht program. That is a perfectly good thing to do and I will do it on a
later rung. But I want to understand the cheaper, stranger phenomenon first, because the whole reason
this benchmark is interesting is the over-parameterized-networks puzzle: those generalize *without* any
explicit regularizer, and I would like to see that mechanism on a case I can reason about. So for the
first rung I deliberately reach for the method that uses *no* explicit penalty and *no* rank cap, and I
let the floor tell me how far that gets me.

Let me make sure I understand the boring case first, gradient descent directly on the estimate $X$. The
objective is $F(X) = \|(X - M^\*)\odot\text{mask}\|_F^2$, convex, with gradient supported only on the
observed entries. Every gradient step is a linear combination of the observation-indicator directions,
so from $X = 0$ every iterate stays in that $m$-dimensional span, and the limit is the point of that
span that fits the data. Which point? It is the minimum-Frobenius-norm solution — among all matrices
fitting the observations, the one with the smallest $\|X\|_F$. And that is exactly the wrong answer: the
minimum-Frobenius completion of a masked matrix is the impute-zeros matrix. Frobenius norm is the wrong
complexity measure for this problem. So whatever I do to bias the search toward low rank, descending on
$X$ directly will not do it — I need to change *which* norm gets implicitly minimized.

The lever that changes it is the parameterization. Instead of optimizing $X$, write $X = W_2 W_1^\top$ —
a depth-2 product of two square, bias-free linear layers — and descend the masked squared error on the
factors $W_1, W_2$. Crucially I take the hidden dimension to be the full $n$, so the factorization
imposes *no* explicit rank constraint: $W_2 W_1^\top$ can represent any $[n,n]$ matrix, and we are back
to the same underdetermined family of fits. And yet the dynamics are different. Track what gradient flow
on the factors does to the product. The flow on $X = UU^\top$ (the symmetric lift; the asymmetric
$W_2 W_1^\top$ reduces to it) is not $\dot X = -\nabla F(X)$, the flat slide that gives min-Frobenius;
it is $\dot X = -\nabla F(X)\,X - X\,\nabla F(X)$ — the *same* residual direction, but now multiplied by
the current $X$ on both sides. That multiplicative $X$ factor is everything. It means the velocity is
suppressed wherever $X$ is small: start near zero and the flow can barely move in directions $X$ has not
already grown into. This is a self-reinforcing, rich-get-richer growth that prefers a few dominant
directions — low rank — and it is a property of the *flow*, not of a penalty I added.

I want to be precise about why this lands on the *nuclear-norm* fit rather than some other low-rank
point. Integrate the flow in the case I can solve — commuting measurements, init $\alpha I$, take
$\alpha\to 0$. The solution has the form $X_t = \exp(s_t A)\,X_0\,\exp(s_t A)$ with $s_t$ the integrated
residual; the two-sided exponential is a power-iteration-like amplifier, and to reach a finite-scale fit
from a vanishing start it must drive $|s_\infty|\to\infty$, at which point $\exp(s_\infty A)$ collapses
onto the top eigenspace of $A$. Reading the limit eigenvalue by eigenvalue gives exactly the
complementary-slackness condition of the minimum-nuclear-norm SDP: on every direction where the limit is
supported, the rescaled dual has eigenvalue $1$, and off the support it stays below $1$. Both nuclear-norm
KKT conditions hold, so the limit minimizes the nuclear norm. The proof only ever used that the flow
stays on the curved manifold $\{\exp(A^\*(s))\,X_0\,\exp(A^\*(s))\}$, which is why the bias is robust to
the choice of loss and to stochastic subsampling. The single-entry indicators of matrix completion do
*not* commute, so this is a theorem in the commuting case and a well-motivated conjecture in general —
but the mechanism (small init $\to$ amplifier collapses onto the top spectrum) is exactly what I am
betting on.

That argument also tells me *why each ingredient must be what it is*, and the harness exposes a knob for
each. Why factorize at all? Because descending on $X$ gives min-Frobenius = impute-zeros; factorizing
swaps the implicit norm from Frobenius to nuclear, the rank-promoting one. Why full hidden dimension
$d = n$? So the factorization imposes no explicit rank cap — the low-rank preference must come entirely
from the dynamics, both because I do not want to rely on knowing the rank and because the whole point is
that over-parameterized models generalize *without* the constraint. Why initialization close to zero?
The theorem is a statement about $\lim_{\alpha\to0}$; concretely, reaching a finite-scale fit from a
vanishing start is what forces the amplifier to collapse onto the top eigenspace and select the
low-nuclear-norm point. Start large and the depth-2 flow behaves like descent on $X$ — a generic,
high-nuclear-norm optimum. Why small step size? The selecting manifold is *curved* — its tangent space
is a different subspace at every point — so only infinitesimal steps stay on it; finite steps and
momentum walk off it. In practice I discretize with a small learning rate and a stable optimizer, but
the bias is a property of the flow and the discretization approximates it.

Now I map those choices onto the literal scaffold edit, and here I have to match the harness, not the
generic paper. The fill builds `nn.Sequential` of two `nn.Linear(n, n, bias=False)` layers. The
initialization is Gaussian with per-layer std `scale = (init_scale ** 0.5) * (n ** -0.5)`: the
`init_scale ** 0.5` is the $1/\text{depth}$ power with depth $2$, spreading the small overall scale
across the two multiplied factors so the *end-to-end* product starts near zero, and the `n ** -0.5` is
fan-in normalization so the product's magnitude is governed by `init_scale` rather than by $n$. The
end-to-end product is formed by `_e2e`, folding the layers left to right — the first layer contributes
its weight transposed, $W_1^\top$, then layer two is applied, giving $W_2 W_1^\top$ — exactly the
ordering of the math. The fit loop is plain: `Adam` on the factors, masked residual `(e2e - target) *
mask`, loss = sum of squared residual divided by the number of observations, an early stop when the
training MSE falls below `train_thres = 1e-7`, and the end-to-end product returned at the end.

Two harness specifics are worth pinning down because they differ from the textbook recipe. First, the
optimizer is **Adam**, not vanilla small-step gradient descent — and the learning rate here is `5e-3`,
larger than the `1e-3` the depth-3 default ships with. Adam's per-coordinate adaptivity is *not* the
gradient flow the implicit-bias theory is built on; it does the fitting, but I should remember the bias
lives in the parameterization plus the near-zero init, and Adam plus a not-tiny step is a coarser
approximation to the flow than the theory would like. That is a real caveat for the *depth-2* rung in
particular, because — and this is the second specific — depth-2 is exactly the case the theory says gives
only the *nuclear-norm* surrogate, with no extra strength. The singular-value dynamics make this
concrete: for a depth-$N$ factorization the per-mode growth rate carries a factor $(\sigma_r^2)^{1-1/N}$,
so for $N=2$ the rate is proportional to $|\sigma_r|$ — a power-law gap between leading and trailing
singular values, some low-rank bias, but only the gentle power-law version. There is no saturating cap on
the small modes the way deeper factorizations get. So I should *expect* this rung to behave like a
nuclear-norm-strength recovery at best, and to be coarsened further by Adam and the larger step.

So at step 1 the method is settled and the edit is the simplest non-trivial one: parameterize the
recovery as a product of two full-dimensional bias-free linear layers, initialize both with tiny Gaussian
weights so the end-to-end product starts a hair above zero, and descend the masked squared error on the
observed entries with Adam — no nuclear-norm penalty, no rank cap, no explicit regularizer (the distilled
module is in the answer).

Now reason about what this floor must do, because that is the entire reason to run it. Where there are
plenty of observations relative to the rank, the implicit bias should comfortably recover $M^\*$ — that is
the regime where even nuclear norm works, so depth-2 should too. The easiest environment, rank3-50 with
$30\%$ observed, sits there: $0.30\cdot 50^2 = 750$ observations for a rank-3 matrix, which has only
$3(2\cdot 50 - 3) \approx 291$ degrees of freedom, so I expect a decent (if not stellar) recovery. The
canonical rank5-100 at $20\%$ is the regime where the depth-2-vs-depth-3 distinction is supposed to bite:
$2000$ observations for $\approx 975$ degrees of freedom, undersampled enough that the *strength* of the
low-rank bias matters. Depth-2 — only the power-law gap — should leave real error on the table here,
worse than what a deeper factorization or even the convex baseline will manage. And the hardest,
rank10-200 at $10\%$ observed, is $4000$ observations for a rank-10 matrix with $\approx 3900$ degrees of
freedom — barely above the information-theoretic floor, in the deeply data-poor regime where *every*
nuclear-norm-strength method is expected to fail. There I expect depth-2 to recover almost nothing: the
implicit bias simply is not sharp enough to pick out the true low-rank matrix from that few entries.

So the falsifiable picture for the floor is: a clean win on rank3-50, a mediocre-to-poor result on
rank5-100 (the test that depth-2 is weaker than what is coming), and a near-total failure on rank10-200.
Whatever the precise numbers, the diagnosis is already pointed at the next rung. If depth-2 leaves error
on rank5-100, the question becomes whether *explicitly* asking for the minimum-nuclear-norm fit — the
classical convex program, run to convergence rather than approximated by an adaptive optimizer from a
finite init — does better than the implicit-bias approximation; and beyond that, whether *more depth*
sharpens the implicit bias past nuclear-norm strength. The floor's job is to establish exactly how much
the bare, penalty-free depth-2 fill buys, so the later rungs have a number to beat.
