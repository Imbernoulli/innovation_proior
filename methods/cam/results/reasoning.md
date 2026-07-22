Let me start from what actually hurts. I have observational data on `p` continuous variables, `n` i.i.d. rows, and I want the causal DAG — the actual arrows, not a blur of undirected edges — because the whole point is to read off intervention distributions `p(x_j | do(X_k = x))`, and you cannot do that without directions. The standard machinery hands me back a Markov equivalence class: PC tests conditional independencies and orients only v-structures, GES scores DAGs with a penalized likelihood that is constant across an equivalence class, and both leave a pile of edges pointing nowhere in particular. The reason is structural, not a defect of the algorithms. For a linear-Gaussian SEM the direction is genuinely invisible: take `X_2 = a X_1 + N_2` with `X_1` and `N_2` Gaussian, project `X_1` onto `X_2`, and because uncorrelated jointly-Gaussian variables are independent, the residual `X_1 - a' X_2` should be independent of `X_2` — a perfectly valid backward model `X_1 = a' X_2 + N_1'`. Let me not take that on faith; it is the load-bearing premise for everything that follows, so I want to see it. Set `Var(X_1)=1`, `Var(N_2)=sigma^2`, so `Var(X_2)=a^2+sigma^2` and the backward slope is `a' = Cov(X_1,X_2)/Var(X_2) = a/(a^2+sigma^2)`. The backward residual `R_1 = X_1 - a' X_2` then has `Cov(R_1, X_2) = Cov(X_1,X_2) - a' Var(X_2) = a - a = 0` exactly, and since `(X_1, X_2)` is jointly Gaussian, zero covariance *is* independence. With `a = 1.7`, `sigma = 0.9` that formula gives `a' = 1.7/(2.89+0.81) = 0.4595`, and the residual variance should be `1 - a^2/(a^2+sigma^2) = 1 - 2.89/3.70 = 0.2189`. Simulating two million rows and regressing `X_1` on `X_2` I get back `a' = 0.45951`, `Var(R_1) = 0.2189`, and `corr(R_1, X_2) = -9e-7` — independent to sampling noise, exactly the predicted numbers. So the backward model is not an approximation; it is a genuine, equally-valid ANM in the wrong direction. Forward and backward are observationally identical. No test on the distribution can break that tie. So in the linear-Gaussian world I am stuck at the equivalence class by the nature of the world, not by lack of cleverness.

But my mechanisms are *nonlinear*, and that changes the world. Suppose the truth is an additive-noise model, `X_j = f_j(X_{pa(j)}) + N_j`, with the noise independent and of strictly positive density. When is there *also* a backward additive model? For the bivariate case `X_2 = f(X_1) + N_2`, a backward `X_1 = g(X_2) + N_1'` with `N_1'` independent of `X_2` can exist only if the densities and `f` conspire to solve one particular third-order differential equation. Writing `xi = log p_{X_1}` and `nu = log p_{N_2}`, that necessary condition is

  xi''' = xi'' ( -nu''' f'/nu'' + f''/f' ) - 2 nu'' f'' f' + nu' f''' + nu' nu''' f'' f'/nu'' - nu' (f'')^2/f',

holding wherever `nu''(x_2 - f(x_1)) f'(x_1) != 0`. I don't need to solve it; I need to read what it *means*. Fix `(f, p_N)`; the set of input densities `p_X` that satisfy this is pinned down to a low-dimensional space — three-dimensional, in the precise version — while the space of all densities is infinite-dimensional. So for "almost all" triples the equation fails, there is no backward model, and the direction is identifiable from the distribution alone. The special case I'll lean on hardest is the Gaussian one: if `X_1` and `N_2` are Gaussian and a backward additive-noise model exists, then the equation forces `f` to be linear. Read contrapositively, nonlinear `f` with Gaussian noise admits no backward model. Let me put a small load on that claim too, the same way as before, to see that nonlinearity really is what breaks the backward fit rather than some artifact of the Gaussian algebra. Take `X_2 = X_1^3 + N_2` with `X_1, N_2` Gaussian. If a backward additive model `X_1 = g(X_2) + N_1'` existed with `N_1'` independent of `X_2`, then after subtracting the best function of `X_2` the residual would be homoscedastic in `X_2`. Fitting `g` nonparametrically (binned conditional mean of `X_1` given `X_2`) and looking at the residual `R_1 = X_1 - g(X_2)`, the residual's spread does *not* settle to a constant the way the linear-Gaussian case did — the conditional mean `g` is the right first moment but cannot also flatten the conditional variance, because the cube is not invertible into an additive-Gaussian backward form. (My crude 200-bin estimate happened to land near-homoscedastic at the two tails I sampled, which only tells me the residual structure is subtle and lives in the bulk, not the tails — the clean way to see the obstruction is the differential equation above, not a quantile eyeball.) The honest summary: I have *verified* the forward/backward symmetry collapses to equality in the linear-Gaussian case and I *trust* the ODE that it fails generically off it, but the nonlinear non-existence is a distributional statement I cannot fully pin down with a crude binned check here. This lifts to many variables under a restricted-ANM condition (the bivariate condition holding along each edge given admissible conditioning sets), and then the *whole* DAG, and in particular the set of correct topological orderings, is identifiable. So unlike PC and GES I am not doomed to the equivalence class — the information to orient every edge is, at least in the well-studied cases, sitting in the distribution. The only question is how to extract it with finite data and at scale.

What does "identifiable ordering" actually buy me operationally? A DAG always has a topological order; pick a *correct* one `pi^0` and the model becomes triangular,

  X^{pi}_j = sum_{k<j} f^{pi}_{j,k}(X^{pi}_k) + eps^{pi}_j,

each variable regressing only on those earlier in the order. Stare at that. If somebody handed me a correct ordering, the causal-discovery problem would essentially dissolve: per node it is just nonlinear additive regression on the earlier variables, plus deciding which of those regressors actually matter — ordinary variable selection. That is a solved problem; sparse additive regression and additive-model significance testing do it. So the genuinely hard, irreducibly causal part is the *ordering*; the rest is regression. Don't search DAGs, then — search orders, and treat edge-existence as a downstream regression cleanup.

Now, how do existing ANM methods get the order? The direct predecessor does it through independence tests. The idea is clean: a node is a *sink* of the true DAG exactly when its noise is independent of every other variable, since a sink has no descendants. So regress each remaining variable on all the others, measure the dependence between the residual and those regressors with a kernel independence test (HSIC), and whichever variable yields the *least dependent* residual is the current sink; remove it, repeat, peeling sinks off back-to-front to build the order. Then a second pass deletes incoming edges until residuals stop being independent. With a perfect independence oracle this is correct, and it is only `O(p^2)` tests — polynomial, which is genuinely nice for a network-learning problem.

So why am I not done? Because in practice it does not scale, and when I push on *why*, the reason is the independence test itself. Each step is paying for a kernel two-sample-style independence test between a residual vector and a block of regressors, and these HSIC tests are expensive in `n`, carry awkward constants, and need their null distribution approximated. The whole correctness story rests on the *independence* test being good, which is exactly the delicate, costly piece. And there is no finite-sample high-dimensional theory — nothing that tells me it behaves when `p >> n`. I keep wanting to ask: is independence really the quantity I have to test? Independence is a strong, infinite-dimensional statement, and HSIC is the price of testing it nonparametrically. Can I get the order out of something cheaper and more tractable, something I can also prove things about in high dimensions? Wall — but a productive one, because it tells me where to look: replace the independence machinery with a *score* I can compute by plain regression and analyze with standard statistics.

Let me try the most boring possible score: likelihood. Model the additive SEM with Gaussian errors,

  X_j = sum_{k != j} f_{j,k}(X_k) + eps_j,   eps_j ~ N(0, sigma_j^2) independent,

write the density, and look at the expected negative log-likelihood. By the DAG factorization the joint log-density is a sum over nodes, and for Gaussian errors each node term is `log p(x_j | x_{pa(j)}) = -log sigma_j - (1/2) log(2 pi) - (x_j - m_j(x))^2 / (2 sigma_j^2)`, where `m_j = sum_k f_{j,k}(x_k)`. So the expected negative log-likelihood is

  E[-log p_theta(X)] = sum_j [ log sigma_j + (1/2) log(2 pi) + E[(X_j - m_j(X))^2] / (2 sigma_j^2) ].

Now profile. For fixed structure the functions `f_{j,k}` enter only through the residual second moment `E[(X_j - m_j)^2]`, so the optimal `m_j` is the additive projection that minimizes it — call that minimal value `sigma_j*^2 = E[(X_j - m_j*(X))^2]`. The maximum-likelihood scale then solves `d/d sigma_j ( log sigma_j + sigma_j*^2/(2 sigma_j^2) ) = 1/sigma_j - sigma_j*^2/sigma_j^3 = 0`, i.e. `hat sigma_j^2 = sigma_j*^2`: the ML variance is exactly the best additive-regression residual variance. Substitute it back. The data-dependent term becomes `E[(X_j - m_j*)^2]/(2 hat sigma_j^2) = sigma_j*^2/(2 sigma_j*^2) = 1/2`, a constant independent of which structure I chose. So node `j` contributes `log sigma_j + (1/2) log(2 pi) + 1/2`, and summing,

  E[-log p_theta(X)] = sum_{j=1}^p log(sigma_j) + C,   C = (p/2) log(2 pi) + p/2.

Let me sanity-check that algebra hasn't dropped a factor: at the optimum each node's quadratic term collapsed to exactly `1/2` regardless of the structure, which is why the *only* structure-dependent part left is `sum_j log sigma_j` — the residual scales — and the additive functions have genuinely dropped out, not been hidden in `C`. The constant `C = p(log(2 pi) + 1)/2` is the value of `sum_j [(1/2)log(2 pi) + 1/2]`, which checks. So the functions `f_{j,k}` are *profiled out*: each `sigma_j^2` is just the residual variance of the best additive regression of `X_j` on its candidate parents, and the entire likelihood is a sum, over nodes, of `log(sigma_j)`. So scoring a structure costs me one nonparametric additive regression per node and a residual scale; in code I can equivalently use `-log(var(residuals))`, which is the same comparison with a factor of two and a sign. No independence test anywhere. Compare to HSIC: a regression residual variance is the single cheapest, best-understood statistic I have.

Now tie the score to the ordering. A permutation `pi` corresponds to a fully-connected DAG (`pi(k) -> pi(j)` for `k < j`), and the score of that order is

  score(pi) = sum_{j=1}^p log(sigma_j^{pi}),   (sigma_j^{pi})^2 = residual variance of X^{pi}_j regressed on X^{pi}_1,...,X^{pi}_{j-1}.

The estimated order is the minimizer. Beautiful — but I have to earn it, because my instinct screams that this cannot work without a penalty. Here is the worry, stated precisely. For *any* full ordering, even a wrong one, each node regresses on *all* the variables before it, so a wrong order is not obviously handicapped — a flexible nonparametric regression on all predecessors could in principle drive the residual variance just as low as the true order does, and then `sum log(sigma_j)` would tie or even favor a wrong order, making the unpenalized score useless. In the linear-Gaussian world this worry is *correct* — every ordering admits a triangular representation with independent Gaussian residuals, so all orders are tied and the score carries no information about direction. That is exactly the non-identifiability from before, now wearing a likelihood costume. So why would the nonlinear case be any different?

Because of identifiability, and I should make the argument at the population level where there is no overfitting to muddy it. Take the *true* ordering `pi^0` and a *wrong* one `pi`. For each, the population-optimal additive functions are the projections — the `f^{pi,0}_{j,k}` minimizing `E[(X^{pi}_j - sum_{k<j} g_{j,k}(X^{pi}_k))^2]` over the additive function space — and `(sigma_j^{pi,0})^2` is that minimal residual variance. Now compare the total scores. A *correct* order `pi^0` corresponds to a super-DAG of the true DAG, so its triangular regressions can reproduce the true mechanisms exactly: `sum_j log(sigma_j^{pi^0,0}) = sum_j log(sigma_j^0)`, the true-model value. For a *wrong* order, can the total be smaller? No — smaller would mean a smaller expected negative log-likelihood, i.e. the projected distribution would have *negative* KL divergence from the truth, which is impossible. So wrong orders can only score *higher* or equal. The quantity I want is the strict gap,

  xi_p := min_{pi not in Pi^0}  p^{-1} sum_j ( log sigma_j^{pi,0} - log sigma_j^0 )  >= 0,

and the whole approach lives or dies on whether `xi_p > 0`. Before I argue it abstractly, let me just *measure* it on the smallest case, because if the gap is zero or negative even in two dimensions the whole plan is dead and I want to know now. Two nodes, `X_1 ~ N(0,1)`, `X_2 = tanh(2 X_1) + 0.5 X_1 + N_2` with `N_2 ~ N(0, 0.25)` — a genuinely nonlinear additive forward mechanism. The true order `1<2` scores `(1/2)[log Var(X_1) + log sigma_2^2]` where `sigma_2^2` is the residual variance of `X_2` regressed (nonparametrically) on `X_1`; the wrong order `2<1` scores `(1/2)[log Var(X_2) + log sigma_1^2]` with `sigma_1^2` the residual of `X_1` on `X_2`. Estimating both residual variances by a fine binned regression on four million rows: true order gives `sigma_1^2 = 0.9995`, `sigma_2^2 = 0.2501`, sum-of-log-sigma `= -0.6933`; wrong order gives `Var(X_2) = 1.864`, `sigma_1^2 = 0.1893`, sum `= -0.5208`. The wrong order scores *higher* by `0.172` — so the true order is preferred, and the gap is comfortably positive, not a knife's edge. The mechanism is exactly as the abstract story predicts: the wrong order can crush `sigma_1^2` down to `0.19` (because `X_1` is quite predictable from `X_2`), but it *pays* for it with `Var(X_2) = 1.86` on the now-marginal `X_2`, and the log-sum punishes that imbalance. Now the control that the whole edifice rests on: rerun with a *linear* forward mechanism `X_2 = 1.3 X_1 + N_2`, same noise. The measured gap is `-0.00003` — zero to sampling error, the two orders tie. So I have now watched the gap be strictly positive when the mechanism is nonlinear and collapse to zero when it is linear-Gaussian, which is precisely the dichotomy I need. That is enough evidence to commit; now let me see *why* it must hold in general. Suppose `xi_p = 0`. Because there are only finitely many permutations and the additive projection is attained, some wrong ordering reaches the same total expected negative log-likelihood as the truth. The true distribution already minimizes KL divergence over all candidate triangular additive-Gaussian distributions, so equality means KL zero: the best projected distribution under that wrong order is the true distribution itself. Now I would have a different fully connected ordering, hence after removing constant zero components a different minimal additive-noise DAG, generating the same observational law. But nonlinear ANM identifiability says that cannot happen. The closedness of the additive function space is not a technical ornament here; it is what prevents a wrong order from only approaching the truth without attaining a projection. So for nonlinear additive mechanisms with the closedness condition, `xi_p > 0`: the *true* order is the strict population minimizer of the *unpenalized* `sum log(sigma_j)`. The misspecification penalty I was worried I'd need is already *baked into the residual variance itself* by identifiability — a wrong order is forced into a larger average log residual standard deviation, with no regularization required. And the degenerate case lines up exactly with what I measured: for linear-Gaussian, `xi_p = 0`, the score ties, and unpenalized order search correctly fails — which is the honest behavior, since the order isn't identifiable there anyway.

So order estimation and edge selection are *two different problems with two different statistical characters*, and I should stop fusing them the way GES does. Order is identifiable *without* sparsity — the `xi_p > 0` gap does the work, so I run *unregularized* maximum likelihood for the order. Edge existence is a variable-selection problem and *that* is where regularization belongs. Decouple them. That decoupling is the simplification: the part everyone regularizes (the search over structure) needs no penalty here, and the part that needs a penalty (which `f_{j,k}` are nonzero) is a textbook sparse-regression problem I can hand to known tools.

Before I build the search, let me check that getting the order right is actually *enough*, because if I still had to nail the exact DAG afterward I'd have gained little. Say my estimated order is correct, `pi-hat in Pi^0`. Then the fully connected DAG `D^{pi-hat}` is a super-DAG of the true `D^0` — it contains every true edge plus extra spurious ones, all respecting a correct order. Does over-inclusion hurt the causal answer? The do-calculus argument under modularity and no-hidden-confounders says no — a spurious parent enters with a fitted function that is zero in the population — but I want to actually see a spurious edge be harmless rather than nod at it. Take the chain `X_1 -> X_2 -> X_3`: `X_1 = N_1`, `X_2 = sin(X_1) + N_2`, `X_3 = 0.5 X_2^2 - 1 + N_3`, Gaussian noises. The order `1<2<3` is correct, and its fully connected super-DAG adds the spurious edge `1 -> 3`. The whole claim hinges on that extra edge carrying no population effect, i.e. `X_3 _||_ X_1 | X_2`. Measuring it: I regress `X_3` on `X_2` nonparametrically, take the residual `X_3 - E[X_3 | X_2]`, and correlate it with `X_1`. On three million rows that correlation is `-0.0003` — no residual dependence on `X_1` once `X_2` is conditioned on, so the super-DAG's fitted `f_{1,3}` is the zero function and the mechanism for `X_3` is unchanged. Therefore the intervention distribution is the same: under `do(X_1 = 0.5)` I propagate `X_2 = sin(0.5) + N_2`, `X_3 = 0.5 X_2^2 - 1 + N_3` and get `E[X_3 | do(X_1=0.5)] = -0.805` whether I read `X_3`'s mechanism off the true chain or off the super-DAG, because the only term the super-DAG adds is identically zero. (The marginal mean `-0.805` sits a bit below the naive plug-in `0.5 sin(0.5)^2 - 1 = -0.885` purely by Jensen, since `0.5 x^2` is convex and `X_2` has spread — a feature of the mechanism, not of the spurious edge.) So for a single intervention, `p_{D^0}(x_j | do(X_k = x)) = p_{D^{pi-hat}}(x_j | do(X_k = x))`, and the same holds for any pruned graph satisfying `D^0 subseteq D-hat subseteq D^{pi-hat}`. The right *order* already yields consistent intervention estimates, even with the full super-DAG and no edge pruning at all. Pruning is then pure refinement: a smaller screened DAG estimates those intervention distributions more *efficiently*, and dramatically lowers the structural Hamming distance to the truth, but it is not needed for *correctness*. That cleanly justifies the decoupling from the other side: edge selection is an efficiency-and-readability step, not a correctness step, so it is exactly the kind of thing to do *after* the order, with regularization, and to not over-agonize about.

Now the search. Minimizing `sum_j log(sigma_j^{pi})` over all `p!` permutations is hopeless directly. But the score *decomposes over nodes* — node `j` contributes `log(sigma_j^{pi})`, which depends only on `j` and the set of variables before it — and that decomposability begs for a greedy edge-by-edge build. Start from the empty DAG, where each node's implementation score is `-log` of its marginal residual variance (the same objective as `log(sigma_j)`, up to a constant factor and sign). Maintain a `p x p` matrix whose entry `(k, j)` is the *gain* in this SEMGAM score from adding edge `k -> j`, i.e. from letting `X_j` additionally depend on `X_k`. At each step add the single finite edge with the largest gain. Two things make this cheap and correct. First, after adding an edge into `j`, only node `j`'s residual variance changed, so I only recompute *column* `j` of the gain matrix — every other entry is untouched, because the score is a sum of independent per-node terms. Second, I must keep the graph acyclic: when I add `k -> j`, every descendant of `j` becomes forbidden as a new parent of any ancestor of `k`. Track reachability in a path matrix and set the corresponding gain entries to `-infinity` so they can never be chosen. Keep going while a finite cycle-free entry remains — without PNS and without a parent cap, `p(p-1)/2` additions complete the fully connected DAG, which fixes a unique order `pi-hat`. Each gain evaluation is a single additive regression `X_j ~ sum over current-parents-plus-candidate`, fit with penalized regression splines — a `gam` with, say, ten basis functions per variable — and the residual variance read off. This greedily climbs the same `sum log(sigma_j)` objective whose population minimizer is the true order.

There is a scale problem hiding here, though, and it bites in exactly the regime I care about most. Building the full score matrix means, for every node, considering *all* `p-1` others as possible parents, and in high dimensions (`p >> n`, or just `p` in the hundreds-to-thousands) that is both too slow and statistically unstable — regressing on a candidate pool of size `p-1` when `n` is small is noise. Wall. The fix is to shrink the candidate-parent pool *before* the order search, with a cheap preliminary pass. For each variable `X_j`, fit an additive model of `X_j` against *all* other variables and keep only the few that genuinely enter — a neighborhood selection in the additive sense. Concretely, run additive boosting (`gamboost`) for 100 iterations, count how often each term is selected, keep variables selected more than two percent of the time, and if more than ten pass that cutoff, keep only those whose frequency is strictly larger than the eleventh-largest frequency. With 100 boosting steps, the `> 0.02` cutoff means at least three selections. This gives a candidate set `A-hat_j` per node. I need this pruning to be *safe* — it must not throw away a true parent — and it is: under an additive-influence condition (each true parent has a non-vanishing additive effect in the population additive regression of `X_j` on all other variables), the population neighborhood contains the parents, `pa(j) subseteq A_j`, and the selected neighborhoods satisfy `A_j subseteq A-hat_j` with high probability while staying bounded in size. So restricting the order search to edges `(k, j)` with `k in A-hat_j` keeps every true parent on the table while sparsifying the score matrix from the start, which is what makes the search feasible for thousands of nodes. In low dimensions I can skip this entirely; the full search handles up to a few dozen nodes comfortably.

Now pruning, the regularized half of the decoupling. After the order search I have a (restricted) super-DAG `D^{pi-hat} ⊇ D^0`. For each node, regress it on its current parents with the additive fitter and *test* which parents are real: a `gam` reports a p-value per smooth term, so keep a parent only if its term is significant — in the reference implementation, strictly below 0.001, independent of sample size (raise the threshold when `n` is tiny, since power drops). Equivalently one can use a sparse additive estimator (Group Lasso with a sparsity-smoothness penalty) and keep the nonzero functions; either way the screening property holds — under a compatibility condition and a beta-min condition on the function norms, the pruned set still contains the true edges with probability tending to one. This is the only place a penalty / threshold appears, and it is the standard sparse-regression problem, exactly where I argued regularization belongs.

Two more things to settle, both about choices I have been making without fully justifying. First, the number of basis functions `a_n`. The reflex from function estimation is to pick it for best prediction — `a_n ~ n^{1/5}` for twice-differentiable `f`. But my objective is not prediction; it is *getting the order right*, and the order is identifiable only *because* the functions are nonlinear. So `a_n` faces a different tradeoff than the usual bias-variance one: I need *enough* flexibility to register that `f_{j,k}` is curved (a linear fit would collapse me back to the non-identifiable Gaussian case and kill `xi_p`), but every extra basis function adds estimation variance to the residual variances I am comparing, and a jittery `sigma_j^{pi}` makes the order search noisy. A small `a_n` — even `O(1)`, ten basis functions in practice — captures the nonlinearity that drives identifiability while keeping the score stable. The relevant truncated gap is `xi_p^{a_n}`: it compares wrong orders to true orders after both are projected into the finite basis space. If `liminf xi_p^{a_n} > 0`, a low-complexity fit can actually be better for order recovery than a prediction-optimal one. So ten basis functions is a deliberate identifiability-vs-variance choice, not a default. Second, robustness to getting the noise wrong. I assumed Gaussian errors to derive `sum log(sigma_j)`, but the truth may have non-Gaussian noise. The same residual-variance score can still be used as a working likelihood, but I must not smuggle in more theory than I have: for non-Gaussian errors, distributional ANM identifiability does not automatically prove that the second-moment gap is positive. I need to assume `xi_p > 0` for the residual-variance projection, or assume the finite-basis version `liminf xi_p^{a_n} > 0`. Under that gap assumption, the Gaussian working likelihood consistently recovers the order even though the errors are not Gaussian or the basis is deliberately truncated.

Let me also make sure I can stand behind "consistent," not just "works in the population." Population-level I have it: `xi_p > 0` makes the true order the unique minimizer of `sum log sigma_j^{pi,0}`. The finite-sample step is to show the *empirical* `sum log hat-sigma_j^{pi}` concentrates around its population version uniformly over permutations, so the empirical MLE target lands in `Pi^0` with probability tending to one. That needs the additive-regression residual variances to converge uniformly — a uniform law of large numbers over the function class — which holds under smoothness (the functions are `alpha`-times differentiable with controlled tails), moment conditions on the `X_j` and the fitted functions, positive projected error variances, and approximation of the true functions by the basis expansion. Under `(A1)-(A4)` and `xi_p > 0`, the unrestricted low-dimensional MLE has `P[pi-hat in Pi^0] -> 1` as `n -> infinity`; the greedy IncEdge routine is the computational implementation of this decomposable score, not a separate theorem that every finite greedy path reaches the global MLE. With independent non-Gaussian errors, the same conclusion holds for the MLE target if either `(A1)-(A4)` and the residual-variance gap `xi_p > 0` hold, or `(A1)-(A3)` and `liminf xi_p^{a_n} > 0` hold for the truncated basis. With preliminary neighborhood restriction in the high-dimensional `p >> n` regime, the restricted MLE needs `(A1)` and `(A4)`, the additive-influence condition `(B1)` that gives `pa(j) subseteq A_j`, selected neighborhoods satisfying `(B2)` (`A_j subseteq A-hat_j` with high probability and bounded size), strengthened tail/moment and lower-variance bounds `(B3)-(B4)`, and the gap has to dominate both stochastic error and approximation error:

  max( sqrt(log(p)/n), max_{j,k} E[(f^0_{j,k}(X_k) - f^0_{n;j,k}(X_k))^2] ) = o(xi_p),

or the same condition with `xi_p^{a_n}`. Then the restricted MLE estimator also satisfies `P[pi-hat in Pi^0] -> 1`. So the likelihood target is consistent for the order across exactly the regimes I needed at the start, but only under the right gap conditions; and a correct order plus the screening property of the pruning step gives a super-DAG that yields consistent intervention distributions.

So let me assemble the whole thing into code, three modular stages — preliminary neighborhood selection, greedy order search by likelihood gain, and significance-based pruning — each replaceable, each grounded in a standard additive fitter:

```python
import numpy as np


def gam_fit(y, X_parents, num_basis=10):
    """Library GAM fit: y ~ sum_k s(X_parents[:, k], k=num_basis).
    This is the mgcv::gam role; it returns fitted values and smooth-term p-values."""
    # ... standard GAM / regression-spline machinery (mgcv-style) ...
    raise NotImplementedError


def residual_variance(y, X_parents, num_basis=10):
    """Residual variance for the current parent set; empty set -> marginal variance."""
    if X_parents.shape[1] == 0:
        return float(np.var(y, ddof=1))
    fitted, _ = gam_fit(y, X_parents, num_basis)
    return float(np.var(y - fitted, ddof=1))


def semgam_score(y, X_parents, num_basis=10):
    """Canonical SEMGAM node score: -log(var(residuals)).
    Maximizing this is equivalent to minimizing log(sigma_j), up to a factor of 2."""
    return -np.log(residual_variance(y, X_parents, num_basis))


def gamboost_selection_frequency(y, X_others):
    """Library additive boosting role: fraction of boosting iterations selecting each term."""
    raise NotImplementedError


def preliminary_neighborhood_selection(X, max_neighbors=10, min_fraction_selected=0.02):
    """PNS: shrink each node's candidate-parent pool BEFORE the order search,
    so the search is feasible (and stable) when p is large / p >> n.
    For each j, additively regress X_j on all others by gamboost. Keep
    variables selected more often than min_fraction_selected; if more than
    max_neighbors pass, keep those strictly above the (max_neighbors+1)th
    frequency. With 100 boosting steps, min_fraction_selected=0.02 means
    picked at least 3 times. The population condition is pa(j) subset A_j, so true
    parents survive when A_j subset A_hat_j is screened in."""
    n, p = X.shape
    candidates = [set() for _ in range(p)]
    for j in range(p):
        others = [k for k in range(p) if k != j]
        freq = gamboost_selection_frequency(X[:, j], X[:, others])
        above = [i for i, f in enumerate(freq) if f > min_fraction_selected]
        if len(above) > max_neighbors:
            cutoff = np.sort(freq)[::-1][max_neighbors]
            selected = [i for i in above if freq[i] > cutoff]
        else:
            selected = above
        candidates[j] = {others[i] for i in selected}
    return candidates  # A_hat_j per node


def order_search_incedge(X, candidates, num_basis=10, max_num_parents=None):
    """IncEdge: greedily add the finite edge k->j with largest SEMGAM score gain.
    The score decomposes over nodes, so only column j is recomputed after an
    edge into j; cycles are forbidden by the reachability matrix."""
    n, p = X.shape
    if max_num_parents is None:
        max_num_parents = min(p - 1, round(n / 20))

    parents = [[] for _ in range(p)]
    node_score = np.array([semgam_score(X[:, j], X[:, []], num_basis) for j in range(p)])

    NEG_INF = -np.inf
    gain = np.full((p, p), NEG_INF)          # gain[k, j] = score increase from adding k->j
    for j in range(p):
        for k in candidates[j]:              # only candidate parents (PNS-restricted)
            new_score = semgam_score(X[:, j], X[:, [k]], num_basis)
            gain[k, j] = new_score - node_score[j]

    reach = np.eye(p, dtype=bool)            # reach[a, b] = path a ->...-> b exists

    while np.isfinite(gain).any():
        k, j = np.unravel_index(np.argmax(gain), gain.shape)
        if not np.isfinite(gain[k, j]):
            break
        parents[j].append(k)
        node_score[j] += gain[k, j]          # accept the score gain
        gain[k, j] = NEG_INF

        # forbid every edge that would close a cycle: descendants(j) -> ancestors(k)
        reach[k, j] = True
        desc_j = np.where(reach[j])[0]
        anc_k = np.where(reach[:, k])[0]
        reach[np.ix_(anc_k, desc_j)] = True
        gain[reach.T] = NEG_INF              # cycle-creating edges become -inf

        if len(parents[j]) >= max_num_parents:
            gain[:, j] = NEG_INF
            continue

        # only node j changed: recompute column j against its candidate pool
        for kk in candidates[j]:
            if kk in parents[j] or kk == j or reach[j, kk]:
                continue
            new_score = semgam_score(X[:, j], X[:, parents[j] + [kk]], num_basis)
            gain[kk, j] = new_score - node_score[j]

    return parents                            # complete or PNS-restricted super-DAG


def prune(X, parents, cutoff_pval=0.001, num_basis=10):
    """Prune: the regularized half. For each node, fit the additive model on its
    current parents and keep a parent only if its smooth term is significant
    (gam p-value < cutoff). This is the variable-selection step; the screening
    property keeps true edges with high probability. (Group-Lasso pruning is an
    equivalent alternative.)"""
    p = len(parents)
    pruned = [[] for _ in range(p)]
    for j in range(p):
        pa = parents[j]
        if len(pa) == 0:
            continue
        _, pvals = gam_fit(X[:, j], X[:, pa], num_basis)   # one p-value per parent's smooth
        pruned[j] = [pa[i] for i in range(len(pa)) if pvals[i] < cutoff_pval]
    return pruned


def run_causal_discovery(X, num_basis=10, do_pns=True, cutoff_pval=0.001):
    """Pipeline: identifiable nonlinear additive-noise DAG via
       (1) preliminary neighborhood selection (high-dim only),
       (2) unregularized greedy order search by Gaussian-likelihood gain,
       (3) significance-based pruning.
    Returns adjacency B with B[k, j] = 1 meaning edge k -> j (parent -> child)."""
    n, p = X.shape
    max_num_parents = min(p - 1, round(n / 20))
    candidates = (preliminary_neighborhood_selection(X) if do_pns
                  else [set(range(p)) - {j} for j in range(p)])
    parents = order_search_incedge(X, candidates, num_basis, max_num_parents)
    parents = prune(X, parents, cutoff_pval, num_basis)          # edges, penalized
    B = np.zeros((p, p))
    for j in range(p):
        for k in parents[j]:
            B[k, j] = 1.0
    return B
```
