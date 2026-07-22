Let me start from what actually hurts. I have observational data `X ∈ R^{n×d}`, `n` samples of `d` variables, and I believe they came from a structural equation model where each variable is some nonlinear function of its parents plus noise, `X_j = f_j(X_{pa(j)}) + z_j`. I want the directed acyclic graph — who causes whom. The thing that makes this miserable is the acyclicity requirement. If I phrase it as score-based learning — pick the graph that best fits the data under some sparsity-penalized score — the feasible set is the discrete space of DAGs, and that space is superexponential in `d`. So every method I know reaches for a combinatorial trick to stay inside it: add edges one at a time and check after each addition that no cycle appeared, or fix a topological ordering and search over the `d!` orderings, fitting a regression within each. Those tricks are exactly what makes these methods slow, brittle, and a pain to implement. What I really want is to not touch the discrete space at all — to write down one continuous objective over real-valued parameters, with acyclicity as a smooth constraint, and hand the whole thing to an off-the-shelf solver that updates the entire graph at once.

There's a reason to think nonlinearity is on my side here, and I should pin it down before anything else, because it tells me whether the problem is even solvable from observational data. If the `f_j` were linear and the noise Gaussian, I'd be stuck: a jointly Gaussian linear SEM is not identifiable — a whole Markov equivalence class of DAGs induces the very same Gaussian distribution, so no amount of data fixes the direction of an edge. The escapes are known. Linear but non-Gaussian noise is identifiable (the LiNGAM result — Shimizu et al. 2006, building on Kagan-Linnik-Rao). And *nonlinear* additive-noise models are identifiable under mild conditions — Hoyer, Janzing, Mooij, Peters & Schölkopf showed this for nonlinear ANMs, and Peters, Mooij, Janzing & Schölkopf nailed the general continuous case; Corollary 31 there says that if each `f_j` is three-times differentiable and not linear in any of its arguments, the DAG is generically recoverable. The intuition I carry from that: regress `X_j` on `X_k` and look at the residual; if the true mechanism is `X_k → X_j` with a nonlinear `f`, the residual comes out independent of `X_k`, but in the wrong direction `X_j → X_k` it does not — the nonlinearity breaks the symmetry that the Gaussian-linear case has no way to break. So the nonlinearity isn't just a modeling burden; it is the thing that makes direction visible. Good. The problem is well-posed. Now I need a continuous machine to actually extract it.

What's the most powerful piece of machinery already on the table? The smooth acyclicity characterization for *linear* SEMs. Let me re-derive it carefully because everything I'm about to do hangs off it, and I want to understand exactly which parts are load-bearing. Start in the binary world: a binary adjacency matrix `B`, edge present iff `B_{ij} = 1`. The classical combinatorial fact is that `(B^k)_{ii}` counts the number of length-`k` closed walks from node `i` back to itself. So the graph has a cycle through `i` of length `k` exactly when `(B^k)_{ii} > 0`, and the graph is acyclic iff `(B^k)_{ii} = 0` for every `k ≥ 1` and every `i`, i.e. `tr(B^k) = 0` for all `k`. That's infinitely many conditions; I want one number. The obvious collapse is a generating function. Sum the powers: `∑_{k≥0} B^k = (I - B)^{-1}` when it converges, and `tr((I - B)^{-1}) = tr(I) + ∑_{k≥1} tr(B^k) = d + ∑_{k≥1} tr(B^k)`. So `tr((I - B)^{-1}) = d` iff acyclic — one equation. But two problems. First, the Neumann series `∑ B^k` only converges when the spectral radius `r(B) < 1`, which an arbitrary graph won't satisfy; that's a strong restriction I can't assume. Second, even setting convergence aside, the walk counts `tr(B^k)` grow fast, so the series is wildly ill-conditioned — the late terms dominate and swamp machine precision. I could instead use a *finite* sum `∑_{k=1}^{d} tr(B^k) = 0`, which is valid because in a `d`-node graph any cycle has length at most `d`, and this needs no spectral-radius condition. But it inherits the second disease in the worst way: the entries of `B^k` overflow for even modest `d`, so the function and its gradient are numerically useless.

So the real requirement crystallizes: I need a power series in `B` that (i) vanishes iff every `tr(B^k) = 0`, (ii) converges for *all* matrices, and (iii) damps the explosive high powers. The series of matrix powers with `1/k!` weights is the matrix exponential, and it has exactly these three properties: `e^B = ∑_{k≥0} B^k / k!`, so `tr(e^B) - d = ∑_{k≥1} tr(B^k)/k!`. Each term is a nonnegative count divided by `k!`, so the whole thing is zero iff every count is zero iff acyclic; the `1/k!` makes it converge for any square matrix; and the factorial weights crush exactly the long-walk terms that blew up before.

I should not take "zero iff acyclic" on faith — it's the load-bearing equivalence, and I want to see it on numbers before I build anything on top of it. Take the smallest nontrivial trap: a 2-node cycle, `B = [[0,1],[1,0]]`. Here `tr(B) = 0` — so the naive "the diagonal of `B` is zero, looks fine" test would *wrongly* call this acyclic. But `B^2 = I`, so `tr(B^2) = 2`: the length-2 closed walk is what gives the cycle away, and any honest acyclicity test has to see it. Does the exponential? Computing `tr(e^B) - 2 = 1.0861...`, comfortably positive — it catches the cycle the diagonal missed. Now the acyclic comparison: `A = [[0,1],[0,0]]` (just `1 → 2`) gives `tr(e^A) - 2 = 0` exactly, since `A^2 = 0` kills every higher power. So far so good. One more, to make sure it's the *first* nonzero power that matters and the weighting is right: the 3-cycle `C` (1→2→3→1) has `tr(C) = tr(C^2) = 0` and `tr(C^3) = 3` (the three nodes each sit on one length-3 closed walk). The leading exponential term is therefore `tr(C^3)/3! = 3/6 = 0.5`, and indeed `tr(e^C) - 3 = 0.50417...`, with the small remainder `0.00417 ≈ tr(C^6)/6! = 3/720` — the series decomposition matches digit for digit. So the equivalence isn't just plausible; it computes correctly, and the factorial weighting is doing visible work (the length-3 term contributes 0.5, the length-6 term a hundredfold less). `tr(e^B) = d` iff `B` is a DAG.

But I want real weights, not a binary `B`, because I want gradients. If I just put a real `W` in place of `B`, the count interpretation breaks — negative weights could cancel, and `tr(e^W) - d` could be zero for a cyclic `W`. The fix is to make every "edge weight" nonnegative without losing the support pattern: square entrywise. Use the Hadamard product `W ∘ W`, with `[W ∘ W]_{kj} = w_{kj}^2 ≥ 0`. The exact same closed-walk argument applies to the nonnegative matrix `W ∘ W`, so

```
h(W) = tr( e^{W ∘ W} ) - d
```

is zero iff `G(W)` is acyclic, where `G(W)` has an edge `k → j` iff `w_{kj} ≠ 0`. And it's smooth, and its value means something — it counts *weighted* closed walks, edge `k→j` carrying weight `w_{kj}^2`, so a larger `h` literally means more or more-heavily-weighted cycles, a "DAG-ness" dial. Let me get its gradient, because I'll need it and because deriving it tells me how to compute it. The trace-exponential differential is `d tr(e^M) = tr(e^M dM)` — that follows from `tr(e^M) = ∑_k tr(M^k)/k!` and the cyclic property of trace, which makes the derivative of `tr(M^k)` equal to `k tr(M^{k-1} dM)`, so summing gives `tr(∑_k M^{k-1}/(k-1)! · dM) = tr(e^M dM)`. With `M = W ∘ W`, `dM = 2W ∘ dW`, so `d h = tr(e^{W∘W} (2W ∘ dW)) = ⟨ (e^{W∘W})^T ∘ 2W, dW ⟩`, giving

```
∇h(W) = (e^{W ∘ W})^T ∘ 2W.
```

So both `h` and its gradient cost a single matrix exponential, `O(d^3)`, a well-studied operation. With this in hand the linear DAG problem is just `min_W (1/2n)||X - XW||_F^2 + λ||W||_1` subject to `h(W) = 0`, solvable by a generic constrained optimizer. That is real machinery, and it is exactly the continuous, all-edges-at-once formulation I said I wanted — except it lives entirely on `W`. A linear SEM *has* a `W` — the regression coefficients. My nonlinear SEM `X_j = f_j(X) + z_j` has no matrix of coefficients anywhere. `h` is a function of a `d×d` real matrix, and I have a collection of functions `f_1, ..., f_d`. There's nothing to feed it. Wall.

Let me not panic and instead ask the narrow question: what does `h` actually *need*? It needs, for each ordered pair `(k, j)`, a single nonnegative number that is zero exactly when `f_j` does not depend on `X_k`, and positive when it does. That's it. `h` is agnostic about where that number came from — it only reads off the support pattern and the weighting. So I don't need a coefficient matrix; I need a *real-valued summary of dependence*: a scalar `[W(f)]_{kj} ≥ 0` measuring "how much does `f_j` use coordinate `k`." If I can manufacture such a matrix `W(f)` out of the functions `f_j`, I can drop it straight into the existing `h` and the whole continuous program comes back to life.

So the question becomes: how do I quantify, with one differentiable nonnegative number, whether a smooth function `f_j` depends on its `k`-th input? The honest definition of "doesn't depend on `X_k`" is: holding the other inputs fixed, varying `u_k` doesn't change the output, i.e. `f_j(..., u_k, ...)` is constant in `u_k`. For a smooth function, constancy in a coordinate is precisely the vanishing of its partial derivative in that coordinate, everywhere. So `f_j` is independent of `X_k` iff `∂_k f_j ≡ 0`. Now I want to turn "the function `∂_k f_j` is identically zero" into "a number is zero," and the natural numeric stand-in for "this function is the zero function" is its `L^2` norm. Work in the Sobolev space `H^1(R^d)` — square-integrable functions with square-integrable derivatives, so the norm `||∂_k f_j||_{L^2}` is well-defined. Then `f_j` is independent of `X_k` iff `||∂_k f_j||_{L^2} = 0`. (This partial-derivative-as-dependence idea isn't out of nowhere — it's the same device used for nonparametric variable selection, e.g. Rosasco et al. 2013.) Define

```
[W(f)]_{kj} := || ∂_k f_j ||_{L^2}.
```

This is a genuine real, nonnegative `d × d` matrix, zero exactly on the non-edges, and it's exactly the kind of object `h` wanted. The program becomes `min_f (1/n) ∑_j ℓ(x_j, f_j(X))` subject to `h(W(f)) = 0`. Before I trust it I should check the one case where I already know the answer: the linear parametric case `f_j(X) = w_j^T X`. There `∂_k f_j = w_{kj}` is constant, so `||∂_k f_j||_{L^2} ∝ |w_{kj}|` — dependence on coordinate `k` disappears exactly when `w_{kj} = 0`, and the surrogate's zero pattern is the usual weighted adjacency matrix. Let me actually run `h` on a small linear `W` to be sure the wiring agrees: for the acyclic `W` with edges `1→2` (weight 0.8) and `2→3` (weight −1.2), `h(W) = tr(e^{W∘W}) - 3 = 0`; add a back-edge `3→1` to close a cycle and `h(W) = 0.115...` turns positive. So on the linear instances the new surrogate reproduces linear NOTEARS exactly — same matrix, same `h`, same zero/positive verdict. The linear method is recovered as a special case rather than competed with, which is what I'd want from a correct generalization. The partial-derivative `L^2` summary is the surrogate `W` for the nonparametric world.

Now, two things are still abstract. The space of `f_j ∈ H^1` is infinite-dimensional, and `||∂_k f_j||_{L^2}` is an integral I can't compute. I have to choose a finite-dimensional family for the `f_j` such that (a) it can approximate the true nonlinear mechanisms, (b) the dependence summary `[W]_{kj}` becomes a tractable, differentiable function of the parameters, and (c) the residual fit and the constraint are both differentiable so a gradient solver can move. The most flexible differentiable family I have is a neural net. Take each `f_j` to be a multilayer perceptron,

```
MLP(u; A^{(1)}, ..., A^{(h)}) = σ( A^{(h)} σ( ··· σ( A^{(1)} u ) ) ),   A^{(1)} ∈ R^{m_1 × d},
```

with a smooth activation `σ`. Universal approximation says enough hidden units capture any smooth `f_j`, and everything is differentiable in the weights. The input layer `A^{(1)}` is `m_1 × d`: it linearly mixes the `d` input coordinates into `m_1` hidden pre-activations before any nonlinearity touches them.

Here's where I have to be careful, because the obvious move is dangerous. I want to know when this MLP is independent of input coordinate `u_k`. The tempting answer: if the `k`-th *column* of `A^{(1)}` is all zeros, then `u_k` never enters the first linear map, so the whole network ignores it. That direction is clearly true. But the worry is the converse and the cost. Suppose I *enforce* independence on `X_k` by zeroing the `k`-th column of `A^{(1)}`. Have I crippled the network? Could there be MLPs that genuinely don't depend on `u_k` yet need a nonzero `k`-th column — using later layers to cancel the `u_k` contribution — so that constraining the column to zero loses expressivity? If so, my dependence summary would be measuring the wrong thing, and forcing sparsity through `A^{(1)}` columns would throw away representable functions. I need to settle this, not hand-wave it.

Let me prove the two function classes coincide. Let `F` be the set of MLPs (with the architecture fixed except the weights) that are independent of `u_k`, and `F_0` the set of MLPs whose `k`-th column of `A^{(1)}` is all zeros. One inclusion is the easy direction I already saw: if `f_0 ∈ F_0`, then `A^{(1)} u` doesn't involve `u_k` at all, so neither does anything downstream, hence `f_0 ∈ F`; `F_0 ⊆ F`. The real content is `F ⊆ F_0`: every MLP that *happens* to be independent of `u_k`, no matter how, can be *rewritten* as an MLP with the `k`-th column of `A^{(1)}` zeroed and the rest of its weights unchanged. Take any `f ∈ F`, so `f(u) = MLP(u; A^{(1)}, ..., A^{(h)})` and `f` is independent of `u_k`. Let `ũ` be `u` with its `k`-th coordinate set to zero, the other coordinates left alone. Since `f` doesn't depend on `u_k` and `u, ũ` differ only in coordinate `k`, `f(u) = f(ũ)`. Now build `Ã^{(1)}` from `A^{(1)}` by zeroing only the `k`-th column and keeping all other columns. Then for each hidden unit `s`,

```
(Ã^{(1)} u)_s = ∑_{k' ≠ k} A^{(1)}_{s k'} u_{k'} = ∑_{k'} A^{(1)}_{s k'} ũ_{k'} = (A^{(1)} ũ)_s,
```

where the middle equality holds because the only term I dropped on the left is the `k'=k` term, and on the right the `k'=k` term is already killed by `ũ_k = 0`. So `Ã^{(1)} u = A^{(1)} ũ` as vectors. Feed both through the *same* later layers:

```
f(u) = f(ũ) = σ(A^{(h)} σ(··· σ(A^{(1)} ũ))) = σ(A^{(h)} σ(··· σ(Ã^{(1)} u))) = MLP(u; Ã^{(1)}, A^{(2)}, ..., A^{(h)}).
```

This last network has the `k`-th column of its first layer zero, so it lies in `F_0`, and it computes the same function `f`. Hence `f ∈ F_0`, so `F ⊆ F_0`, and combined with the easy inclusion, `F = F_0`. (Biases don't change the argument — they're constants that ride along.) That settles it: enforcing independence on `X_k` by zeroing the `k`-th column of `A^{(1)}` loses *no* expressivity. There is no clever cancellation in later layers I'm forbidding; every `u_k`-independent function the architecture can express is already reachable with a zero column.

This proposition is doing more than reassurance — it hands me the dependence summary for free and tells me something important about cost. Independence on `X_k` is controlled *entirely* by the `k`-th column of `A^{(1)}`, regardless of how deep the network is. So the right scalar dependence measure for the MLP version of `f_j` is the norm of that column: `[W(θ)]_{kj} = ||\,k\text{-th column of } A_j^{(1)}\,||_2`. And note the consequence for the constraint: `h(W(θ))` depends on the network only through the `d` first-layer columns, so the acyclicity machinery is *independent of network depth* — I can stack as many hidden layers as I like and the acyclicity computation doesn't grow. That is the payoff of proving `F = F_0` rather than tracking dependence path-by-path through every layer, which would have made the constraint scale linearly with depth (the alternative route some neural DAG methods take, e.g. Lachapelle et al. 2019). My acyclicity cost is just the cost of `h` on a `d×d` matrix, no matter how rich the regressors.

So now I can write a concrete program. Let `θ_j = (A_j^{(1)}, ..., A_j^{(h)})` be the parameters of the `j`-th MLP, `θ = (θ_1, ..., θ_d)`. The fit term is least squares — for an additive-noise model, regressing `x_j` on `MLP(X; θ_j)` with squared loss is the natural (Gaussian-noise) negative log-likelihood up to constants, and any other differentiable loss / GLM link `g_j` would slot in the same way. For sparsity I penalize the first-layer weights, since those are what carry the parent structure; an `ℓ_1` penalty `||A_j^{(1)}||_{1,1}` drives whole columns to zero, which is exactly "drop a parent." And acyclicity is `h(W(θ)) = 0` with `[W(θ)]_{kj} = ||\,k\text{-th col of } A_j^{(1)}\,||_2`:

```
min_θ  (1/n) ∑_{j=1}^d ℓ( x_j, MLP(X; θ_j) ) + λ ∑_{j=1}^d ||A_j^{(1)}||_{1,1}    s.t.   h(W(θ)) = 0.
```

Let me get the surrogate matrix into a form `h` can eat directly, because there's a small but real subtlety. `h` wants the nonnegative matrix `W ∘ W` — entries `w_{kj}^2`. My `[W(θ)]_{kj}` is a column *norm* `sqrt(∑_b (A_j^{(1)})_{bk}^2)`. So `[W(θ)]_{kj}^2 = ∑_b (A_j^{(1)})_{bk}^2`, the *squared* column norm, should be exactly the `(k,j)` entry of `W ∘ W` — meaning I never need the square root to evaluate the constraint. Let me put numbers on it and on the column-dependence claim at once, with a tiny first layer: `d = 3` inputs, `m_1 = 2` hidden units, `A_j^{(1)} = [[0, 2, -1],[0, 0.5, 3]]` (column 0, the input `X_0`, deliberately zeroed). Summing squares down each column gives `[0², 2²+0.5², (−1)²+3²] = [0, 4.25, 10]` — so `[W∘W]_{·j} = (0, 4.25, 10)`: zero on `X_0` exactly, positive on the inputs that actually enter. And feeding two inputs that differ only in coordinate 0, say `(10, 1, 2)` and `(−99, 1, 2)`, through the full MLP (with a downstream layer) returns the same output `−0.69999...` both times — a concrete instance of the independence the column norm is supposed to detect: the zeroed column really does make the network blind to `X_0`. So the identity holds and the surrogate measures what I claimed. Define `A_{kj} := ∑_b (A_j^{(1)})_{bk}^2` (sum of squares down the column), which is `[W∘W]_{kj}`, and compute `h = tr(e^{A}) - d` straight on `A`. The square root only appears at the very end, when I want to report the weighted adjacency `W = sqrt(A)` to threshold and read off edges. Keeping the constraint on the squared quantity also keeps it smooth in `θ` (no `sqrt` near zero), which the solver will thank me for.

Now, how do I actually solve a smooth objective with a smooth *equality* constraint `h = 0`? The constraint surface `{h(θ) = 0}` is nonconvex, so I'm going to be content with stationary points. The clean classical tool is the augmented Lagrangian: solve a sequence of unconstrained problems while pushing the constraint to zero. The augmented Lagrangian is

```
L^ρ(θ, α) = (smooth fit) + (ρ/2) |h(W(θ))|^2 + α h(W(θ)),
```

plus the `ℓ_1` term. The `(ρ/2) h^2` is a quadratic penalty that punishes constraint violation; the `α h` is the Lagrange-multiplier (dual) term. Why both, rather than just cranking a pure quadratic penalty? Because a pure penalty needs `ρ → ∞` to drive `h` exactly to zero, and large `ρ` makes the subproblem brutally ill-conditioned. The augmented Lagrangian, by carrying the multiplier `α`, recovers the constrained solution *without* sending `ρ` to infinity — the multiplier absorbs the residual so a finite `ρ` suffices (the standard augmented-Lagrangian fact). The outer loop is dual ascent: since the dual function `D(α) = min_θ L^ρ(θ, α)` has derivative `∇D(α) = h(θ_α^*)` (the constraint value at the inner minimizer), I update `α ← α + ρ h`. And I only need to grow `ρ` when the constraint is being stubborn: if after an inner solve `h` hasn't shrunk to at most a fraction (say a quarter) of its previous value, multiply `ρ` by ten and resolve; otherwise accept the step. Terminate when `h` is below a tiny tolerance (machine-precision-ish, `1e-8`) or `ρ` hits a ceiling.

Each inner subproblem is an unconstrained smooth-plus-`ℓ_1` minimization. The smooth part — least squares plus the `(ρ/2)h^2 + αh` penalty plus an optional `ℓ_2` weight decay — is exactly the shape a quasi-Newton method eats. L-BFGS-B is the natural choice: limited-memory BFGS with box constraints. But I have a nonsmooth `ℓ_1` term and two structural constraints (no self-loops; and I want the penalty to behave like a clean linear term), and L-BFGS-B only handles *box* bounds, not an `ℓ_1` prox. The trick — inherited from the linear formulation — is variable splitting. Write each first-layer weight as a difference of two nonnegative parts, `A^{(1)} = A^{(1)}_+ - A^{(1)}_-` with `A^{(1)}_+, A^{(1)}_- ≥ 0`. Then `|A^{(1)}_{bk}| = (A^{(1)}_+)_{bk} + (A^{(1)}_-)_{bk}` at the optimum (because at least one of the two parts is zero when they're both pushed nonnegative and penalized), so the `ℓ_1` norm becomes a *linear* function `∑ (A_+ + A_-)` of nonnegative variables — and a linear penalty under nonnegativity box constraints is exactly what L-BFGS-B can optimize. The box bounds do double duty: lower-bound both split parts at zero to realize the splitting, and pin the *diagonal* entries to `(0,0)` so a variable can never be its own parent (no self-loops, baked in for free).

Let me also sanity-check the architecture knobs against what the problem needs. How many hidden units? Zero hidden units is just a linear model — and I expect a sharp jump in recovery the instant I allow even a few units, because that's the difference between seeing the nonlinear cause/effect asymmetry and not; the identifiability argument was *about* nonlinearity. Too many units, though, and with scarce samples (say `n = 200`) I'll overfit each regressor and read spurious dependence into columns that should be zero. So a modest single hidden layer — on the order of ten units — sits in the sweet spot: enough nonlinearity to expose the asymmetry, few enough parameters to estimate from limited data. Activation `σ` should be smooth so the partial derivatives (and the whole objective) are differentiable for the gradient solver and so the "constancy ⇔ zero derivative" logic is clean — a sigmoid does fine. The final hard threshold is a post-processing step, not part of the continuous constraint: after the augmented Lagrangian, `h` is only `≤ 1e-8`, not symbolically zero, so the learned `W = sqrt(A)` can contain tiny numerical or weak spurious entries. Hard-thresholding `|W| < ω` to zero removes those weak edges; deleting edges cannot create cycles, and the canonical implementation uses `ω = 0.3` as a fixed practical default.

Let me put the whole thing into the code I'd actually run, filling the empty slots in the harness — the per-node MLP family, the map from its first-layer weights to the squared-column-norm matrix that the acyclicity function consumes, and the `ℓ_1` penalty — and using the variable-split trick so a bound-constrained quasi-Newton solver handles everything:

```python
import math
import numpy as np
import scipy.linalg as slin
import torch
import torch.nn as nn


class TraceExpm(torch.autograd.Function):
    """h's core: f = tr(exp(A)). d tr(exp(A)) = tr(exp(A) dA), so the gradient is exp(A)^T."""
    @staticmethod
    def forward(ctx, input):
        E = slin.expm(input.detach().numpy())     # matrix exponential, O(d^3)
        f = np.trace(E)
        E = torch.from_numpy(E)
        ctx.save_for_backward(E)
        return torch.as_tensor(f, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (E,) = ctx.saved_tensors
        return grad_output * E.t()                 # (e^A)^T

trace_expm = TraceExpm.apply


class LocallyConnected(nn.Module):
    """One independent linear map per node j (so the d MLPs run batched). [n,d,m1] -> [n,d,m2]."""
    def __init__(self, num_linear, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_linear, in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(num_linear, out_features)) if bias else None
        k = 1.0 / in_features
        nn.init.uniform_(self.weight, -math.sqrt(k), math.sqrt(k))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -math.sqrt(k), math.sqrt(k))

    def forward(self, x):                           # [n,d,1,m1] @ [1,d,m1,m2]
        out = torch.matmul(x.unsqueeze(2), self.weight.unsqueeze(0)).squeeze(2)
        if self.bias is not None:
            out = out + self.bias
        return out


class NotearsMLP(nn.Module):
    """Each f_j is an MLP; dependence of f_j on X_k = norm of the k-th column of its first layer."""
    def __init__(self, dims, bias=True):           # dims = [d, m1, ..., 1]
        super().__init__()
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # First layer, variable-split for an L1 that a bound-constrained solver can handle:
        # effective W1 = fc1_pos - fc1_neg, both kept >= 0.  Shape [d*m1, d] = [j*m1, i].
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # Deeper layers: one local linear map per node.
        self.fc2 = nn.ModuleList(
            [LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias)
             for l in range(len(dims) - 2)])

    def _bounds(self):
        # Nonneg parts everywhere; diagonal (i==j) pinned to 0 to forbid self-loops.
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for _ in range(self.dims[1]):
                for i in range(d):
                    bounds.append((0, 0) if i == j else (0, None))
        return bounds

    def forward(self, x):                           # [n,d] -> [n,d]
        x = self.fc1_pos(x) - self.fc1_neg(x)       # [n, d*m1], effective first layer
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2:
            x = torch.sigmoid(x)                     # smooth activation
            x = fc(x)
        return x.squeeze(2)                          # [n, d]

    def _squared_colnorms(self):
        # [W(theta)^2]_{kj} = sum_b (A_j^{(1)})_{bk}^2 = squared k-th-column norm = [W o W]_{kj}.
        d = self.dims[0]
        w1 = (self.fc1_pos.weight - self.fc1_neg.weight).view(d, -1, d)  # [j, m1, i]
        return torch.sum(w1 * w1, dim=1).t()        # [i, j] = [k, j]

    def h_func(self):
        A = self._squared_colnorms()                # already W o W (nonnegative)
        return trace_expm(A) - self.dims[0]         # tr(exp(W o W)) - d

    def l2_reg(self):
        reg = torch.sum((self.fc1_pos.weight - self.fc1_neg.weight) ** 2)
        for fc in self.fc2:
            reg = reg + torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        # L1 of the (split) first-layer weights -> linear in the nonneg parts -> drops parents.
        return torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)

    @torch.no_grad()
    def fc1_to_adj(self):                            # report W = sqrt(W o W) for thresholding
        return torch.sqrt(self._squared_colnorms()).cpu().numpy()


def squared_loss(output, target):                   # ANM least squares = Gaussian NLL up to const
    n = target.shape[0]
    return 0.5 / n * torch.sum((output - target) ** 2)


def dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h_prev, rho_max):
    """One augmented-Lagrangian subproblem solved with bound-constrained L-BFGS-B."""
    optimizer = LBFGSBScipy(model.parameters())     # uses the per-weight .bounds above
    X_t = torch.from_numpy(X)
    h_new = None
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            loss = squared_loss(model(X_t), X_t)
            h = model.h_func()
            penalty = 0.5 * rho * h * h + alpha * h  # (rho/2) h^2 + alpha h
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            obj = loss + penalty + l2_reg + l1_reg
            obj.backward()
            return obj
        optimizer.step(closure)
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h_prev:                    # constraint not shrinking -> stiffen penalty
            rho *= 10
        else:
            break
    alpha += rho * h_new                             # dual ascent: alpha += rho * h
    return rho, alpha, h_new


def notears_nonlinear(model, X, lambda1=0., lambda2=0., max_iter=100,
                      h_tol=1e-8, rho_max=1e16, w_threshold=0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    W = model.fc1_to_adj()                           # W = sqrt(W o W)
    W[np.abs(W) < w_threshold] = 0                   # cut weak edges after h is small
    return W
```

(`LBFGSBScipy` is a thin wrapper that flattens the parameters, reads each weight's `.bounds`, and calls `scipy.optimize.minimize(method='L-BFGS-B', jac=True)` — the existing bound-constrained quasi-Newton solver doing the inner minimization.)

Let me trace the chain back through once and check that each link is one I actually settled rather than assumed. I started needing the DAG behind nonlinear observational data, blocked by a superexponential combinatorial acyclicity constraint that forces every existing method into edge-by-edge or ordering search. Identifiability told me the problem is solvable — nonlinear additive noise breaks the cause/effect symmetry that sinks the Gaussian-linear case — so the goal is a single continuous program, not a combinatorial one. The linear world already had the key device: `tr(B^k)` counts closed walks, the matrix exponential `tr(e^{W∘W}) - d` collapses "no cycle of any length" into one smooth, numerically-stable equation with a clean `(e^{W∘W})^T ∘ 2W` gradient — and I checked the equivalence on numbers (the 2-cycle scores 1.086 where the diagonal test fails, the 3-cycle's 0.504 decomposes as `tr(C^3)/3! + tr(C^6)/6! + ...`), so I'm confident the constraint does what it claims. But it needs a weighted adjacency matrix, which a nonparametric SEM doesn't have. The way past that wall: `h` only needs a nonnegative number per ordered pair saying "does `f_j` use `X_k`," and a smooth function ignores a coordinate iff its partial derivative there vanishes, so `[W(f)]_{kj} = ||∂_k f_j||_{L^2}` is the surrogate matrix `h` wants — and on the linear instances I ran, `h` on this surrogate reproduces linear NOTEARS exactly. To make it finite and differentiable I modeled each `f_j` as an MLP, and proving that the MLPs independent of `X_k` are precisely those with a zeroed `k`-th first-layer column showed that the squared column norm `∑_b (A_j^{(1)})_{bk}^2` is the right dependence summary, costs nothing in expressivity, and makes acyclicity depth-independent — and it equals `[W∘W]_{kj}` directly (I verified the sum-of-squares matches the squared column norm on a small layer, and that the zeroed column really blinds the network to that input), so `h` runs on it without a square root. Sparsity is an `ℓ_1` on the first-layer columns (drop a parent), realized by variable-splitting into nonnegative parts so a bound-constrained L-BFGS-B can carry it, with the diagonal pinned to zero to kill self-loops and optional `ℓ_2` regularization carried exactly as in the implementation. The constraint is driven to zero by an augmented Lagrangian — quadratic penalty plus multiplier so finite `ρ` suffices — growing `ρ` by ten only when `h_new > 0.25 h_prev`, and a final hard threshold cuts weak residual edges after the constraint is small. One continuous program, generic solver, the whole graph updated at once.
