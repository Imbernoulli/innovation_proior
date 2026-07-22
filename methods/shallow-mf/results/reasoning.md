Let me start from the thing that actually bothers me. I have a matrix `X` I want to recover, I get `m` linear measurements `A(X) = y` with `A(X)_i = <A_i, X>`, and `m` is tiny compared to `n^2`. Matrix completion is the case where each `A_i` is an indicator of a single entry, so I just see a handful of entries and want the rest. The honest first thing to say is that fitting the data is trivial and useless. The set `{X : A(X) = y}` is a huge affine family — `n^2` unknowns, `m ≪ n^2` constraints — and "minimize `||A(X) - y||^2`" hits zero on all of them. In completion I can set every unobserved entry to zero and I have a perfectly valid zero-loss solution that recovers nothing. So the question was never "can I fit", it's "which fitting `X` do I end up at", and that is entirely a property of the algorithm, not the objective.

The clean classical answer is to add structure: assume the truth is low rank and ask for the lowest-rank `X` that fits. That's NP-hard, so the field relaxes rank to its convex envelope, the nuclear norm `||X||_* = Σ_k σ_k`, and solves `min ||X||_* s.t. A(X) = y` as an SDP — Recht–Fazel–Parrilo and Candès–Recht give exact recovery from `O(nr)` measurements, and Srebro–Shraibman tell me low nuclear norm is exactly the quantity that controls generalization for this kind of prediction. So I know *what* I want: a low-nuclear-norm fit. The question is whether I have to put it in by hand as an explicit penalty, or whether something cheaper produces it on its own.

Let me look at what the over-parameterized models people actually train are doing, because there's a nagging empirical fact. The Burer–Monteiro trick writes a PSD `X = U U^T` and optimizes the factor `U ∈ R^{n×d}`; if `d < n` that hard-caps `rank(X) ≤ d` and recovery works when `d` matches the rank. Fine. But take `d = n`. Now `X = U U^T` ranges over *all* PSD matrices, the factorization imposes nothing, the `U`-problem is equivalent to the `X`-problem, and we're back to the same underdetermined mess with the same family of bad global optima. And yet — this is the part I can't ignore — if I just run gradient descent on `U` from an initialization close to zero with a small step size, the matrix it lands on is not a random member of that family. It reconstructs a planted low-rank `X*` almost exactly, and its nuclear norm sits right on top of the minimum-nuclear-norm SDP value. Push the initialization away from zero, or take big steps, and it drifts off that point. So there's a strong implicit preference here, it depends on *how* I descend, and it seems to be aimed at exactly the nuclear-norm solution I wanted. I want to understand why, because if I do, I get nuclear-norm recovery for free out of plain gradient descent on an unconstrained factorization.

Before I touch the factorization, let me make sure I understand the boring case, gradient descent directly on `X`. The objective `F(X) = ||A(X) - y||^2` is convex and `∇F(X) = A^*(A(X) - y)`, where `A^*(r) = Σ_i r_i A_i` is the adjoint. The key structural fact: every gradient is a linear combination of the `A_i`, so it lives in the `m`-dimensional subspace `L = {A^*(s) : s ∈ R^m}`. If I start at `X = 0`, every iterate stays in `L`, and so does the limit. Now which point of `L ∩ {A(X) = y}` is it? Consider `min_X ||X||_F^2 s.t. A(X) = y`. Its KKT conditions are `A(X) = y` and `∃ν : X = A^*(ν)` — the gradient of the Frobenius objective, `2X`, must be in the constraint span. On `L` the second condition holds automatically, and if I reach zero error the first holds too. So gradient descent on `X` from zero converges to the **minimum-Frobenius-norm** solution. That's a clean result, and it's also exactly the wrong answer: for completion the minimum-Frobenius fit is the impute-zeros matrix. Frobenius norm is the wrong complexity measure. So descending on `X` gives me the bias I don't want; whatever the factorization is doing, it must be changing *which* norm gets implicitly minimized.

So let me descend on `U` instead and track what happens to `X = U U^T`. Gradient flow — the small-step-size idealization, `U̇ = -∇f(U)` — gives

```
U̇_t = -A^*(A(U_t U_t^T) - y) U_t = -A^*(r_t) U_t ,   r_t = A(X_t) - y .
```

I am using the standard gradient-flow normalization here. If the implemented loss is written as an unnormalized or averaged squared error, a positive scalar appears in front of the right-hand side; that only rescales time and can be absorbed into `s_t`, so it does not change the limit argument. Now the move I care about is what this does to `X_t = U_t U_t^T`, not to `U` itself. Chain rule:

```
Ẋ_t = U̇_t U_t^T + U_t U̇_t^T = -A^*(r_t) U_t U_t^T - U_t U_t^T A^*(r_t)
    = -A^*(r_t) X_t - X_t A^*(r_t) .
```

Stare at that for a second. The `U` dropped out — the dynamics of `X` depend only on `X` itself and the residual, not on the particular factorization. That's reassuring (the answer can't depend on an arbitrary square root) but it's also the crux of why this is different from descending on `X`. There the flow was `Ẋ = -∇F(X) = -A^*(r_t)`: the velocity is just an element of the flat span `L`, so `X` slides around inside the affine subspace `L` and lands at the min-Frobenius point. Here the velocity is `-A^*(r_t) X_t - X_t A^*(r_t)` — the same residual direction `A^*(r_t)`, but now **multiplied by the current `X` on both sides**. That `X` factor is everything. It means the velocity is no longer confined to a fixed subspace; the directions available to the flow depend on where it currently is, and crucially the velocity is suppressed wherever `X` is small. Start near zero and the flow can barely move in the directions `X` hasn't already grown into. So this is some kind of self-reinforcing, multiplicative growth, and that smells like it should prefer a few dominant directions — low rank. I need to make that precise.

Let me state the thing I think is true and then try to prove it. For a full-rank `X_init` (which needs the full-dimensional `d = n` so `U_init` can be full rank), if I scale the initialization down and take the limit, `X̂ = lim_{α→0} X_∞(α X_init)`, and if that limit is a zero-error global optimum, then

```
X̂ ∈ argmin_{X ⪰ 0} ||X||_*  s.t.  A(X) = y .
```

The global-optimality assumption is mild: for almost every initialization gradient flow goes to a local minimizer (Lee et al.), and at `d = n` every local minimizer of the factorized problem is a global minimizer of the convex problem (Journée et al.), and since we're underdetermined the global optimum has zero error. So the open part is genuinely *which* zero-error solution, and I'm claiming it's the nuclear-norm one. Now prove it — or at least prove it where I can.

Start as simple as possible, `m = 1`, a single measurement `A_1 = A`, `y_1 = y`. Then `r_t` is a scalar and the flow is `Ẋ_t = -r_t (A X_t + X_t A)`. This is a linear (in `X`) time-varying ODE with the two generators `A·` and `·A` that commute with each other, so I can integrate it: let `s_T = -∫_0^T r_t dt`, then

```
X_t = exp(s_t A) X_0 exp(s_t A) .
```

Quick check by differentiating: `d/dt exp(s_t A) = ṡ_t A exp(s_t A) = -r_t A exp(s_t A)` since `A` commutes with its own exponential, and applying the product rule to `exp(s_t A) X_0 exp(s_t A)` reproduces `-r_t(A X_t + X_t A)`. Good. Now what does this tell me about the limit? The KKT conditions for the nuclear-norm SDP `min_{X⪰0} ||X||_* s.t. A(X)=y` are: `A(X) = y`, `X ⪰ 0`, and a dual `ν` with `A^*(ν) ⪯ I` and `(I - A^*(ν)) X = 0`. The first I'm assuming, the second the factorization gives me for free, so the content is the last two: `X̂` must be spanned by the eigenvectors where the scaled dual matrix reaches eigenvalue `1`. In the one-measurement case that means the top eigenspace of the signed and rescaled matrix `ν A`, not necessarily the top eigenspace of `A` with a fixed sign. The exponential solution makes that almost visible: `X_t = exp(s_t A) X_0 exp(s_t A)` amplifies `X_0` along the eigenvectors with the largest `s_∞ · (eigenvalue of A)`. For the limit to fit a nonzero `y` as the product initialization goes to zero, I need `|s_∞| → ∞` (the tiny `X_0` has to be blown up to finite size), and once `|s_∞| → ∞` the exponential is dominated by exactly that signed leading eigenspace. That's the complementary-slackness condition. So at `m = 1` the nuclear-norm KKT conditions hold. The mechanism is clear: the multiplicative `exp(s A)` on both sides is a power-iteration-like amplifier, and small init + the need to reach finite scale forces it to amplify only the dominant signed direction.

Now push to general `m` but keep the case I can actually solve: the `A_i` **commute**, `A_i A_j = A_j A_i`. Then they're simultaneously diagonalizable, and `A^*(s) = Σ_i s_i A_i` is too, and the same exponential ansatz works. Define the vector integral `s_t = -∫_0^t r_t dt`; I claim

```
X_t = exp(A^*(s_t)) X_0 exp(A^*(s_t)) .
```

Differentiate to confirm: because all the `A_i` commute, `A^*(s_t)` commutes with its own time derivative `A^*(ṡ_t) = -A^*(r_t)`, so `d/dt exp(A^*(s_t)) = -A^*(r_t) exp(A^*(s_t))`, and the product rule again gives `Ẋ_t = -A^*(r_t) X_t - X_t A^*(r_t)`. Good, the ansatz solves the flow.

Now the limit argument, carefully, because this is where signs and factors have to be right. I need to distinguish the scale of the product matrix from the scale of the factors. Write the product initialization as `X_0 = γ I`. If I initialize two square factors as `α I`, then `γ = α^2`; if I state the conjecture directly in `X`-space as `X_0 = α I`, then `γ = α`. To keep the exponential algebra clean, set `β := -1/2 log γ`, so `γ = exp(-2β)`; in the two-factor identity-initialization convention this is exactly `β = -log α`. Since the `A_i` commute and are symmetric, pick a common eigenbasis `v_1, ..., v_n`; `A^*(s)` is diagonal in it for every `s`, so `X_∞(γ I)` and its limit `X̂` share that eigenbasis. As `γ → 0`, `X_∞(γ I) → X̂`, so each `v_k^T X_∞(γ I) v_k → v_k^T X̂ v_k`, i.e. the eigenvalue along `v_k`, call it `λ_k(X_∞(γ I)) → λ_k(X̂)` (here `λ_k` means the eigenvalue *for eigenvector `v_k`*, not the `k`-th largest). And from the exponential form,

```
λ_k(X_∞(γ I)) = γ · exp(2 λ_k(A^*(s_∞))) = exp(2 λ_k(A^*(s_∞)) - 2β) .
```

The product-scale `γ` out front became `exp(-2β)`; the two exponentials on either side, both `exp(λ_k(A^*(s_∞)))` along `v_k`, multiply to `exp(2 λ_k(A^*(s_∞)))`. Good. As `γ → 0`, `β → ∞`. Now split on whether the limiting eigenvalue is positive or zero.

For a `k` with `λ_k(X̂) > 0`: take logs (continuous on the positives),

```
2 λ_k(A^*(s_∞(β))) - 2β - log λ_k(X̂) → 0 .
```

Divide by `2β`:

```
λ_k(A^*(s_∞(β))/β) - 1 - log λ_k(X̂)/(2β) → 0 .
```

The last term `→ 0` because `β → ∞`, so with the rescaled dual `ν(β) := s_∞(β)/β`,

```
lim_{β→∞} λ_k(A^*(ν(β))) = 1 .
```

That's exactly the complementary-slackness condition `(I - A^*(ν))X̂ = 0` reading eigenvalue by eigenvalue: on every direction where `X̂` is supported, `A^*(ν)` has eigenvalue 1.

For a `k` with `λ_k(X̂) = 0`: now `exp(2 λ_k(A^*(s_∞(β))) - 2β) → 0`. Rewrite the exponent as `2β · (λ_k(A^*(ν(β))) - 1)`, so the quantity is `(exp(λ_k(A^*(ν(β))) - 1))^{2β} → 0`. For this to go to zero with `β → ∞`, the base must be below 1: for every `ε ∈ (0,1]`, for `β` large enough, `exp(λ_k(A^*(ν(β))) - 1) < ε^{1/(2β)} < 1`, hence `λ_k(A^*(ν(β))) < 1`. So on the null space of `X̂`, the eigenvalues of `A^*(ν)` stay strictly below 1 and their limit is `≤ 1`.

Put the two together: `lim A^*(ν(β)) ⪯ I` (eigenvalues 1 on the support, `≤ 1` off it) — that's dual feasibility — and `lim A^*(ν(β)) X̂ = X̂` — that's complementary slackness. Both nuclear-norm KKT conditions hold (the other two were given), so `X̂` minimizes the nuclear norm. Done with the commutative case.

What's striking, and worth pausing on, is what the proof actually used. It never touched the specific form of the residual control `r_t` — it only used that the flow stays on the set

```
M = { exp(A^*(s)) X_init exp(A^*(s)) : s ∈ R^m } .
```

Because the `A_i` commute, the tangent space of `M` at a point `X` is `span{A_i X + X A_i}`, which is exactly the set of allowed velocities `-A^*(r) X - X A^*(r)`, so the flow can never leave `M`. Any control `r_t` that drives the residual to zero, following `Ẋ = -A^*(r) X - X A^*(r)`, lands on a nuclear-norm minimizer. That means the bias is robust: replace the Euclidean loss with another reasonable one, or use only a subset of measurements per step the way stochastic gradient descent does, and as long as you reach a zero-error optimum the limit is still the nuclear-norm solution.

But the same observation tells me why the *step size* matters and why I can't just take big greedy steps. The manifold `M` is **curved** — its tangent space is a different subspace at every point. Gradient descent takes finite-length steps along the current tangent, which, on a curved surface, walk you *off* the manifold. The min-Frobenius story for descent on `X` worked precisely because that manifold was *flat* (an affine subspace `L`), so any linear combination of gradients — momentum, conjugate gradient, Nesterov — stays on it and lands at the same min-norm point. Here, the moment I use a finite step or accumulate momentum, I shoot off `M` and lose the guarantee. The only way to stay on the curved manifold is to take infinitesimal steps, i.e. to actually approximate the gradient flow. That's the principled reason behind "small step size": it isn't a tuning nicety, it's what keeps the trajectory on the manifold whose intersection with the feasible set is the nuclear-norm solution. In practice I'll discretize with a small learning rate and let a stable optimizer keep the descent well-behaved, but I should understand that the bias is a property of the *flow*, and the discretization is an approximation to it.

Now the honest hole: when the `A_i` **don't** commute — which is the generic case for off-diagonal or lifted completion measurements, even though purely diagonal coordinate measurements are commutative — the exponential ansatz fails. `d/dt exp(F_t) = Ḟ_t exp(F_t)` only when `Ḟ_t` commutes with `F_t`, and here it doesn't, so `X_t = exp(A^*(s_t)) X_0 exp(A^*(s_t))` is no longer a solution. The true solution is a **time-ordered exponential**,

```
X_t = lim_{ε→0} ( ∏_{τ = t/ε}^{1} exp(-ε A^*(r_{τε})) ) X_0 ( ∏_{τ = 1}^{t/ε} exp(-ε A^*(r_{τε})) ) ,
```

where the order of the factors matters. If the `A_i` commuted, the product of exponentials would collapse to one exponential of the sum and I'd recover the manifold `M`. They don't, so the path leaves `M`. Can I just build a bigger manifold `M'` containing all the directions the flow can take, so I can run the same KKT argument there? The directions reachable by combining infinitesimal commuting-looking steps are governed by the Lie algebra generated by `{A_i}` under the commutator `[A_i, A_j] = A_i A_j - A_j A_i`. And here's the wall: even for `m = 2`, the closure of two generic non-commuting symmetric matrices under the commutator is typically *all* of `R^{n×n}`. So `M'` would be the entire space — there's no nontrivial manifold to confine the flow, and I can't conclude the KKT conditions for an arbitrary control. The argument that worked by "the path stays on a small manifold" simply has no small manifold to use.

So is the bias actually gone in the non-commutative case, or is the abstract worst case just not what gradient flow does? Look at the cost of actually *exploiting* a commutator direction. To make finite progress along `[A_i, A_j] X + X [A_i, A_j]^T`, I'd have to use a wildly non-smooth control — loop on the order of `1/ε^2` times through `ε`-steps in the directions `A_i, A_j, -A_i, -A_j`, each four-step loop netting only an `ε^2` move in the commutator direction. The residual control `r_t = A(X_t) - y` of an actual gradient flow is nothing like that — it's smooth and slowly varying. For smooth control the non-commutative terms in the time-ordered-exponential expansion are higher order in the step relative to the direct `A^*(s)` term, and they vanish faster as `X_init → 0`. So the heuristic is: as initialization shrinks, the flow can't accumulate enough wiggle to move off the commutative manifold by a non-negligible amount, and the commutative analysis governs the limit. This isn't a proof — I'll keep it as a conjecture for the non-commutative case — but it's a well-motivated one, and it's the kind of statement I'd want to check numerically against the actual residual controls and against random smooth controls before trusting it. The clean theorem I *do* have is the commutative one; the general claim is the conjecture that small init + small step on the full-dimensional factorization yields the minimum-nuclear-norm fit.

There's also a tidy degenerate corollary that sanity-checks the whole picture. Take `X_init` and all the `A_i` diagonal. Then the matrix problem decouples into a vector problem: parameterize `x(u)_i = u_i^2`, minimize `||A x(u) - y||^2` over `u`, and the gradient-flow limit from `α·1` as `α → 0` is the minimum-`ℓ_1` nonnegative solution, `argmin_{x ≥ 0} ||x||_1 s.t. A x = y`. That's the vector shadow of the nuclear-norm result — nuclear norm on diagonal matrices *is* the `ℓ_1` norm of the diagonal — and it's exactly the "implicit `ℓ_1` from squaring the parameterization" phenomenon, which I already believe. Consistent.

Let me also be precise about *why each ingredient must be what it is*, because the recipe is deceptively plain. Why factorize at all instead of descending on `X`? Because descent on `X` gives min-Frobenius = impute-zeros, the wrong bias; factorizing swaps the implicit norm from Frobenius to nuclear, which is the rank-promoting one I want. Why full dimension `d = n`? So the factorization imposes *no* explicit rank constraint — I want the low-rank preference to come entirely from the dynamics, not from a hard cap, both because I usually don't know the true rank and because the whole point is that over-parameterized models generalize *without* the constraint. Why initialization close to zero? The theorem is a statement about `lim_{α→0}`; concretely, reaching a finite-scale `X̂` from a vanishing start forces `|s_∞| → ∞`, which is precisely what makes the `exp(A^*(s))` amplifier collapse onto the eigenspace where the rescaled dual saturates at eigenvalue `1` and select the low-nuclear-norm point. Start large and you behave like descent on `X` — a generic, high-nuclear-norm optimum. Why small step size? The selecting manifold is curved, and only infinitesimal steps stay on it; finite steps and momentum shoot off it.

I can even see the rank-selection mechanism at the level of singular values, which makes the "low rank" tangible. Write the end-to-end matrix as `W = W_2 W_1^T` in the PyTorch layer convention, equivalently `W_2 W_1` in the bare factor convention. The asymmetric depth-2 factorization lifts through `[[·, W],[W^T, ·]]` to the symmetric `U U^T` flow above, so the analysis transfers. Under balanced near-zero initialization the singular values `σ_r` of `W` evolve, for a depth-`N` factorization, by

```
σ̇_r = -N (σ_r^2)^{1 - 1/N} <∇ℓ(W), u_r v_r^T> .
```

For the plain convex case (no factorization, `N = 1`) the prefactor is just 1: every singular value moves at a rate set only by its alignment with the gradient, no size dependence — flat, Frobenius-like. Add one layer of depth, `N = 2`, and the prefactor becomes `(σ_r^2)^{1/2} = |σ_r|`: **the rate is proportional to the singular value itself.** Large singular values accelerate, small ones are throttled. So a few directions that get an early lead run away to finite size while the rest stay pinned near zero — the spectrum self-organizes into a few large values and many near-zero ones, which is an effectively low-rank, low-nuclear-norm matrix. To see the gap concretely, take two singular values; from `σ̇_r ∝ (σ_r^2)^{1/2} = σ_r` (for `N = 2`, with stationary singular vectors so the alignment factor is a shared scalar times each value's coupling), `d log σ_{r_1} / d log σ_{r_2}` is the constant ratio `α_{r_1,r_2}` of their gradient couplings, which integrates to `σ_{r_1} = const · σ_{r_2}^{α_{r_1,r_2}}` — a *power law* between singular values, versus the *linear* `σ_{r_1} = α·σ_{r_2} + const` you get at `N = 1`. The power law is what opens a multiplicative gap between the leading and trailing singular values. That's the same low-rank bias the nuclear-norm argument predicted, now visible directly in the spectral dynamics, and it's exactly why this is different from descending on `X`.

So the method, finally, is just this: parameterize the recovered matrix as a product of two full-dimensional bias-free linear layers, initialize both with tiny Gaussian weights so the end-to-end product starts a hair above zero, and descend the masked squared error on the observed entries with a small step size. No nuclear-norm penalty, no rank cap, no explicit regularizer — the bias toward the minimum-nuclear-norm (low-rank) fit is supplied entirely by gradient descent on the factorization from near-zero. Let me write it the way it actually runs.

```python
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


class GroupRMSprop(Optimizer):
    """RMSprop with one global gradient scale, matching the reference implementation."""

    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-4):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, adjusted_lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            state = self.state
            if len(state) == 0:
                state["step"] = 0
                state["square_avg"] = torch.tensor(0.0)

            square_avg = state["square_avg"]
            alpha = group["alpha"]
            square_avg.mul_(alpha)
            state["step"] += 1

            for param in group["params"]:
                if param.grad is not None:
                    square_avg.add_((1.0 - alpha) * param.grad.detach().pow(2).sum().cpu())

            avg = square_avg.div(1.0 - alpha ** state["step"]).sqrt_().add_(group["eps"])
            adjusted_lr = group["lr"] / avg
            group["adjusted_lr"] = adjusted_lr

            for param in group["params"]:
                if param.grad is not None:
                    param.data.add_(-adjusted_lr.to(param.device) * param.grad.data)
        return loss


class MatrixRecoveryStrategy:
    def recover(self, observed_values, observed_mask, n, rank_hint,
                device, max_iters, log_iters):
        raise NotImplementedError


class ShallowMatrixFactorization(MatrixRecoveryStrategy):
    """Recover X by gradient descent on a full-dimensional depth-2 factorization
    X = W2 @ W1^T, starting from a near-zero initialization."""

    def __init__(self, init_scale=1e-3, lr=1e-3, train_thres=1e-6):
        self.init_scale = float(init_scale)   # near-zero start: the limit alpha -> 0
        self.lr = float(lr)                    # reference GroupRMSprop config uses 1e-3
        self.train_thres = float(train_thres)

    def _build(self, n, device):
        # two square, bias-free linear layers => end-to-end product is unconstrained
        # (full dimension d = n: no explicit rank cap; the bias comes from the dynamics)
        layers = [nn.Linear(n, n, bias=False) for _ in range(2)]
        # std chosen so the end-to-end matrix W2 W1^T has Frobenius norm ~ init_scale * sqrt(n),
        # i.e. starts just above zero: per-layer std = init_scale^(1/depth) * n^(-1/2), depth=2
        scale = (self.init_scale ** 0.5) * (n ** (-0.5))
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0.0, std=scale)
        return nn.Sequential(*layers).to(device)

    @staticmethod
    def _e2e(model):
        # end-to-end product W2 @ W1^T, formed left-to-right: start from W1^T, apply layer 2
        out = None
        for layer in model:
            out = layer.weight.t() if out is None else layer(out)
        return out

    def recover(self, observed_values, observed_mask, n, rank_hint,
                device, max_iters, log_iters):
        model = self._build(n, device)
        optimizer = GroupRMSprop(model.parameters(), lr=self.lr, eps=1e-4)
        mask = observed_mask.to(device)
        target = observed_values.to(device)
        denom = max(int(mask.sum().item()), 1)

        log_every = max(int(log_iters), 1)
        for it in range(max_iters):
            e2e = self._e2e(model)
            residual = (e2e - target) * mask          # error only on observed entries
            loss = residual.pow(2).sum() / denom       # masked MSE, normalized by #obs
            optimizer.zero_grad()
            loss.backward()                            # gradient flow on the factors

            if it % log_every == 0 or it == max_iters - 1 or loss.item() <= self.train_thres:
                print(f"TRAIN_METRICS iter={it} train_mse={loss.item():.6e}", flush=True)
                if loss.item() <= self.train_thres:
                    break
            optimizer.step()

        with torch.no_grad():
            return self._e2e(model).detach().cpu()     # the recovered matrix on all entries
```
