OK, let me think this through from scratch. I have a convex function f on R^n and I want to minimize it, and the functions I actually care about are things like `max_i(a_iᵀx+b_i)`, the ℓ1 norm, the dual of a constrained problem, the largest eigenvalue of a matrix that depends on x. Every one of these has kinks — and annoyingly, the minimum often *sits* at a kink, because that's where several pieces meet and balance. So the one tool I'd reach for first, gradient descent, is already in trouble: at a kink there's no ∇f to step against. And it's worse than "undefined at a measure-zero set, just perturb" — if I evaluate the gradient slightly to one side of a kink, that gradient points as if the kink weren't there, straight past the corner I'm trying to settle into. So I can't just patch smooth gradient descent. I need to rethink what first-order information even means here.

What does convexity hand me at a kink? Not a gradient, but the next best thing: a supporting hyperplane to the graph. Convexity says that at any point x I can find a vector g such that

  f(z) ≥ f(x) + gᵀ(z − x)   for all z.

That g is a *subgradient*: the affine function on the right sits underneath f everywhere and touches it at x. Geometrically `(g,−1)` supports the epigraph at `(x,f(x))`. This always exists in the interior of the domain — it's just the supporting hyperplane theorem applied to the convex set epi f at the boundary point `(x,f(x))`, and the slope of that hyperplane is g. Where f is smooth there's exactly one such g and it's ∇f. Where there's a kink there's a whole *set* of them — for `|x|` at zero, every slope in `[−1,1]` works; for `max_i f_i`, the convex hull of the gradients of the active pieces; for `‖x‖₁`, all g with `‖g‖_∞≤1` and `gᵀx=‖x‖₁`. Call this set the subdifferential ∂f(x). Good — so at a kink I don't have *a* gradient, I have a set, and I can always grab some member of it cheaply (for `max_i f_i`, just pick a piece that's active and take its gradient).

So the obvious move: pick any `g ∈ ∂f(x)` and run `x⁺ = x − α g`, exactly like gradient descent but with a subgradient in place of the gradient. Let me see if the usual convergence story survives.

The usual story for smooth descent is: `−∇f` is a descent direction, so the function value goes down each step, and I prove convergence by watching f(x_k) decrease. Does `−g` go downhill here? The directional derivative of a convex f in a direction v is `f'(x;v) = sup_{h∈∂f(x)} hᵀv`. So along `−g`,

  f'(x; −g) = sup_{h∈∂f(x)} hᵀ(−g) = − inf_{h∈∂f(x)} hᵀg.

And I hit the wall here. g is just *one* element of ∂f(x). There can easily be another `h ∈ ∂f(x)` with `hᵀg < 0`, and then that supremum is *positive* — meaning `−g` points *uphill*. Concretely at a kink the subdifferential is fat, and the particular g I happened to grab can be almost orthogonal to, or even anti-aligned with, the steepest-descent member of the set. So `−g` is **not** a descent direction in general, and a step `x − αg` can *increase* f. Let me not paper over this: it's not "increases at large α," it's that even infinitesimally the value can rise. The entire foundation of the smooth proof — "f(x_k) decreases" — is gone. Not "harder to arrange"; structurally false.

So I can't track f. And I can't line-search either, for the same reason: a line search exists to find an α that decreases f along the search direction, but along `−g` there may be *no* such α. Whatever step sizes I use will have to come from an explicit rule, not from a decrease condition. Fine — but then what on earth is the quantity that *does* improve, that I can build a proof on?

Let me stare at the subgradient inequality again, but plug in the minimizer. Let x* be any minimizer, f* = f(x*). The inequality at x, evaluated at z = x*, says

  f* = f(x*) ≥ f(x) + gᵀ(x* − x),

so rearranging,

  gᵀ(x − x*) ≥ f(x) − f* ≥ 0.

Look at that. Even though `−g` isn't a descent direction for f, it has a strictly positive component along the direction `x* − x` that points from x toward the optimum — and that component is at least `f(x)−f*`, the suboptimality. So `−g` doesn't reliably take me downhill in *value*, but it reliably takes me toward x* in *space*. The thing that's monotone-ish isn't `f(x)`; it's `‖x − x*‖`, the distance to the minimizer. That's the substitution. Stop watching the function value, watch the distance to optimum.

Let me make it precise. With `x⁺ = x − αg`,

  ‖x⁺ − x*‖² = ‖x − αg − x*‖²
            = ‖x − x*‖² − 2α gᵀ(x − x*) + α²‖g‖².

That middle term is exactly the inner product I just bounded below. Substitute `gᵀ(x−x*) ≥ f(x) − f*`:

  ‖x⁺ − x*‖² ≤ ‖x − x*‖² − 2α (f(x) − f*) + α²‖g‖².    (★)

Let me sanity-check the signs, because the whole edifice rests on this one line. The cross-term came in as `−2α gᵀ(x−x*)`. I'm replacing `gᵀ(x−x*)` by something *smaller or equal*, namely `f(x)−f*`; replacing a quantity by a smaller one inside a `−2α(·)` makes the whole expression *larger or equal*, which is the right direction for an upper bound — so the inequality `≤` is correct. And `f(x)−f* ≥ 0`, so `−2α(f(x)−f*) ≤ 0`: the distance to x* shrinks by an amount proportional to how suboptimal I am, and grows back by `α²‖g‖²`, the overshoot from taking a finite step. So there's a tension: the step earns me `2α(f(x)−f*)` of progress toward x* but pays `α²‖g‖²` for overshooting. For small α the linear gain beats the quadratic cost and I genuinely get closer. That's the mechanism. The function value can bounce around all it likes; the distance to x* is being driven down.

One more thing falls out of "not a descent method": since f(x_k) isn't monotone, the *last* iterate need not be the best one I've seen. So I should keep the best-so-far value, `f_best^{(k)} = min_{i≤k} f(x_i)`, and report that. In a descent method this is automatic; here it's a necessity.

Now let me turn (★) into an actual rate. Write it at step i with step size α_i and subgradient g_i, and telescope. From

  ‖x_{i+1} − x*‖² ≤ ‖x_i − x*‖² − 2α_i (f(x_i) − f*) + α_i²‖g_i‖²,

sum i = 1 … k:

  ‖x_{k+1} − x*‖² ≤ ‖x_1 − x*‖² − 2 Σ_{i=1}^k α_i (f(x_i) − f*) + Σ_{i=1}^k α_i²‖g_i‖².

The left side is ≥ 0, and let R be a bound on the initial distance, `‖x_1 − x*‖ ≤ R`. Drop the left side and move the middle sum over:

  2 Σ_{i=1}^k α_i (f(x_i) − f*) ≤ R² + Σ_{i=1}^k α_i²‖g_i‖².    (1)

The left side is a weighted sum of nonnegative suboptimalities, with weights α_i. A weighted sum is at least (total weight) × (minimum term):

  Σ_{i=1}^k α_i (f(x_i) − f*) ≥ (Σ_{i=1}^k α_i) · min_{i≤k}(f(x_i) − f*) = (Σ_{i=1}^k α_i)(f_best^{(k)} − f*).

Combine with (1):

  f_best^{(k)} − f* ≤ ( R² + Σ_{i=1}^k α_i²‖g_i‖² ) / ( 2 Σ_{i=1}^k α_i ).

And if the subgradients are bounded — `‖g_i‖ ≤ G` for all i, which holds exactly when f is
G-Lipschitz, since then every subgradient has norm at most G — I can replace `‖g_i‖²` by `G²`:

  f_best^{(k)} − f* ≤ ( R² + G² Σ_{i=1}^k α_i² ) / ( 2 Σ_{i=1}^k α_i ).    (B)

This single inequality is the whole convergence theory; every step-size rule is just a way of reading it off.

So what do I want from the step sizes? Look at (B). I want the right-hand side to go to zero as k → ∞. The numerator has the constant R² plus `G² Σα_i²`; the denominator is `2 Σα_i`. For the bound to vanish I need the denominator `Σα_i → ∞` — I have to be *able* to travel arbitrarily far, because x* might be far and each step is finite. But I also can't let the numerator blow up faster than the denominator. If I demand `Σα_i² < ∞` while `Σα_i = ∞`, then the numerator converges to the finite constant `R² + G²Σα_i²` and the denominator grows without bound, so the ratio → 0. There it is: `Σα_k = ∞, Σα_k² < ∞` makes `f_best^{(k)} → f*`, for *any* G-Lipschitz convex f. A canonical such schedule is `α_k = a/k` (or `a/(b+k)`): square-summable, not summable. Remarkable — a couple of lines of algorithm and a couple of lines of proof, and I converge on every convex function I can compute a subgradient of.

Let me double-check the boundary cases so I understand the tension. Constant step `α_k = α`: then `Σα_i = αk`, `Σα_i² = α²k`, and

  f_best^{(k)} − f* ≤ (R² + G²α²k)/(2αk) = R²/(2αk) + G²α/2.

As k → ∞ this does *not* go to zero — it plateaus at `G²α/2`. So a constant step converges only to within `G²α/2` of optimal; smaller α means a lower floor but slower approach (the `R²/(2αk)` term takes longer to fade). That's the price of not letting the step shrink: `Σα² = ∞`, so the overshoot accumulates and never lets the bound close. Conversely a diminishing nonsummable `α_k → 0, Σα_k = ∞` does drive it to zero (the numerator's tail can be made `≤ ε·Σα`, the standard ε-split), but possibly slowly. The square-summable rule is the clean middle: it converges exactly.

Now the sharper question — how *fast*, at best? Suppose I'm going to run a fixed budget of K steps and I get to pick the step sizes knowing K. Which choice minimizes the bound (B)? With `‖g‖≤G` the right side is `(R² + G² Σα_i²)/(2 Σα_i)`, a symmetric convex function of `(α_1,…,α_K)` — swapping any two α's leaves both sums unchanged, and the quadratic-over-linear form is convex on positive step sizes. So the minimizer has all α_i equal, say to α. Then the bound is

  (R² + G²Kα²)/(2Kα).

Minimize over α: differentiate, `d/dα [ R²/(2Kα) + G²α/2 ] = −R²/(2Kα²) + G²/2 = 0`, giving `α² = R²/(G²K)`, i.e.

  α = (R/G)/√K.

Plug back in: `R²/(2K·(R/G)/√K) + G²(R/G)/(2√K) = (R G √K)/(2K) + (R G)/(2√K) = RG/(2√K) + RG/(2√K) = RG/√K`. So

  f_best^{(K)} − f* ≤ R G / √K.

That's the rate. After K steps the best value is within `RG/√K` of optimal; to guarantee ε-accuracy I need about `K = (RG/ε)²` steps. And `RG` has a clean meaning: it's a bound on the initial suboptimality `f(x_1) − f*` (Lipschitz constant G times initial distance R). So the number of steps to cut my uncertainty about f* by a given factor scales like the *square* of that factor — to gain three digits, `10⁶` iterations. The method is slow. But it's slow in a very controlled way: `1/√K`, no dependence on dimension n, on *any* Lipschitz convex problem.

Is `1/√K` just my proof being loose, or is it real? Let me see whether an adversary can force it. I'm a first-order black box: at each queried point I get f and *one* subgradient, adversarially chosen, and my next iterate lives in the starting point plus the span of the subgradients I've seen. Suppose the adversary uses the function

  f_{K+1}(x) = γ · max_{1≤i≤K+1} x^{(i)} + (μ/2)‖x‖²,

whose subdifferential is `∂f_{K+1}(x)=μx+γ conv{e_i : i is active in the max}`. The resisting oracle chooses the first active coordinate. Start at `x_0=0`; the returned subgradient is `μx_0+γe_1=γe_1`, so `x_1` can only use coordinate 1. If the current point lives in the first p coordinates, then among the still-zero coordinates the first possible maximizer is no later than `p+1`, and the returned subgradient `μx_i+γe_{i*}` lives in the first `p+1` coordinates. Inductively, after i queries the iterate is still zero in coordinates `i+1,…,K+1`. Therefore at `x_K` there is an untouched coordinate with value zero, the max term is at least zero, and the quadratic term is nonnegative, so `f_{K+1}(x_K) ≥ 0`. With the parameters tuned as `γ=√(K+1)M/(2+√(K+1))` and `μ=M/((2+√(K+1))R)`, the minimizer has its first `K+1` coordinates equal to `−γ/(μ(K+1))`, its distance from `x_0` is R, the function is M-Lipschitz on the radius-R ball around the minimizer, and

  f* = − M R / (2(2 + √(K+1))).

Since `f_{K+1}(x_K) ≥ 0`, this gives, for any first-order scheme,

  f(x_K) − f* ≥ M R / (2(2 + √(K+1))).

That's `Ω(MR/√K)`. So `1/√K` is not an artifact of a lazy proof — it's a floor no first-order method can break on this class. My upper bound `RG/√K` (with G the Lipschitz constant, the same role as M) matches it up to the constant. The simple subgradient method with `α = (R/G)/√K` is, in this sense, *optimal* for nonsmooth Lipschitz convex minimization. That's the surprise: no first-order method can improve the worst-case rate by more than a constant factor over "step against a subgradient with a `1/√K`-scale step."

There's still something unsatisfying. The optimal step `α=(R/G)/√K` needs R and G, and worse, my best convergence used a *fixed* horizon K. Can I get a step rule that adapts to the function itself and doesn't need me to know R or G? Go back to the per-step inequality (★),

  ‖x⁺ − x*‖² ≤ ‖x − x*‖² − 2α (f(x) − f*) + α²‖g‖².

For a given x and g, which α makes the right side smallest — i.e. which step buys the most distance-to-x* per step? It's a quadratic in α, `α²‖g‖² − 2α(f(x)−f*) + const`, minimized at

  α = (f(x) − f*) / ‖g‖².

This is beautiful: it's exactly the suboptimality `f(x)−f*` divided by `‖g‖²`. It needs no R, no horizon, no global G — just the current gap to f* and the current subgradient. (And there's a smooth-case intuition: if f were differentiable and I linearized, `f(x − αg) ≈ f(x) − α gᵀg`, and asking that the linearized value hit f* gives `α = (f(x)−f*)/‖g‖²` — the step that, to first order, lands me at the optimal value.) Of course it assumes I *know* f*. Sometimes I do — feasibility problems have f* = 0; a min-max LP I can bound. Let me check it converges. Substitute `α_i = (f(x_i)−f*)/‖g_i‖²` into the telescoped (1):

  2 Σ_i α_i(f(x_i)−f*) − Σ_i α_i²‖g_i‖² ≤ R².

With this α_i, `2α_i(f(x_i)−f*) = 2(f(x_i)−f*)²/‖g_i‖²` and `α_i²‖g_i‖² = (f(x_i)−f*)²/‖g_i‖²`, so the left side is `Σ_i (f(x_i)−f*)²/‖g_i‖²`. Hence

  Σ_{i=1}^k (f(x_i)−f*)² / ‖g_i‖² ≤ R²,

and using `‖g_i‖ ≤ G`,

  Σ_{i=1}^k (f(x_i)−f*)² ≤ R² G².

The partial sums of a series of nonnegative terms are bounded, so the terms `(f(x_i)−f*)² → 0`, i.e. `f(x_i) → f*`. Also, among the first k iterates, the smallest squared gap is at most the average squared gap, so `f_best^{(k)}−f* ≤ RG/√k`; equivalently, the number of steps with gap at least ε is at most `R²G²/ε²`. The Polyak step matches the optimal rate, self-tunes to the geometry, and needs no horizon.

And when f* isn't known? Estimate it. Replace f* with `f_best^{(k)} − γ_k`, where γ_k is my running guess of how suboptimal I still am — take γ_k > 0, `γ_k → 0`, `Σγ_k = ∞`:

  α_k = (f(x_k) − f_best^{(k)} + γ_k) / ‖g_k‖².

The `+γ_k` keeps the step from collapsing to zero just because the current point ties the best-so-far. Let `δ_i=f(x_i)−f_best^{(i)}+γ_i`. Substituting `α_i=δ_i/‖g_i‖²` into (1) gives

  R² ≥ Σ_{i=1}^k δ_i ((f(x_i)−f*) + (f_best^{(i)}−f*) − γ_i) / ‖g_i‖².

If the best value refused to converge, then for some ε > 0 it would satisfy `f_best^{(i)}−f* ≥ ε` forever after some point. Choose N so that `γ_i ≤ ε` for i ≥ N. For those i, the second factor is at least ε, and the first factor is at least `γ_i` because `f(x_i) ≥ f_best^{(i)}`. With `‖g_i‖≤G`, the tail of the right side is at least `(ε/G²)Σ_{i=N}^k γ_i`, which diverges because `Σγ_i=∞`. That cannot stay bounded by R². So even without f* the best value must converge to f*; with f* I get the optimal rate.

Let me make sure I haven't smuggled in any smoothness. Existence of g: supporting hyperplane theorem, needs only convexity. The inequality `gᵀ(x−x*) ≥ f(x)−f*`: just the subgradient definition. The bound `‖g‖≤G`: just Lipschitz. The telescoping: pure algebra. Nowhere did I need ∇f, a descent step, or a line search. The price for that generality is the `1/√k` rate — and I just showed that price is unavoidable. Everything is consistent.

So here's where it all lands. The method is: initialize x_1, and repeat

  pick any g_k ∈ ∂f(x_k);   x_{k+1} = x_k − α_k g_k;   f_best ← min(f_best, f(x_k)),

with a step-size rule — `Σα_k=∞, Σα_k²<∞` for guaranteed convergence on any
Lipschitz convex f, `α_i=(R/G)/√K` for `i=1,…,K` to get the optimal `RG/√K` rate over a fixed budget, or the Polyak
step `α_k=(f(x_k)−f*)/‖g_k‖²` (or its estimated-f* form) to self-tune to the optimal rate without a
horizon. The proof is the single chain: the update expands `‖x_{k+1}−x*‖²` into a recursion whose
cross-term the subgradient inequality bounds by the suboptimality `f(x_k)−f*` — so even though f
itself isn't monotone and `−g_k` need not point downhill, the distance-to-minimizer recursion can be controlled;
telescoping that recursion and minimizing over iterates bounds `f_best^{(k)}−f*` by
`(R²+G²Σα²)/(2Σα)`, which the step-size choices drive to zero, at best like `RG/√k` — and a resisting
oracle shows no first-order method can beat `1/√k`, so the subgradient method is optimal for this
class.

```python
import numpy as np

def subgradient_method(f, subgrad, x0, num_iters, step,
                       f_star=None, G=None, R=None):
    """Minimize a convex (possibly nonsmooth) f by stepping against a subgradient.
       'step' in {'sqsum', 'sqrt', 'polyak', 'polyak_est'}."""
    x = np.array(x0, dtype=float)
    f_best, x_best = np.inf, x.copy()
    for k in range(1, num_iters + 1):
        fx = f(x)
        if fx < f_best:                      # not a descent method: keep best-so-far
            f_best, x_best = fx, x.copy()     # (last iterate need not be best)
        g = subgrad(x)                        # ANY g in the subdifferential at x

        if step == 'sqsum':                   # sum alpha_k = inf, sum alpha_k^2 < inf
            alpha = 1.0 / k                   # converges for any Lipschitz convex f
        elif step == 'sqrt':                  # fixed-budget optimal step -> RG/sqrt(num_iters)
            alpha = (R / G) / np.sqrt(num_iters)
        elif step == 'polyak':                # f* known: minimizes the per-step
            alpha = (fx - f_star) / g.dot(g)  # ||x-x*||^2 bound exactly -> optimal rate
        elif step == 'polyak_est':            # f* unknown: estimate it as f_best - gamma_k
            gamma = 1.0 / k                   # gamma_k -> 0, sum gamma_k = inf
            alpha = (fx - f_best + gamma) / g.dot(g)

        x = x - alpha * g                     # step against the subgradient
    return x_best, f_best


# canonical nonsmooth test: f(x) = max_i (a_i^T x + b_i); subgradient = active row a_j
def make_piecewise_linear(A, b):
    f = lambda x: np.max(A @ x + b)
    def subgrad(x):
        j = int(np.argmax(A @ x + b))         # an active piece; weak calculus for max_i f_i
        return A[j].copy()
    return f, subgrad
```
