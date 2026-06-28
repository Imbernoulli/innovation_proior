I start with the object that still exists when the gradient disappears. I have a convex function, and at a kink I cannot ask for one tangent slope, but I can ask for a supporting affine lower bound. A vector `g` is useful if `f(z) >= f(x) + g^T(z-x)` for every `z`. That is exactly the slope of a hyperplane supporting the epigraph at `(x,f(x))`. This matters because it is global information, not just a local approximation: if I plug in an optimum `x*`, the support line must still sit below `f(x*)`.

My first temptation is to imitate gradient descent. If `g` is the only first-order vector I have, I try moving in direction `-g`. But before trusting that, I should ask whether the smooth proof of descent survives. In the smooth case `-\nabla f(x)` is a descent direction because the directional derivative is `-||\nabla f(x)||^2 <= 0`. For a convex nonsmooth function the directional derivative is not a single inner product; it is `f'(x;v) = sup_{h in partial f(x)} h^T v`. Along `-g` this becomes `sup_h h^T(-g)`, and another valid supporting slope `h` can make this positive.

I want to know whether that is a real obstruction or just a worry. Take `f(x) = max(x, -2x)` on the line. At the kink `x=0` both pieces are active, so `partial f(0) = [-2, 1]`. The value `g = 1` is a legitimate subgradient there. Now move along `-g`, i.e. let `x` decrease. For small `t>0`, `f(-t) = max(-t, 2t) = 2t`, while `f(0) = 0`. So `f` increased, and the directional derivative along `-g` is `+2`. The step pointed strictly uphill, infinitesimally, from a correct subgradient. This is not a nuisance caused by a step that is too large; shrinking `t` does not help, because the slope `+2` persists in the limit. It is a structural failure of the descent-direction idea.

That also breaks the usual line-search story. A line search assumes the chosen ray contains a lower value if I look close enough or far enough along it. The example just showed the ray can be locally uphill from the start. So I should not build the algorithm around monotone decrease of `f(x_k)`. I need a quantity that the supporting inequality actually controls, even when the value misbehaves.

I look again at the support inequality and evaluate it at a minimizer. Since `f(x*) = f*`, I get `f* >= f(x) + g^T(x* - x)`. Rearranging gives `g^T(x - x*) >= f(x) - f*`. So the chosen supporting slope, whatever it does to the objective value, has a guaranteed nonnegative component pointing away from the optimum, and that component is at least the current suboptimality. Stepping along `-g` is justified spatially: it is trying to reduce distance to `x*`, not value at the very next point. In the `max(x,-2x)` example this is consistent: `g=1` at `x=0` gives `g^T(x-x*) = 0` since we are already at `x*=0`, and indeed `f(x)-f* = 0` there, so the inequality is tight, not violated.

Now I can test the update. Let `x_{k+1} = x_k - alpha_k g_k`, with `g_k in partial f(x_k)`. I expand the squared distance to an arbitrary minimizer:

`||x_{k+1}-x*||^2 = ||x_k-x*||^2 - 2 alpha_k g_k^T(x_k-x*) + alpha_k^2 ||g_k||^2`.

The cross term is where convexity enters. From the supporting inequality, `g_k^T(x_k-x*) >= f(x_k)-f*`, so

`||x_{k+1}-x*||^2 <= ||x_k-x*||^2 - 2 alpha_k (f(x_k)-f*) + alpha_k^2 ||g_k||^2`.

I read the signs. The middle term is negative and linear in the step size; it is the progress earned from being suboptimal. The final term is positive and quadratic; it is the price of taking a finite step. So small steps can be safe even without descent in objective value, and steps cannot stay large forever if I want exact convergence.

The objective values can still bounce, so the last iterate is not the right reported point. I keep `f_best^(k) = min_{1 <= i <= k} f(x_i)`. Summing the distance inequality from `i=1` to `k`, dropping the nonnegative final distance `||x_{k+1}-x*||^2 >= 0`, and using `||x_1-x*|| <= R`, I get

`2 sum_i alpha_i (f(x_i)-f*) <= R^2 + sum_i alpha_i^2 ||g_i||^2`.

If every queried subgradient has norm at most `G`, which is the Lipschitz case, then

`2 sum_i alpha_i (f(x_i)-f*) <= R^2 + G^2 sum_i alpha_i^2`.

The weighted average of the gaps is at least the smallest gap times the total weight, and `f_best^(k)-f*` is no larger than that smallest gap, so

`f_best^(k)-f* <= (R^2 + G^2 sum_i alpha_i^2) / (2 sum_i alpha_i)`.

This one inequality is supposed to dictate the step schedule, so let me make sure it really separates the good schedules from the bad. If `sum alpha_i` diverges while `sum alpha_i^2` stays finite, the numerator is bounded but the denominator grows, so the bound goes to `0`. A schedule like `alpha_k = a/k` satisfies both. A constant step `alpha_i = alpha` does not: then the bound is `R^2/(2 alpha k) + G^2 alpha/2`, whose first term vanishes but whose second term is a fixed floor `G^2 alpha/2`.

I want to confirm that the floor is a real behavior and not an artifact of a loose bound. Take `f(x) = |x| = max(x,-x)` on the line, so `f* = 0` at `x*=0`, `G=1`, and run a constant step `alpha = 0.4` from `x_0 = 1.5`. Tracing the update `x_{k+1} = x_k - 0.4\,sign(x_k)`:

```text
x:  1.5  1.1  0.7  0.3 -0.1  0.3 -0.1  0.3 -0.1  0.3 -0.1  0.3
f:  1.5  1.1  0.7  0.3  0.1  0.3  0.1  0.3  0.1  0.3  0.1  0.3
```

The iterates march in until they straddle the kink, then lock into a `0.3, -0.1` cycle and never settle. So the last iterate genuinely does not converge, which is why returning it would be wrong and `f_best` is needed. The best value seen is `0.1`, and the predicted floor is `G^2 alpha/2 = 0.5 \cdot 0.4 = 0.2`. The observed `0.1 <= 0.2`, so the bound holds and is the right order of magnitude. The cycle also matches the spatial picture: with a fixed step the method keeps overshooting the kink by a fixed amount.

If I commit in advance to exactly `K` steps, I can minimize the bound itself rather than just check schedules. The numerator carries `sum alpha_i^2` and the denominator `sum alpha_i`. Restricting to equal steps `alpha_i = alpha` gives `b(alpha) = R^2/(2K alpha) + G^2 alpha/2`. Differentiating, `b'(alpha) = -R^2/(2K alpha^2) + G^2/2`, which is zero at `alpha = (R/G)/sqrt(K)`, and `b''>0` so this is the minimum. Substituting back:

`b((R/G)/sqrt(K)) = R^2/(2K) \cdot (G sqrt(K)/R) + (G^2/2)(R/(G sqrt(K))) = RG/(2 sqrt(K)) + RG/(2 sqrt(K)) = RG/sqrt(K)`.

Let me sanity-check the arithmetic numerically with `R=2, G=3`. For `K=16`, the formula gives `alpha = (2/3)/4 = 0.16667` and `b = RG/sqrt(K) = 6/4 = 1.5`; evaluating `R^2/(2K alpha) + G^2 alpha/2 = 4/(32\cdot0.16667) + 9\cdot0.16667/2 = 0.75 + 0.75 = 1.5`, matching. For `K=100`, `alpha=0.06667`, `b=0.6`, again matching `RG/sqrt(K)=6/10`. A grid scan over `alpha` at `K=16` puts the minimizer at `0.1666`, indistinguishable from the closed form `0.1667`. So the fixed-horizon rate is `f_best^(K)-f* <= RG/sqrt(K)`, and the iteration count for an `epsilon` guarantee scales like `(RG/epsilon)^2`.

I should ask whether this `1/sqrt(K)` rate is just a loose analysis that a cleverer algorithm could beat. A resisting-oracle construction argues it cannot. Using a function like `gamma max_i x^(i) + (mu/2)||x||^2`, an adversarial oracle answers each query with a worst active subgradient, exposing essentially one useful coordinate per query. After `k` queries a first-order method has touched too few coordinates, and the residual gap stays on the order of `MR/sqrt(k)`. I will treat the matching lower bound as the reason the simple schedule is not wasteful, while noting that the constant in the two `1/sqrt(k)` expressions need not coincide.

There is one more step-size rule hidden in the same inequality. Holding the current point fixed, the right side of the distance recursion is a quadratic in `alpha_k`: `alpha_k^2 ||g_k||^2 - 2 alpha_k(f(x_k)-f*)` plus a constant. If I happen to know `f*`, the minimizing step is `alpha_k = (f(x_k)-f*)/||g_k||^2`, the Polyak step. It needs no `R`, `G`, or horizon. I test it on `|x|` from `x_0 = 1.5`: there `f(x)-f* = |x|`, `||g||^2 = 1`, so `alpha = |x|`, and `x_1 = x - |x|\,sign(x) = 0`. One step to the exact minimizer, which is the best possible — a useful sanity check that the rule is doing something sharp rather than merely convergent.

I want to confirm the Polyak step inherits the same global rate. Substituting `alpha_k = (f(x_k)-f*)/||g_k||^2` into the distance inequality, the right side becomes `||x_k-x*||^2 - (f(x_k)-f*)^2/||g_k||^2`. Telescoping and dropping the final nonnegative distance gives `sum_i (f(x_i)-f*)^2/||g_i||^2 <= R^2`. Let me verify this on a less trivial case where the answer is not reached in one step: `f(x) = ||x||_inf = max(x_1,-x_1,x_2,-x_2)` in 2D, started at `x_0=(3,1)`, so `R = sqrt(10)` and `f*=0`. Running the Polyak iteration and accumulating `sum (f_i)^2/||g_i||^2` along the actual trajectory, I get `10.0`, against `R^2 = 10.0`. The inequality holds, and here it is tight — the bound is exactly saturated, which tells me the telescoping lost nothing in this instance. With `||g_i|| <= G`, this same sum bound forces a smallest term of order `(RG)^2/k`, i.e. the same `RG/sqrt(k)` best-value scale.

So the method assembles itself from the spatial inequality. At each point I take any supporting slope `g_k in partial f(x_k)`, move against it with `x_{k+1} = x_k - alpha_k g_k`, keep `f_best`, and choose the step by one of the rules the bound singled out: square-summable-not-summable for asymptotic convergence, equal `(R/G)/sqrt(K)` for a fixed horizon, or the Polyak step when `f*` is known. A subgradient does not behave like a gradient — the `max(x,-2x)` example showed `-g` can climb — but a supporting hyperplane certifies enough alignment with the direction to `x*` to control squared distance, and the step-size conditions are exactly those that let the linear progress term outrun the quadratic overshoot.
