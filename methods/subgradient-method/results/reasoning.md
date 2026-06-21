I start with the object that still exists when the gradient disappears. I have a convex function, and at a kink I cannot ask for one tangent slope, but I can ask for a supporting affine lower bound. A vector `g` is useful if `f(z) >= f(x) + g^T(z-x)` for every `z`. That is exactly the slope of a hyperplane supporting the epigraph at `(x,f(x))`. This matters because it is global information, not just a local approximation: if I plug in an optimum `x*`, the support line must still sit below `f(x*)`.

My first temptation is to imitate gradient descent. If `g` is the only first-order vector I have, I try moving in direction `-g`. But I have to stop and check the smooth proof before trusting the update. In the smooth case `-\nabla f(x)` is a descent direction because the directional derivative is `-||\nabla f(x)||^2`. For a convex nonsmooth function the directional derivative is not one inner product; it is `sup_{h in partial f(x)} h^T v`. Along `-g` this becomes `sup_h h^T(-g)`, and another valid supporting slope `h` can make this positive. So the move can point uphill even infinitesimally. This is not a nuisance caused by a step that is too large. It is a structural failure of the descent-direction idea.

That also breaks the usual line-search story. A line search assumes that the chosen ray contains a lower value if I look close enough or far enough. Here the ray can be locally uphill. So I should not build the algorithm around monotone decrease of `f(x_k)`. I need a quantity that the supporting inequality actually controls.

I look again at the support inequality and evaluate it at a minimizer. Since `f(x*) = f*`, I get `f* >= f(x) + g^T(x* - x)`. Rearranging gives `g^T(x - x*) >= f(x) - f*`. This is the key switch. The chosen supporting slope may not point in a descent direction for the objective value, but it has a nonnegative component away from the optimum, and the component is at least the current suboptimality. So stepping along `-g` is justified spatially: it is trying to reduce distance to `x*`, not necessarily value at the very next point.

Now I can test the update. Let `x_{k+1} = x_k - alpha_k g_k`, with `g_k in partial f(x_k)`. I expand the squared distance to an arbitrary minimizer:

`||x_{k+1}-x*||^2 = ||x_k-x*||^2 - 2 alpha_k g_k^T(x_k-x*) + alpha_k^2 ||g_k||^2`.

The cross term is where convexity enters. From the supporting inequality, `g_k^T(x_k-x*) >= f(x_k)-f*`, so

`||x_{k+1}-x*||^2 <= ||x_k-x*||^2 - 2 alpha_k (f(x_k)-f*) + alpha_k^2 ||g_k||^2`.

I check the sign carefully. The middle term is negative and linear in the step size; it is the progress earned from being suboptimal. The final term is positive and quadratic; it is the price of taking a finite step. This is why small steps can be safe even without descent in objective value, and it is also why steps cannot stay too large forever if I want exact convergence.

The objective values can still bounce, so the last iterate is not the right reported point. I keep `f_best^(k) = min_{1 <= i <= k} f(x_i)`. Summing the distance inequality from `i=1` to `k`, dropping the nonnegative final distance, and assuming `||x_1-x*|| <= R`, I get

`2 sum_i alpha_i (f(x_i)-f*) <= R^2 + sum_i alpha_i^2 ||g_i||^2`.

If every queried subgradient has norm at most `G`, which is the Lipschitz case, then

`2 sum_i alpha_i (f(x_i)-f*) <= R^2 + G^2 sum_i alpha_i^2`.

The weighted average of the gaps is at least the smallest gap times the total weight, so

`f_best^(k)-f* <= (R^2 + G^2 sum_i alpha_i^2) / (2 sum_i alpha_i)`.

This one inequality tells me how to choose steps. If `sum alpha_i` diverges, the fixed initial-distance term can be washed out. If `sum alpha_i^2` is finite, the accumulated overshoot remains bounded. Thus a square-summable but not summable schedule, such as `alpha_k = a/k`, drives the best objective value to `f*`. A constant step has a different meaning: with `alpha_i = alpha`, the bound becomes `R^2/(2 alpha k) + G^2 alpha/2`, so it approaches only a neighborhood whose size is proportional to the step. That matches the spatial picture: the method keeps overshooting the kink.

If I know in advance that I will take exactly `K` steps, I can minimize the bound itself. The numerator contains the sum of squared steps and the denominator the sum of steps. By symmetry and convexity of this expression over positive step sizes, the best choice has all steps equal. With `alpha_i = alpha`, the bound is `R^2/(2K alpha) + G^2 alpha/2`. Differentiating gives `alpha = (R/G)/sqrt(K)`, and substituting gives `f_best^(K)-f* <= RG/sqrt(K)`. So the iteration count for an `epsilon` guarantee scales like `(RG/epsilon)^2`.

I ask whether that slow rate is just a loose analysis. The black-box lower-bound construction says it is not. An oracle can use a function of the form `gamma max_i x^(i) + (mu/2)||x||^2` and reveal only one useful coordinate per query by selecting a worst active subgradient. After `k` queries, a first-order method has not touched enough coordinates, and the gap remains at least on the order of `MR/sqrt(k)`. So the simple bound is aligned with the worst possible nonsmooth Lipschitz convex problem.

There is one more step-size idea hidden in the same inequality. For a fixed current point, the right side of the distance recursion is a quadratic in `alpha_k`: `alpha_k^2 ||g_k||^2 - 2 alpha_k(f(x_k)-f*)` plus a constant. If I know the optimal value, the minimizing step is `alpha_k = (f(x_k)-f*)/||g_k||^2`. This is the Polyak step. It does not need `R`, `G`, or a horizon, but it does need `f*`. Substituting it into the telescoped inequality gives `sum_i (f(x_i)-f*)^2/||g_i||^2 <= R^2`, and with `||g_i|| <= G` this implies the same `RG/sqrt(k)` best-value scale.

Now the method is clear. At each point I take any supporting slope, move against it with a step size chosen by one of these rules, and keep the best value seen. The insight is not that a subgradient behaves like a gradient. It does not. The insight is that a supporting hyperplane certifies enough alignment with the direction to the optimum to control squared distance, and the step-size conditions are exactly the conditions that make the linear progress dominate the quadratic overshoot over time.
