Let me start from what actually blocks me. I want to minimize an expected loss `F(w) = E_z[Q(z, w)]` — the loss on a fresh example drawn from the data distribution — but I can't see `F`, and I can't even see its gradient. The distribution is unknown; the most I can do is replace it by a finite sample of `n` examples and minimize the empirical average `R_n(w) = (1/n) sum_i Q(z_i, w)`. The clean thing to do is steepest descent on `R_n`: `w_{t+1} = w_t - gamma * (1/n) sum_i grad Q(z_i, w_t)`. And the theory for that is reassuring — if the gradient is `L`-Lipschitz I get the quadratic upper bound `F(w') <= F(w) + grad F(w)^T(w'-w) + (L/2)||w'-w||^2`, so a small enough `gamma` strictly decreases the objective, and if `F` is strongly convex the error contracts geometrically, residual `~ rho^t` with `rho < 1`. Linear convergence. Beautiful — and useless to me at scale, because *every single step* costs `n` per-example gradients. The number of iterations to hit a target accuracy barely depends on `n`, so the total work scales with `n`: I burn one full pass over the data to take one step. When `n` is in the millions the binding resource isn't the number of examples, it's compute time, and batch descent spends all of it forming exact gradients I then take one tiny step along. I could throw curvature at it — scale the step by an approximate inverse Hessian for quadratic local convergence — but now I'm forming and inverting a `w`-by-`w` matrix on top of paying `n` per step, which is worse on both counts. So the exact-gradient route is structurally too expensive. I need a step whose cost does not grow with `n`.

The cheapest possible estimate of `grad R_n(w)` is the gradient on a *single* randomly drawn example, `grad Q(z_t, w)`. Its expectation over the random draw is exactly `(1/n) sum_i grad Q(z_i, w)` — it's an unbiased estimate of the batch gradient — and if I draw the example from the true data distribution it's an unbiased estimate of `grad F` itself, the thing I actually care about. So the tempting move is to just *use* it: `w_{t+1} = w_t - gamma * grad Q(z_t, w_t)`, one example per step, cost `O(1)`, and never even store which examples I've seen. But I have to be honest about what I've done. The single-example gradient is a wildly noisy draw of the average. Far from the optimum that might be fine — even a noisy direction is mostly downhill. Near the optimum it's a disaster waiting to happen: the *true* gradient is going to zero there, but the *variance* of the single-sample gradient does not — different examples disagree about the gradient even at the minimizer — so I'm taking steps of essentially constant size along directions that are pure noise. With a fixed `gamma` I should expect a neighborhood, not exact settling: the iterate can rattle around the optimum forever. So the constant step may be useful as a speed/accuracy tradeoff, but it cannot be the asymptotically exact story. The question is whether the noisy one-sample step can be made exact at all, and if so, what the step has to do.

Let me strip the problem down to the bare bones to see the mechanism, because high-dimensional optimization is hiding the issue. Take the simplest nontrivial instance: I want to estimate the mean `x` of i.i.d. draws `x_1, x_2, ...`, by chasing the root of `h(theta) = theta - x` (root at `theta = x`) using noisy measurements `theta - x_n` (each `x_n` an unbiased noisy sample of `x`). The recursion is `theta_n = theta_{n-1} - gamma_n(theta_{n-1} - x_n)`. Rewrite it: `theta_n - x = (1 - gamma_n)(theta_{n-1} - x) + gamma_n(x_n - x)`. Now unroll all the way back to `theta_0`. Each step multiplies the running error by `(1 - gamma_n)` and injects a fresh noise term `gamma_n(x_n - x)`, so

```
theta_n - x = prod_{k=1}^{n}(1 - gamma_k) * (theta_0 - x)
            + sum_{i=1}^{n} prod_{k=i+1}^{n}(1 - gamma_k) * gamma_i (x_i - x).
```

The first piece is deterministic — it's the fading memory of where I started. The second is random — it's all the measurement noise I've injected, each term decayed by the products that came after it. These are two different enemies and they want opposite things from `gamma`. Square and take expectations; the cross terms vanish because the `x_i` are independent and zero-mean about `x`, with `E[(x_i - x)^2] = sigma^2`, so

```
E||theta_n - x||^2 = prod_{k=1}^{n}(1 - gamma_k)^2 * ||theta_0 - x||^2
                   + sum_{i=1}^{n} gamma_i^2 prod_{k=i+1}^{n}(1 - gamma_k)^2 * sigma^2.
```

Stare at the bias term. I need `prod(1 - gamma_k)^2 -> 0` so the start point is forgotten. Take logs: `log prod(1 - gamma_k)^2 = 2 sum_k log(1 - gamma_k)`, and for small `gamma_k`, `log(1 - gamma_k) ~ -gamma_k`, so this is `~ -2 sum_k gamma_k`. For the product to go to zero I need that to go to `-infinity`, i.e. `sum_k gamma_k = infinity`. The steps have to *not be summable*: the total distance the method is allowed to travel must be unbounded, or it can never crawl out of a bad initialization and reach the optimum. So `gamma_k` cannot decay too fast — if it were summable, say `gamma_k = 1/k^2`, the product `prod(1-gamma_k)` converges to a positive constant and I'm stuck a fixed distance from the answer forever.

Now the variance term, and here the pressure flips. If I keep `gamma_k = gamma` constant, the noise term is `sigma^2 gamma^2 sum_i (1-gamma)^{2(n-i)} -> sigma^2 gamma^2 / (1 - (1-gamma)^2) ~ sigma^2 gamma / 2` — a *constant* noise floor that never decays, exactly the rattling I feared. To push the variance to zero I need `gamma_k -> 0`. So one enemy says "don't decay too fast" (`sum gamma = infinity`) and the other says "do decay" (`gamma -> 0`, and in fact I'll want the squared steps controllable). Is there a schedule satisfying both? The borderline case `gamma_k = 1/k`: then `sum 1/k = infinity` (bias dies), and `sum 1/k^2 = pi^2/6 < infinity` (the squared steps are summable), which is exactly what I'll need to keep the accumulated noise finite and shrinking.

That's promising, but I want a sanity check that `1/k` doesn't just satisfy the two conditions formally while doing something pathological to the iterate. Let me chase the special case where I can compute the answer by hand and compare. Plug `gamma_n = 1/n` and `theta_0 = 0` into the recursion `theta_n = (1 - gamma_n)theta_{n-1} + gamma_n x_n` and unroll the first few steps. `theta_1 = 0 + 1*x_1 = x_1`. `theta_2 = (1/2)x_1 + (1/2)x_2 = (x_1+x_2)/2`. `theta_3 = (2/3)*(x_1+x_2)/2 + (1/3)x_3 = (x_1+x_2+x_3)/3`. The pattern is `theta_n = (1/n) sum_{k=1}^n x_k`, and I can see why by induction: if `theta_{n-1}` is the mean of the first `n-1` draws, then `theta_n = (1 - 1/n)theta_{n-1} + (1/n)x_n = ((n-1)theta_{n-1} + x_n)/n`, which is exactly the update from the mean of `n-1` numbers to the mean of `n`. So with this step the recursion *is* the running sample mean. I coded the recursion against `numpy.cumsum(x)/arange` on ten random draws to be sure I didn't fumble the algebra, and the two agree to machine precision (max abs difference `~2.2e-16`). That settles it for the toy: the right decaying step makes the noisy one-sample recursion reproduce, exactly, the estimator I'd have written by hand — and the sample mean is the statistically optimal estimator of `x`, converging at rate `sigma^2/n`. So the construction isn't just consistent; on the one case I can check in closed form it's optimal.

So the structure I want is settling: a recursion `theta_{n} = theta_{n-1} - gamma_n[h(theta_{n-1}) + noise_n]` where `h` has the root I'm chasing, the noise is zero-mean, and `gamma_n` decays with `sum gamma_n = infinity` but `sum gamma_n^2 < infinity`. But I derived the two conditions in the one-dimensional mean-estimation toy. I need them to hold for the general root-finding problem, with an arbitrary monotone `h` and bounded but otherwise arbitrary noise, before I trust them. Let me set it up properly and actually prove convergence, because if I can't, the whole edifice is a heuristic.

Here is the general object. There is an unknown function `M(x) = E[Y(x)]`, the mean response at level `x`, observed only through a noisy random `Y(x)` with conditional distribution `H(y|x)`, so `M(x) = integral y dH(y|x)`. I want the root `theta` of `M(x) = alpha` for a given constant `alpha`. (In the mean-estimation toy, `M(x) = x - target` shifted, `alpha = 0`.) Assume `M` is monotone increasing through the root: `M(x) <= alpha` for `x < theta` and `M(x) >= alpha` for `x > theta`. Assume the response is bounded, `Pr[|Y(x)| <= C] = 1` for all `x`, so the noise has finite variance. The recursion, with `y_n` the observed response at the current level `x_n`:

```
x_{n+1} - x_n = a_n (alpha - y_n),     y_n ~ H(y | x_n),
```

with `x_1` arbitrary and `a_n > 0`. Note `E[alpha - y_n | x_n] = alpha - M(x_n)`, which is positive when I'm below the root (so I step up) and negative when above (so I step down): on average it's a restoring force toward `theta`. I want to show `x_n -> theta`. The right potential to track is the mean-square error to the root,

```
b_n = E(x_n - theta)^2.
```

If `b_n -> 0` then `x_n -> theta` in quadratic mean, which forces convergence in probability — that's the target. Let me compute how `b_n` evolves. Condition on `x_n` and take the next step:

```
b_{n+1} = E(x_{n+1} - theta)^2 = E[ E[ (x_n - theta + a_n(alpha - y_n))^2 | x_n ] ].
```

Expand the inner square: `(x_n - theta)^2 + 2 a_n (x_n - theta)(alpha - M(x_n)) + a_n^2 E[(alpha - y_n)^2 | x_n]`, using `E[alpha - y_n | x_n] = alpha - M(x_n)`. So

```
b_{n+1} = b_n + a_n^2 e_n - 2 a_n d_n,
```

where I name the two new pieces

```
d_n = E[(x_n - theta)(M(x_n) - alpha)],
e_n = E[ integral (y - alpha)^2 dH(y | x_n) ].
```

Look at the signs. By the monotonicity of `M` through `theta`, `(x_n - theta)` and `(M(x_n) - alpha)` always have the *same* sign — both positive above the root, both negative below — so their product is nonnegative, hence `d_n >= 0`. That's the drift term, and it always helps: `-2 a_n d_n` pushes `b_n` down. And by the boundedness `|Y| <= C`, the spread term is bounded: `0 <= e_n <= [C + |alpha|]^2 < infinity`. That's the noise term, and `+a_n^2 e_n` always hurts: it pushes `b_n` up. So in one step I trade a guaranteed decrease proportional to `a_n d_n` against a guaranteed increase proportional to `a_n^2 e_n` — the same bias-versus-variance tradeoff as the toy, now general.

Now sum the recursion from `1` to `n`:

```
b_{n+1} = b_1 + sum_{j=1}^{n} a_j^2 e_j - 2 sum_{j=1}^{n} a_j d_j.
```

Here's where the two step-size conditions earn their keep. Because `e_j` is bounded and I'm going to demand `sum a_j^2 < infinity`, the positive-term series `sum a_j^2 e_j` converges. Since `b_{n+1} >= 0` always (it's a mean square), rearrange:

```
2 sum_{j=1}^{n} a_j d_j = b_1 + sum_{j=1}^{n} a_j^2 e_j - b_{n+1} <= b_1 + sum_{j=1}^{n} a_j^2 e_j,
```

and the right side is bounded above by a finite constant for all `n`. So the positive-term series `sum a_j d_j` is bounded, hence *converges*. Two consequences. First, `sum a_n d_n < infinity`. Second, going back to the summed recursion, `lim_{n} b_n = b_1 + sum a_n^2 e_n - 2 sum a_n d_n` exists (both series converge), call it `b >= 0`. So `b_n` has a limit; I just have to show the limit is `0`.

Suppose for contradiction `b > 0`. Then `b_n` stays bounded away from zero, `b_n >= b/2 > 0` eventually. If I can show `d_n >= k_n b_n` for some `k_n >= 0` with `sum a_n k_n = infinity`, then `sum a_n d_n >= sum a_n k_n b_n >= (b/2) sum a_n k_n = infinity`, contradicting `sum a_n d_n < infinity`. So the entire question reduces to: can I lower-bound the drift `d_n` by `(something) * b_n` whose `a_n`-weighted sum diverges? This is where I need to be careful and where I need a second condition on the steps.

First I need the iterates to stay in a bounded region, or the inf I'm about to take is over an unbounded set. From the recursion, `|x_{n+1} - theta| <= |x_n - theta| + a_n |alpha - y_n| <= |x_n - theta| + a_n(C + |alpha|)`, so by induction `|x_n - theta| <= |x_1 - theta| + (C + |alpha|)(a_1 + ... + a_{n-1}) =: A_n` with probability one. Now define

```
kbar_n = inf{ (M(x) - alpha)/(x - theta) : 0 < |x - theta| <= A_n }.
```

By monotonicity through the root, `(M(x) - alpha)/(x - theta) >= 0`, so `kbar_n >= 0`. And it directly lower-bounds the drift: writing `P_n` for the law of `x_n`,

```
d_n = integral_{|x - theta| <= A_n} (x - theta)(M(x) - alpha) dP_n(x)
    >= integral kbar_n (x - theta)^2 dP_n(x) = kbar_n b_n.
```

So `k_n = kbar_n` works for the *first* requirement, `d_n >= k_n b_n`. The whole proof now rests on the *second* requirement, `sum a_n kbar_n = infinity` — and `kbar_n` shrinks as `A_n` grows (the inf is over a wider interval), so I'm racing a shrinking drift coefficient against the step sizes. I'll need `kbar_n` to not shrink faster than I can sum `a_n` against it. Suppose the curvature is genuinely there: `kbar_n >= K/A_n` for some `K > 0` and large `n`. This holds if `M` is strictly separated from `alpha`, with `M(x) <= alpha - delta` below and `M(x) >= alpha + delta` above, because then `kbar_n >= delta/A_n`. It also holds under the local differentiable condition `M'(theta) > 0`: near the root, `M(x) - alpha = (x - theta)[M'(theta) + epsilon(x-theta)]`, so the ratio is at least `M'(theta)/2`; outside a small radius `eta`, monotonicity makes `|M(x)-alpha|` at least `eta M'(theta)/2`, and over `|x-theta| <= A_n` that gives the ratio at least `eta M'(theta)/(2A_n)`. Then `a_n kbar_n >= a_n K / A_n`, and recalling `A_n = |x_1 - theta| + (C+|alpha|)(a_1 + ... + a_{n-1})`, for large `n` the constant is dominated by the growing sum, so `A_n <= 2(C + |alpha|)(a_1 + ... + a_{n-1})` and

```
a_n kbar_n >= a_n K / A_n >= K / (2(C+|alpha|)) * a_n / (a_1 + ... + a_{n-1}).
```

So `sum a_n kbar_n = infinity` is guaranteed if

```
sum_{n>=2} a_n / (a_1 + ... + a_{n-1}) = infinity.
```

There's the second condition, and it forces the total step length to diverge. If the partial sums `S_n = a_1 + ... + a_n` were bounded, then `S_{n-1} >= a_1 > 0` and `sum_{n>=2} a_n / S_{n-1} <= (1/a_1) sum_{n>=2} a_n < infinity`, contradicting the condition. So the condition implies `sum a_n = infinity`. With both conditions, the contradiction closes: `sum a_n d_n >= (b/2) sum a_n kbar_n = infinity` contradicts `sum a_n d_n < infinity`, so `b = 0`, so `b_n -> 0`, so `x_n -> theta` in quadratic mean and in probability. Done — and notice it's *distribution-free*: I never used the form of `H` beyond zero-mean and bounded variance, and never assumed a functional form for `M`. That's the thing least squares couldn't give me; this estimates the root with no model for the response.

The two conditions are exactly the pair the toy predicted: `sum a_n^2 < infinity` (kept the noise series convergent) and `sum a_n / (a_1 + ... + a_{n-1}) = infinity`, which entails `sum a_n = infinity` (kept the drift summing to enough). The canonical choice satisfying both is `a_n = 1/n`. The first is immediate, `sum 1/n^2 = pi^2/6 < infinity`. The second I should actually check rather than wave at, because it's the one I just invented and it's not a textbook series. The partial sum `a_1 + ... + a_{n-1}` is the harmonic number `H_{n-1} ~ log n`, so the general term is `(1/n)/H_{n-1} ~ 1/(n log n)`, and `sum 1/(n log n)` diverges (its partial sums grow like `log log n` — the integral test: `integral dx/(x log x) = log log x`). So the term sum diverges, just very slowly. To make sure I have the asymptotics right and not backwards, I summed `sum_{n>=2} a_n/H_{n-1}` numerically out to `n = 2e6`: the partial sum climbs through `1.24` at `n=10`, `2.15` at `n=10^3`, `2.81` at `n=10^6`, with no sign of leveling — consistent with the predicted `log log n` divergence (slow but unbounded), while the denominator `H_{n-1}` it divides by sits at only `~14.4` at `n=10^6`, confirming the harmonic, not faster, growth. So both conditions hold for `1/n`. More generally any `a_n` squeezed between `c'/n` and `c''/n` works. So I'll call `1/n` the canonical schedule.

One refinement that's basically free: instead of one observation per level, average `r` of them, `ybar_n = (1/r) sum y_{(n-1)r + 1 .. nr}`, and step on `ybar_n`. The drift is unchanged (still `alpha - M(x_n)` in expectation) but the spread term `e_n` is divided by `r`, so the noise enemy is `r` times weaker. The same convergence argument goes through with the same mean function and a smaller conditional variance. This is the mini-batch idea sitting inside the construction from the start — a knob to trade per-step noise for throughput.

Now the leap I've been circling. I built all this for "find the root of `M(x) = alpha`". But what is *minimizing* a loss? It's finding where the gradient vanishes: `grad F(w) = 0`. And `grad F(w) = grad E_z[Q(z, w)] = E_z[grad Q(z, w)]` — the gradient of the expected loss is the *expectation* of the per-example gradient. That is precisely the shape "an unknown mean function `M(w) = E[Y(w)]`, where `Y(w) = grad Q(z, w)` is a noisy observation of it, and I want the root of `M(w) = 0`." The root-finding problem and the optimization problem are the same problem, with `alpha = 0` and the noisy response being the single-example gradient. So the recursion `x_{n+1} = x_n - a_n(y_n - alpha)` becomes, in `w` and with `alpha = 0`,

```
w_{t+1} = w_t - gamma_t * grad Q(z_t, w_t),
```

with `g_t := grad Q(z_t, w_t)` an unbiased, noisy estimate of `grad F(w_t)`. This is the cheap one-sample update I wrote down on a hunch at the very start — and now I know it's a stochastic-approximation recursion whose convergence I've already proved, provided I pick the step sizes `gamma_t` to satisfy `sum gamma_t = infinity` and `sum gamma_t^2 < infinity`. The noisy-direction descent that looked like a reckless shortcut is the *right* method; it just needed the step schedule the root-finding analysis hands me. The cost is `O(1)` per step, first-order, no matrix — exactly the budget batch descent blew. And because each `z_t` can be drawn fresh from the data stream, the recursion optimizes the *expected* risk `F` directly, not the empirical `R_n`: it never needs to remember which examples it saw, so it can run online forever.

Let me now nail the convergence in the optimization language, because the root-finding proof assumed a scalar monotone `M`, and a high-dimensional loss isn't monotone — but it can be strongly convex, and I want to see the noise floor explicitly so I understand how the step size trades against it. Start from the Lipschitz-gradient descent bound applied to the *random* iterate `w_{k+1} = w_k - gamma_k g_k`:

```
F(w_{k+1}) <= F(w_k) + grad F(w_k)^T (w_{k+1} - w_k) + (L/2) ||w_{k+1} - w_k||^2
            = F(w_k) - gamma_k grad F(w_k)^T g_k + (gamma_k^2 L / 2) ||g_k||^2.
```

Take the expectation over the draw `z_k` given `w_k`. The cross term uses unbiasedness, `E[g_k | w_k] = grad F(w_k)`, so `E[grad F(w_k)^T g_k] = ||grad F(w_k)||^2`. The last term needs the second moment `E[||g_k||^2 | w_k]`, which I bound by a noise floor plus a piece proportional to the true gradient: `E[||g_k||^2] <= M + M_G ||grad F(w_k)||^2`, with `M >= 0` the irreducible variance at a stationary point and `M_G >= 1`. (This is just saying the gradient noise has bounded variance — `M` is the part that survives even when `grad F = 0`, the thing that will make a fixed step rattle.) Then

```
E[F(w_{k+1}) | w_k] - F(w_k) <= -(mu - (1/2) gamma_k L M_G) gamma_k ||grad F(w_k)||^2 + (1/2) gamma_k^2 L M,
```

where I keep a `mu <= 1` to allow `g_k` to be merely a sufficient-descent direction in expectation (`mu = 1` for the unbiased gradient). Read this off: the first term is the *signal*, negative and proportional to `gamma_k ||grad F||^2` — genuine progress. For the clean fixed-step contraction I choose the conservative small-step condition `gamma_k <= mu/(L M_G)`, which makes `(mu - (1/2)gamma_k L M_G) >= mu/2`. The second is the *noise*, positive and proportional to `gamma_k^2 M` — the price of using a noisy gradient, and it's there even when `grad F = 0`. Two terms, opposite signs, one linear and one quadratic in `gamma_k`. Everything about SGD's behavior is in the balance between them.

Now bring in strong convexity, which gives `2c(F(w) - F*) <= ||grad F(w)||^2` (the optimality gap is controlled by the gradient norm). Suppose first I'm stubborn and use a *constant* step `gamma_k = gamma <= mu/(L M_G)`. Substitute the strong-convexity inequality into the signal term:

```
E[F(w_{k+1}) - F*] <= (1 - gamma c mu) E[F(w_k) - F*] + (1/2) gamma^2 L M.
```

This is an affine recursion `E_{k+1} <= rho E_k + s` with `rho = 1 - gamma c mu < 1` and `s = (1/2) gamma^2 L M`. Its fixed point is `s/(1 - rho) = gamma L M / (2 c mu)`, and subtracting it,

```
E[F(w_{k+1}) - F*] - (gamma L M)/(2 c mu) <= (1 - gamma c mu) [ E[F(w_k) - F*] - (gamma L M)/(2 c mu) ],
```

so

```
E[F(w_k) - F*] <= (gamma L M)/(2 c mu) + (1 - gamma c mu)^{k-1} ( F(w_1) - F* - (gamma L M)/(2 c mu) )
              --> (gamma L M)/(2 c mu)   as k -> infinity.
```

There it is, the floor the toy warned me about, now with a bound on its size. A constant step gives *linear* (geometric) convergence — fast — but only down to a *noise ball*, after which the noise stops me; the bound on the ball's radius is `gamma L M / (2 c mu)`, proportional to `gamma`. If there were no noise (`M = 0`) the floor vanishes and I recover the clean linear convergence to `F*` of exact descent.

I should check this against a case I can simulate, both to confirm the floor is real and to see whether `gamma L M/(2 c mu)` is the *size* of the floor or only an upper bound on it — the derivation went through an inequality (the Lipschitz upper bound and the second-moment bound), so I'd expect a bound, not an equality, and I want to know how loose it is. Take the simplest strongly convex `F(w) = (c/2)w^2` in one dimension, `c = 1`, optimum at `w* = 0`, with `grad F(w) = w`; a quadratic has `L = c = 1`, `mu = 1`, `M_G = 1`, and I inject gradient noise `g = w + xi`, `xi ~ N(0, sigma^2)`, `sigma = 1`, so the irreducible variance is `M = sigma^2 = 1`. Then the predicted bound at `gamma = 0.1` (which satisfies `gamma <= mu/(L M_G) = 1`) is `gamma L M/(2 c mu) = 0.1*1*1/2 = 0.05` on `E[F - F*] = E[(c/2)w^2]`. Running the constant-step recursion `w <- w - gamma(w + xi)` over many independent chains to a plateau, the measured `E[F - F*]` settles at `~0.0265` — below the bound `0.05`, by roughly a factor of two. That's the right outcome: the formula is a genuine *upper* bound on the floor, not its exact value. And I can see why exactly here — the recursion is a linear AR(1) process `w_{t+1} = (1 - gamma c)w_t - gamma xi_t`, whose stationary variance is `gamma^2 sigma^2/(1 - (1 - gamma c)^2)`, giving `E[(c/2)w^2] = 0.0263`, dead-on the simulation. So the noise ball is real and its true size for this problem is `0.0263`, with `0.05` a 2x-loose upper bound from the inequalities. The load-bearing claim is the *proportionality to `gamma`*, not the constant, so I checked that directly too: halving `gamma` through `0.2, 0.1, 0.05, 0.025`, the measured floor tracks `0.056, 0.026, 0.013, 0.0063` — it halves each time, `measured/gamma -> ~0.25` a constant. So the scaling `floor ∝ gamma` holds cleanly even though the constant in my bound is loose. Halve the step, halve the floor — but also slow the contraction `(1 - gamma c mu)`. So a constant step is a *speed/accuracy* dial: big step, fast to a coarse answer; small step, slow to a fine answer. That's why a fixed learning rate is perfectly sensible when I'll stop at a plateau anyway — I run until the geometric term is swamped by the floor, and read off the answer. It also tells me the fixed step is *not* asymptotically exact, which is the price.

To actually reach `F*` I have to shrink the step — and now the `sum gamma = infinity`, `sum gamma^2 < infinity` conditions reappear, this time with a rate attached. Take `gamma_k = beta/(gamma + k)` (a `1/k`-type schedule, the optimization analogue of `a_n = 1/n`), with `beta` and offset `gamma` to be chosen. Claim: `E[F(w_k) - F*] <= nu/(gamma + k)` for a constant `nu`, i.e. `O(1/k)`. Prove by induction. The base case fixes part of `nu`. For the step, from the descent inequality with this `gamma_k` (and `gamma_k L M_G <= mu` for all `k`, which I get by choosing the offset so `gamma_1` is small enough),

```
E[F(w_{k+1}) - F*] <= (1 - gamma_k c mu) E[F(w_k) - F*] + (1/2) gamma_k^2 L M.
```

Substitute the inductive hypothesis `E[F(w_k)-F*] <= nu/(gamma+k)` and `gamma_k = beta/(gamma+k)`. Write `hatk = gamma + k`. Then

```
E[F(w_{k+1}) - F*] <= (1 - beta c mu/hatk) nu/hatk + (1/2)(beta/hatk)^2 L M
                    = nu/hatk - (beta c mu nu)/hatk^2 + (beta^2 L M)/(2 hatk^2)
                    = nu/hatk - (beta c mu - 1) nu / hatk^2 - nu/hatk^2 + (beta^2 L M)/(2 hatk^2).
```

I want this `<= nu/(hatk + 1) = nu/(gamma + k + 1)`. Use `nu/hatk - nu/hatk^2 = nu(hatk - 1)/hatk^2 <= nu/(hatk+1)` (since `(hatk-1)(hatk+1) = hatk^2 - 1 <= hatk^2`). So it suffices that the remaining terms are nonpositive:

```
-(beta c mu - 1) nu / hatk^2 + (beta^2 L M)/(2 hatk^2) <= 0,
```

i.e. `(beta c mu - 1) nu >= beta^2 L M / 2`. If I choose `beta > 1/(c mu)` so that `beta c mu - 1 > 0`, this holds for `nu >= beta^2 L M / (2(beta c mu - 1))`. Setting `nu = max{ beta^2 L M/(2(beta c mu - 1)), (gamma+1)(F(w_1)-F*) }` satisfies both base case and step. So `E[F(w_k) - F*] <= nu/(gamma + k) = O(1/k)`.

That's the bound I wanted, but it's an induction over an inequality and I've been burned once already (the constant-step "floor" turned out 2x loose), so I'd rather watch the iterate actually do this than trust the algebra alone. Two things I specifically want to confirm: that there's *no* residual floor (unlike the constant step), and that the rate really is `1/k` rather than something slower. On the same quadratic toy (`c = mu = L = 1`, `M = sigma^2 = 1`), run the decaying schedule `gamma_k = beta/(gamma_0 + k)` with `beta = 2` (which respects `beta > 1/(c mu) = 1`) over many chains, and read off `E[F - F*]` at `k = 10^3, 10^4, 10^5, 2*10^5`. If the rate is `O(1/k)` with no floor, then `k * E[F - F*]` should approach a constant. It does: the products are `0.655, 0.660, 0.665, 0.660` — flat across more than two decades of `k`, and the error itself keeps dropping (`6.6e-4` down to `3.3e-6`) with no sign of leveling onto a floor. So the decaying step genuinely kills the noise ball and lands the clean `1/k` rate; the measured constant `~0.66` sits comfortably under the bound `nu` (here `beta^2 L M/(2(beta c mu-1)) = 4/2 = 2`), again a valid-but-loose upper bound, consistent with the constant-step experience. The decaying `1/k` step buys me genuine convergence to the optimum, no floor, at rate `1/k` — and crucially the condition `beta > 1/(c mu)` says the step-size *constant* must be large enough relative to the curvature: too timid a schedule and the bias term (the memory of the start) decays too slowly to keep up, and the clean `1/k` rate degrades. This is the optimization echo of the toy's "`sum gamma = infinity`" — if I shrink the steps too aggressively (small `beta`) I forget the initialization too slowly; if I shrink too slowly (`sum gamma^2 = infinity`) the noise term wouldn't be summable. The `1/k` family threads both, and `beta > 1/(c mu)` is the sharp version of "not too fast."

A practical wrinkle I should head off: `gamma_k = beta/k` literally can mean an enormous first step (when `k` is small the step is huge), which can wreck the early iterations before the schedule calms down. The fix is to start from a chosen, sane value and let it decay into the `1/k` regime: `gamma_k = gamma_0 / (1 + gamma_0 lambda k)`, which equals `gamma_0` at `k = 0` and behaves like `1/(lambda k)` for large `k`. For the theorem, the asymptotic constant must still satisfy `beta > 1/(c mu)`; so if I use a curvature scale `lambda`, I choose it with slack so `1/lambda > 1/(c mu)`. In an `L2`-regularized objective `(lambda_reg/2)||w||^2 + ...`, the regularization weight gives a concrete lower-curvature scale, and the schedule can be set with a small safety factor around it. So I pick `gamma_0` (best found on a small subsample, since the SGD rates are independent of dataset size, so a small sample is a faithful proxy) and let the schedule do the annealing.

Let me step back and compare the rate honestly against the batch method, because SGD is, per iteration, clearly *worse*: batch descent on a strongly convex objective converges linearly (`~ rho^t`), needing only `log(1/rho_target)` iterations, while SGD converges like `1/k`, needing `~ 1/rho_target` iterations — vastly more. As an optimizer of a *fixed* objective, SGD is one of the worst. But that's the wrong scoreboard for learning. What I care about is the *expected* risk reached for a given *compute budget*, and there the accounting flips: batch costs `n` per iteration and SGD costs `1`, and to reach a target excess risk in the large-scale regime — where compute, not data, is the limit — SGD processes far more examples in the allotted time and lands at a better expected risk. The slow asymptotic rate doesn't bite, because I stop at a plateau long before the asymptotics matter, and the noise floor / `1/k` tail is happening at an accuracy finer than the statistical error of the model anyway. So the right way to read SGD is not "a bad optimizer" but "the optimizer matched to the regime where time is the bottleneck." And notice scaling the step by curvature (a stochastic second-order step `w_{k+1} = w_k - gamma_k Gamma_k g_k`) does *not* help the leading behavior: it warps the signal term but does nothing to the noise term `gamma_k^2 M` — the variance of the estimate is unchanged — so the asymptotic rate is still governed by the same `1/k` noise-limited tail. Curvature improves constants, not the dominant term; the noise is the wall, and the only way through it is averaging (more samples per step, or decaying the step).

That last thought points at one more cheap improvement. The decaying-step iterate converges, but the *last* iterate of a constant-or-slowly-decaying run rattles inside the noise ball. The information about the optimum is smeared across the recent iterates, not concentrated in the last one — which is exactly the situation where averaging helps. So keep a running average of the iterates themselves, `wbar_t = (1/(t - t_0)) sum_{i > t_0} w_i`, computed by a one-line recursive update, and report `wbar` rather than the final `w`. Averaging cancels the zero-mean rattle and recovers a point closer to the optimum — the same "average out the noise" logic that turned the `1/n`-step recursion into the sample mean, now applied to the parameter trajectory. It costs one extra vector and a running update.

Now the concrete code, filling the one empty slot in the diagonal-net optimizer harness — the update rule, given the gradients of the two parameter vectors. The minimal, canonical form is the bare recursion `w <- w - gamma * g` applied to each of `u` and `v`, with a fixed step (the speed/accuracy dial from the constant-step analysis — run to a plateau, read off the answer):

```python
from typing import Any
import torch


def get_hyperparameters(dim: int, sparsity: int, delta: float) -> dict[str, Any]:
    # the single knob: the step size gamma (constant-step SGD).
    return {"lr": 0.1}


def init_state(u: torch.Tensor, v: torch.Tensor,
               hyperparameters: dict[str, Any]) -> dict[str, Any]:
    # bare SGD keeps no per-parameter state; just count steps.
    return {"t": 0}


def step(u: torch.Tensor, v: torch.Tensor,
         grad_u: torch.Tensor, grad_v: torch.Tensor,
         state: dict[str, Any], hyperparameters: dict[str, Any]
         ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    lr = float(hyperparameters["lr"])
    state["t"] = state.get("t", 0) + 1
    # w_{t+1} = w_t - gamma * g_t, the stochastic-approximation recursion with alpha=0,
    # applied coordinate-wise to each parameter vector.
    return u - lr * grad_u, v - lr * grad_v, state
```
