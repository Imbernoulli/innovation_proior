Let me start from what actually hurts. I have a slot machine with `K` arms, each paying an i.i.d. reward bounded in `[0, 1]` with an unknown mean `mu_a`, and I want to minimize regret, which I can write as `R_n = sum_{a : mu_a < mu*} Delta_a E[N_a(n)]` with `Delta_a = mu* - mu_a` the gap and `N_a(n)` the number of times I pulled arm `a`. So the regret is just a gap-weighted tally of my sub-optimal pulls. Everything reduces to one thing: keep `E[N_a(n)]` small for every bad arm. And I want to do it with an *index* policy — for each arm compute one number from that arm's own data, pull the largest — because that's `O(K)` per round, horizon-free, and dead simple to deploy. The optimistic reading of an index is the one I trust: don't use the bare estimate `mu_hat_a = S_a / N_a`, use an *upper confidence bound* on `mu_a`, and act as if every arm is as good as its data still permit. Then under-sampled arms get explored automatically (wide bounds, high optimistic value) and well-sampled good arms get exploited (tight bounds sitting near their true high mean).

The workhorse I have today is UCB1: pull each arm once, then at step `t` pull `argmax_a mu_hat_a + sqrt(2 log t / N_a)`. The additive radius `sqrt(2 log t / N_a)` is a Hoeffding bonus — treat a `[0, 1]` reward as having variance proxy `1/4`, and the half-width that fails with probability about `t^{-4}` is `sqrt(2 log t / N_a)`. It works: Auer, Cesa-Bianchi and Fischer prove `E[R_n] <= sum_a 8 log n / Delta_a + (1 + pi^2/3) sum_a Delta_a`, i.e. `E[N_a(n)] <= 8 log n / Delta_a^2 + C`. That's `O(log n)`, horizon-free, no tuning, holds for any `[0, 1]`-bounded reward. Four boxes ticked. But stare at that constant `8 / Delta_a^2`, and stare at the bonus. The bonus depends only on `N_a` and the range `[0, 1]`; it never looks at *where* `mu_hat_a` sits. So an arm with `mu_hat = 0.5` and an arm with `mu_hat = 0.02` get the *same* confidence half-width at equal `N_a`. That can't be right. A coin that comes up heads two times in a hundred is intrinsically far less variable than a fair coin — its sample mean concentrates much harder — so my interval around `0.02` should be much narrower than my interval around `0.5`. UCB1 throws that away. It uses a *symmetric*, range-only width, which is exactly the inverse of the quadratic Pinsker bound, and Pinsker is loosest precisely near the boundaries `0` and `1`. On low-reward arms — the rare-event regime you actually meet in advertising or clinical trials, where the optimal arm might pay `0.1` and the rest pay `0.01` — UCB1 over-explores badly. The disease is the symmetric, variance-blind width.

Is there a floor I'm trying to reach, so I know how far off `8` is? Yes — Lai and Robbins proved one, and Burnetas and Katehakis generalized it. For any *uniformly good* policy (regret `o(n^alpha)` for every `alpha` on every instance), `liminf_n E[N_a(n)] / log n >= 1 / Kinf(nu_a, mu*)`, where `Kinf(nu_a, x) = inf{ KL(nu_a, nu') : E[nu'] > x }`. The argument behind it is a change of measure: to justify abandoning arm `a`, I must gather enough evidence to rule out the most confusing alternative world — one where I nudge `a`'s distribution up just past `mu*` so it becomes the best arm — and the number of samples it takes to tell the true world from that nudged twin is set by a Kullback-Leibler divergence. The smallest such divergence, over all upward nudges, is `Kinf`. For Bernoulli arms this `Kinf` collapses to the Bernoulli KL `d(mu_a, mu*)`, with `d(p, q) = p log(p/q) + (1 - p) log((1-p)/(1-q))`. So the per-arm floor is `1 / d(mu_a, mu*)`, and the regret floor is `sum_a Delta_a / d(mu_a, mu*)`. Now compare. Pinsker says `d(mu_a, mu*) >= 2 Delta_a^2`, so `1 / d <= 1 / (2 Delta_a^2)`; UCB1 pays `8 / Delta_a^2`, sixteen times the Pinsker-level constant `1/2`, and worse against the true `d`. And the gap between `d` and the Pinsker parabola `2(p-q)^2` *grows* as the means approach `0` or `1` — a Taylor expansion gives `d(p, q) = (p - q)^2 / (2 p (1 - p)) + o((p-q)^2)`, so the local curvature is `1 / (2 p (1 - p))`, which blows up near the boundary while Pinsker keeps the flat `2`. That's the quantitative version of "the rare-event arm should be cheap to rule out, and UCB1 doesn't notice." The whole loss is that UCB1's confidence width is the inverse of a *quadratic*, when it should be the inverse of `d` itself.

So the obvious thing to want: a confidence width built from `d`, not from its Pinsker lower bound. Where would that come from? From the actual large-deviation tail of the sample mean, not the Hoeffding one. For i.i.d. Bernoulli `X_1, ..., X_n` with mean `mu`, Chernoff's method gives `P(mu_hat >= mu + eps) <= exp(-n d(mu + eps, mu))`. Let me actually run the Chernoff method so I trust the rate. For `lambda > 0`, `P(mu_hat >= mu + eps) = P(exp(lambda sum (X_t - mu)) >= exp(lambda n eps)) <= E[exp(lambda sum(X_t - mu))] / exp(lambda n eps) = (mu exp(lambda(1 - mu - eps)) + (1 - mu) exp(-lambda(mu + eps)))^n`. Minimizing over `lambda` — set the derivative to zero, the optimizer is `lambda = log( (mu + eps)(1 - mu) / (mu(1 - mu - eps)) )` — and substituting back, the base collapses, after a couple of lines of algebra, to exactly `exp(-d(mu + eps, mu))`. So the whole tail is `exp(-n d(mu + eps, mu))`, and symmetrically `P(mu_hat <= mu - eps) <= exp(-n d(mu - eps, mu))`. That's the *exact* exponential rate; Pinsker (hence Hoeffding) would only give `exp(-2 n eps^2)`, which is the same near `mu = 1/2` but enormously weaker near the boundaries. So `n d(mu_hat, mu)` is the natural "nats of surprise" in an observed deviation, and I should build my interval by thresholding *that*.

Concretely: invert the tail. Fix a budget `a >= 0` and define `U(a) = max{ u in [0, 1] : d(mu_hat, u) <= a }`. Because `d(mu_hat, .)` is `0` at `u = mu_hat`, strictly convex, and strictly increasing on `[mu_hat, 1]`, this `U(a)` is the right endpoint of the feasible interval: either the root where `d(mu_hat, U(a)) = a`, or `1` if the whole right branch is still feasible. I claim `mu <= U(a)` with probability at least `1 - exp(-n a)`. If `mu > U(a)`, then `mu` lies to the right of the maximal feasible endpoint, so necessarily `mu_hat < mu` and `d(mu_hat, mu) > a`. That event is contained in `{ d(mu_hat, mu) > a, mu_hat < mu }`, whose probability the left-tail Chernoff bound caps at `exp(-n a)`. So `P(mu > U(a)) <= exp(-n a)`, i.e. `U(a)` is a valid upper confidence bound of risk `exp(-n a)`. And — this is the payoff — as `mu_hat` rises toward `1`, the function `d(mu_hat, .)` gets steeper, so the width `U(a) - mu_hat` *shrinks on its own*, exactly the variance-adaptation UCB1 was missing. The interval is asymmetric and self-tightening near the boundary, for free, with no variance estimate to plug in.

That gives me the index. To make the failure risk decay like the inverse of (a bit more than) `t`, I want `exp(-n a) ~ 1 / t`, i.e. budget `a = (something like log t) / N_a`. So the index for arm `a` at time `t` is

```
U_a(t) = max{ q in [0, 1] : N_a d(mu_hat_a, q) <= log t (+ a little more) },
```

and I pull `argmax_a U_a(t)`. Call this the KL ball: the upper bound is the largest mean `q` whose Bernoulli-KL distance from `mu_hat_a` costs at most the deviation budget `log t / N_a`. This *is* the optimism principle, just with the honest large-deviation geometry instead of the quadratic proxy. UCB1 is literally the special case where I replace `d(p, q)` by its Pinsker lower bound `2(p - q)^2`: then `N_a 2(mu_hat_a - q)^2 <= log t` solves to `q = mu_hat_a + sqrt(log t / (2 N_a))`, the additive sqrt bonus. So I'm not inventing a new family, I'm using the tight member of the one UCB1 already lives in.

Now the part that worried me at first and turns out to be the deepest point. My derivation of the index leaned on the Bernoulli Chernoff rate `d`. But the problem only promises rewards *bounded in `[0, 1]`* — they could be any distribution on the interval, not Bernoulli at all. So why should a *Bernoulli* divergence control a general bounded reward's deviations? I almost reached for a different rate function per distribution, and then realized that would wreck the whole point: I don't know the distribution, and I want one universal index. Let me check whether the Bernoulli rate is actually an *upper bound* on everyone's deviations. Take any `X` in `[0, 1]` with mean `mu`, and look at its moment generating function `E[exp(lambda X)]`. Consider `f(x) = exp(lambda x) - x(exp(lambda) - 1) - 1`. It's convex in `x` (second derivative `lambda^2 exp(lambda x) >= 0`), and `f(0) = 1 - 0 - 1 = 0`, `f(1) = exp(lambda) - (exp(lambda) - 1) - 1 = 0`. A convex function that vanishes at both endpoints of an interval is `<= 0` throughout the interval. So `f(x) <= 0` on `[0, 1]`, i.e. `exp(lambda X) <= X(exp(lambda) - 1) + 1` pointwise. Take expectations: `E[exp(lambda X)] <= mu(exp(lambda) - 1) + 1 = 1 - mu + mu exp(lambda)`. And the right-hand side is *exactly the MGF of a Bernoulli with mean `mu`*. So among all `[0, 1]` variables with a given mean, the Bernoulli has the largest MGF — it is, in this exact sense, the *least concentrated* bounded variable, the worst case. Therefore its Chernoff rate `d(.,.)` *dominates* the deviations of any `[0, 1]` reward, and my Bernoulli-KL index is a valid upper confidence bound for *all* of them. That's the lemma that makes the method distribution-free while still using `d`. It's the analogue of "variance `1/4` is the max for `[0, 1]`," lifted from second moments to the whole MGF.

But here's a subtlety I can't gloss over. The clean Chernoff bound I derived was for a *fixed* number `n` of i.i.d. summands. In the bandit, `N_a(t)` is random — it depends on the whole history through my own adaptive choices — and I'm taking a union over all the times I might evaluate the bound. A naive union bound over `t = 1, ..., n` would cost a factor `n` and ruin the rate. I need a deviation inequality that holds for the *self-normalized* mean `S_a(t) / N_a(t)` with a random, data-dependent number of summands, uniformly in time, and only loses a tiny factor. So let me build that.

The right object is a supermartingale. For a sequence of independent `[0, 1]` variables `X_s` with mean `mu`, let `psi_mu(lambda) = log(1 - mu + mu exp(lambda))`, the Bernoulli log-MGF that upper-bounds the true log-MGF by the bounded-to-Bernoulli lemma. Let `epsilon_s in {0, 1}` be the *previsible* indicator "did I pull this arm at step `s`" (it's `F_{s-1}`-measurable — I decide whether to pull before seeing the reward), and set `S(t) = sum_{s<=t} epsilon_s X_s`, `N(t) = sum_{s<=t} epsilon_s`. Define `W_0^lambda = 1` and `W_t^lambda = exp(lambda S(t) - N(t) psi_mu(lambda))`. I claim this is a supermartingale. Compute `E[exp(lambda(S(t+1) - S(t))) | F_t] = E[exp(lambda epsilon_{t+1} X_{t+1}) | F_t]`. Since `epsilon_{t+1}` is known given `F_t` and is `0` or `1`: if it's `0` the factor is `1`; if it's `1`, the lemma gives `E[exp(lambda X_{t+1})] <= exp(psi_mu(lambda))`. Either way it is at most `exp(epsilon_{t+1} psi_mu(lambda)) = exp((N(t+1) - N(t)) psi_mu(lambda))`. Multiply through by `exp(lambda S(t) - N(t+1) psi_mu(lambda))` and rearrange: `E[W_{t+1}^lambda | F_t] <= W_t^lambda`. So `W^lambda` is a supermartingale with `E[W_t^lambda] <= E[W_0^lambda] = 1`. Good — that's the engine, and it tolerates the random `N(t)`.

Now turn the supermartingale into a uniform-in-time bound on `P(u(n) < mu)`, where `u(n) = max{ q > mu_hat(n) : N(n) d(mu_hat(n), q) <= delta }` is my upper bound at deviation budget `delta`. First, `u(n) < mu` happens exactly when the truth is under-estimated badly: `mu_hat(n) < mu` *and* `N(n) d(mu_hat(n), mu) > delta`. If I tried a single Markov bound on `W_n` at one cleverly chosen `lambda`, I'd be stuck because the right `lambda` depends on the unknown random `N(n)`. So I slice. This is the peeling trick (Massart's lectures): cut the range `{1, ..., n}` of possible values of `N(n)` into geometric blocks and handle each block with a `lambda` tuned to that block. Assume `delta > 1` (otherwise the bound is trivial). Set `eta = 1/(delta - 1)`, `t_0 = 0`, `t_k = floor((1 + eta)^k)`, and let `D = ceil(log n / log(1 + eta))` be the number of slices needed to cover `n`. Let `A_k = { t_{k-1} < N(n) <= t_k } cap { u(n) < mu }`. Then `P(u(n) < mu) <= P(union_k A_k) <= sum_{k=1}^D P(A_k)`. If a slice is so early that `delta / N(n)` would exceed the maximum possible left-tail divergence `d(0, mu)`, the event is empty; I only need to bound the slices where the tilted point below `mu` exists.

On slice `k`, I want a single `lambda` that works for every `N(n)` in `(t_{k-1}, t_k]`. Pick `z < mu` with `d(z, mu) = delta / (1 + eta)^k`, and let `lambda(z) = log( z(1 - mu) / (mu(1 - z)) ) < 0` be the Chernoff-optimal tilt for `z`, the one for which `d(z, mu) = lambda(z) z - psi_mu(lambda(z))` (this is the Bernoulli Legendre duality `d(z, mu) = sup_lambda { lambda z - psi_mu(lambda) }` evaluated at its maximizer). Now suppose I'm on `A_k`: `N(n) <= t_k <= (1 + eta)^k`, so `delta / N(n) >= delta / (1+eta)^k = d(z, mu)`, and the under-estimation event forces `d(mu_hat(n), mu) > delta / N(n) >= d(z, mu)` with `mu_hat(n) < mu`. Because `d(., mu)` is decreasing on `[0, mu]`, this puts `mu_hat(n) <= z`. Since `lambda(z) < 0`, the inequality reverses in the useful direction: `mu_hat(n) <= z` gives `lambda(z) mu_hat(n) >= lambda(z) z`, and therefore `lambda(z) mu_hat(n) - psi_mu(lambda(z)) >= lambda(z) z - psi_mu(lambda(z)) = d(z, mu)`. Also `N(n) > floor((1 + eta)^{k-1})`, hence `N(n) > (1 + eta)^{k-1}`, so `d(z, mu) = delta / (1+eta)^k >= delta / ((1+eta) N(n))`. Stringing it together, on `A_k`,

```
lambda(z) S(n) - N(n) psi_mu(lambda(z)) = N(n)(lambda(z) mu_hat(n) - psi_mu(lambda(z))) >= N(n) d(z, mu) >= delta/(1+eta),
```

so `A_k subset { W_n^{lambda(z)} >= exp(delta/(1+eta)) }`. Markov on the supermartingale gives `P(A_k) <= exp(-delta/(1+eta))`. Sum over the `D` slices:

```
P(u(n) < mu) <= D exp(-delta/(1+eta)).
```

Now plug in `eta = 1/(delta - 1)`, so `1 + eta = delta/(delta - 1)` and `delta/(1+eta) = delta - 1`, giving `exp(-(delta - 1)) = e exp(-delta)`. And `D = ceil(log n / log(1 + 1/(delta-1)))`; using `log(1 + 1/(delta-1)) >= 1/delta`, I get `D <= ceil(delta log n)`. Therefore

```
P(u(n) < mu) <= e ceil(delta log n) exp(-delta).
```

That's the self-normalized deviation bound — uniform over the random `N(n)`, paying only a `delta log n` polynomial factor in front of the exponential. The symmetric statement for over-estimation follows identically, so `P(N(n) d(mu_hat(n), mu) > delta) <= 2 e ceil(delta log n) exp(-delta)`.

This tells me what exploration budget `delta` to use. I want the probability that I under-estimate the *best* arm, summed over all `t`, to be a low-order term. Take `delta = log t + c log log t`. Then `P(mu_1 > u_1(t)) <= e ceil(log t (log t + c log log t)) exp(-log t - c log log t) = e ceil(log t^2 + c log t log log t) / (t (log t)^c)`. For this to sum over `t` to something like `O(log log n)`, I need the `(log t)^c` in the denominator to beat the `log t^2` in the ceiling — and `c = 3` does it: the partial sum of `e ceil(log t^2 + 3 log t log log t) / (t (log t)^3)` up to `n` is `O(log log n)` (a constant `C_1' <= 7` suffices). So `c = 3` in the exploration function `log t + 3 log log t` is exactly what makes the optimal-arm under-estimation term negligible. Lower `c` would let that sum diverge faster than I can afford in the proof.

With the deviation bound in hand the regret proof is short. Fix a sub-optimal arm `a`, take `a* = 1`. I split the count of `a`-pulls into "the best arm was under-estimated" and "it wasn't but I still pulled `a`":

```
E[N_n(a)] <= sum_t P(mu_1 > u_1(t)) + E[ sum_t 1{A_t = a, mu_1 <= u_1(t)} ].
```

The first sum is `O(log log n)` by the above. For the second, note that if I pull `a` while `mu_1 <= u_1(t)`, then by my own greedy rule `u_a(t) >= u_1(t) >= mu_1`, and since `u_a(t)` is inside the confidence set for `mu_hat_a`, that means `d^+(mu_hat_a(t), mu_1) <= d(mu_hat_a(t), u_a(t)) <= (log t + 3 log log t) / N_a(t)`, where `d^+(x, y) = d(x, y) 1{x < y}` (only an under-estimate of a *better* mean counts). Re-indexing by the number of samples `s` of arm `a` — each value `s` of `N_a` occurs for at most one pull — collapses the time sum into a sum over `s` (this is the bookkeeping lemma): the on-policy count is bounded by `sum_{s=1}^n 1{ s d^+(mu_hat_{a,s}, mu_1) < log n + 3 log log n }`, where `mu_hat_{a,s}` is the mean of the first `s` samples of arm `a`. Now split this sum at

```
K_n = floor( (1+eps)/d^+(mu_a, mu_1) (log n + 3 log log n) ).
```

The first `K_n` terms contribute at most `K_n`. For `s > K_n`, `s d^+(mu_hat_{a,s}, mu_1) < log n + 3 log log n` together with `s > K_n` forces `d^+(mu_hat_{a,s}, mu_1) < d(mu_a, mu_1)/(1+eps)`, which (since `d(., mu_1)` is decreasing toward `mu_1`) means `mu_hat_{a,s} > r(eps)`, where `r(eps) in (mu_a, mu_1)` solves `d(r(eps), mu_1) = d(mu_a, mu_1)/(1+eps)`. But `mu_hat_{a,s} > r(eps) > mu_a` is an *over*-estimate of arm `a`'s own mean, so Chernoff gives `P(mu_hat_{a,s} > r(eps)) <= exp(-s d(r(eps), mu_a))`, and summing the geometric tail from `K_n + 1`,

```
sum_{s > K_n} exp(-s d(r(eps), mu_a)) <= exp(-K_n d(r(eps), mu_a)) / (1 - exp(-d(r(eps), mu_a))) = C_2(eps)/n^{beta(eps)},
```

with `beta(eps) = (1+eps) d(r(eps), mu_a)/d(mu_a, mu_1)` and `C_2(eps) = (1 - exp(-d(r(eps), mu_a)))^{-1}`. Since `r(eps) = mu_a + O(eps)`, `d(r(eps), mu_a) = O(eps^2)`, so `C_2(eps) = O(eps^{-2})` and `beta(eps) = O(eps^2)`. Putting the three pieces together,

```
E[N_n(a)] <= (log n)/d(mu_a, mu_1) (1 + eps) + C_1 log log n + C_2(eps)/n^{beta(eps)}.
```

Let `eps -> 0` after `n -> inf` and the leading term is `log n / d(mu_a, mu*)`, so

```
limsup_n E[R_n]/log n <= sum_{a : mu_a < mu*} (mu* - mu_a) / d(mu_a, mu*).
```

For Bernoulli rewards that *is* the Lai-Robbins floor, so this index policy is asymptotically optimal in the binary case — and by the bounded-to-Bernoulli lemma the very same bound holds for *every* `[0, 1]` reward, because the proof only ever used the Bernoulli MGF as an upper bound, never the assumption that the rewards are binary. So I get optimality for Bernoulli and a leading-constant improvement over UCB for all bounded rewards, from one index, with no tuning. The improvement over UCB falls right out of Pinsker: replacing `d` by its quadratic lower bound `2(p-q)^2` in the *same* proof gives a correctly tuned UCB the bound `E[N_n(a)] <= log n / (2 Delta_a^2) (1+eps) + ...` — recovering the optimal `1/2` constant for the quadratic divergence as a by-product — and since `d(mu_a, mu*) >= 2 Delta_a^2` always, with the gap large near the boundary, the KL leading term is smaller.

Let me settle the constant for practice, because the `3 log log t` correction is a theorem artifact. The proof needed `c >= 3` in `log t + c log log t` to make `sum_t 1/(t (log t)^{c-2})`-type series grow only like `log log n` or better. But I could instead use an exploration function `(1+eps) log t` and similar results hold, since `(1+eps) log t >= log t + 3 log log t` once `t` is large enough — except "large enough" is absurd: for `eps = 0.1` it only kicks in past `t > 2 * 10^51`, way beyond any real horizon. So the proof-friendly finite-time statement keeps `log t + 3 log log t`, while the practical implementation drops the `log log t` correction and uses a pure `c_impl log t` budget. The default I want in code is `c_impl = 1`, i.e. plain `log t / N_a`; the heavier `log t + 3 log log t` is what I keep for the proven guarantee.

Now, how do I actually compute `U_a(t) = max{ q : N_a d(mu_hat_a, q) <= c_impl log t }`? There's no closed form for the inverse of `d` in its second argument, but the geometry is friendly: for fixed `p = mu_hat_a`, `q -> d(p, q)` is strictly convex, zero at `q = p`, and strictly increasing on `[p, 1]`. So on `[p, 1]` the equation `N_a d(p, q) = c_impl log t`, i.e. `d(p, q) = (c_impl log t)/N_a =: budget`, has a unique root, and `U_a(t)` is that root (or the right endpoint if the whole bracket is inside the ball). I can find it by bisection on `[p, 1]`, or by Newton since `d(p, .)` is convex. To bisect efficiently I want a good upper endpoint: Pinsker gives one almost for free. `d(p, q) >= 2(p - q)^2`, so any `q` inside the KL ball must satisfy `q <= p + sqrt(budget / 2)`. That is exactly the Gaussian/Pinsker seed: `upper = min(1, p + sqrt(budget/2))`, then bisect between `p` and `upper`, pushing the lower endpoint up when the midpoint is still inside the ball and pulling the upper endpoint down when it is outside. A handful of iterations — each one call to `d` — and I have the index to the policy tolerance. I'll guard the logarithms by clipping `p` and `q` off the exact `0/1` endpoints (where `d` is `+inf` by convention) to a tiny `eps`, so the arithmetic never sees `log 0`.

Before I write it down, one generalization I get almost for nothing, because I notice the proof never used that the rewards were binary or even bounded once I have the right MGF control. Suppose instead each arm's reward is from a one-parameter canonical exponential family, density `p_theta(x) = exp(x theta - b(theta) + c(x))` with log-partition `b` twice differentiable, so the mean is `mu(theta) = b'(theta)` and `b''(theta) = Var(X) > 0` makes `mu(.)` one-to-one. The right deviation rate is the Legendre transform of the cumulant generating function, `d(x, mu(theta)) = sup_lambda { lambda x - log E_theta[exp(lambda X)] }`. Let me verify this equals the family's own KL divergence, because that's what makes it Lai-Robbins-optimal for that family. The MGF of the exponential family shifts the parameter: `E_theta[exp(lambda X)] = int exp(x(theta + lambda) - b(theta) + c(x)) dx = exp(b(theta + lambda) - b(theta))`. So `lambda x - log E_theta[exp(lambda X)] = lambda x - b(theta + lambda) + b(theta)`, a smooth concave function of `lambda`, maximized where its derivative `x - b'(theta + lambda) = x - mu(theta + lambda) = 0`. If `x = mu(beta)`, then since `mu` is one-to-one, `theta + lambda* = beta`, and the value at the max is `d(mu(beta), mu(theta)) = (beta - theta) mu(beta) - b(beta) + b(theta)`. Meanwhile the family's KL is `KL(p_beta, p_theta) = int p_beta(x) [x(beta - theta) - b(beta) + b(theta)] dx = mu(beta)(beta - theta) - b(beta) + b(theta)` — the same expression. So the Legendre rate *is* the KL divergence of the family. Swap that `d` into the index, and every line of the regret proof goes through unchanged (it only ever invoked the MGF), giving Lai-Robbins optimality for that family. The recipes drop out: exponential rewards give `d(x, y) = x/y - 1 - log(x/y)`; Poisson gives `d(x, y) = y - x + x log(x/y)`; Gaussian with fixed variance gives back the additive sqrt bonus, i.e. UCB itself. And I don't even need the *exact* family rate — an upper bound on `d` still gives a valid (slightly looser) confidence bound, so I can use the simple Bernoulli `d` on, say, bounded exponential rewards and still do well. The Bernoulli case isn't special; it's just the worst-case-bounded instance of a single principle.

So let me write the index policy I'd actually ship — filling the one empty slot, the per-arm index, with the KL-ball upper bound and its bisection inverse, seeded by the Pinsker bracket:

```python
import math
import numpy as np

_EPS = 1e-15  # keep means off the exact 0/1 endpoints where d = +inf


def kl_bernoulli(p, q):
    # d(p, q) = p log(p/q) + (1-p) log((1-p)/(1-q)) -- the Bernoulli/Chernoff rate
    p = min(max(p, _EPS), 1 - _EPS)
    q = min(max(q, _EPS), 1 - _EPS)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def kl_ucb_index(mu_hat, n, budget, precision=1e-6, max_iter=50):
    # U = max{ q in [mu_hat, 1] : kl_bernoulli(mu_hat, q) <= budget / n }
    # The whole confidence width is here: q -> d(mu_hat, q) is convex & increasing on
    # [mu_hat, 1], so there is one root; the interval self-tightens as mu_hat -> 0 or 1.
    if n == 0:
        return 1.0
    threshold = budget / n
    # Pinsker bracket: d(p, q) >= 2(p - q)^2, so the true KL ball sits inside this.
    upper = min(1.0, mu_hat + math.sqrt(threshold / 2.0))
    lower = mu_hat
    for _ in range(max_iter):
        if upper - lower <= precision:
            break
        mid = 0.5 * (lower + upper)
        if kl_bernoulli(mu_hat, mid) > threshold:   # mid is outside the ball -> pull down
            upper = mid
        else:                                       # mid is inside -> push up
            lower = mid
    return 0.5 * (lower + upper)


class BanditPolicy:
    """KL-UCB index policy: pull argmax_a U_a(t), the KL-ball upper confidence bound."""

    def __init__(self, K, context_dim=0, c=1.0):
        self.K = K
        self.c = c                              # implementation multiplier; c=1 gives
        self.counts = np.zeros(K)               #   the plain log(t) budget. (The proof uses
        self.rewards = np.zeros(K)              #   log(t) + 3 log log(t).)

    def reset(self):
        self.counts[:] = 0
        self.rewards[:] = 0

    def select_arm(self, t, context=None):
        # Pull each arm once first, then any still-unpulled arm gets +inf index.
        if t < self.K:
            return t
        best_arm, best_index = 0, -math.inf
        budget = self.c * math.log(max(t, 1))        # delta = c log t (the exploration function)
        for a in range(self.K):
            if self.counts[a] == 0:
                return a
            mu_hat = self.rewards[a] / self.counts[a]
            index = kl_ucb_index(mu_hat, self.counts[a], budget)   # U_a(t)
            if index > best_index:
                best_index, best_arm = index, a
        return best_arm

    def update(self, arm, reward, context=None):
        self.counts[arm] += 1
        self.rewards[arm] += reward
```

Here is the causal chain. I started with UCB1, an optimistic index that ticks every box except the constant: its additive `sqrt(2 log t / N_a)` bonus is symmetric and range-only, so it ignores that a near-`0`-or-`1` arm concentrates much harder, and it pays `8/Delta_a^2` against a floor of `1/d(mu_a, mu*)`. The floor itself, from Lai and Robbins via a change-of-measure argument, is governed by the Bernoulli KL `d`, and `d` is exactly the *true* Chernoff large-deviation rate of the sample mean — which I derived by minimizing the MGF over the tilt `lambda` — whereas Hoeffding only delivers the quadratic Pinsker lower bound `2(p-q)^2`, loosest near the boundary. So the fix is to replace the additive bonus with a KL ball: `U_a(t) = max{ q : N_a d(mu_hat_a, q) <= c_impl log t }` in the practical code, whose width self-tightens near `0` and `1` for free; UCB1 is the Pinsker-relaxed special case of this. The index is distribution-free because of the convexity lemma `E[exp(lambda X)] <= 1 - mu + mu exp(lambda)` for any `[0, 1]` variable — the Bernoulli is the least concentrated bounded law, so its `d` dominates everyone's deviations. To bound the regret with a random number of pulls I built a self-normalized supermartingale `W_t = exp(lambda S(t) - N(t) psi_mu(lambda))` and a peeling argument over geometric slices of `N(n)`, yielding `P(u(n) < mu) <= e ceil(delta log n) exp(-delta)`; choosing `delta = log t + 3 log log t` makes the optimal-arm-underestimation term `O(log log n)`. The regret decomposition then splits sub-optimal pulls into best-arm underestimates (controlled by that bound) and a sample-count sum split at `K_n`, whose tail is a geometric Chernoff sum, giving `E[N_n(a)] <= (1+eps) log n / d(mu_a, mu*) + C_1 log log n + C_2(eps)/n^{beta(eps)}` and hence the Lai-Robbins-matching `limsup E[R_n]/log n <= sum_a Delta_a / d(mu_a, mu*)`. The same proof, untouched, generalizes to any one-parameter exponential family by swapping `d` for that family's Legendre/KL rate (exponential, Poisson, Gaussian recipes drop out). The proven exploration is `log t + 3 log log t`; the shipped index uses the practical pure-log budget with `c_impl = 1`. The index is computed by bisection on the convex, increasing `q -> d(mu_hat_a, q)`, seeded by the Pinsker/Gaussian endpoint `mu_hat_a + sqrt((c_impl log t)/(2 N_a))`, to the policy tolerance in a handful of `d`-evaluations.
