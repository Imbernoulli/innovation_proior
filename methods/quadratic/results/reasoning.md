Let me start from the thing that actually bothers me. I have a network with far more parameters than I have training examples, and on MNIST it generalizes beautifully — a few percent test error. But every guarantee I can write down for it is garbage. VC-dimension, Rademacher complexity, any uniform-convergence argument: they all scale with the capacity of the function class, and with a million parameters and sixty thousand examples the complexity term is enormous, the bound on the error comes out above 1, and "your error is at most 130%" is not a guarantee. So I have a model that works and no number that says so. What I want is a single number, computed from the training data, that is a true high-probability upper bound on the error on unseen data, and is small enough — close enough to the test error — to be worth printing.

The escape from the capacity trap is to stop certifying a fixed weight vector and instead certify a *distribution* over weights. If I commit to a distribution `Q` over weight space and predict by drawing a fresh `W ~ Q` each time, then the object I care about is the `Q`-averaged true risk `L(Q) = E_{W~Q}[L(W)]`, and the empirical side is `Lhat_S(Q) = E_{W~Q}[Lhat_S(W)]`. The reason this helps is the form the complexity term takes in this randomized setting: it is `KL(Q || Q0)`, the Kullback-Leibler divergence from a reference distribution `Q0` fixed before I look at the data, to my learned `Q`. That is *not* a parameter count. It measures how far my learned distribution had to move from the reference, in nats. If my posterior stays near its prior, `KL(Q || Q0)` is small no matter how many parameters there are. That is the whole reason this route survives over-parameterization where uniform convergence dies. So `Q0` is a "prior" and `Q` a "posterior" in the PAC-Bayes sense — just a fixed reference and an unrestricted data-dependent distribution, no likelihood factor between them, nothing Bayesian.

Now I need the actual inequality. The strongest classical one controls not the difference but the binary KL between the two risks. For `q, p in [0,1]`, write `kl(q || p) = q log(q/p) + (1-q) log((1-q)/(1-p))`, the relative entropy of Bernoulli(`q`) from Bernoulli(`p`). The PAC-Bayes-kl bound says: with probability at least `1 - delta` over the size-`n` sample, simultaneously for all `Q`,

  kl( Lhat_S(Q) || L(Q) )  <=  ( KL(Q || Q0) + log( 2 sqrt(n) / delta ) ) / n.

Let me make sure I believe this and understand where every piece comes from, because everything I build will be a relaxation of it and I had better know what I'm relaxing. Start per-hypothesis. For a fixed `h`, the empirical loss `M(h(S))` is an average of `n` i.i.d. `[0,1]` variables with mean the true loss `h(D)`. The clean way to control `kl(M(h(S)) || h(D))` is the exponential moment: I want `E_S[ e^{ n kl( M(h(S)), h(D) ) } ]` to be small. The claim I keep seeing quoted is that for `n >= 8` this expectation is at most `2 sqrt(n)`. I don't want to take the `sqrt(n)` order on faith — it controls the whole `log` term — so let me reduce it to something I can actually compute. The integrand is largest when `M(h(S))` deviates from `h(D)`; the worst case is a Bernoulli loss, and reducing the general `[0,1]` case to Bernoulli is a convexity argument — `e^{n kl}` is convex, so its expectation over `[0,1]` variables is dominated by the expectation over the corresponding Bernoulli variables with the same mean. For Bernoulli `M(h(S))` takes values `k/n`, so the whole expectation is the explicit finite sum `sum_{k=0}^{n} C(n,k) mu^k (1-mu)^{n-k} e^{n kl(k/n || mu)}` for `mu = h(D)`. That I can just evaluate. Sweeping `mu` over `0.1, 0.3, 0.5` and `n` over `8, 32, 128, 512`, the worst-over-`mu` value comes out `4.25, 7.77, 14.86, 29.03`. Lay those next to `sqrt(n) = 2.83, 5.66, 11.31, 22.63` and `2 sqrt(n) = 5.66, 11.31, 22.63, 45.25`: each value sits between `sqrt(n)` and `2 sqrt(n)`, and the ratio of successive values (`29.03/14.86 ≈ 1.95`, `14.86/7.77 ≈ 1.91`) tracks the `sqrt(4) = 2` you'd expect from quadrupling `n` — so the growth really is `Θ(sqrt(n))`, not constant and not linear, and `2 sqrt(n)` really is a valid envelope on every row, never violated. Good — that pins down the order I cared about. The heuristic behind it is that after the kl exponential cancels the binomial probability in each term, Stirling leaves something of order `1/sqrt(n)` per `k` summed over `n+1` values of `k`; the numbers above are the version of that I trust. The upshot is that the `2 sqrt(n)` in the numerator's logarithm is the sharp price of the moment inequality, and it makes the bound's `log` term grow like `log sqrt(n) = (1/2) log n` rather than `log n` — a real halving over the older constant, and free, so I keep it.

Now lift from a fixed `h` to a data-dependent `Q`. This is the only subtle step. I have a per-hypothesis moment bound under `P` (the fixed reference), but I want a statement about `E_{W~Q}` where `Q` depends on the sample. The tool is a change of measure. For any `Q` absolutely continuous w.r.t. `P` and any function `f`,

  E_{h~Q}[ f(h) ]  =  E_{h~Q}[ log( e^{f} ) ]  <=  KL(Q || P) + log E_{h~P}[ e^{f(h)} ].

That inequality is just the variational (Donsker-Varadhan) form: the gap between `E_Q[f]` and `log E_P[e^f]` is exactly `KL(Q || P)` minus a nonnegative KL to the Gibbs measure, so dropping that nonnegative term gives the inequality. Apply it with `f(h) = n kl( M(h(S)), h(D) )`. Then

  n E_{h~Q}[ kl(...) ] - KL(Q || P)  <=  log E_{h~P}[ e^{ n kl(...) } ].

Exponentiate, take `E_S` of both sides, and on the right I can swap `E_S` and `E_{h~P}` (Fubini — `P` is data-free, the order is legal) and use the per-hypothesis moment bound inside: `E_S E_{h~P}[ e^{n kl} ] = E_{h~P} E_S[ e^{n kl} ] <= E_{h~P}[ 2 sqrt(n) ] = 2 sqrt(n)`. So

  E_S[ exp( n E_{h~Q}[kl] - KL(Q || P) ) ]  <=  2 sqrt(n),

and now Markov: the probability that the exponent exceeds `log(2 sqrt(n)/delta)` is at most `delta`. Rearranged, with probability `1 - delta`, for all `Q` simultaneously (the `KL(Q||P)` is inside, so the statement is already uniform over `Q`):

  E_{h~Q}[ kl( Lhat_S(h), L(h) ) ]  <=  ( KL(Q || P) + log(2 sqrt(n)/delta) ) / n.

One more move to get the displayed form: `kl(. || .)` is jointly convex, so by Jensen `E_{h~Q}[ kl(Lhat_S(h), L(h)) ] >= kl( E_Q[Lhat_S], E_Q[L] ) = kl( Lhat_S(Q), L(Q) )`. That gives exactly the displayed pb-kl. Now I can name the chain that produced it: change of measure to move `Q` onto the fixed `P`, the `2 sqrt(n)` moment bound (which I just checked numerically) applied per hypothesis, Markov, Jensen. Looking back over the four steps, the only inequality that is genuinely lossy is Markov — the moment bound is tight to within the factor `2` I measured, the change of measure is an identity minus a dropped nonnegative KL, and Jensen on a convex function. So if my eventual certificate is loose, the looseness will not be hiding in this derivation; it will be in whatever I do to `kl(Lhat || L)` next to make it optimizable.

The trouble with pb-kl as a thing to *use* is that it bounds `kl(Lhat_S(Q) || L(Q))`, not `L(Q)` itself. As a certificate I can invert it — find the largest `L(Q)` consistent with `kl(Lhat_S(Q) || L(Q)) <= c` — and that's the right thing to do for the final number. But as a *training objective*, `kl(Lhat_S(Q) || L(Q))` is implicit in `L(Q)` and there's no closed form to differentiate; I can't just hand SGD `kl(...)` and ask it to push `L(Q)` down. So the practical move everyone makes is to *relax* pb-kl into an explicit upper bound on `L(Q)` that I can write down, plug a surrogate loss into, and minimize directly. The question is which relaxation, and the answer is going to decide whether my certificate is tight.

The textbook relaxation is Pinsker's inequality, `kl(qh || p) >= 2(p - qh)^2`. Lower-bound the left side of pb-kl by this: `2(L(Q) - Lhat_S(Q))^2 <= (KL + log(2 sqrt(n)/delta))/n`. Solve for `L(Q)` (take the upper root):

  L(Q)  <=  Lhat_S(Q) + sqrt( ( KL(Q || Q0) + log(2 sqrt(n)/delta) ) / (2n) ).

Call the complexity quantity `kl_term = (KL + log(2 sqrt(n)/delta)) / (2n)`. This is clean, it's explicit, it's optimizable — replace the 0-1 loss with a smooth surrogate and minimize over `Q`. This is essentially the objective that gave the first non-vacuous neural-net bounds. But stare at it in the regime I actually live in. A trained network drives `Lhat_S(Q)` toward zero. Then the bound becomes `0 + sqrt(kl_term)` — it collapses to `sqrt(kl_term)`. The complexity enters through a *square root*. If `kl_term` is small, say `1e-3`, the certificate is `sqrt(1e-3) ≈ 0.032`. The empirical error is essentially zero and the network might have a couple percent test error, yet my certificate sits at 3%+, dominated entirely by `sqrt(complexity)`, and it refuses to get smaller as I train harder because `Lhat_S(Q) -> 0` already saturated the additive term. The certificate is non-vacuous but it is loose, and it is loose *exactly where it matters*. Wall.

Why is it loose there? My suspicion is that Pinsker is the loose lower bound on `kl` in the small-`p` regime. `kl(qh || p) >= 2(p - qh)^2` is a *parabola* lower bound, symmetric in `p` around `qh`. But the true `kl` is asymmetric — for `qh < p`, as `p` shrinks toward `qh`, `kl` grows faster than a symmetric parabola allows. There is a sharper, asymmetric Pinsker variant for that side: `kl(qh || p) >= (p - qh)^2 / (2p)`, valid for `qh < p`. Comparing the two prefactors algebraically — `1/(2p)` for the refined one versus `2` for the standard — refined should beat standard exactly when `1/(2p) > 2`, i.e. `p < 1/4`. Before I rebuild the bound around that claim I want to see the actual numbers, including that I haven't fooled myself with an inequality that *overshoots* the true `kl` (which would be invalid as a lower bound). Take `qh = 0` and tabulate `kl`, the standard bound `2p^2`, and the refined bound `p^2/(2p) = p/2` at `p = 0.05, 0.1, 0.2, 0.25, 0.3, 0.4`:

  p=0.05:  kl=0.0513  std=0.0050  ref=0.0250  -> ref larger, both < kl
  p=0.10:  kl=0.1054  std=0.0200  ref=0.0500  -> ref larger, both < kl
  p=0.20:  kl=0.2231  std=0.0800  ref=0.1000  -> ref larger, both < kl
  p=0.25:  kl=0.2877  std=0.1250  ref=0.1250  -> equal, both < kl
  p=0.30:  kl=0.3567  std=0.1800  ref=0.1500  -> std larger, both < kl
  p=0.40:  kl=0.5108  std=0.3200  ref=0.2000  -> std larger, both < kl

Two things land. First, in every row both candidate bounds stay strictly below the true `kl`, so both are legitimate lower bounds — the refined one is not cheating by exceeding `kl`. Second, the crossover is exactly where the prefactor algebra said: refined is the larger (tighter) lower bound for `p < 0.25`, they coincide at `p = 0.25` (both `0.1250`), and standard takes over above. And `p < 1/4` — a true risk below 25% — is precisely the regime a network that works lives in. So I've been relaxing pb-kl with the wrong inequality for my regime; I should relax with the refined one.

Substitute `kl(qh || p) >= (p - qh)^2/(2p)` with `qh = Lhat_S(Q)`, `p = L(Q)` into pb-kl. Let `C = (KL + log(2 sqrt(n)/delta))/n` — the bare pb-kl right-hand side, divided by `n`, with no extra factor of 2 yet. Then `(L - Lhat)^2 / (2L) <= C`, i.e. `(L - Lhat)^2 <= 2 L C`. If `L <= Lhat`, the empirical term already upper-bounds the risk; the certificate case I need to solve is `L >= Lhat`. Taking that root, `L - Lhat <= sqrt(2 L C)`, so

  L(Q)  <=  Lhat_S(Q) + sqrt( 2 L(Q) C ).        (★)

And there's the catch that everyone hits next: `L(Q)` appears on *both* sides — under the square root on the right. So (★) is not an explicit upper bound, I can't just minimize the right side because the right side contains the thing I'm bounding. This is exactly why this inequality "is not immediately useful for optimization." Wall, again, but a different kind — this one is algebraic, not a regime problem, and algebraic walls have a way out.

The way out: (★) is a quadratic inequality, just not in `L`. It's quadratic in `sqrt(L)`. Set `x = sqrt(L)`, so `L = x^2`. Then (★) reads `x^2 <= Lhat + sqrt(2C) x`, i.e.

  x^2 - sqrt(2C) x - Lhat  <=  0.

A quadratic in `x` with a positive leading coefficient is `<= 0` between its roots, so `x` is at most the larger root:

  x  <=  ( sqrt(2C) + sqrt( 2C + 4 Lhat ) ) / 2.

Square it to get back `L = x^2`:

  L  <=  ( ( sqrt(2C) + sqrt(2C + 4 Lhat) ) / 2 )^2.

Let me clean this up, because I suspect it has a nicer form. Pull the `1/2` inside each square root: `sqrt(2C)/2 = sqrt(2C/4) = sqrt(C/2)`, and `sqrt(2C+4Lhat)/2 = sqrt((2C+4Lhat)/4) = sqrt(C/2 + Lhat)`. So

  L(Q)  <=  ( sqrt( Lhat_S(Q) + C/2 ) + sqrt( C/2 ) )^2.

And `C/2 = (KL + log(2 sqrt(n)/delta))/(2n)` — the same `kl_term` shape as the classic bound, with the factor `2n` now explained: the `n` is from pb-kl, the `2` is the `2` that came out of `(L - Lhat)^2 <= 2 L C`. I check the algebra by expanding both forms. The compact form gives `(sqrt(Lhat + C/2) + sqrt(C/2))^2 = Lhat + C + 2 sqrt((Lhat + C/2)(C/2)) = Lhat + C + sqrt(C(2Lhat + C))`. The larger-root form gives `(sqrt(2C) + sqrt(2C+4Lhat))^2/4 = Lhat + C + sqrt(2C(2C+4Lhat))/2 = Lhat + C + sqrt(C(2Lhat + C))`. Same expression, so the two derivations agree.

Algebraic agreement only tells me I simplified consistently; it doesn't tell me I solved the *right* equation. The real test is whether the value I get actually saturates (★): plug `L` back into `(L - Lhat)^2/(2L)` and see if it returns `C`. Let me evaluate the compact form and that residual at a few `(Lhat, C)`. At `(0, 0.001)`: `L = 0.002000`, and `(0.002 - 0)^2/(2·0.002) = 0.000004/0.004 = 0.001000 = C`. At `(0.02, 0.005)`: `L = 0.040000`, and `(0.04 - 0.02)^2/(2·0.04) = 0.0004/0.08 = 0.005000 = C`. At `(0.1, 0.2)`: `L = 0.582843`, and `(0.582843 - 0.1)^2/(2·0.582843) = 0.233137/1.165685 = 0.200000 = C`. The residual hits `C` to all printed digits in every case, and the three forms (compact, larger-root, `Lhat + C + sqrt(C(2Lhat+C))`) all return the identical `L`. So the quadratic was solved correctly: this `L` is the exact value at which refined-Pinsker-relaxed pb-kl holds with equality.

The crucial thing about this form: `L(Q)` is gone from the right side. It is an explicit upper bound on `L(Q)` in terms of `Lhat_S(Q)`, `KL`, `n`, `delta` — exactly what I need to minimize by gradient descent. So solving the refined-Pinsker relaxation as a quadratic in `sqrt(L)` simultaneously (i) used the inequality that's tight in my small-loss regime and (ii) eliminated the implicit `L`. The bound is `(sqrt(Lhat + kl_term) + sqrt(kl_term))^2` with `kl_term = (KL + log(2 sqrt(n)/delta))/(2n)`.

Now I want to confirm it actually buys me what the wall was about — tightness at small loss — by comparing it head to head with the classic bound. At `Lhat = 0`: classic gives `sqrt(kl_term)`; this quadratic bound gives `(sqrt(kl_term) + sqrt(kl_term))^2 = (2 sqrt(kl_term))^2 = 4 kl_term`. So at zero empirical loss the comparison is `4 kl_term` versus `sqrt(kl_term)`. Tabulating both across the `kl_term` values a trained network actually reaches:

  kl_term=1e-4:  f_quad=4 kl_term=0.00040   f_classic=sqrt(kl_term)=0.01000   ratio 25.0x
  kl_term=1e-3:  f_quad=0.00400            f_classic=0.03162                 ratio  7.9x
  kl_term=1e-2:  f_quad=0.04000            f_classic=0.10000                 ratio  2.5x

So the quadratic certificate is `7.9×` tighter at `kl_term = 1e-3` and `25×` tighter at `1e-4`, and — this is the diagnostic that matters — the advantage *grows* as `kl_term` shrinks. That growing gap is the signature of the rate difference: `f_quad`'s complexity contribution is `O(kl_term)`, linear, while `f_classic`'s is `O(sqrt(kl_term))`, a square root. The ratio is `sqrt(kl_term)/(4 kl_term) = 1/(4 sqrt(kl_term))`, which blows up as `kl_term -> 0` — and `25 = 1/(4·sqrt(1e-4)) = 1/(4·0.01)` checks against the table exactly. This is the fast (`1/n`, realizable-case) rate instead of the slow `1/sqrt(n)` one, and it is the direct payoff of the refined Pinsker inequality being sharp in the small-`p` regime. I should be honest about the boundary: the refined lower bound only beats the standard one while the true-risk argument `p = L(Q)` stays below `1/4` (the crossover I tabulated above); once the true risk crosses `1/4`, the standard Pinsker relaxation is the tighter one and this bound would actually be the looser choice. That is fine here — my regime is small loss — but it means the bound is not universally better, only better where it counts.

So as a training objective I take this bound and plug in a differentiable surrogate for the loss. The 0-1 loss is hopeless for SGD — piecewise constant, gradient zero almost everywhere. The standard surrogate is cross-entropy, `-log sigma(u)_y` with `sigma` the softmax, and it's justified as a surrogate because it upper-bounds the mistake probability: using `log x <= x - 1`, if I draw a label `Y ~ sigma(u)` then `P(Y != y) = 1 - sigma(u)_y <= -log sigma(u)_y`. Fine. But there's a snag: every PAC-Bayes inequality I used assumed a loss in `[0,1]`, and cross-entropy is unbounded above — as `sigma(u)_y -> 0` it goes to infinity. If I feed an unbounded loss into a `[0,1]`-loss bound, the bound is simply false; the formula and the objective are mis-calibrated. I have to bound the loss. The clean fix is to floor the predicted probability: replace `sigma(u)_y` with `max(sigma(u)_y, pmin)` for a small `pmin > 0`. Then the bounded cross-entropy `-log max(sigma(u)_y, pmin)` lives in `[0, log(1/pmin)]`. To get it into `[0,1]` I rescale by `1/log(1/pmin)`. So my surrogate empirical loss is

  Lhat_ce(Q) = ( 1 / log(1/pmin) ) * E_{W~Q}[ mean over data of -log max(sigma(h_W(x))_y, pmin) ].

This rescaling is not cosmetic — it's load-bearing. The factor `1/log(1/pmin)` is exactly what maps the surrogate into the range the `(sqrt(...) + sqrt(...))^2` formula assumes; without it the empirical-risk term and the kl_term are on incompatible scales, the formula no longer upper-bounds anything, the optimizer chases a wrong objective, and in practice the posterior drifts far from the prior, blowing up `KL` and loosening the very certificate I'm trying to tighten. So `pmin` and its `1/log(1/pmin)` scale are part of the method, not a detail. Concretely I clamp the log-probabilities at `log(pmin)` before the NLL and multiply the result by `_loss_scale = 1/log(1/pmin)`.

Now the optimization machinery. I want a distribution `Q` over weights, and I want to backprop through `E_{W~Q}` of the objective. I parametrize `Q` as a diagonal Gaussian per weight: `W = mu + sigma odot V` with `V` standard Gaussian noise drawn fresh each step, and `sigma = log(1 + exp(rho))` so `sigma` is always positive under an unconstrained `rho` (softplus). Drawing `V` and differentiating `objective(mu + sigma odot V)` w.r.t. `(mu, rho)` is the pathwise (reparameterization) gradient — unbiased and low-variance, far better than a score-function estimator, and the gradients for `mu` and `rho` are just the ordinary backprop gradients of the network, shifted and scaled by the reparameterization. The KL between two diagonal Gaussians has a closed form I can differentiate: per coordinate, `KL = (1/2)(log(b0/b1) + (mu1 - mu0)^2/b0 + b1/b0 - 1)` with `b = sigma^2`, summed over coordinates by independence (and a similar closed form if I use Laplace instead). So `net.compute_kl()` is exact and differentiable, and a training step is: stochastic forward pass `net(data, sample=True, clamping=True, pmin=pmin)`, form the bounded-CE empirical surrogate, read off the KL, plug both into the bound formula, and return that scalar for SGD to descend.

One numerical guard on the formula itself. The quantity under each square root must be non-negative. `kl_term` is a KL plus a log, divided by a positive number — it's non-negative as long as `2 sqrt(n)/delta >= 1`, which always holds, but a tiny negative from floating point would crash `sqrt`. And `Lhat + kl_term` is a sum of non-negatives. So I clamp both at zero before the square roots, purely defensively. That keeps `compute_bound` returning `(sqrt(clamp(Lhat + kl_term)) + sqrt(clamp(kl_term)))^2`.

The KL term dominates this bound — it's the lever for tightness, since `Lhat` is already near zero. So I want `KL(Q || Q0)` small, which means I want the prior `Q0` centered near where the posterior ends up. A random-init prior is centered far from a good solution, so `KL` is large. The fix is to learn the prior from data — but pb-kl requires `Q0` to be data-free *with respect to the sample the bound is evaluated on*. So I split the training set: use part of it to train a prior mean by plain ERM on the surrogate loss (with dropout during prior-learning only, to keep the prior from overfitting), and the disjoint remainder to learn the posterior and evaluate the certificate. On the remainder the prior is genuinely data-free, so pb-kl is valid. I initialize the posterior at the prior (both `mu` and `rho`), so the posterior starts at the ERM minimizer inside a broad region and only has to move a little — keeping `KL` small from the outset. Keeping the two subsets disjoint is what lets me avoid the heavier differential-privacy corrections that re-using the same data for prior and bound would force.

Now the final certificate — the number I actually report — which is *not* the training objective. For the headline guarantee I don't want the quadratic relaxation; I want to invert the tightest bound I have, pb-kl itself, on the 0-1 loss. I need the kl-inversion explicitly: define `f*(q, c) = sup{ p in [q,1] : kl(q || p) <= c }`, the largest true risk consistent with empirical risk `q` and complexity budget `c`. It's well-defined because, for fixed `q`, `kl(q || p)` is strictly increasing in `p` on `[q, 1]` (the derivative `(p - q)/(p(1-p))` is positive there), so `{ p : kl(q || p) <= c }` is an interval `[q, p*]` and `f* = p*`. Inverting pb-kl on the 0-1 loss gives `L(Q) <= f*( Lhat_S(Q), c )` with `c = (KL + log(2 sqrt(n)/delta))/n` — note `/n`, not `/2n`, because this is the bare pb-kl; the `2` only appeared in the quadratic *relaxation*, never in the inversion. The obstacle is that `Lhat_S(Q) = E_{W~Q}[Lhat_S(W)]` is an expectation over `Q`, not computable. So I Monte-Carlo it: draw `m` fresh weight samples `W_1, ..., W_m ~ Q` and average the 0-1 empirical loss to get `Lhat_S(Qhat_m)`. That MC estimate is itself random, and the principled way to keep the guarantee valid is to charge its own sampling error — the estimate is an average of `m` i.i.d. `[0,1]` losses, so a binary-kl (Chernoff) tail gives `kl( Lhat_S(Qhat_m) || Lhat_S(Q) ) <= log(2/delta')/m` with probability `1 - delta'`, which inverts to `Lhat_S(Q) <= f*( Lhat_S(Qhat_m), log(2/delta')/m )` — an inner inversion before the outer one. The reported 0-1 risk certificate is therefore `inv_kl(empirical_risk_01, c)` only after `empirical_risk_01` has already been replaced by the inner inversion of the Monte-Carlo estimate. The same two-level correction applies to the bounded-CE empirical risk. The quadratic relaxation drives training; the sharp nested pb-kl inversion delivers the reported risk certificate.

I need `f*` numerically, and there's no closed form. Monotonicity hands me bisection: bracket `p` between `q` and `1 - epsilon`, halve, move the bracket according to the sign of `c - kl(q || p)`, stop when the relative width is tiny. No derivatives, and monotonicity guarantees convergence. I handle the `q = 0` and `q = 1` edges where one of the two kl terms is `0 * log` and should be dropped.

Let me trace the routine once on a concrete input to be sure it returns what I think it does, since the whole certificate hangs on it. Take an empirical risk `q = 0.01` and a budget `c = 0.05` — roughly the scale of a small posterior on the bound set. Running the bisection to its `1e-5` relative-width tolerance returns `p = 0.078098`. The test of correctness isn't the procedure, it's the output: does `kl(0.01 || 0.078098)` actually equal the budget `0.05`? Computing it, `0.01·log(0.01/0.078098) + 0.99·log(0.99/0.921902) = 0.050000`. It lands on the budget to all printed digits, and `0.078 > 0.01 = q`, so it returned the *upper* root, the largest true risk consistent with the budget — exactly the `sup` in the definition of `f*`. The certificate inflates the empirical `1%` to a guaranteed `7.8%` at this budget, which is the right direction and a sane magnitude. So the inverter is doing its job.

Let me also keep a cross-entropy-style training-bound number for diagnostics: take the inner-corrected bounded-CE empirical risk and pass it through the quadratic `compute_bound`. That's a secondary metric; the headline is the 0-1 risk certificate from the nested pb-kl inversion.

The empty slot is now concrete: `compute_bound` is the quadratic formula, `train_step` is the differentiable objective, `compute_risk_certificate` is the final nested-kl evaluation:

```python
import math
import torch
import torch.nn.functional as F


class BoundObjective:
    """PAC-Bayes-quadratic (fquad) objective and risk certificate."""

    def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
                 pmin=1e-5, delta=0.025, delta_test=0.01,
                 mc_samples=1000, kl_penalty=1.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prior_sigma = prior_sigma
        self.pmin = pmin
        self.delta = delta
        self.delta_test = delta_test
        self.mc_samples = mc_samples
        self.kl_penalty = kl_penalty
        self._loss_scale = 1.0 / math.log(1.0 / pmin)

    def compute_empirical_risk(self, outputs, targets, bounded=True):
        empirical_risk = F.nll_loss(outputs, targets)
        if bounded:
            empirical_risk = empirical_risk * self._loss_scale
        return empirical_risk

    def compute_losses(self, net, data, target, clamping=True):
        outputs = net(data, sample=True, clamping=clamping, pmin=self.pmin)
        loss_ce = self.compute_empirical_risk(outputs, target, clamping)
        pred = outputs.max(1, keepdim=True)[1]
        loss_01 = 1.0 - pred.eq(target.view_as(pred)).sum().item() / target.size(0)
        return loss_ce, loss_01, outputs

    def compute_bound(self, empirical_risk, kl, n, delta):
        kl = kl * self.kl_penalty
        kl_term = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * n)
        inner = torch.clamp(empirical_risk + kl_term, min=0.0)
        kl_term_clamped = torch.clamp(kl_term, min=0.0)
        return (torch.sqrt(inner) + torch.sqrt(kl_term_clamped)) ** 2

    def train_step(self, net, data, target, n_posterior, clamping=True):
        kl = net.compute_kl()
        loss_ce, loss_01, outputs = self.compute_losses(net, data, target, clamping)
        train_obj = self.compute_bound(loss_ce, kl, n_posterior, self.delta)
        return train_obj, kl / n_posterior, outputs, loss_ce, loss_01

    def mc_sampling(self, net, input=None, target=None, data_loader=None,
                    device="cuda", clamping=True):
        error, cross_entropy = 0.0, 0.0
        if data_loader is not None:
            batches = 0
            for data_batch, target_batch in data_loader:
                data_batch = data_batch.to(device)
                target_batch = target_batch.to(device)
                ce_mc, err_mc = 0.0, 0.0
                for _ in range(self.mc_samples):
                    loss_ce, loss_01, _ = self.compute_losses(
                        net, data_batch, target_batch, clamping
                    )
                    ce_mc += loss_ce
                    err_mc += loss_01
                cross_entropy += ce_mc / self.mc_samples
                error += err_mc / self.mc_samples
                batches += 1
            return cross_entropy / batches, error / batches

        ce_mc, err_mc = 0.0, 0.0
        for _ in range(self.mc_samples):
            loss_ce, loss_01, _ = self.compute_losses(net, input, target, clamping)
            ce_mc += loss_ce
            err_mc += loss_01
        return ce_mc / self.mc_samples, err_mc / self.mc_samples

    def compute_risk_certificate(self, net, n_posterior, n_bound, input=None,
                                 target=None, data_loader=None, device="cuda",
                                 clamping=True):
        kl = net.compute_kl()
        error_ce, error_01 = self.mc_sampling(
            net, input, target, data_loader, device, clamping
        )

        mc_c = math.log(2.0 / self.delta_test) / self.mc_samples
        empirical_risk_ce = inv_kl(float(error_ce.item()), mc_c)
        empirical_risk_01 = inv_kl(float(error_01), mc_c)

        train_bound = self.compute_bound(
            torch.tensor(empirical_risk_ce, device=kl.device),
            kl,
            n_posterior,
            self.delta,
        )

        # Outer certificate budget is the PAC-Bayes-kl budget: divide by n, not 2n.
        c = (kl.item() + math.log(2.0 * math.sqrt(n_bound) / self.delta)) / n_bound
        risk_ce = inv_kl(empirical_risk_ce, c)
        risk_01 = inv_kl(empirical_risk_01, c)
        return (
            train_bound.item(),
            kl.item() / n_bound,
            empirical_risk_ce,
            empirical_risk_01,
            risk_ce,
            risk_01,
        )


def inv_kl(qs, ks):
    izq, dch = qs, 1 - 1e-10
    qd = 0
    while (dch - izq) / dch >= 1e-5:
        p = (izq + dch) * 0.5
        if qs == 0:
            ikl = ks - ((1 - qs) * math.log((1 - qs) / (1 - p)))
        elif qs == 1:
            ikl = ks - (qs * math.log(qs / p))
        else:
            ikl = ks - (
                qs * math.log(qs / p)
                + (1 - qs) * math.log((1 - qs) / (1 - p))
            )
        if ikl < 0:
            dch = p
        else:
            izq = p
        qd = p
    return qd
```

The causal chain I end with is this. Over-parameterized networks generalize but every capacity-based certificate for them is vacuous, because capacity dwarfs the sample. PAC-Bayes escapes that by certifying a randomized predictor and charging complexity as `KL(Q || Q0)`, which is small when the posterior stays near the prior regardless of parameter count. The tightest classical statement, pb-kl, controls `kl(Lhat || L)` — I re-derived it from a change of measure, the sharp `2 sqrt(n)` moment bound, Markov, and Jensen — but it bounds a binary kl, not `L`, so it isn't directly optimizable. Relaxing it with the standard Pinsker inequality gives the classic additive bound `Lhat + sqrt(kl_term)`, whose complexity enters as a square root that doesn't shrink as `Lhat -> 0`, so the certificate stalls loose exactly in the small-loss regime a trained network reaches. The diagnosis: standard Pinsker is the looser lower bound on `kl` when the true-risk argument is below `1/4`; the refined Pinsker `kl(qh || p) >= (p - qh)^2/(2p)` is the tighter lower bound there. Relaxing pb-kl with the refined inequality gives `L <= Lhat + sqrt(2 L C)`, which has `L` on both sides — but it's a quadratic in `sqrt(L)`, and solving it yields the explicit, optimizable `(sqrt(Lhat + kl_term) + sqrt(kl_term))^2` with `kl_term = (KL + log(2 sqrt(n)/delta))/(2n)`. At zero empirical loss this is `4 kl_term`, linear in the complexity, versus the classic `sqrt(kl_term)` — the fast rate, a factor `1/(4 sqrt(kl_term))` tighter (`7.9×` at `kl_term=1e-3`, `25×` at `1e-4` by the table I worked out), valid in the below-`1/4` risk regime. To use it I bound and rescale the cross-entropy surrogate into `[0,1]` via the `pmin` floor and the `1/log(1/pmin)` factor, learn a diagonal-Gaussian posterior with the reparameterization gradient and closed-form KL, keep the KL small with a data-dependent prior trained on a disjoint split, and report the headline 0-1 certificate by first upper-bounding the Monte-Carlo empirical risk with an inner `inv_kl`, then applying the outer pb-kl inversion with the `/n` certificate budget. The quadratic relaxation drives training; the nested sharp inversion delivers the number.
