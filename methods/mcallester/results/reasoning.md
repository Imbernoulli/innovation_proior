Let me start from what I actually have and what I actually want. I have a learning algorithm that, after seeing `n` i.i.d. examples `S`, hands me back not a single classifier but a distribution `Q` over hypotheses — to predict, I draw `h ~ Q` and apply it. I can compute the empirical risk of this randomized predictor, `E_{h~Q}[r(h)]`, where `r(h) = (1/n) Σ_i l(h, z_i)` is the training error of a draw. What I want is a guarantee on the thing I can't see, the true risk `E_{h~Q}[R(h)]` with `R(h) = E_{z~D}[l(h, z)]`, and I want it as a number I can print: with probability at least `1 − delta` over the draw of `S`, the true risk is at most something computable. The catch that kills the naive approach is that `Q` is chosen *after* and *because of* `S`, so any concentration statement I prove for a fixed hypothesis doesn't transfer to the one the algorithm actually picked.

So let me see exactly how far the fixed-hypothesis statement gets me, because I want to feel the wall. For a single fixed `h`, `r(h)` is the mean of `n` i.i.d. `[0,1]` variables with mean `R(h)`. Hoeffding hands me `P_S( R(h) − r(h) > t ) ≤ e^{−2 n t²}`, so setting the right side to `delta` gives `R(h) ≤ r(h) + sqrt( log(1/delta) / (2n) )` with probability `1 − delta`. Clean — but it's about *one* `h` fixed before the data. The learner reads `S` and then chooses; the `h` it returns is a random function of `S`, and this inequality says nothing about a data-dependent choice. The standard patch for a finite hypothesis class is to make the statement hold for *all* `h` at once by a union bound: pay `log|H|`, and get `R(h) ≤ r(h) + sqrt( (log|H| + log(1/delta)) / (2n) )` uniformly. Now it covers whatever the learner picks. But the price is the size of the *whole* space. For an infinite `H` — anything continuously parameterized, certainly a neural network — `log|H|` is unbounded and the bound degenerates to "true error ≤ 1," which is no statement at all. The complexity I'm being charged is the complexity of the *space*, not of the *solution*. That's the thing I have to escape: I need a penalty that knows how much the algorithm actually committed to, not how big the menu was.

Here's the reframing that might break it. Don't certify a single `h` at all — certify the *distribution* `Q`. Fix some reference distribution `P` over `H` before I see the data (call it the prior, though there's no Bayesian likelihood in sight — it's just a yardstick fixed in advance). Let the learner return any `S`-dependent `Q` it likes. The object I bound is the `Q`-average of the risks. Now what's the complexity term? It should measure how far `Q` had to move from the reference `P`, because *that* is what "the algorithm committed to specific hypotheses" means quantitatively. The natural measure of that move is the relative entropy `KL(Q || P) = E_{h~Q}[ log (dQ/dP) ]`: zero when `Q = P` (I learned nothing, committed to nothing), growing as `Q` concentrates somewhere `P` was diffuse, and `+∞` if `Q` ventures where `P` has no mass. If I can get a bound whose penalty is `KL(Q || P)` instead of `log|H|`, then a `Q` near `P` is cheap no matter how huge `H` is, and only a `Q` that genuinely fled the prior to fit the data pays a lot. That's exactly the locality I want.

But how do I *transport* a per-hypothesis concentration statement, which I can only prove for a fixed `h`, into a statement about the data-dependent average over `Q`? This is the crux, and I keep circling it. I have, for each fixed `h`, control of some exponential moment under the data — and I want to swap the integration over `h` from the prior `P` (which I can handle, because `P` is fixed) to the posterior `Q` (which I can't directly handle, because `Q` depends on `S`). The device that does precisely this swap is the change-of-measure inequality. Let me write it down and convince myself. For any reference `P`, any `Q` absolutely continuous w.r.t. `P`, and any bounded `phi: H → R`,

  E_{h~Q}[ phi(h) ]  ≤  KL(Q || P)  +  log E_{h~P}[ e^{phi(h)} ].

Why is this true, and why is `KL` exactly the right currency? Start from the Gibbs measure `dP_phi/dP (h) = e^{phi(h)} / E_{h'~P}[e^{phi(h')}]` and just expand `KL(Q || P_phi)`, which I know is `≥ 0`:

  KL(Q || P_phi) = E_{h~Q}[ log (dQ/dP_phi) ] = E_{h~Q}[ log (dQ/dP) − phi(h) + log E_{h'~P}[e^{phi(h')}] ]
                 = KL(Q || P) − E_{h~Q}[phi(h)] + log E_{h~P}[ e^{phi(h)} ].

Nonnegativity of the left side rearranges to exactly the inequality, with equality when `Q = P_phi`, i.e. when the posterior is the Gibbs tilt of the prior by `phi`. So the change-of-measure inequality isn't a trick I have to memorize — it's just `KL ≥ 0` applied to the gap between `Q` and the tilted prior, and `KL(Q || P)` appears because that's literally what you pick up when you re-express `dQ/dP_phi` in terms of `dQ/dP`. The shape it gives me is the one I wanted: the left side is a `Q`-average of whatever quantity I plug into `phi`, the right side has a `KL(Q || P)` complexity term plus a moment computed *under the fixed prior `P`*, which is the part I can integrate against the data.

So now I need to choose `phi(h)` to be the per-hypothesis discrepancy whose exponential moment under the data I can control, build the moment under `P`, and turn the whole thing into a probability statement with Markov. Let me try the most obvious `phi` first and see where it lands, because I suspect it gives a known but not-yet-tight bound. Take `phi(h) = lambda (R(h) − r(h))` for some fixed `lambda > 0` — the raw gap between true and empirical risk, scaled. Hoeffding in its exponential form says, for a single fixed `h`, `E_S[ e^{lambda (R(h) − r(h)) } ] ≤ e^{lambda² C² / (8n)}` where `C = 1` is the loss range (this is the sub-Gaussian moment of a bounded-variable average — the `lambda²/(8n)` is the Hoeffding lemma constant). Now I want the moment under `P`. Take expectation over `S` of the change-of-measure inequality with this `phi`, but I have to be careful: the inequality holds pointwise in `S` for the `S`-dependent `Q`, so

  E_{h~Q}[ lambda (R(h) − r(h)) ]  ≤  KL(Q || P)  +  log E_{h~P}[ e^{lambda (R(h) − r(h))} ].

The annoying part is that the log of an expectation is concave, so I can't just push `E_S` through. Cleaner to work with the exponential directly. Consider `E_S E_{h~P}[ e^{lambda(R(h)−r(h))} ]`; by Tonelli I swap the order to `E_{h~P} E_S[ e^{lambda(R(h)−r(h))} ] ≤ E_{h~P}[ e^{lambda²C²/(8n)} ] = e^{lambda²C²/(8n)}`, since the Hoeffding bound is uniform in `h`. So the prior-averaged moment is at most `e^{lambda²C²/(8n)}`, a constant. Then the standard route: the change-of-measure step actually gives me `E_S[ e^{ sup_Q ( lambda E_{h~Q}[R−r] − KL(Q||P) ) } ] = E_S E_{h~P}[ e^{lambda(R−r)} ] ≤ e^{lambda²C²/(8n)}` — that's the variational form of change of measure, the supremum over `Q` is attained at the Gibbs tilt and equals the prior moment. Now Markov on this single nonnegative random variable: `P_S( sup_Q [ lambda E_Q[R−r] − KL(Q||P) ] > log(M/delta) ) ≤ delta`, with `M = e^{lambda²C²/(8n)}`. Rearranging, with probability `1 − delta`, simultaneously for all `Q`,

  E_{h~Q}[R(h)]  ≤  E_{h~Q}[r(h)]  +  lambda C² / (8n)  +  ( KL(Q || P) + log(1/delta) ) / lambda.

There it is — the change of measure plus Hoeffding plus Markov gives me a uniform-over-`Q` bound with a `KL` complexity term instead of `log|H|`. I escaped the union bound. And notice the supremum being attained at the Gibbs measure tells me the *optimal* posterior for fixed `lambda` is `dQ ∝ e^{−lambda r} dP` — minimizing the right side over `Q` is exactly the Donsker–Varadhan variational problem, so the bound even names the best `Q` for me.

But stare at this bound and I'm not happy. The empirical risk enters *linearly* with a coefficient `1`, and the penalty is `g(lambda) = lambda/(8n) + (KL + log(1/delta))/lambda` (taking `C = 1`). Let me actually minimize it instead of eyeballing the rate: `g'(lambda) = 1/(8n) − (KL + log(1/delta))/lambda² = 0` gives `lambda* = sqrt( 8n (KL + log(1/delta)) )`, and plugging back, the two terms become equal — each is `sqrt((KL + log(1/delta))/(8n))` — so `g(lambda*) = 2 sqrt((KL + log(1/delta))/(8n)) = sqrt((KL + log(1/delta))/(2n))`. So the optimized penalty is `Θ(1/sqrt(n))`, with no dependence on the empirical risk at all — it's the same whether `E_Q[r]` is `0.4` or `0`. That's the wall here: in the regime I most care about for these over-parameterized nets — where the predictor fits the training data almost perfectly, `E_Q[r] ≈ 0` — I *should* be able to do better than `1/sqrt(n)`. A realizable problem has a fast `1/n` rate classically; this linear bound can't see it. The reason is structural: by collapsing the comparison between empirical and true risk into the *raw difference* `R − r` and bounding it sub-Gaussianly, I threw away the fact that for small risks the relevant fluctuation scale is much smaller (a near-zero-mean Bernoulli has tiny variance). I need a discrepancy measure between `r` and `R` that is naturally sharp near `0`, not a symmetric quadratic gap. And `lambda` is a free knob I have to guess or grid-search, and the grid itself costs a `log n`. Two reasons to look for something better.

The discrepancy that is sharp near the boundary is the binary relative entropy itself. For two Bernoulli parameters `p, q ∈ [0,1]`, define `kl(p || q) = p log(p/q) + (1−p) log((1−p)/(1−q))`. This is asymmetric and steep: near `q = 0` it behaves like `p log(p/q)`, exploding, which is exactly the sensitivity I want when the true risk is small. So instead of `phi(h) = lambda(R − r)`, let me try to control the exponential moment of `n · kl(r(h) || R(h))` per hypothesis, and run it through the same change-of-measure machinery. The reason this is the right object: `kl(r || R)` *is* the large-deviation rate for the empirical mean `r` of `n` Bernoulli`(R)` variables, so `e^{n kl(r || R)}` is precisely the quantity whose expectation Sanov/Chernoff-type arguments keep finite. Let me see if I can make a general statement that covers both this `phi` and the linear one, so I only do the change-of-measure-and-Markov dance once.

Let `D: [0,1]² → R` be any convex function, and set `phi(h) = n · D(r(h), R(h))`. The same three steps go through verbatim. Define the prior-averaged moment `M(P) = E_S E_{h~P}[ e^{n D(r(h), R(h))} ]` (Tonelli to swap, so I can also write it `E_{h~P} E_S[...]`). The variational form of change of measure gives `E_S[ e^{ sup_Q ( n E_Q[D(r,R)] − KL(Q||P) ) } ] = M(P)`. Multiply through by `delta / M(P)` and apply Markov: with probability `1 − delta`, `sup_Q [ n E_Q[D(r,R)] − KL(Q||P) − log(M(P)/delta) ] ≤ 0`, i.e. for all `Q`,

  E_{h~Q}[ D(r(h), R(h)) ]  ≤  ( KL(Q || P) + log( M(P)/delta ) ) / n.

And now the one extra move that buys everything: `D` is *convex*, so by Jensen
`D( E_{h~Q}[r(h)], E_{h~Q}[R(h)] ) ≤ E_{h~Q}[ D(r(h), R(h)) ]`. Therefore

  D( E_{h~Q}[r(h)], E_{h~Q}[R(h)] )  ≤  ( KL(Q || P) + log( M(P)/delta ) ) / n.

This is the master bound. It says: for *any* convex discrepancy `D`, the discrepancy between the `Q`-averaged empirical risk and the `Q`-averaged true risk is at most `(KL + log(M(P)/delta))/n`, where the only `D`-specific quantity is the prior moment `M(P)`. The linear bound from before is the special case `D(x,y) = lambda(y − x)/n`-ish; the bound I actually want is `D = kl`. Everything now reduces to one question: **how big is `M(P)` when `D = kl`?**

So I need `M(P) = E_S E_{h~P}[ e^{n kl(r(h) || R(h))} ]`. By Tonelli that's `E_{h~P}` of `E_S[ e^{n kl(r(h) || R(h))} ]`. The inner expectation can vary with the fixed hypothesis through the distribution of its loss, but what I need is a uniform upper bound that depends only on `n`; once I have that, the prior average is bounded by the same scalar. So the entire constant in the certificate comes down to this worst-case concentration fact: for a fixed `h`, with `X_i = l(h, z_i) ∈ [0,1]` i.i.d., mean `mu = R(h)`, and `M(X) = (1/n) Σ X_i = r(h)`, how large can

  E[ e^{n kl( M(X) || mu )} ]  ?

Let me actually compute it, because the constant is the whole game — getting `2n` versus `2 sqrt(n)` here directly halves the additive log term in the final bound. Define `f(x) = exp( n kl( (1/n) Σ x_i, mu ) )` on `[0,1]^n`. First, is `f` convex? The inner map `x ↦ (1/n)Σx_i` is linear; `kl(·, mu)` is convex in its first argument (relative entropy is convex); the exponential is nondecreasing and convex; composition of a convex-increasing with a convex is convex, and convex-precomposed-with-linear stays convex. So `f` is convex, and it's clearly symmetric under permuting the `x_i`. Convexity plus the fact that `[0,1]^n` is the convex hull of its `{0,1}^n` corners means I can push the expectation to the *Bernoulli* case and only increase it. Concretely, any point `x` is a convex combination of corners `eta ∈ {0,1}^n` with weights `Π_{i: eta_i=0}(1−x_i) Π_{i: eta_i=1} x_i`, so `f(x) ≤ Σ_eta (those weights) f(eta)`; taking expectations and using independence with `E[X_i] = mu`,

  E[ f(X) ]  ≤  Σ_{k=0}^{n} C(n,k) (1−mu)^{n−k} mu^k f(theta(k)),

where `theta(k)` is any corner with exactly `k` ones (using symmetry of `f`). This is the convexity lemma — it reduces the worst case over all `[0,1]`-valued losses to Bernoulli losses, which is precisely why I can pretend the loss is `0/1` for the purpose of the constant. Now plug in the explicit `f` at a corner: at `theta(k)` the empirical mean is `k/n`, so `f(theta(k)) = exp( n kl(k/n, mu) ) = ( (n−k)/(n(1−mu)) )^{n−k} ( k/(n mu) )^k`. Substituting,

  E[ f(X) ]  ≤  Σ_{k=0}^{n} C(n,k) (1−mu)^{n−k} mu^k · ( (n−k)/(n(1−mu)) )^{n−k} ( k/(n mu) )^k.

The `(1−mu)^{n−k}` cancels against the `(1−mu)^{−(n−k)}` and `mu^k` against `mu^{−k}` — the `mu`-dependence vanishes entirely, which already tells me something: for Bernoulli losses this moment depends only on `n`, not on the true risk. What survives is

  E[ f(X) ]  ≤  Σ_{k=0}^{n} C(n,k) (k/n)^k ((n−k)/n)^{n−k}.

The `k=0` and `k=n` terms each contribute `1` (with the convention `0^0 = 1`), so peel them off as `+2` and write the middle as `(n!/n^n) Σ_{k=1}^{n−1} [k^k / k!] [(n−k)^{n−k}/(n−k)!]`. Now Stirling: `m! = sqrt(2 pi m) (m/e)^m e^{g(m)/12m}` with `0 < g < 1`, so `m^m/m! = e^m / ( sqrt(2 pi m) e^{g/12m} ) ≤ e^m / sqrt(2 pi m)`, and `n!/n^n ≤ sqrt(2 pi n) e^{−n} e^{1/12n}`. Multiply the three Stirling factors — the `e^k`, `e^{n−k}`, `e^{−n}` cancel exactly, and I'm left with

  E[ f(X) ]  ≤  e^{1/12n} sqrt( n / (2 pi) ) Σ_{k=1}^{n−1} 1/ sqrt( k (n−k) ) + 2.

The remaining sum `c_n = Σ_{k=1}^{n−1} 1/sqrt(k(n−k))` is a Riemann sum for `∫_0^1 dt / sqrt(t(1−t)) = pi` (substitute `t = cos²θ`). I want a clean uniform bound `c_n ≤ pi`, and I should check that the Riemann-sum direction goes the right way: `1/sqrt(t(1−t))` is convex (it's a product of two convex-ish reciprocal-square-roots, and a direct second-derivative check on `[0,1]` confirms it stays positive away from the endpoints) with minimum `2` at `t = 1/2`, so the rectangle rule *under*-approximates the integral, giving `c_n ≤ pi`. The lower side `c_n ≥ 1` is loose but enough: the single central-ish term already exceeds `1`. So

  E[ e^{n kl(M(X) || mu)} ]  ≤  e^{1/12n} sqrt( pi n / 2 ) + 2.

I don't want to take "≤ 2 sqrt(n)" on faith — the constant is the whole point of this detour, so let me actually compare the two sides at the boundary case and at a larger `n`. At `n = 8`: `sqrt(pi·8/2) = sqrt(4 pi) ≈ 3.545`, times `e^{1/96} ≈ 1.0105` gives `≈ 3.582`, plus `2` is `≈ 5.58`; meanwhile `2 sqrt(8) ≈ 5.657`. Just under, by about `0.07`. The worry is that for small `n` the additive `+2` might dominate the `sqrt(n)` growth and the inequality could flip somewhere. So I check `n = 100` too: `sqrt(pi·100/2) = sqrt(50 pi) ≈ 12.533`, times `e^{1/1200} ≈ 1.0008` is `≈ 12.544`, plus `2` is `≈ 14.54`, against `2 sqrt(100) = 20` — now the slack is huge. The reason the gap widens rather than closes: my bound grows like `sqrt(pi/2) sqrt(n) ≈ 1.2533 sqrt(n)` plus a fixed `+2`, while the target grows like `2 sqrt(n)`; since `1.2533 < 2`, the `sqrt(n)` terms alone open a gap of `0.747 sqrt(n)`, which already beats the constant `+2` once `sqrt(n) > 2.68`, i.e. `n > 7.2` — exactly why `n = 8` is the first integer where it holds and why it never flips back. So I can state the clean form

  E[ e^{n kl(r(h) || R(h))} ]  ≤  2 sqrt(n),    for n ≥ 8.

This is the constant. A cruder route that drops the `sqrt` and carries `2n` instead is available, and it's worth knowing how much I'd lose by being lazy: the constant enters the certificate only through `log(2 sqrt(n)/delta)` versus `log(2n/delta)`, a difference of `½ log n` in the additive term, so for the `n ≈ 30000` bound sets I have in mind that's about `½ log 30000 ≈ 5.2` nats saved — not nothing once divided by `n`, but the real reason to keep the `sqrt` is that it's free now that I've done the Stirling bookkeeping.

Before I commit to `sqrt(n)` as the *order*, I should rule out that I've been sloppy in the upper direction and the true moment is actually `O(1)` or `O(log n)`, which would mean my `2 sqrt(n)` is throwing away a real improvement. So let me bound the same binomial sum from *below*. Applying Stirling the other way, `m^m/m! ≥ e^m/(sqrt(2 pi m) e^{1/12m}) ≥ e^m/(sqrt(2 pi m) e^{1/6})` for `m ≥ 1`, and `n!/n^n ≥ sqrt(2 pi n) e^{−n}`, so the middle sum is at least `e^{−1/6} sqrt(n/(2 pi)) c_n`; adding back the two endpoint `1`'s gives the binomial sum `≥ e^{−1/6} sqrt(n/(2 pi)) c_n + 2`. Here I have to keep the *actual* `c_n`, not relax it to `c_n ≥ 1` — I tried that first and it's too lossy: at `n = 100`, `e^{−1/6} sqrt(100/(2 pi)) + 2 ≈ 5.38`, which sits *below* `sqrt(100) = 10`, so the crude `c_n ≥ 1` version fails to even establish the `sqrt(n)` lower bound. Keeping `c_n` itself (`c_n ≈ 2.10` at `n = 8`, `≈ 2.85` at `n = 100`, `→ pi`), the lower bound is `≈ 4.01` at `n = 8` against `sqrt(8) ≈ 2.83`, and `≈ 11.62` at `n = 100` against `10` — both clear `sqrt(n)`. Asymptotically the leading coefficient is `e^{−1/6}/sqrt(2 pi) · c_n → 0.337·pi ≈ 1.06 > 1`, so the moment really does grow like `sqrt(n)` (times a constant just above `1`), confirming it from below. So the moment is sandwiched between `sqrt(n)` and `2 sqrt(n)`: the `sqrt(n)` factor is genuinely in the answer, not slack I introduced, and no amount of tightening this route removes it. `M(P) = 2 sqrt(n)` it is.

Feed that back into the master bound with `D = kl`:

  kl( E_{h~Q}[r(h)] || E_{h~Q}[R(h)] )  ≤  ( KL(Q || P) + log( 2 sqrt(n) / delta ) ) / n.

This is the tightest certificate on this route — the relative-entropy form. It controls the binary KL between the empirical and true risks of the randomized predictor, directly, with the sharp `2 sqrt(n)` constant. Compare to the linear bound: there the rate was `1/sqrt(n)` no matter what; here `kl(p || q)` blows up as `q` leaves `p` and is asymmetric near `0`, so I'd expect the certificate to get fast behavior when the empirical risk is near zero. Let me confirm that on the realizable extreme rather than assert it. If `E_Q[r] = 0`, then `kl(0 || q) = −log(1−q)`, which is `≥ q` for all `q ∈ [0,1)` (since `−log(1−q) = q + q²/2 + q³/3 + … ≥ q`; e.g. `−log(0.9) = 0.1054 ≥ 0.1`), so the constraint `kl(0 || E_Q[R]) ≤ c` with `c = (KL + log(2 sqrt(n)/delta))/n` forces `E_Q[R] ≤ −log(1 − E_Q[R]) ≤ c`, i.e. `E_Q[R] ≤ c`, which is `O(1/n)` rather than `O(1/sqrt(n))`. Put numbers on it to feel the size of the speedup: take `n = 30000`, `delta = 0.025`, and a modest `KL = 100`. Then `c = (100 + log(2 sqrt(30000)/0.025))/30000 = (100 + log(13856))/30000 = (100 + 9.54)/30000 ≈ 0.00365`, so the `kl`-form certifies `E_Q[R] ≲ 0.0037`. The Pinsker/linear form at the same numbers would give `sqrt(c/2) = sqrt(0.00365/2) ≈ 0.043` — an order of magnitude worse. So in the near-realizable regime the `kl`-form is the one that produces a tight number; keeping the comparison in `kl`-form rather than in raw differences is what buys it.

But the relative-entropy form has a usability problem: it's *implicit* in the true risk. The certificate is "`E_Q[R]` is at most the upper end of the `kl`-ball of radius `c = (KL + log(2 sqrt(n)/delta))/n` around `E_Q[r]`," i.e. `E_Q[R] ≤ sup{ p ≥ E_Q[r] : kl(E_Q[r] || p) ≤ c }`. To *report* a number I'd invert `kl` numerically — fine, and I'll do exactly that for the final certificate. But to *train* against it, I want something differentiable in `Q` and additive, a closed expression I can backprop through. So I need an explicit upper bound on the `kl`-inverse.

The simplest explicit relaxation is Pinsker's inequality. Pinsker says `kl(p || q) ≥ 2 (p − q)²`. So if `kl(E_Q[r] || E_Q[R]) ≤ c`, then `2 (E_Q[R] − E_Q[r])² ≤ c`, i.e. `(E_Q[R] − E_Q[r])² ≤ c/2`, i.e.

  E_{h~Q}[R(h)]  ≤  E_{h~Q}[r(h)]  +  sqrt( c / 2 )  =  E_{h~Q}[r(h)]  +  sqrt( ( KL(Q || P) + log( 2 sqrt(n) / delta ) ) / (2n) ).

Let me make sure the `2n` in the denominator is right and I haven't dropped a factor. `c = (KL + log(2 sqrt(n)/delta))/n`. Then `sqrt(c/2) = sqrt( (KL + log(2 sqrt(n)/delta)) / (2n) )`. Yes — the `2` from Pinsker's `2(p−q)²` lands in the denominator inside the square root, turning the `/n` of the kl-form into `/(2n)` of the square-root form. This is the classic additive PAC-Bayes certificate:

  E_{h~Q}[R(h)]  ≤  E_{h~Q}[r(h)]  +  sqrt( ( KL(Q || P) + log( 2 sqrt(n) / delta ) ) / (2n) ).

It's exactly what I wanted for optimization: closed, additive, and differentiable in `Q` through `E_Q[r]` and `KL(Q || P)`. The price for that closed form is the Pinsker relaxation — I went from the sharp `kl`-ball to its quadratic under-estimate, so this bound is strictly looser than the `kl`-form, and in particular it shows the `1/sqrt(n)` rate rather than the `1/n` it came from. I'll pay that price for *training* but not for *reporting*: I'll train against this differentiable surrogate, then report the tighter `kl`-inverted number. (A momentary temptation: there's a refined Pinsker `kl(p || q) ≥ (p − q)²/(2q)` valid for `p < q`, which is sharper when `q < 1/4` and keeps an `E_Q[R]` under the root — but that puts the unknown true risk on the right-hand side, so it's a quadratic *in the thing I'm solving for*, not directly usable as an additive training objective. That's a different bound; here I want the plain additive one, so plain Pinsker is the right relaxation.)

Now I have to turn three abstractions into code, and each hides a real obstacle. First, the loss inside all of this must be in `[0,1]` — every step, from Hoeffding through the convexity lemma to the `2 sqrt(n)` moment, assumed `l(h, z) ∈ [0,1]`. The 0-1 classification loss is bounded, good, but it's *piecewise constant in the weights*, so its gradient is zero almost everywhere and SGD can't move. The standard surrogate is cross-entropy, `−log p_y` where `p_y` is the softmax probability of the true label. For a label drawn from the network softmax, `P(Y ≠ y) = 1 − p_y ≤ −log p_y` by `log x ≤ x − 1`, so this is a usable differentiable upper-bound surrogate for stochastic classification loss. But cross-entropy is *unbounded* — as `p_y → 0` it diverges, violating the `[0,1]` assumption the whole derivation rests on. I fix that by flooring the probability at a small `pmin`: replace `p_y` with `max(p_y, pmin)`, so the loss is at most `log(1/pmin)`, then rescale by `1/log(1/pmin)` to land back in `[0,1]`. In code I clamp `log p_y` at `log(pmin)` before the NLL and then divide the NLL by `log(1/pmin)`. Both parts matter: the clamp makes the loss finite, and the rescale is what makes the PAC-Bayes statement apply with range one and keeps the empirical-risk/KL tradeoff calibrated.

Second, `KL(Q || P)`. The posterior over each weight is a Gaussian `N(mu_q, sigma_q²)` and the prior is `N(mu_p, sigma_p²)`; for one coordinate the relative entropy is `½ [ log(sigma_p²/sigma_q²) + (sigma_q² + (mu_q − mu_p)²)/sigma_p² − 1 ]`, and because the weights are independent across coordinates the total is the sum over all weights. That total is what dominates the certificate, so it had better be small — which is the whole point of building `P` data-dependently: train a deterministic model by ERM on a held-out half of the data, center `P` (and initialize `Q`) at those weights, evaluate the bound only on the *other* half (the `n` examples `P` never saw, so `P` is still data-free with respect to them), and then `Q` starts at `P` with `KL = 0` and only grows as fitting demands. The harness already provides this — a per-layer analytic `_kl` and a `get_total_kl` that sums them — so I just call it.

Third, the empirical risk in the *reported* certificate. `E_{h~Q}[r(h)]` is itself not computable in closed form — it's an average over the posterior I can only sample. So I estimate it by Monte Carlo: draw `m` weight vectors `h_1, ..., h_m ~ Q` and average each sampled classifier's empirical 0-1 error on the bound set. This must be the Gibbs risk average, not a majority vote over the sampled classifiers, because the theorem certifies the randomized predictor. But a Monte Carlo estimate still is not the true `E_Q[r]`; it has its own sampling error, and a valid certificate must account for it. The same `kl`-concentration applies one level down: with probability `1 − delta'` over the `m` draws, `kl( emphat_m || E_Q[r] ) ≤ log(2/delta') / m`, so I `kl`-invert that to get a high-probability upper bound on the true empirical risk before I plug it into the outer certificate. The fully honest certificate is therefore a *nested* inversion — invert the inner `kl` to bound `E_Q[r]` from the MC estimate, then invert the outer `kl` with `c = (KL + log(2 sqrt(n)/delta))/n` to bound `E_Q[R]` — and it holds with probability `1 − delta − delta'`. For the reported 0-1 risk certificate I use exactly that nested inversion.

Let me write it so it drops straight into the three empty methods of the harness. The combining rule first — that's the certificate `E_Q[r] + sqrt((KL + log(2 sqrt(n)/delta))/(2n))`, used both as the training objective's functional form and as the cross-entropy bound I report alongside:

```python
import math
import torch
import torch.nn.functional as F


class BoundOptimizer:
    """PAC-Bayes certificate: train against the additive square-root bound
    (Pinsker relaxation of the kl-form), report via nested binary-kl inversion."""

    def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
                 pmin=1e-5):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prior_sigma = prior_sigma
        self.pmin = pmin

    def compute_bound(self, empirical_risk, kl, n, delta):
        # E_Q[r] + sqrt( (KL + log(2*sqrt(n)/delta)) / (2n) )
        # the 2*sqrt(n) is the exponential-moment constant; the /(2n) is Pinsker.
        kl_term = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * n)
        return empirical_risk + torch.sqrt(kl_term)

    def train_step(self, model, data, target, device, n_bound, delta):
        # 0-1 loss has no gradient -> bounded cross-entropy surrogate.
        output = model(data, sample=True)              # draw h ~ Q, forward
        log_probs = F.log_softmax(output, dim=1)
        log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
        nll = F.nll_loss(log_probs, target) / math.log(1.0 / self.pmin)
        kl = get_total_kl(model)                        # sum of per-coordinate Gaussian KL(Q||P)
        return self.compute_bound(nll, kl, n_bound, delta)   # minimize the certificate itself

    def compute_risk_certificate(self, model, bound_loader, device, delta=0.025,
                                 mc_samples=1000):
        model.eval()
        n_bound = len(bound_loader.dataset)
        delta_mc = 0.01

        # empirical Gibbs risks by Monte Carlo over h ~ Q, not majority vote
        total_01, total_nll, total_samples = 0.0, 0.0, 0
        with torch.no_grad():
            for data, target in bound_loader:
                data, target = data.to(device), target.to(device)
                for _ in range(mc_samples):
                    output = model(data, sample=True)
                    pred = output.argmax(dim=1)
                    total_01 += (pred != target).sum().item()
                    log_probs = torch.clamp(F.log_softmax(output, dim=1),
                                            min=math.log(self.pmin))
                    batch_nll = F.nll_loss(log_probs, target, reduction="sum")
                    total_nll += (batch_nll / math.log(1.0 / self.pmin)).item()
                total_samples += target.size(0)
        emp_risk_01_mc = total_01 / (total_samples * mc_samples)
        emp_nll_mc = total_nll / (total_samples * mc_samples)

        # inner MC correction: bound true E_Q empirical risk from m posterior samples
        c_mc = math.log(2.0 / delta_mc) / mc_samples
        emp_risk_01 = inv_kl(emp_risk_01_mc, c_mc)
        emp_nll = inv_kl(emp_nll_mc, c_mc)

        # KL(Q||P) from one stochastic pass
        with torch.no_grad():
            dummy = next(iter(bound_loader))[0][:1].to(device)
            model(dummy, sample=True)
            kl = get_total_kl(model).item()

        # outer PAC-Bayes-kl inversion on the MC-corrected empirical 0-1 risk
        c = (kl + math.log(2.0 * math.sqrt(n_bound) / delta)) / n_bound
        risk_cert_01 = inv_kl(emp_risk_01, c)

        # also report the additive square-root bound on the CE risk
        ce_bound = self.compute_bound(torch.tensor(emp_nll),
                                      torch.tensor(kl), n_bound, delta).item()

        metrics = {
            "empirical_01_risk_mc": emp_risk_01_mc,
            "empirical_01_risk": emp_risk_01,
            "empirical_nll_mc": emp_nll_mc,
            "empirical_nll": emp_nll,
            "kl_divergence": kl,
            "ce_bound": ce_bound,
            "delta_mc": delta_mc,
        }
        return risk_cert_01, metrics
```

The causal chain, start to finish: I needed a non-vacuous, computable, uniform-over-`Q` certificate for a randomized classifier, and the union-bound penalty `log|H|` was hopeless for rich `H`. Replacing "cover all hypotheses" with "measure how far the chosen `Q` moved from a fixed reference `P`" made `KL(Q || P)` the natural complexity term, and the change-of-measure inequality — which is just `KL ≥ 0` against the Gibbs tilt — let me transport per-hypothesis concentration onto the data-dependent `Q`. Running Hoeffding through that gave a linear bound with a free `lambda`, but it was stuck at `1/sqrt(n)` and blind to the realizable regime, so I switched the per-hypothesis discrepancy to the binary relative entropy `kl(r || R)`, which is sharp near zero. A single convex-`D` master bound packaged the change-of-measure-and-Markov step once and reduced everything to one constant, the prior-averaged exponential moment `E[e^{n kl(r || R)}]`; bounding it via the convexity-to-Bernoulli reduction, the binomial sum, Stirling, and the `∫ dt/sqrt(t(1−t)) = pi` Riemann sum gives `2 sqrt(n)` for `n ≥ 8` (sandwiched below by `sqrt(n)`, so the order is forced). That delivered the tight `kl`-form certificate; Pinsker's `kl(p||q) ≥ 2(p−q)²` relaxed it into the closed, differentiable additive form `E_Q[r] + sqrt((KL + log(2 sqrt(n)/delta))/(2n))`, which I minimize during training with a `pmin`-floored and range-rescaled cross-entropy surrogate. For the final number I use the sharper binary-`kl` inversion on the MC-corrected Gibbs 0-1 empirical risk — inner inversion for posterior-sampling error, outer inversion for sample generalization — reporting the tightest valid certificate while training against the most tractable one.
