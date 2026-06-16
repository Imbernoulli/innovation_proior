Let me start from what actually goes wrong when I run the standard thing. I have a smooth nonconvex `F` I can only touch through a stochastic oracle `f(x, xi)` with `E[f(x,xi)] = F(x)`, and I want a critical point, `nabla F(x) = 0`. Plain SGD, `x_{t+1} = x_t - eta_t g_t` with `g_t = nabla f(x_t, xi_t)`, gets me `E[||nabla F(x_t)||] <= O(T^{-1/4})`, and the thing strangling that rate is gradient variance: even as `nabla F(x_t)` shrinks toward zero near a critical point, the sampled gradient keeps a floor `E[||g_t - nabla F(x_t)||^2] <= sigma^2` that does not shrink, so the iterate rattles and progress stalls. So the whole game is to manufacture, out of single stochastic gradients, a direction whose error is *smaller* than that floor.

The variance-reduction people already beat the floor, and I should understand exactly how, because their machinery is what I want to keep the spirit of and throw away the cost of. The lever every one of them pulls is the same, and it is just smoothness. If I evaluate the *same* sample `xi` at two nearby points `x` and `x'`, then `||nabla f(x, xi) - nabla f(x', xi)|| <= L||x - x'||`. That difference is a random quantity, but it is *small* whenever the two points are close, and — this is the subtle part — it is small *in variance*, not just in mean, because the randomness in `xi` largely cancels between the two terms (the same draw appears in both). So a control variate built from a two-point same-sample difference is a low-noise object precisely in the regime an optimizer that is converging lives in, where consecutive iterates are close.

SVRG uses this with a *fixed anchor*. Pick a snapshot `w_0`, pay for its full gradient `nabla F(w_0)` once with a whole pass over the data, then estimate `v_t = nabla f_i(w_t) - nabla f_i(w_0) + nabla F(w_0)`. Take conditional expectation over `i`: `E[v_t] = nabla F(w_t) - nabla F(w_0) + nabla F(w_0) = nabla F(w_t)`, so it is *unbiased*, and its variance is controlled by `||w_t - w_0||`, which is small while we stay near the snapshot. SARAH keeps the same skeleton but moves the anchor every step: `v_t = nabla f_i(w_t) - nabla f_i(w_{t-1}) + v_{t-1}`, re-anchoring at the *previous iterate* rather than a frozen snapshot, with `v_0` the full gradient. This one is *biased* — `E[v_t | F_t] = nabla F(w_t) - nabla F(w_{t-1}) + v_{t-1}`, which is not `nabla F(w_t)` — but the correction term now uses the genuinely tiny gap `||w_t - w_{t-1}||` between consecutive steps, so the per-step injected error is smaller, and the variance of `v_t` drives toward zero within an inner loop, which SVRG's does not. SPIDER and SNVRG run this biased recursive estimator with carefully sized mega-batches and hit the optimal nonconvex rate `O(T^{-1/3})`.

So why am I not done? Because all of them, SVRG, SARAH, SPIDER, share two costs I cannot stomach. The first is the checkpoint: they must periodically stop producing iterates and burn a fresh batch of `N` samples — `N` as large as `O(T)`, never smaller than `O(T^{2/3})` — to compute a full or near-full gradient at one anchor point, spending all those samples to learn about a *single* point while making zero progress. SARAH's `v_0` is exactly this; SPIDER's mega-batch is exactly this. The second is the learning rate: their step sizes are non-adaptive and have to be set by balancing `L`, `sigma`, the batch size, and the checkpoint frequency against each other, so you must know those constants or sweep them, and a mis-set schedule loses the guarantee. And empirically the whole family is reported to underperform plain SGD on real deep nets. I want the `O(T^{-1/3})` rate with none of this — no checkpoint, a step size that sets itself from observed gradients, and no need to know `sigma`.

Now the recursive estimator structure keeps nagging at me, because it looks so close to something cheap. Look at SARAH again: `v_t = nabla f_i(w_t) - nabla f_i(w_{t-1}) + v_{t-1}`. Rewrite it as `v_t = nabla f_i(w_t) + (v_{t-1} - nabla f_i(w_{t-1}))`. The only reason it needs a checkpoint is the initialization `v_0 = full gradient`, which seeds the recursion with a low-noise anchor and is what the whole inner loop leans on; the bias accumulates from there and is held in check by restarting at a fresh full gradient. What if I never restart, and never seed with a full gradient? Then `v_t` would be a recursion of single-sample objects with nothing low-noise anywhere in it, and the bias would just compound. I need something *in* the recursion itself that contracts the error rather than letting it pile up.

Let me look at the other heuristic on the table, the one nobody can explain in the noisy case: momentum. SGD with momentum keeps `d_t = (1-a) d_{t-1} + a nabla f(x_t, xi_t)`, an exponential moving average of past gradients, `a` small. People use it everywhere and it works, but the folklore result is that in the *stochastic* setting the noise nullifies its theoretical advantage — there is no general theorem that momentum improves the convergence *rate* over plain SGD with noise. So it is a heuristic with an unexplained practical success. Here is what strikes me staring at the two formulas side by side. SARAH's recursion `v_t = nabla f_i(w_t) + (v_{t-1} - nabla f_i(w_{t-1}))` *carries the old estimate forward* with a same-sample correction; momentum's recursion `d_t = (1-a) d_{t-1} + a nabla f(x_t, xi_t)` *carries the old estimate forward* with exponential weight. Both are "keep the running thing, fold in the new gradient." What if momentum's averaging is the contraction I need, and the SARAH correction is the bias-removal, and I should be running *both at once* rather than choosing?

So let me try to graft the SARAH correction onto the momentum recursion. Momentum alone is `d_t = (1-a) d_{t-1} + a nabla f(x_t, xi_t)`. The SARAH correction term is `nabla f(x_t, xi_t) - nabla f(x_{t-1}, xi_t)` — same sample, two consecutive points. I'll add a copy of it, weighted by `(1-a)` to match the part of `d` that is "old," and see what happens:

  `d_t = (1-a) d_{t-1} + a nabla f(x_t, xi_t) + (1-a)(nabla f(x_t, xi_t) - nabla f(x_{t-1}, xi_t))`,
  `x_{t+1} = x_t - eta d_t`.

The only difference from vanilla momentum is that last `(1-a)(nabla f(x_t,xi_t) - nabla f(x_{t-1},xi_t))` term. And notice immediately: if the algorithm is converging, `x_t ~ x_{t-1}`, that correction term goes to zero, and `d_t` collapses back to plain momentum. So whatever I'm building behaves like classic momentum SGD toward the end of optimization — it is not some exotic object, it is momentum with an extra correction that matters most early when steps are large. Let me also write the same thing in the form I'd actually run, collecting terms. Group the two `nabla f(x_t, xi_t)` pieces: `a g + (1-a) g = g` where `g = nabla f(x_t, xi_t)`, leaving `d_t = (1-a) d_{t-1} + g - (1-a) nabla f(x_{t-1}, xi_t) = nabla f(x_t, xi_t) + (1-a)(d_{t-1} - nabla f(x_{t-1}, xi_t))`. With a step shift this is `d_{t+1} = nabla f(x_{t+1}, xi_{t+1}) + (1-a)(d_t - nabla f(x_t, xi_{t+1}))`, which is *exactly* SARAH's recursion with the `(1-a)` weight on the carry-forward instead of a hard `1`. So SARAH is the `a = 0` corner of this, pure carry-forward; momentum is the corner where I drop the correction term. I've found the two-parameter object that contains both, and the interesting region is the interior, `a` small but nonzero.

Now I have to check whether this actually reduces variance, because "it looks like both ancestors" is not a proof. The right object to track is the error in the direction I'm stepping along. Define `epsilon_t := d_t - nabla F(x_t)`. This measures exactly how wrong my update direction is relative to the true (unknown) gradient. For plain SGD the analogous error is `g_t - nabla F(x_t)` with `E[||.||^2] <= sigma^2` — flat, no decay. If I can show `E[||epsilon_t||^2]` *decreases* over time, that *is* variance reduction. Subtract `nabla F(x_t)` from my update. Start from `d_t = nabla f(x_t, xi_t) + (1-a)(d_{t-1} - nabla f(x_{t-1}, xi_t))`:

  `epsilon_t = nabla f(x_t, xi_t) + (1-a)(d_{t-1} - nabla f(x_{t-1}, xi_t)) - nabla F(x_t)`.

I want `(1-a) d_{t-1}` to turn into `(1-a) epsilon_{t-1}`, so I add and subtract `(1-a) nabla F(x_{t-1})`, and I split the leading `nabla f(x_t, xi_t)` into a part that pairs with `a` and a part that pairs with `(1-a)`:

  `epsilon_t = (1-a) epsilon_{t-1} + a(nabla f(x_t, xi_t) - nabla F(x_t)) + (1-a)(nabla f(x_t, xi_t) - nabla f(x_{t-1}, xi_t) - (nabla F(x_t) - nabla F(x_{t-1})))`.

Let me read these three terms, because the whole method lives in them. The first, `(1-a) epsilon_{t-1}`, is a *contraction* of the previous error by factor `(1-a)` — this is the momentum averaging doing its job, shrinking accumulated error. The second, `a(nabla f(x_t, xi_t) - nabla F(x_t))`, is fresh single-sample noise, magnitude `~ sigma`, but multiplied by the small `a` — I can make this as small as I like by shrinking `a`. The third is the genuinely good one: `nabla f(x_t, xi_t) - nabla f(x_{t-1}, xi_t)` minus its mean `nabla F(x_t) - nabla F(x_{t-1})`. This is the same-sample two-point difference, centered, so by smoothness it is `O(||x_t - x_{t-1}||) = O(eta ||d_{t-1}||)` — small whenever the step is small. So heuristically `||epsilon_t|| ~ (1-a)||epsilon_{t-1}|| + Z`, where `Z` is small, and a contraction-plus-small-input recursion settles at the fixed point `||epsilon|| ~ Z/a`. So the error stabilizes at `Z/a`. The whole challenge is now sharp and quantitative: I have to choose `eta` and `a` to make `Z/a` tiny — drive the numerator `Z` (which scales with `a sigma` and with `eta`) down while keeping the denominator `a` from being so small that `Z/a` blows up. That tension between `Z` and `a` in `Z/a` is the entire design problem; everything else is making it precise.

Before I make it precise, the step size and the momentum coefficient. SVRG/SARAH set these by hand against `L` and `sigma`; I refuse to. The adaptive-learning-rate idea says: set the step from the gradients you've seen. AdaGrad uses `eta_t = k / (sum_{i<=t} g_i^2)^{1/2}`. Let me reach for the same shape but I am *not* going to assume the `1/2` power — the exponent should be whatever the convergence analysis demands, so I'll leave it free and pin it down by matching rates. Write `eta_t = k / (w + sum_{i=1}^t G_i^2)^{p}` with `G_t = ||nabla f(x_t, xi_t)||` the gradient norm at the sample, `w` a small offset to keep things finite at `t = 0`, and `p` to be determined. And I want the momentum coefficient to *also* be adaptive and to *decay*, because the heuristic `Z/a` analysis said `a` controls the contraction and the noise injection — early on I want `a` larger (strong fresh-noise suppression is less important than tracking), later smaller. Let me tie it to the step size: `a_{t+1} = c eta_t^2`. Why `eta^2` and not `eta`? Because in the error recursion the noise term enters as `a` times noise, but it will get squared in `E[||epsilon||^2]`, and — I'll see this drop out of the algebra in a moment — the term that wants to be controlled is `a^2 = c^2 eta^4`, and pairing `a` with `eta^2` is what makes the noise contribution scale like `eta^4`, i.e. summable. Let me carry `p` and the `eta^2` coupling and let the analysis confirm them.

Let me get the exponents by the equilibrium heuristic, sharpened. The gradient-norm-squared `G_t^2 = ||nabla f(x_t,xi_t)||^2` does *not* go to zero even at a critical point, because of the variance floor — `E[G_t^2] >= sigma^2 > 0` roughly — so `sum_{i<=t} G_i^2 ~ Theta(t)`, and `eta_t ~ k / t^p`, and `a_t = c eta_{t-1}^2 ~ t^{-2p}`. Now sharpen the error recursion. From the three terms, `E[||epsilon_t||^2]` picks up: from the noise term, `~ a_t^2 sigma^2 = eta^4 ~ t^{-4p}`; from the contraction, `(1-a_t)^2 E[||epsilon_{t-1}||^2]`; from the smoothness term, `~ eta_{t-1}^2 ||d_{t-1}||^2 ~ eta^2(||epsilon||^2 + ||nabla F||^2)`. Set `E[||epsilon_t||^2] = E[||epsilon_{t-1}||^2]` (equilibrium), use `(1-a_t) ~ 1 - a_t`, and the `a_t` from the contraction balances the inputs: `a_t ||epsilon||^2 ~ t^{-4p} + t^{-2p}||nabla F||^2`, so `||epsilon||^2 ~ t^{-4p}/t^{-2p} + ||nabla F||^2 = t^{-2p} + ||nabla F||^2`. The error floor is `t^{-2p}`. Now the descent side will trade `eta_t ||nabla F||^2` against `eta_t ||epsilon||^2`, and the iterate stops making progress when `||nabla F||^2` drops to the error floor `t^{-2p}`. So `||nabla F||^2 ~ T^{-2p}`, `||nabla F|| ~ T^{-p}`. I want the optimal `T^{-1/3}`, so `p = 1/3`. There it is: the step-size exponent must be `1/3`, the *cube* root of accumulated squared gradients, not AdaGrad's square root. AdaGrad's `1/2` would give `T^{-1/2}` on the easy/low-noise part but cannot deliver the variance-reduced `T^{-1/3}` in the noisy regime; the `1/3` is exactly tuned to the equilibrium. And with `p = 1/3`, `a_t ~ t^{-2/3}` — decaying, as I wanted. So:

  `eta_t = k / (w + sum_{i=1}^t G_i^2)^{1/3}`,  `a_{t+1} = c eta_t^2`,
  `d_{t+1} = nabla f(x_{t+1}, xi_{t+1}) + (1 - a_{t+1})(d_t - nabla f(x_t, xi_{t+1}))`,  `x_{t+1} = x_t - eta_t d_t`,

seeded by a single sample, `d_1 = nabla f(x_1, xi_1)`, `eta_0 = k / w^{1/3}`. No checkpoint anywhere — the recursion's only seed is one stochastic gradient, and the contraction `(1-a)` plus the smoothness correction do the variance reduction that SARAH bought with a full-gradient `v_0`.

Now I have to actually prove the rate, because the equilibrium hand-wave is not a theorem, and I need to nail the constants `k`, `c`, `w`. The standard Lyapunov function for smooth nonconvex SGD is just `Phi_t = F(x_t)` — show it decreases on average. That cannot work for me, because my direction `d_t` is *biased*, so a descent lemma on `F` alone leaves an uncontrolled `||epsilon_t||^2` term. I need the potential to also carry the error. So take `Phi_t = F(x_t) + z_t ||epsilon_t||^2` for some weight `z_t`. The error term in the potential is what lets the analysis "pay for" the bias. And the weight should be *time-varying* — I'll discover the exact `z_t` by what makes the telescoping close, but I can already see why constant `z_t` would fail: with constant weight the `||epsilon_t||^2` ledger doesn't track the shrinking step size, and you end up needing a low-noise anchor (a checkpoint) to start it. A growing `z_t` is precisely what removes the checkpoint.

Let me build the two pieces I need: a descent lemma for `F` with a biased direction, and a recursion for `E[||epsilon_t||^2]`.

First the descent lemma. By `L`-smoothness, `F(x_{t+1}) <= F(x_t) + nabla F(x_t) . (x_{t+1} - x_t) + (L/2)||x_{t+1} - x_t||^2 = F(x_t) - eta_t nabla F(x_t) . d_t + (L eta_t^2 / 2)||d_t||^2`. Now `d_t = nabla F(x_t) + epsilon_t`, so the inner product is `nabla F(x_t).d_t = ||nabla F(x_t)||^2 + nabla F(x_t).epsilon_t`. So

  `E[F(x_{t+1})] <= E[F(x_t) - eta_t ||nabla F(x_t)||^2 - eta_t nabla F(x_t).epsilon_t + (L eta_t^2/2)||d_t||^2]`.

The cross term `-eta_t nabla F.epsilon_t` I tame with Young's inequality `|a.b| <= ||a||^2/2 + ||b||^2/2`: `-nabla F.epsilon_t <= (1/2)||nabla F||^2 + (1/2)||epsilon_t||^2`, costing me half a `||nabla F||^2` and giving half a `||epsilon||^2`. And `||d_t||^2 = ||nabla F + epsilon_t||^2 <= 2||nabla F||^2 + 2||epsilon_t||^2`. Substitute:

  `<= E[F(x_t) - eta_t||nabla F||^2 + (eta_t/2)||nabla F||^2 + (eta_t/2)||epsilon_t||^2 + L eta_t^2||epsilon_t||^2 + L eta_t^2||nabla F||^2]`.

Collect. The `||nabla F||^2` coefficient is `-eta_t + eta_t/2 + L eta_t^2 = -eta_t/2 + L eta_t^2`; if `eta_t <= 1/(4L)` then `L eta_t^2 <= eta_t/4`, so the coefficient is `<= -eta_t/4`. The `||epsilon||^2` coefficient is `eta_t/2 + L eta_t^2 <= eta_t/2 + eta_t/4 = 3 eta_t/4`. So

  `E[F(x_{t+1}) - F(x_t)] <= E[-(eta_t/4)||nabla F(x_t)||^2 + (3 eta_t/4)||epsilon_t||^2]`,   provided `eta_t <= 1/(4L)`.

That `3/4` versus `1/4` asymmetry is the price of the bias: I only get a quarter of the gradient-norm decrease that unbiased SGD would, and I owe three quarters of an error term. So I had better make `||epsilon_t||^2` small, which is the point of the whole construction.

Now the error recursion. I want to bound `E[||epsilon_t||^2 / eta_{t-1}]` (the `1/eta_{t-1}` weighting is foreshadowing the `z_t`). Start from `epsilon_t = a_t(nabla f(x_t,xi_t) - nabla F(x_t)) + (1-a_t)(nabla f(x_t,xi_t) - nabla f(x_{t-1},xi_t) - (nabla F(x_t) - nabla F(x_{t-1}))) + (1-a_t) epsilon_{t-1}`. Call the three summands `P`, `Q`, `R`. The key fact that makes this tractable: `P` and `Q` both have *conditional mean zero* given `xi_1, ..., xi_{t-1}`, because they are centered functions of the fresh sample `xi_t`, while `epsilon_{t-1}` and `x_t` are determined by `xi_1,...,xi_{t-1}` and so are independent of `xi_t`. So when I expand `||epsilon_t||^2 = ||P||^2 + ||Q||^2 + ||R||^2 + 2 P.Q + 2 P.R + 2 Q.R` and take expectations, the cross terms `2 P.R` and `2 Q.R` vanish: `E[P . (1-a_t) epsilon_{t-1}] = E[E[P | xi_{1:t-1}] . (1-a_t) epsilon_{t-1}] = 0`, and the same for `Q`. (This is exactly the "unbiasedness substitute" — `d_t` is biased, but the *increment's* noise pieces are conditionally centered, which is all the squared expansion needs.) The `P.Q` cross term I don't get for free, so I fold it in with `||P+Q||^2 <= 2||P||^2 + 2||Q||^2`. Concretely, group the centered pieces and bound:

  `E[||epsilon_t||^2] <= E[2||P||^2 + 2||Q||^2 + (1-a_t)^2 ||epsilon_{t-1}||^2]`.

Take `P` first: `||P||^2 = a_t^2 ||nabla f(x_t,xi_t) - nabla F(x_t)||^2`. I want to convert `||nabla f - nabla F||^2` into something I can read off. Conditioning on `xi_{1:t-1}`, `E[||nabla f(x_t,xi_t) - nabla F(x_t)||^2] = E[||nabla f||^2] - ||nabla F(x_t)||^2 <= E[||nabla f(x_t,xi_t)||^2] = E[G_t^2]` (variance equals second moment minus squared mean, drop the subtracted square). And `a_t = c eta_{t-1}^2`, so `2||P||^2 = 2 a_t^2 ||nabla f - nabla F||^2`, and `2 a_t^2 = 2 c^2 eta_{t-1}^4`. Carrying the `1/eta_{t-1}` weight I'm after, `2||P||^2 / eta_{t-1} = 2 c^2 eta_{t-1}^3 G_t^2` in expectation. That is the first term, and the `eta^4 = eta^3 . eta` against the `1/eta` weight gives the `eta^3` I expected for the noise contribution.

Now `Q`: `(1-a_t)^2 ||nabla f(x_t,xi_t) - nabla f(x_{t-1},xi_t) - (nabla F(x_t) - nabla F(x_{t-1}))||^2`. Same conditional-variance trick — the centered version has expectation `<=` the uncentered `(1-a_t)^2 ||nabla f(x_t,xi_t) - nabla f(x_{t-1},xi_t)||^2`. And *here* smoothness bites: `||nabla f(x_t,xi_t) - nabla f(x_{t-1},xi_t)||^2 <= L^2 ||x_t - x_{t-1}||^2`, the same-sample two-point difference. But `x_t - x_{t-1} = -eta_{t-1} d_{t-1}`, so `||x_t - x_{t-1}||^2 = eta_{t-1}^2 ||d_{t-1}||^2`, and with the `1/eta_{t-1}` weight this is `2 (1-a_t)^2 L^2 eta_{t-1} ||d_{t-1}||^2`. Finally `||d_{t-1}||^2 = ||epsilon_{t-1} + nabla F(x_{t-1})||^2 <= 2||epsilon_{t-1}||^2 + 2||nabla F(x_{t-1})||^2`. Putting `P`, `Q`, `R` together with the `1/eta_{t-1}` weight:

  `E[||epsilon_t||^2 / eta_{t-1}] <= E[2 c^2 eta_{t-1}^3 G_t^2 + (1-a_t)^2 (1 + 4 L^2 eta_{t-1}^2)||epsilon_{t-1}||^2 / eta_{t-1} + 4(1-a_t)^2 L^2 eta_{t-1} ||nabla F(x_{t-1})||^2]`.

Let me sanity-read this against the heuristic. Drop constants, use `(1-a_t)^2 <= 1-a_t`, recall `sum G^2 ~ t` so `eta_t ~ t^{-1/3}`, `a_t ~ t^{-2/3}`: multiplying through by `eta_{t-1}`, the first term is `~ eta^4 ~ t^{-4/3}`, the middle is `(1 - t^{-2/3})||epsilon_{t-1}||^2` (the `4L^2 eta^2` is lower order), and the last is `~ eta^2 ||nabla F||^2 ~ t^{-2/3}||nabla F||^2`. So `E[||epsilon_t||^2] ~ t^{-4/3} + (1-t^{-2/3})||epsilon_{t-1}||^2 + t^{-2/3}||nabla F(x_{t-1})||^2`. Set the two consecutive `epsilon` terms equal and solve: `t^{-2/3}||epsilon||^2 ~ t^{-4/3} + t^{-2/3}||nabla F||^2`, so `||epsilon||^2 ~ t^{-2/3} + ||nabla F||^2`. Good, the exact lemma reproduces the heuristic: the estimator error falls to the same scale as the squared gradient plus the `t^{-2/3}` floor, so the method should stop improving only when `||nabla F|| ~ T^{-1/3}`.

Now I build the potential and force the constants. Take `Phi_t = F(x_t) + z_t ||epsilon_t||^2` with `z_t = 1/(32 L^2 eta_{t-1})`. The `1/eta_{t-1}` weighting is exactly why I carried `||epsilon||^2/eta_{t-1}` in the lemma; the `1/(32L^2)` constant I'll see pay for the smoothness terms. Let me bound the error part of `Phi_{t+1} - Phi_t`, i.e. `z_t (||epsilon_{t+1}||^2/?) ...` — carefully, the error weight at step `t+1` is `1/(32L^2 eta_t)`, so I need `(1/(32L^2)) (||epsilon_{t+1}||^2/eta_t - ||epsilon_t||^2/eta_{t-1})`. Apply the lemma (shifted by one) to `||epsilon_{t+1}||^2/eta_t`:

  `E[||epsilon_{t+1}||^2/eta_t - ||epsilon_t||^2/eta_{t-1}] <= E[A_t + B_t + C_t]`,
  `A_t = 2 c^2 eta_t^3 G_{t+1}^2`,
  `B_t = (eta_t^{-1}(1-a_{t+1})(1 + 4 L^2 eta_t^2) - eta_{t-1}^{-1})||epsilon_t||^2`,
  `C_t = 4 L^2 eta_t ||nabla F(x_t)||^2`.

`A_t` is the noise term, `B_t` the contraction-minus-previous-weight term, `C_t` the gradient feed-in. I sum each over `t`.

`A_t` first. `sum_t 2 c^2 eta_t^3 G_{t+1}^2 = sum_t 2 k^3 c^2 G_{t+1}^2 / (w + sum_{i<=t} G_i^2)`, using `eta_t^3 = k^3/(w + sum_{i<=t}G_i^2)`. The denominator is missing the `G_{t+1}^2` term, but if `w >= 2 G^2 >= G^2 + G_{t+1}^2`, I can replace `w` by `G^2` and add `G_{t+1}^2` into the sum to get `<= sum_t 2 k^3 c^2 G_{t+1}^2/(G^2 + sum_{i<=t+1}G_i^2)`. Now this is exactly the shape `sum a_t/(a_0 + sum_{i<=t}a_i)` with `a_0 = G^2`, `a_t = G_t^2`, and by the standard log lemma (from concavity of `ln`: `ln(a_0 + sum_{i<=t}a_i) - ln(a_0 + sum_{i<t}a_i) >= a_t/(a_0 + sum_{i<=t}a_i)`, telescope), `sum_t a_t/(a_0 + sum_{i<=t}a_i) <= ln(1 + sum a_i/a_0) <= ln(T+2)` after bounding each `G_t^2/G^2 <= 1`. So `sum_t A_t <= 2 k^3 c^2 ln(T+2)`. The noise term is only logarithmic — that is the variance-reduction payoff in the math: a *summable* (log) noise contribution where SGD would have a linear-in-`T` one.

`B_t` is where `c` gets pinned. `B_t = (eta_t^{-1}(1-a_{t+1})(1+4L^2 eta_t^2) - eta_{t-1}^{-1})||epsilon_t||^2`. Expand: `eta_t^{-1}(1-a_{t+1})(1+4L^2 eta_t^2) <= eta_t^{-1}(1 + 4L^2 eta_t^2 - a_{t+1})` (dropping the cross product `a_{t+1} . 4L^2 eta_t^2 >= 0`), `= eta_t^{-1} + eta_t(4L^2) - eta_t^{-1}a_{t+1}`. With `a_{t+1} = c eta_t^2`, `eta_t^{-1}a_{t+1} = c eta_t`. So `B_t <= (eta_t^{-1} - eta_{t-1}^{-1} + eta_t(4L^2 - c))||epsilon_t||^2`. I need this to come out *negative* (so the error in the potential is being burned, not accumulated), and that is a race between the step-size-increment `eta_t^{-1} - eta_{t-1}^{-1}` (which is positive — the inverse step size grows) and the `-c eta_t` from the momentum coupling. So I must show `eta_t^{-1} - eta_{t-1}^{-1}` is itself `O(eta_t)` and then choose `c` big enough.

Bound the increment. `eta_t^{-1} - eta_{t-1}^{-1} = (1/k)[(w + sum_{i<=t}G_i^2)^{1/3} - (w + sum_{i<t}G_i^2)^{1/3}]`. The map `x -> x^{1/3}` is concave, so `(x+y)^{1/3} <= x^{1/3} + (y/3) x^{-2/3}`; with `x = w + sum_{i<t}G_i^2`, `y = G_t^2`, the increment is `<= G_t^2/(3k(w + sum_{i<t}G_i^2)^{2/3})`. Now I want to express the denominator in terms of `eta_t`, i.e. `(w + sum_{i<=t}G_i^2)`. Since `w >= 2G^2`, `w + sum_{i<t}G_i^2 = (w - G_t^2) + sum_{i<=t}G_i^2 >= (w - G^2) + sum_{i<=t}G_i^2 >= w/2 + sum_{i<=t}G_i^2 >= (1/2)(w + sum_{i<=t}G_i^2)`. So `(w + sum_{i<t}G_i^2)^{2/3} >= 2^{-2/3}(w + sum_{i<=t}G_i^2)^{2/3}`, and the increment `<= 2^{2/3} G_t^2/(3k(w+sum_{i<=t}G_i^2)^{2/3}) = (2^{2/3}/(3k)) G_t^2 . (eta_t/k)^2 = (2^{2/3}G_t^2/(3k^3)) eta_t^2`. Now `G_t^2 <= G^2`, and `eta_t <= 1/(4L)` (I'll guarantee that with `w`), so `eta_t^2 <= eta_t/(4L)`, giving `eta_t^{-1} - eta_{t-1}^{-1} <= (2^{2/3}G^2/(12 L k^3)) eta_t <= (G^2/(7 L k^3)) eta_t` (since `2^{2/3}/12 < 1/7`). So the increment is bounded by `(G^2/(7Lk^3)) eta_t`. Now choose `c = 28 L^2 + G^2/(7 L k^3)`. Then `eta_t(4L^2 - c) = eta_t(4L^2 - 28L^2 - G^2/(7Lk^3)) = -24 L^2 eta_t - (G^2/(7Lk^3)) eta_t`, and the `-(G^2/(7Lk^3))eta_t` exactly cancels the increment bound, leaving `B_t <= -24 L^2 eta_t ||epsilon_t||^2`. So `c` is forced into two pieces with two distinct jobs: the `28 L^2` produces the surplus `-24L^2 eta_t` that will dominate the descent lemma's `+3eta_t/4` error term, and the `G^2/(7Lk^3)` exactly eats the step-size-increment. Nothing arbitrary.

`C_t` just stays `sum_t 4 L^2 eta_t ||nabla F(x_t)||^2`.

Now assemble. Multiply the telescoped error-potential change by `1/(32L^2)`:

  `(1/(32L^2)) sum_{t=1}^T (||epsilon_{t+1}||^2/eta_t - ||epsilon_t||^2/eta_{t-1}) <= (1/(32L^2))[sum A_t + sum B_t + sum C_t]`
  `<= (k^3 c^2/(16 L^2)) ln(T+2) + sum_t[(eta_t/8)||nabla F(x_t)||^2 - (3 eta_t/4)||epsilon_t||^2]`,

where `(1/(32L^2))(2 k^3 c^2 ln(T+2)) = k^3 c^2/(16L^2) ln(T+2)`, `(1/(32L^2))(-24L^2 eta_t ||epsilon||^2) = -(3eta_t/4)||epsilon||^2`, and `(1/(32L^2))(4L^2 eta_t ||nabla F||^2) = (eta_t/8)||nabla F||^2`. Add the descent lemma `E[F(x_{t+1}) - F(x_t)] <= -(eta_t/4)||nabla F||^2 + (3eta_t/4)||epsilon_t||^2` to this, summed over `t`, and the full potential change telescopes:

  `E[Phi_{T+1} - Phi_1] = E[sum_t (F(x_{t+1}) - F(x_t)) + (1/(32L^2)) sum_t(||epsilon_{t+1}||^2/eta_t - ||epsilon_t||^2/eta_{t-1})]`
  `<= E[(k^3c^2/(16L^2)) ln(T+2) + sum_t(-(eta_t/4) + (eta_t/8))||nabla F(x_t)||^2 + sum_t((3eta_t/4) - (3eta_t/4))||epsilon_t||^2]`
  `= E[(k^3c^2/(16L^2)) ln(T+2) - sum_t (eta_t/8) ||nabla F(x_t)||^2]`.

The two `(3eta_t/4)||epsilon_t||^2` terms cancel exactly — the error the descent lemma *spends* is precisely the error the potential's `B_t` term *burns*. That cancellation is the whole reason for the `1/(32L^2 eta_{t-1})` weight in `z_t` and for the `28L^2` piece of `c`: I tuned them to make it land. Reorder, using `Phi_{T+1} >= F^*` (since `||epsilon||^2 >= 0` and `F >= F^*`):

  `E[sum_t eta_t ||nabla F(x_t)||^2] <= 8 E[Phi_1 - Phi_{T+1}] + (k^3 c^2/(2L^2)) ln(T+2) <= 8(F(x_1) - F^*) + 8 z_1 E[||epsilon_1||^2] + (k^3c^2/(2L^2))ln(T+2)`.

And `z_1 = 1/(32L^2 eta_0)`, `epsilon_1 = d_1 - nabla F(x_1) = nabla f(x_1,xi_1) - nabla F(x_1)`, so `E[||epsilon_1||^2] <= sigma^2`, and `8 z_1 sigma^2 = sigma^2/(4 L^2 eta_0) = w^{1/3} sigma^2/(4 L^2 k)` since `eta_0 = k/w^{1/3}`. So

  `E[sum_t eta_t ||nabla F(x_t)||^2] <= 8(F(x_1)-F^*) + w^{1/3}sigma^2/(4L^2k) + (k^3c^2/(2L^2))ln(T+2) =: kM`,

where I name the bracket `kM`, `M = (1/k)[8(F(x_1)-F^*) + w^{1/3}sigma^2/(4L^2k) + (k^3c^2/(2L^2))ln(T+2)]`. The single seed `epsilon_1` carries a `sigma^2`, not a full-gradient's near-zero variance — that is the checkpoint I was determined to avoid, replaced by one ordinary stochastic gradient, and the analysis swallows it because it is divided by `eta_0` and the rest of the run contracts it away.

Last step: turn `sum eta_t ||nabla F||^2` into `sum ||nabla F||^2`, which is what I actually want to bound. `eta_t` is decreasing, so `sum_t eta_t ||nabla F(x_t)||^2 >= eta_T sum_t ||nabla F(x_t)||^2`. But `eta_T` is itself random (it depends on the gradients seen), so I cannot just divide. Use Cauchy-Schwarz in the form `E[A^2]E[B^2] >= E[AB]^2` with `A = sqrt(eta_T sum_t ||nabla F(x_t)||^2)` and `B = sqrt(1/eta_T)`: then `E[1/eta_T] . E[eta_T sum_t ||nabla F||^2] >= E[sqrt(sum_t ||nabla F||^2)]^2`. So `E[sqrt(sum ||nabla F||^2)]^2 <= E[1/eta_T] . kM` — wait, more carefully, `E[sum eta_t ||nabla F||^2] >= E[eta_T sum ||nabla F||^2]`, and combining, `E[sqrt(sum ||nabla F||^2)]^2 <= E[1/eta_T] E[eta_T sum||nabla F||^2] <= E[1/eta_T] . kM`. Now `1/eta_T = (w + sum_{t<=T}G_t^2)^{1/3}/k`, so `E[1/eta_T] . kM = M E[(w + sum G_t^2)^{1/3}]`. Bound `G_t^2 = ||nabla F(x_t) + zeta_t||^2 <= 2||nabla F(x_t)||^2 + 2||zeta_t||^2` with `zeta_t = nabla f(x_t,xi_t) - nabla F(x_t)`, `E||zeta_t||^2 <= sigma^2`. Then by `(a+b)^{1/3} <= a^{1/3} + b^{1/3}` and concavity (Jensen, to push `E` inside the roots),

  `E[X]^2 <= M(w + 2T sigma^2)^{1/3} + 2^{1/3} M E[X]^{2/3}`,   where `X = sqrt(sum_t ||nabla F(x_t)||^2)`.

This is a self-referential inequality in `E[X]`. Either the first term dominates — `E[X]^2 <= 2 M(w + 2T sigma^2)^{1/3}` — or the second does — `E[X]^2 <= 2 . 2^{1/3} M E[X]^{2/3}`, i.e. `E[X]^{4/3} <= 2^{4/3}M`, `E[X] <= 2 M^{3/4}`. Taking both cases, `E[X] <= sqrt(2M)(w + 2T sigma^2)^{1/6} + 2 M^{3/4}`. Finally Cauchy-Schwarz once more: `(1/T) sum_t ||nabla F(x_t)|| <= (1/sqrt T) sqrt(sum_t ||nabla F||^2) = X/sqrt T`, so

  `E[(1/T) sum_t ||nabla F(x_t)||] <= (sqrt(2M)(w + 2T sigma^2)^{1/6} + 2 M^{3/4})/sqrt T <= (w^{1/6}sqrt(2M) + 2M^{3/4})/sqrt T + 2 sigma^{1/3}/T^{1/3}`,

splitting `(w + 2Tsigma^2)^{1/6} <= w^{1/6} + (2T)^{1/6}sigma^{1/3}` (using `(a+b)^{1/6} <= a^{1/6}+b^{1/6}` then absorbing `(2T)^{1/6}/sqrt T = 2^{1/6} T^{-1/3}` and constants into the `2`). That is the theorem: the randomly chosen iterate satisfies `E[||nabla F(x_hat)||] <= (w^{1/6}sqrt(2M) + 2M^{3/4})/sqrt T + 2 sigma^{1/3}/T^{1/3}` with `M = (8/k)(F(x_1)-F^*) + w^{1/3}sigma^2/(4L^2k^2) + (k^2c^2/(2L^2))ln(T+2)` (the `1/k` from `kM` distributes into each piece). When `sigma = 0` everything is `O(ln T/sqrt T)`; when `sigma != 0` the dominant term is `2 sigma^{1/3}/T^{1/3}`, the optimal nonconvex rate, and I never had to know `sigma` — it came out of the adaptive `eta_t`.

I should close the choice of `k` and `w`. Write `k = b G^{2/3}/L` for a free dial `b > 0`; this makes the units work (`eta_0 = k/w^{1/3}` has the dimension of an inverse Lipschitz constant) and lets me read the constants in `G` and `L`. Then `c = 28L^2 + G^2/(7Lk^3) = L^2(28 + 1/(7b^3))` after substituting `k`, and `w = max((4Lk)^3, 2G^2, (ck/(4L))^3) = G^2 max((4b)^3, 2, (28b + 1/(7b^2))^3/64)`. The three pieces of the `max` are the three constraints the proof used: `w >= (4Lk)^3` makes `eta_t <= 1/(4L)` (descent lemma); `w >= 2G^2` was the `A_t` and increment bound; `w >= (ck/(4L))^3` makes `a_{t+1} = c eta_t^2 <= ck/(4Lw^{1/3}) <= 1`, so the momentum coefficient stays a valid `(1-a) in [0,1]`. Let me also kill the worry that `G -> 0` blows up `M` (the `1/k = L/(bG^{2/3})` factor diverges): both `F(x_1)-F^* = O(G)` and `sigma = O(G)` shrink at least as fast as `1/k` grows, so `M` stays bounded — the bound degrades gracefully, not catastrophically, as the problem gets flat. And `L -> 0` sending `M -> infinity` is correct, not a bug: `L = 0` means all gradients are equal everywhere, so there is no critical point to find. `M` is morally an `O(log T)` hardness constant.

There is one assumption I leaned on that the SVRG/SARAH analyses do not: that each `f(x,xi)` is `G`-Lipschitz, `||nabla f|| <= G`. I used it to make the step size adaptive — `eta_t` is built from the observed `G_t = ||nabla f(x_t,xi_t)||`, and I needed `G_t <= G` and `w >= 2G^2` to control the increments. If I am willing to *give up* the `sigma`-adaptivity and instead be told `sigma`, I can drop the Lipschitz assumption entirely: replace every `G` and `G_t` in the schedule by `sigma`, making `eta_t = k/(w + sigma^2 t)^{1/3}` a *deterministic* schedule. The error recursion then uses `E[eta_{t-1}^3 ||nabla f - nabla F||^2] <= eta_{t-1}^3 sigma^2` directly instead of going through `G_t^2`, and because `eta_t` is now deterministic and independent of the gradients, the final Cauchy-Schwarz dance is unnecessary — I just divide `E[sum eta_t ||nabla F||^2] >= eta_T E[sum ||nabla F||^2]` by `T eta_T` to get `(1/T) E[sum ||nabla F||^2] <= M_det w^{1/3}/(kT) + M_det sigma^{2/3}/(k T^{2/3})`, where `M_det = 8(F(x_1)-F^*) + w^{1/3}sigma^2/(4L^2 k) + (k^3 c^2/2L^2)ln(T+2)`. Same `O(T^{-1/3})`-flavored guarantee, no Lipschitz bound, at the cost of needing `sigma`. Good to have both ends of the trade.

Now the implementation, because the clean update rule hides a few decisions a real optimizer must make. I have one sample per step, two parameter sets in play (current `x_t` and previous `x_{t-1}`), and I need *the same sample's* gradient at *both* points. In a static TensorFlow graph I can keep a slot containing the previous iterate and use graph replacement to ask for the current loss gradient with reads of each variable swapped to that slot. I keep the slots that the recursion actually needs: previous iterate, running gradient estimate `d`, an elementwise squared-gradient accumulator, a scalar running maximum gradient norm for clipping, and a diagnostic sum of squared estimates. The theorem writes one scalar `eta_t = k/(w + sum G_i^2)^{1/3}`; the implementation uses the same cube-root law elementwise, `sum_grad_squared += grad^2`, which is the diagonal preconditioning version practitioners expect from AdaGrad/Adam-style code. The coefficient is `beta = min(1, momentum * eta^2)`, the implementation name for `a`; the cap is the numerical version of the proof's `a <= 1` constraint. The update is exactly the compact formula `grad + (1-beta)(grad_estimate - grad_at_prev_iterate)`, then a scalar-norm clip to the running gradient scale, then save the current variable as the next previous iterate before applying `var += -eta * new_grad_estimate`. The actual optimizer also carries optional summary hooks, records the local smoothness diagnostic `||grad - grad_at_prev_iterate|| / (0.0001 + ||var - previous_iterate||)`, and keeps the `compute_gradients` signature compatible with the surrounding Tensor2Tensor training code. Mapping names: `lr` is `k`, `eta` is the denominator offset `w`, `momentum` is `c`, and `g_max` initializes the gradient-scale quantities.

```python
import tensorflow.compat.v1 as tf
from tensorflow.contrib import graph_editor as contrib_graph_editor
from tensorflow.contrib.optimizer_v2 import optimizer_v2

GATE_OP = 1

PREVIOUS_ITERATE = "previous_iterate"
GRAD_ESTIMATE = "grad_estimate"
SUM_GRAD_SQUARED = "sum_grad_squared"
MAXIMUM_GRADIENT = "maximum_gradient"
SUM_ESTIMATES_SQUARED = "sum_estimates_squared"


class StormOptimizer(optimizer_v2.OptimizerV2):
  def __init__(self, lr=1.0, g_max=0.01, momentum=100.0, eta=10.0,
               output_summaries=False, use_locking=False,
               name="StormOptimizer"):
    super(StormOptimizer, self).__init__(use_locking, name)
    self.lr = lr
    self.g_max = g_max
    self.momentum = momentum
    self.eta = eta
    self.output_summaries = output_summaries

  def _find_read_tensors(self, outputs, target):
    read_tensors, visited = set(), set()

    def dfs(parent):
      for x in parent.op.inputs:
        if x.name not in visited:
          if x.name == target.name:
            read_tensors.add(parent)
          visited.add(x.name)
          dfs(x)

    for output in outputs:
      dfs(output)
    return read_tensors

  def _make_replace_dict(self, state, grads, var_list):
    replace_dict = {}
    for var in var_list:
      previous_iterate = tf.convert_to_tensor(state.get_slot(var, PREVIOUS_ITERATE))
      for tensor in self._find_read_tensors(grads, var):
        replace_dict[tensor] = previous_iterate
    return replace_dict

  def _recompute_gradients(self, state):
    return contrib_graph_editor.graph_replace(
        self.grads, self._make_replace_dict(state, self.grads, self.vars))

  def _create_slot_with_value(self, state, var, value, name):
    state.create_slot(
        var, tf.constant(value, shape=var.shape, dtype=var.dtype.base_dtype), name)

  def _create_vars(self, var_list, state):
    for var in var_list:
      state.create_slot(var, var.initialized_value(), PREVIOUS_ITERATE)
      self._create_slot_with_value(state, var, self.g_max ** 3, SUM_GRAD_SQUARED)
      state.zeros_slot(var, GRAD_ESTIMATE)
      state.create_slot(
          var, tf.constant(self.g_max, dtype=var.dtype.base_dtype), MAXIMUM_GRADIENT)
      self._create_slot_with_value(state, var, 0.01, SUM_ESTIMATES_SQUARED)

  def _prepare(self, state):
    self.grads = []
    self.vars = []

  def _resource_apply_dense(self, grad, var, state):
    return self._apply_dense(grad, var, state)

  def _apply_dense(self, grad, var, state):
    self.grads.append(grad)
    self.vars.append(var)
    return tf.no_op()

  def _finish(self, state):
    update_ops = []
    grads_at_prev_iterate = self._recompute_gradients(state)

    for var, grad, grad_at_prev_iterate in zip(
        self.vars, self.grads, grads_at_prev_iterate):
      sum_grad_squared = state.get_slot(var, SUM_GRAD_SQUARED)
      previous_iterate = state.get_slot(var, PREVIOUS_ITERATE)
      maximum_gradient = state.get_slot(var, MAXIMUM_GRADIENT)
      grad_estimate = state.get_slot(var, GRAD_ESTIMATE)
      sum_estimates_squared = state.get_slot(var, SUM_ESTIMATES_SQUARED)

      maximum_gradient_updated = tf.assign(
          maximum_gradient, tf.maximum(maximum_gradient, tf.norm(grad)))
      update_ops.append(maximum_gradient_updated)

      sum_grad_squared_updated = tf.assign_add(
          sum_grad_squared, tf.pow(tf.abs(grad), 2.0))
      update_ops.append(sum_grad_squared_updated)

      smoothness = tf.norm(grad - grad_at_prev_iterate) / (
          0.0001 + tf.norm(var - previous_iterate))
      eta = self.lr * tf.pow(self.eta + sum_grad_squared_updated, -1.0 / 3.0)
      beta = tf.minimum(1.0, self.momentum * tf.square(eta))

      new_grad_estimate = grad + (1.0 - beta) * (
          grad_estimate - grad_at_prev_iterate)
      new_grad_estimate = tf.clip_by_value(
          new_grad_estimate, -maximum_gradient_updated, maximum_gradient_updated)

      if self.output_summaries:
        tf.summary.scalar(self._name + "/smoothness/" + var.name, smoothness)
        tf.summary.scalar(self._name + "/max_grad/" + var.name,
                          maximum_gradient_updated)
        tf.summary.scalar(self._name + "/average_beta/" + var.name,
                          tf.reduce_mean(beta))
        tf.summary.scalar(self._name + "/iterate_diff/" + var.name,
                          tf.norm(var - previous_iterate))
        tf.summary.scalar(self._name + "/grad_diff/" + var.name,
                          tf.norm(grad - grad_at_prev_iterate))
        tf.summary.scalar(self._name + "/vr_grad_estimate_norm/" + var.name,
                          tf.norm(new_grad_estimate))
        tf.summary.scalar(self._name + "/grad_norm/" + var.name, tf.norm(grad))

      grad_estimate_updated = tf.assign(grad_estimate, new_grad_estimate)
      update_ops.append(grad_estimate_updated)

      update_ops.append(tf.assign_add(
          sum_estimates_squared, tf.square(new_grad_estimate)))

      with tf.control_dependencies([grad_at_prev_iterate]):
        previous_iterate_updated = tf.assign(previous_iterate, var)
        update_ops.append(previous_iterate_updated)

      with tf.control_dependencies([previous_iterate_updated]):
        update_ops.append(tf.assign_add(var, -eta * grad_estimate_updated))

    return tf.group(*update_ops)

  def compute_gradients(self, loss, var_list=None, gate_gradients=GATE_OP,
                        aggregation_method=None, grad_loss=None,
                        stop_gradients=None, colocate_gradients_with_ops=False,
                        scale_loss_by_num_replicas=None):
    return super(StormOptimizer, self).compute_gradients(
        loss, var_list, gate_gradients, aggregation_method, grad_loss,
        stop_gradients, scale_loss_by_num_replicas)
```

So the causal chain, start to finish. SGD is stuck at `O(T^{-1/4})` because gradient noise has a floor that does not vanish near a critical point. Variance reduction beats the floor with a smoothness control variate — a same-sample two-point gradient difference, small precisely when consecutive iterates are close — but SVRG and SARAH and SPIDER all anchor that control variate with a full-gradient checkpoint (a mega-batch costing up to `O(T)` samples for one point) and ride a non-adaptive step size tuned against unknown constants. Staring at SARAH's recursive estimator next to heavy-ball momentum, both are "carry the running estimate forward and fold in the new gradient," so I graft the SARAH same-sample correction onto the momentum recursion, `d_t = (1-a)d_{t-1} + a nabla f(x_t,xi_t) + (1-a)(nabla f(x_t,xi_t) - nabla f(x_{t-1},xi_t))`, which contains both ancestors as corners (`a=0` is SARAH, drop the correction is momentum) and reduces to plain momentum once the iterates settle. Tracking the error `epsilon_t = d_t - nabla F(x_t)` gives a recursion with a contraction `(1-a)`, a tiny `a sigma` noise injection, and a smoothness term `O(eta||d||)`, so the error settles at `Z/a` and the design problem is to make `Z/a` small. Building the step size adaptively as `eta_t = k/(w + sum G^2)^{1/3}` and `a_{t+1} = c eta_t^2`, the equilibrium heuristic fixes the cube-root exponent as exactly what yields the optimal `||nabla F|| ~ T^{-1/3}` floor. A time-varying Lyapunov potential `Phi_t = F(x_t) + ||epsilon_t||^2/(32L^2 eta_{t-1})` — time-varying being what removes the checkpoint — lets me telescope: the descent lemma spends `(3eta/4)||epsilon||^2` and the potential's error term burns exactly `(3eta/4)||epsilon||^2`, a cancellation I force by choosing `z_t propto 1/eta` and `c = 28L^2 + G^2/(7Lk^3)` (the `28L^2` covers the descent surplus, the second piece eats the step-size increment). The noise term sums to only `ln(T+2)` via the log lemma, the seed error `epsilon_1` costs a single `sigma^2` instead of a checkpoint, and a Cauchy-Schwarz pass converts `sum eta_t||nabla F||^2` into the final `E[||nabla F||] <= O(1/sqrt T) + 2sigma^{1/3}/T^{1/3}` — optimal, adaptive to `sigma` without knowing it. Dropping the Lipschitz assumption is available by hardwiring `sigma` into a deterministic schedule. And the whole thing ships as a single-sample optimizer with per-variable slots for the previous iterate, the estimate, the squared-gradient accumulator, a gradient-scale maximum, and diagnostics, plus a same-sample second gradient evaluation at the previous iterate, per-coordinate adaptive steps, and a scalar gradient-scale clip; in the TensorFlow training setup, fixing `k` and `w` leaves `c` as the swept knob.
