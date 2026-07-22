Let me start from what is actually about to break. I have a loss law that works beautifully on the runs I've seen: for a transformer with `N` parameters trained on `D` tokens, `L(N, D) = E + A/N^α + B/D^β`. Three pieces, each one I can read off physically — `E` is the entropy floor of the text, no model gets under it; `A/N^α` is the gap a finite-capacity model leaves below the ideal generative process; `B/D^β` is the gap from training on a finite sample for finite steps. Fit five constants in log space and I can extrapolate loss, and minimizing it under the compute constraint `6ND = C` hands me the compute-optimal allocation in closed form: `N_opt = G(C/6)^a`, `D_opt = G^{-1}(C/6)^b` with `a = β/(α+β)`, `b = α/(α+β)`, and since `α ≈ β` empirically, `a ≈ b ≈ 0.5` — scale parameters and tokens together. Hoffmann and colleagues nailed this and it overturned Kaplan's earlier `a ≈ 0.73` "spend it all on a bigger model" prescription, which I now understand was an artifact of early stopping and a learning-rate schedule that didn't match each run's token horizon.

But push the allocation forward and the arithmetic gets uncomfortable. Equal scaling says a 530B model wants on the order of 11 trillion tokens — call it thirty-odd terabytes of text. The stock of high-quality English on the internet is, by the better estimates, going to run dry around the middle of this decade at the current trend, and for essentially every language that isn't English the corpus is already orders of magnitude too small. So the regime I'm walking into is: a fixed pool of unique tokens `U`, and a compute budget so large that there is simply no fresh data left to spend it on. What do I do when I run out of data?

The obvious lever is to repeat — train for multiple epochs over the pool I have. Ordinary in machine learning generally, but the large-LM world has mostly trained for a single epoch, and some people argue reuse actively hurts. There's one suggestive data point of a science model trained for about four epochs whose validation loss kept dropping the whole way — but it was never run head-to-head against a one-epoch unique-data baseline, so I can't extract from it the thing I need, which is the *trade-off*: is buying the second epoch worth as much as buying twice the data? And there's a theoretical nudge from the deep-bootstrap line — good online learners are good offline generalizers — which whispers that for a *small* number of passes a repeated token can't be all that different from a fresh one. None of this is a model. I have no quantitative handle on how a token's value decays as I show it again and again.

So let me just try to use the law I have, and see exactly where it breaks. The fastest thing: treat `D` as the total number of tokens *processed*, repeats included, and plug it straight into `L = E + A/N^α + B/D^β`. Now stare at what that says. Under `B/D^β`, the millionth token and the millionth-plus-one token — which is the same token on its second pass — contribute identically to driving the loss down. The law literally cannot tell a fresh token from a re-read one. Take it to the limit to be sure it's broken and not just imprecise: loop forever over a single sentence, so `U` stays tiny but `D → ∞`. Then `B/D^β → 0` and the predicted loss falls to `E + A/N^α`, the entropy floor plus the capacity gap — i.e. the law claims I can squeeze a single sentence down to the irreducible entropy of the whole language just by re-reading it enough times. That's not a small error, it's qualitatively impossible. And it traces to one assumption: `D` is fresh tokens, a fit done only on one-epoch runs. That single assumption is what the entire compute-optimal edifice silently rests on, and it's exactly the one that fails in the regime I now care about. Wall. The law isn't *wrong* — it's blind to the only variable that matters now.

So I need to teach it the difference between fresh and repeated. Let me name the variables cleanly. `N` parameters, `U` the unique tokens — the size of the pool, the thing that's capped. I'll let `R_D` be the number of *repetitions*, i.e. epochs minus one, so `R_D = 0` is the single-epoch base case where total processed equals `U`. The question I actually want to answer is: if I repeat my `U` unique tokens `R_D` times, that's worth *as much as* having had how many fresh unique tokens? Call that quantity the effective data, `D'`. If I can compute `D'`, I don't have to touch the law at all — I just feed `D'` into the `B/D'^β` slot, because `D'` is by construction "the amount of fresh data that would have done the same job." Notice the constraint this puts on `D'`: it must equal `U` exactly when `R_D = 0`, so that the new law collapses to the old one in the single-epoch case. Any candidate for `D'` that fails that test is disqualified up front, because I am not allowed to break agreement with the validated single-epoch fits.

Now I need a model of how a token's worth decays with re-reading. Let me think about what the model gains from one pass over a token. The first time it sees the token, it extracts most of the information in it. The second time, a lot of that information is already in the weights, so it can only extract from the *remaining* novelty — less. The third time, less still. The simplest assumption with that shape: each pass, the model learns a fixed fraction `1 - δ` of whatever value is left, for some constant `0 ≤ δ ≤ 1`. Test the assumption at its boundaries before I build on it. If `δ = 0`, nothing is lost per pass — repeats are as good as fresh, which is the optimistic deep-bootstrap end. If `δ = 1`, the model has already absorbed everything after the first look, so the second pass is worthless — the pessimistic end. Both endpoints behave the way intuition demands, so this one-parameter family at least spans the right qualitative range. The phrase "a fixed fraction of what remains, every pass" is geometric decay: the value the model gets from the `k`-th pass over a token is `(1-δ)^k` times what a fresh token gives. I'm not measuring this — I'm positing it, because it's the simplest law with the right monotone-decay shape and, crucially, it'll sum in closed form so I can fold it back into the existing law rather than fitting a new curve.

So the effective value of one unit of unique data taken through `R_D + 1` passes — the original pass plus `R_D` repeats — is a geometric series. Summing over my `U` unique tokens:

  `D' = U + (1-δ)U + (1-δ)^2 U + ... + (1-δ)^{R_D} U`.

The first term `U` is the fresh first pass at full value; each subsequent term discounts by another factor `(1-δ)`. I want to keep the `U` (the first, full-value pass) separate and only sum the *repeats*, so write it as the full first pass plus the discounted tail:

  `D' = U + U Σ_{k=1}^{R_D} (1-δ)^k`.

The tail is a geometric series with ratio `r = (1-δ)`, first term `(1-δ)U` (that's `k=1`), and `R_D` terms. Using `S = a(1 - r^n)/(1 - r)` with `a = (1-δ)U`, `r = (1-δ)`, `n = R_D`, and `1 - r = 1 - (1-δ) = δ`:

  `D' = U + (1-δ)U · (1 - (1-δ)^{R_D}) / δ`.

This form is perfectly happy with a *non-integer* `R_D`, which I'll need, because real runs don't land on whole epochs. Run the test I committed to: at `R_D = 0` the tail sum is empty, so `D' = U`. ✓ — the base case recovers the pool exactly, so the new law will collapse to the old single-epoch law. And the limit: as `R_D → ∞`, `(1-δ)^{R_D} → 0`, so `D' → U + (1-δ)U/δ`. It *plateaus* at a finite value. No matter how many times I loop, the effective data saturates — which is exactly the diminishing-returns behavior the naive law was missing, now made quantitative: there's a hard ceiling on how much signal a fixed pool can ever yield, and that ceiling is what rules out the looping-a-single-sentence absurdity above.

I could stop here, learn `δ` directly, and use this. But `δ` is an awkward thing to report — "fraction of remaining value lost per pass" is fine but not vivid. The plateau is the interpretable object, so let me reparameterize around it. The tail of `D'` tends to `(1-δ)U/δ`, so define `R*_D = (1-δ)/δ`. Then the effective data plateaus at `U + R*_D · U` as `R_D → ∞`. So `R*_D` reads directly as "the number of *extra* full-value epochs' worth of data I can ever squeeze out of this pool by repeating" — a repetition budget, a half-life-like constant. If `R*_D` is large, repeats stay valuable for a long time; if small, they die fast. That's a much better thing to fit and report than a raw `δ`.

Now I want the cleaner closed form in terms of `R*_D`. Here's where I lean on `δ` being small — which I should check is the regime I'm actually in, and I'll come back to verify it after fitting, since right now I have no idea what `δ` will turn out to be. If `δ` is small, two approximations line up. First, `1/R*_D = δ/(1-δ) ≈ δ` (the denominator is ≈ 1). Second, the Taylor expansion `e^x = 1 + x + x²/2! + ... ≈ 1 + x` for small `x`: setting `x = -δ`,

  `1 - δ ≈ e^{-δ} ≈ e^{-1/R*_D}`.

So `(1-δ)^{R_D} ≈ e^{-R_D/R*_D}`. And the prefactor `(1-δ)U/δ` is just `R*_D · U`. Substitute both into the geometric closed form:

  `D' = U + U · R*_D · (1 - e^{-R_D/R*_D})`.

That's the candidate effective-data law. I won't trust it until I re-run the checks on this smoothed version, because I've now made an approximation and approximations are where errors hide. `R_D = 0`: the exponential is `e^0 = 1`, the bracket is zero, `D' = U`. ✓ — single epoch still recovers the base, so the smoothing didn't cost me the one property I refused to give up. `R_D ≪ R*_D`: `1 - e^{-R_D/R*_D} ≈ R_D/R*_D` (small-argument expansion again), so `D' ≈ U + U·R*_D·(R_D/R*_D) = U(1 + R_D) = D`. So in the few-epochs regime, effective data equals total processed data — repeated tokens are worth almost exactly fresh ones. I didn't put that in by hand; it's precisely what the deep-bootstrap intuition predicted, and it fell out of the algebra, which makes me more confident the model has the right shape rather than just the right endpoints. `R_D ≫ R*_D`: the exponential vanishes, `D' → U + U·R*_D`, the plateau. And at `R_D = R*_D` exactly, `1 - e^{-1} = 0.632`, so the repeated-token block is worth on average about `1 - 1/e` of fresh — the half-life reading of `R*_D` made precise.

Now the one shortcut in the whole derivation — the small-`δ` smoothing — needs an actual error bound, not a hand-wave, because I'm about to build the headline result on top of it. Let me push it deliberately *outside* the regime it assumes and measure the damage. Take `δ = 0.25`, four times what "small" would mean, so a repeat retains 75% of its value. One unit `U = 1`, repeated four times, `R_D = 4`. The exact geometric form: `D' = 1 + 0.75·(1 - 0.75^4)/0.25`. Compute `0.75^4 = 0.31640625`, so `1 - 0.31640625 = 0.68359375`; times `0.75/0.25 = 3` gives `2.0508`, plus the leading `1` is `D'_exact = 3.0508`. The smoothed form with `R*_D = (1-δ)/δ = 3`: `D' = 1 + 1·3·(1 - e^{-4/3})`, and `e^{-4/3} = 0.26360`, so the bracket is `0.73640`, times `3` is `2.2092`, plus `1` gives `D'_smooth = 3.2092`. So the approximation overshoots `D'` by `3.2092/3.0508 - 1 = 5.19%`. That sounds like a lot — but `D'` doesn't enter the loss raw; it enters as `B/D'^β` with `β ≈ 0.35`, and a power below one compresses errors. The relative effect on the loss term is `(3.2092/3.0508)^{0.35} - 1 = (1.0519)^{0.35} - 1 = 1.79%`. So even at a `δ` four times larger than "small," the smoothed law is within 1.8% on the quantity that actually feeds the loss. The approximation is safe in any plausible regime, and I'll still confirm post-hoc that the fitted `δ` really is small. (And the plateau survives the smoothing: `R_D = 100`, `D' = 1 + 3(1 - e^{-100/3}) ≈ 1 + 3·(1 - 3e-15) = 4.000`, matching `U + R*_D·U = 4`.)

Now — do I touch only the data term? Let me look at the parameter term `A/N^α` under the same lens. It says every parameter has the same marginal value no matter how much data I have: going from 1B to 10B parameters drops loss by the same absolute amount whether my dataset is one token or a billion tokens. Push that to the extreme the way I pushed the data term. If I have a single token, the first handful of parameters can already memorize everything there is to know about it; the next billion parameters cannot possibly extract more, because there *is* no more information in the data for them to capture. So excess parameters — parameters beyond what the available data can justify — must decay in value the same way repeated tokens do: the same feature gets relearned, adding nothing. Keeping the data term saturating but leaving the parameter term linear-in-value would be an arbitrary asymmetry, and the single-token thought-experiment shows the parameter term has the identical pathology. So I'll mirror the construction. Define `U_N` as the number of parameters that are "fresh" for this data — the compute-optimal parameter count for the unique pool — and `R_N` as how many times I'm effectively repeating that allocation: `R_N = max(N/U_N - 1, 0)`. Then by the identical geometric-decay argument,

  `N' = U_N + U_N · R*_N · (1 - e^{-R_N/R*_N})`,

with its own learned plateau constant `R*_N`. The `max(·, 0)` is because `N` can legitimately be *below* the data-justified count (then there are no excess parameters, `R_N = 0`, `N' = N`), whereas `D` can never be below `U` — you always do at least one pass.

I need `U_N`, the data-justified parameter count for a pool of `U_D` unique tokens. That's exactly what the compute-optimal frontier gives: find the compute budget at which `U_D` is the optimal token count, and read off the optimal parameter count there. Invert `D_opt(C) = G^{-1}(C/6)^b` to get the budget at which `D_opt = U_D`, then feed that budget into `N_opt(C) = G(C/6)^a`. Doing the algebra — from `D_opt = G^{-1}(C/6)^b` and `N_opt = G(C/6)^a`, eliminate `C/6` to get `N_opt = (D_opt · G)^{a/b} · G`, and with `a/b = β/α`,

  `U_N = min{ (U_D · G)^{β/α} · G, N },   G = (αA/βB)^{1/(α+β)}`,

capped at `N` because if the model is smaller than the data justifies, all its parameters are fresh. So the full law, replacing `N` with `N'` and `D` with `D'` in the single-epoch form:

  `L(U_N, U_D, R_N, R_D) = A / (U_N + U_N R*_N (1 - e^{-R_N/R*_N}))^α + B / (U_D + U_D R*_D (1 - e^{-R_D/R*_D}))^β + E`.

I should check that this is genuinely a generalization and not a different law wearing the old one's clothes. With no data repetitions and no excess parameters — `R_D = 0` and `N ≤ U_N` so `R_N = 0` — both effective quantities collapse (`D' = U`, `N' = N`) and it is `E + A/N^α + B/D^β` exactly. And if I send the two decay constants `R*_N, R*_D → ∞`, then `1 - e^{-R/R*} → R/R* `, so `N' → U_N(1 + R_N) = N` and `D' → U(1 + R_D) = D` for *all* `N, D`, recovering the old law everywhere, not just at the base point. So it inherits everything the single-epoch law already got right, adding only the two decay constants. That's the cleanest extension I can see — two new parameters, both interpretable as repetition budgets.

Before I fit, let me think about whether I've left out a behavior I should model. The empirical contours, when you push parameters or epochs far past optimal, eventually turn *upward* — too much of either can hurt, not just plateau. My law can't represent that: `D'` and `N'` are monotone increasing and saturating, so loss only ever decreases toward a floor. Should I build the upturn in? There's a tempting way to do it: let the *exponents* `α, β` decay with repetition toward zero instead of decaying `N, D`, because `lim_{α→0} N^α = 1` pushes the `A/N^α` term back up toward `A`, making loss climb. Let me take that alternative seriously enough to see why I reject it rather than waving it off. Mechanically it would fit the upturn. But it has no information-theoretic story — "each pass learns a fraction of the remaining value" is a real account of *why* value decays, whereas "the exponent of the power law slowly shrinks with epochs" is just a knob chosen because it bends the curve the right way. And it costs the property I worked to keep: a decaying exponent does not collapse to the single-epoch power law cleanly, so I'd lose exact agreement with the validated base fit. So the deciding question is whether the upturn is even something I need to *predict*. If excess epochs start to hurt, the fix is to stop training; if excess parameters hurt, remove them. Nobody wants an accurate forecast of *how much* extra loss they'd eat by overtraining — they want to know when returns plateau so they can quit. So I'll keep the monotone-saturating form, declare explicitly that it models plateau-not-upturn, and drop from the fit the runs where loss demonstrably went back up (and the double-descent cases where it dips, climbs, then dips again), since the form can't represent them and they'd inject noise. I'll keep the exponent-decay idea in my pocket and actually fit it later as a check, so the rejection rests on a number rather than on taste.

Now to fit `R*_N` and `R*_D`. I want to follow the same methodology as the single-epoch fit so the two are comparable. I do *not* refit `A, B, E, α, β` — those are pinned to the single-epoch values, because the new law must agree with the validated single-epoch behavior; the only new degrees of freedom are the two decay constants. The objective: minimize, over `R*_N, R*_D`, the Huber loss between predicted and observed log-loss across all runs,

  `min_{R*_N, R*_D} Σ_i Huber_δ( LSE(a - α log N'_i, b - β log D'_i, e) - log L_i )`,

where `a = log A`, `b = log B`, `e = log E`, and `LSE` is log-sum-exp. Why this shape? I fit in *log* space because loss is power-law/multiplicative, so log-errors are the natural homoscedastic residual. `LSE(a - α log N', b - β log D', e)` is exactly `log(A/N'^α + B/D'^β + E)` — the log-sum-exp evaluates the log of the sum of the three additive terms without ever forming the sum directly, which keeps it numerically stable when the terms span many orders of magnitude. And Huber instead of plain squared error, with a small `δ` (`δ = 1e-3`), because the low-compute and double-descent runs are noisy outliers and Huber stops them from dominating the fit. I'll minimize with L-BFGS — the surface in `(R*_N, R*_D)` is smooth but non-convex, so I run it from a grid of initializations (each of `R*_N, R*_D` swept over `{0, 4, 8, ..., 20}`) and keep the best local optimum.

But the fit needs the single-epoch constants `A, B, E, α, β`, and the prior work reported, for this web corpus, only that the optimal allocation is exactly `a = b = 0.5`, i.e. `α = β`, without publishing the coefficients themselves. So I refit the single-epoch form on the corpus's runs with `α` and `β` *tied* equal (forced by `a = b = 0.5`), minimizing the same LSE-Huber objective over a grid. That yields `a = 6.255`, `b = 7.305`, `e = 0.625`, `α = β = 0.3527`, i.e. `A = e^a = 520.8`, `B = e^b = 1487.7`, `E = e^e = 1.869`:

  `L(N, D) = 1.87 + 521/N^{0.353} + 1488/D^{0.353}`.

Before freezing these, I should validate them the way the original work did — by checking that they reproduce the known compute-optimal point on this corpus, not just that they fit the cloud of runs. From these constants, `G = (αA/βB)^{1/(α+β)} = (520.8/1487.7)^{1/0.7053} = 0.2258` (the `α=β` cancel inside). At the Gopher compute budget `C = 5.76e23` FLOPs, with `a = b = 0.5`, I get `N_opt = G(C/6)^{0.5} = 0.2258·(9.6e22)^{0.5} = 7.00e10` and `D_opt = G^{-1}(C/6)^{0.5} = (9.6e22)^{0.5}/0.2258 = 1.372e12`. So 70.0B parameters and 1.37T tokens. The IsoFLOP curves fit on this corpus report roughly 73B / 1.3T at that budget — my numbers land on top of those without my having tuned to them, which is the independent check I wanted. So these are the constants I freeze before fitting the two decay terms.

When I run the two-parameter decay fit on the body of runs — order a couple hundred, parameters from millions up to billions, epochs from one into the hundreds, with the upturn/double-descent outliers removed — I get `R*_N = 5.31` and `R*_D = 15.39`. First, the consistency check I promised when I made the small-`δ` smoothing. From `1/R*_D ≈ δ`, `R*_D = 15.39` gives `δ ≈ 0.065`, comfortably small; for `R*_N = 5.31`, `δ ≈ 0.188`, still inside the regime where I measured the loss-term error at under 1.8% (that test used `δ = 0.25`, larger than either of these). So the smoothing was self-consistent — I didn't fit my way into a regime where my own approximation breaks.

Now I want to read the *content* of those two numbers carefully, because the headline rests on the comparison and I'd rather see it as a number than assert it. The claim I'm tempted to make is "data decays slower than parameters, so spend on epochs." Let me make that mechanical. At a point where I've matched epochs and excess-parameter passes, so `R_N = R_D = r`, the marginal remaining value of the next parameter is `e^{-r/R*_N}` and of the next repeated token is `e^{-r/R*_D}`. At `r = 5`: parameter value `e^{-5/5.31} = 0.390`, token value `e^{-5/15.39} = 0.723`. At `r = 10`: parameter `e^{-10/5.31} = 0.152`, token `e^{-10/15.39} = 0.522`. So at the *same* number of "repeats," the parameter has already decayed roughly twice as far as the token, and the gap widens as `r` grows. The ratio of the decay constants is `R*_D/R*_N = 15.39/5.31 = 2.90`, so the data axis stays alive almost 3× longer than the parameter axis. That's the whole payoff, and now it's an arithmetic fact about the fitted constants rather than a slogan: in the data-constrained regime, where I'm forced to repeat, additional compute should go to *more epochs* faster than to *more parameters* — the exact opposite of the single-epoch prescription, where you scale them together. Equal computational cost, unequal remaining value, so tilt toward data.

Let me assemble the concrete law I'll predict with. The data-justified parameter count collapses nicely under `α = β`: the exponent `β/α = 1`, so `U_N = (U_D·G)·G = U_D·G² = U_D·(0.2258)² = 0.05099·U_D`, which I'll write `min(N, 0.051·U_D)`. Substituting the frozen single-epoch constants and the two fitted decay constants:

  `L = 521 / (U_N + 5.3 U_N (1 - e^{-R_N/5.3}))^{0.35} + 1488 / (U_D + 15.4 U_D (1 - e^{-R_D/15.4}))^{0.35} + 1.87`,   with `U_N = min(N, 0.051 · U_D)`.

Two more sanity passes against alternatives, to be sure the smoothed exponential form earns its place. What if I'd skipped the `e^{-x}` smoothing and used the raw geometric `D'` (and the symmetric raw `N'`) directly, fitting `δ` rather than `R*`? That fits marginally *better* — the smoothed form carries the ~1.8% approximation I measured, so the raw one lands a hair higher in `R²`. But the raw form's learned constant is `δ`, which is harder to read than "the number of bonus epochs you can extract," and the fit difference is within the run-to-run noise. Interpretability wins by a margin I'm comfortable defending. What if I decay only `D` and not `N`? The fit collapses badly — decaying data alone beats decaying neither, but leaving the parameter term un-decayed loses most of the structure, because excess parameters in the heavily-repeated runs genuinely stop helping, exactly as the single-token thought-experiment predicted. Decaying both is necessary, not symmetric for symmetry's sake. And the exponent-decay variant I kept in my pocket for the upturn? Fitting it for real, it fits the overall surface clearly worse and gets the *return* curve wrong — it predicts repetition value diminishing far too slowly — which is the number I wanted before rejecting it. So: decay both `N` and `D`, via the smoothed exponential-of-geometric form, fitting two interpretable plateau constants.

Now let me put this into the code I'd actually fit and predict with, filling the single empty slot in the regression harness — the loss law and its fit. The fit minimizes the log-space LSE-Huber objective with L-BFGS; the predict path computes the repeats, the data-justified parameter count, and the two effective quantities, then evaluates the law. I'll mirror the canonical structure exactly.

```python
import numpy as np
import torch


A_LOG, B_LOG, E_LOG = 6.255414, 7.3049974, 0.6254804
ALPHA = BETA = 0.3526596
A, B, E = np.exp(A_LOG), np.exp(B_LOG), np.exp(E_LOG)          # ~521, ~1488, ~1.87
G = ((ALPHA * A) / (BETA * B)) ** (1.0 / (ALPHA + BETA))


def optimal_N(C):
    # N_opt(C) = G (C/6)^{beta/(alpha+beta)}  (compute-optimal parameter count)
    a = BETA / (ALPHA + BETA)
    return G * (C / 6.0) ** a


def D_to_C(D):
    # invert D_opt(C) = G^{-1}(C/6)^b  ->  the compute budget at which D is optimal
    b = ALPHA / (ALPHA + BETA)
    return ((G * D) ** (1.0 / b)) * 6.0


def _fit_loss(inp, params):
    # inp columns: [U_N, U_D, R_D, R_N, L];  params: a, b, e, alpha, beta, rd_star, rn_star
    a, b, e, alpha, beta, ep_star, n_star = params
    # effective parameters N' and effective data D' (geometric decay, smoothed to e^{-x})
    n_eff = inp[:, 0] + inp[:, 0] * n_star * (1 - torch.exp(-inp[:, 3] / n_star))
    d_eff = inp[:, 1] + inp[:, 1] * ep_star * (1 - torch.exp(-inp[:, 2] / ep_star))
    # LSE(a - alpha*log N', b - beta*log D', e) = log(A/N'^alpha + B/D'^beta + E)
    pre = torch.stack([a - alpha * torch.log(n_eff),
                       b - beta * torch.log(d_eff),
                       e.expand(inp.shape[0])])
    pred = torch.logsumexp(pre, dim=0)
    # Huber in log space, robust to the noisy low-compute / double-descent runs
    return torch.nn.functional.huber_loss(pred, torch.log(inp[:, 4]),
                                          delta=1e-3, reduction="none").sum()


def _fit_decay_from(inp, init, steps=50):
    # Fix a,b,e,alpha,beta (single-epoch fit); learn only R*_D (ep_star), R*_N (n_star).
    p = torch.nn.Parameter(torch.tensor(init, dtype=torch.float32))
    lbfgs = torch.optim.LBFGS([p], lr=1e-1, history_size=10, max_iter=20,
                              line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad()
        l = _fit_loss(inp, p)
        l.backward()
        p.grad[:5] = 0          # do not move the frozen single-epoch coefficients
        return l

    for _ in range(steps):
        lbfgs.step(closure)
    with torch.no_grad():
        return float(_fit_loss(inp, p)), p.detach().numpy()


def fit_decay(inp, steps=50):
    best_loss, best_params = float("inf"), None
    for rd0 in np.arange(0.0, 24.0, 4.0):
        for rn0 in np.arange(0.0, 24.0, 4.0):
            init = [A_LOG, B_LOG, E_LOG, ALPHA, BETA,
                    max(rd0, 1e-6), max(rn0, 1e-6)]
            loss, params = _fit_decay_from(inp, init, steps=steps)
            if np.isfinite(loss) and loss < best_loss:
                best_loss, best_params = loss, params
    return best_params


PARAMS = [A_LOG, B_LOG, E_LOG, ALPHA, BETA, 15.387756, 5.309743]


def scaling_law(N, D, U, params=PARAMS):
    a, b, e, alpha, beta, rd_star, rn_star = params
    A_, B_, E_ = np.exp(a), np.exp(b), np.exp(e)
    R_D = np.maximum((D / U) - 1.0, 0.0)                 # data repetitions = epochs - 1
    U_N = np.minimum(N, optimal_N(D_to_C(U)))            # data-justified parameter count
    R_N = np.maximum((N / U_N) - 1.0, 0.0)              # excess-parameter "repeats"
    # effective data and effective parameters: geometric value-decay, plateauing at U(1+R*)
    D_eff = U + U * rd_star * (1 - np.exp(-R_D / rd_star))
    N_eff = U_N + U_N * rn_star * (1 - np.exp(-R_N / rn_star))
    return E_ + A_ / N_eff ** alpha + B_ / D_eff ** beta
```

Let me trace this code once on a concrete data-constrained run to make sure the pieces wire together and the numbers come out where the algebra says they should. Take a unique pool `U = 1e9`, a model `N = 1e9` params, trained for 10 epochs so `D = 10e9`. Then `R_D = D/U - 1 = 9`. The data-justified count is `U_N = min(1e9, 0.051·1e9) = 0.051e9 = 5.10e7` — so a billion-parameter model on a billion-token pool is genuinely over-parameterized by this measure, which is the regime the parameter-decay term exists to handle. Then `R_N = N/U_N - 1 = 1e9/5.10e7 - 1 = 18.6`. Effective data `D_eff = 1e9·(1 + 15.39·(1 - e^{-9/15.39})) = 1e9·(1 + 15.39·0.4423) = 7.81e9` — well below the total `10e9` processed (the repeats are discounted) and well below the plateau `16.39e9` (not yet saturated at 10 epochs). Effective params `N_eff = 5.10e7·(1 + 5.31·(1 - e^{-18.6/5.31})) = 5.10e7·6.15 = 3.14e8` — the billion params behave like ~0.31 billion fresh ones, heavily discounted because they're way past justified. Plug in: `L = 1.87 + 521/(3.14e8)^{0.353} + 1488/(7.81e9)^{0.353} = 1.87 + 521/745 + 1488/2330 = 1.87 + 0.699 + 0.639 = 2.88`. A sensible held-out cross-entropy, every intermediate quantity finite and on the right side of its bound. And the base-case wiring: set `N` to exactly the data-justified count and `D = U`, so `R_N = R_D = 0`; then `N_eff = U_N`, `D_eff = U`, and the law returns `E + A/U_N^α + B/U^β` with no decay applied — exactly the single-epoch power law, confirming the collapse holds in the actual code and not just on paper.
