Let me start from what is actually about to break. I have a loss law that works beautifully: for a transformer with `N` parameters trained on `D` tokens, `L(N, D) = E + A/N^α + B/D^β`. Three pieces, each one I can read off physically — `E` is the entropy floor of the text, no model gets under it; `A/N^α` is the gap a finite-capacity model leaves below the ideal generative process; `B/D^β` is the gap from training on a finite sample for finite steps. Fit five constants in log space and I can extrapolate loss, and minimizing it under the compute constraint `6ND = C` hands me the compute-optimal allocation in closed form: `N_opt = G(C/6)^a`, `D_opt = G^{-1}(C/6)^b` with `a = β/(α+β)`, `b = α/(α+β)`, and since `α ≈ β` empirically, `a ≈ b ≈ 0.5` — scale parameters and tokens together. Hoffmann and colleagues nailed this and it overturned Kaplan's earlier `a ≈ 0.73` "spend it all on a bigger model" prescription, which I now understand was an artifact of early stopping and a learning-rate schedule that didn't match each run's token horizon.

But push the allocation forward and the arithmetic gets uncomfortable. Equal scaling says a 530B model wants on the order of 11 trillion tokens — call it thirty-odd terabytes of text. The stock of high-quality English on the internet is, by the better estimates, going to run dry around the middle of this decade at the current trend, and for essentially every language that isn't English the corpus is already orders of magnitude too small. So the regime I'm walking into is: a fixed pool of unique tokens `U`, and a compute budget so large that there is simply no fresh data left to spend it on. What do I do when I run out of data?

The obvious lever is to repeat — train for multiple epochs over the pool I have. Ordinary in machine learning generally, but the large-LM world has mostly trained for a single epoch, and some people argue reuse actively hurts. There's one suggestive data point of a science model trained for about four epochs whose validation loss kept dropping the whole way — but it was never run head-to-head against a one-epoch unique-data baseline, so I can't extract from it the thing I need, which is the *trade-off*: is buying the second epoch worth as much as buying twice the data? And there's a theoretical nudge from the deep-bootstrap line — good online learners are good offline generalizers — which whispers that for a *small* number of passes a repeated token can't be all that different from a fresh one. None of this is a model. I have no quantitative handle on how a token's value decays as I show it again and again.

So let me just try to use the law I have. The fastest thing: treat `D` as the total number of tokens *processed*, repeats included, and plug it straight into `L = E + A/N^α + B/D^β`. Stare at what that says. Under `B/D^β`, the millionth token and the millionth-plus-one token — which is the same token on its second pass — contribute identically to driving the loss down. The law literally cannot tell a fresh token from a re-read one. So it predicts the eleventh epoch is worth exactly as much as the first, and that I can drive loss to the floor just by looping forever over a single sentence. That's plainly false, and it's false in a way that matters: it's the single assumption — `D` is fresh tokens, fit only on one-epoch runs — that the whole compute-optimal edifice silently rests on. Wall. The law isn't wrong, it's just blind to the only variable that matters now.

So I need to teach it the difference between fresh and repeated. Let me name the variables cleanly. `N` parameters, `U` the unique tokens — the size of the pool, the thing that's capped. I'll let `R_D` be the number of *repetitions*, i.e. epochs minus one, so `R_D = 0` is the single-epoch base case where total processed equals `U`. The question I actually want to answer is: if I repeat my `U` unique tokens `R_D` times, that's worth *as much as* having had how many fresh unique tokens? Call that quantity the effective data, `D'`. If I can compute `D'`, I don't have to touch the law at all — I just feed `D'` into the `B/D'^β` slot, because `D'` is by construction "the amount of fresh data that would have done the same job." And `D' = U` exactly when `R_D = 0`, which means the new law collapses to the old one in the single-epoch case for free. That's the property I refuse to give up.

Now I need a model of how a token's worth decays with re-reading. Let me think about what the model gains from one pass over a token. The first time it sees the token, it extracts most of the information in it. The second time, a lot of that information is already in the weights, so it can only extract from the *remaining* novelty — less. The third time, less still. The natural assumption: each pass, the model learns a fixed fraction `1 - δ` of whatever value is left, for some constant `0 ≤ δ ≤ 1`. If `δ = 0`, nothing is lost — repeats are as good as fresh. If `δ = 1`, the second pass is worthless. The phrase "a fixed fraction of what remains, every pass" *is* geometric decay: the value the model gets from the `k`-th pass over a token is `(1-δ)^k` times what a fresh token gives. I'm not measuring this — I'm positing it, because it's the simplest law with the right qualitative shape (monotone decay toward zero) and, crucially, it'll sum in closed form.

So the effective value of one unit of unique data taken through `R_D + 1` passes — the original pass plus `R_D` repeats — is a geometric series. Summing over my `U` unique tokens:

  `D' = U + (1-δ)U + (1-δ)^2 U + ... + (1-δ)^{R_D} U`.

The first term `U` is the fresh first pass at full value; each subsequent term discounts by another factor `(1-δ)`. Geometric series, first term... let me be careful here, because I want to keep the `U` (the first, full-value pass) separate and only sum the *repeats*. Write it as the full first pass plus the discounted tail:

  `D' = U + U Σ_{k=1}^{R_D} (1-δ)^k`.

The tail is a geometric series with ratio `r = (1-δ)`, first term `(1-δ)U` (that's `k=1`), and `R_D` terms. Using `S = a(1 - r^n)/(1 - r)` with `a = (1-δ)U`, `r = (1-δ)`, `n = R_D`:

  `D' = U + (1-δ)U · (1 - (1-δ)^{R_D}) / δ`,

since `1 - r = 1 - (1-δ) = δ`. Good — and notice this form is perfectly happy with a *non-integer* `R_D`, which I'll need, because real runs don't land on whole epochs. Sanity check the base case: `R_D = 0` gives an empty tail sum, `D' = U`. ✓. And the limit: as `R_D → ∞`, `(1-δ)^{R_D} → 0`, so `D' → U + (1-δ)U/δ`. It *plateaus*. No matter how many times I loop, the effective data saturates at a finite value. That's exactly the diminishing-returns intuition made quantitative — there's a ceiling on how much signal a fixed pool can ever yield.

I could stop here, learn `δ` directly, and use this. But `δ` is an awkward thing to interpret — "fraction of remaining value lost per pass" is fine but not vivid. Let me reparameterize around the plateau, because the plateau is the interpretable object. The tail of `D'` tends to `(1-δ)U/δ`, so define `R*_D = (1-δ)/δ`. Then the effective data plateaus at `U + R*_D · U` as `R_D → ∞`. So `R*_D` reads directly as "the number of *extra* full-value epochs' worth of data I can ever squeeze out of this pool by repeating" — a repetition budget, a half-life-like constant. If `R*_D` is large, repeats stay valuable for a long time; if small, they die fast. Much better to fit and report than a raw `δ`.

Now I want the cleaner closed form in terms of `R*_D`. Here's where I lean on `δ` being small — which I should check is the regime I'm actually in, and I'll come back to verify it after fitting. If `δ` is small, two approximations line up. First, `1/R*_D = δ/(1-δ) ≈ δ` (the denominator is ≈ 1). Second, the Taylor expansion `e^x = 1 + x + x²/2! + ... ≈ 1 + x` for small `x`: setting `x = -δ`,

  `1 - δ ≈ e^{-δ} ≈ e^{-1/R*_D}`.

So `(1-δ)^{R_D} ≈ e^{-R_D/R*_D}`. And the prefactor `(1-δ)U/δ` is just `R*_D · U`. Substitute both into the geometric closed form:

  `D' = U + U · R*_D · (1 - e^{-R_D/R*_D})`.

That's the effective-data law I want. Let me re-run all the checks on this smoothed version. `R_D = 0`: the exponential is `e^0 = 1`, the bracket is zero, `D' = U`. ✓ — single epoch recovers the base. `R_D ≪ R*_D`: `1 - e^{-R_D/R*_D} ≈ R_D/R*_D` (small-argument expansion again), so `D' ≈ U + U·R*_D·(R_D/R*_D) = U(1 + R_D) = D`. So in the few-epochs regime, effective data equals total processed data — repeated tokens are worth almost exactly fresh ones. That's precisely what the deep-bootstrap intuition predicted, and now it falls out of the algebra instead of being asserted. `R_D ≫ R*_D`: the exponential vanishes, `D' → U + U·R*_D`, the plateau. And at `R_D = R*_D` exactly, `1 - e^{-1} ≈ 0.63`, so the repeated tokens are worth on average about `1 - 1/e` of fresh ones — that's the half-life reading of `R*_D` made precise.

I should make sure the small-`δ` approximation didn't cost me much, because it's the one shortcut in the whole derivation. Take a deliberately *not* small `δ = 0.25`, so a repeat retains 75% of its value. One unit `U = 1`, repeated four times, `R_D = 4`. The exact geometric form `D' = U + (1-δ)U(1-(1-δ)^{R_D})/δ` gives `D' = 1 + 0.75·(1 - 0.75^4)/0.25`. Now `0.75^4 = 0.3164`, so `1 - 0.3164 = 0.6836`, times `0.75/0.25 = 3`, gives `3 · 0.6836 = 2.05`, plus the leading `1` is `3.05`. The smoothed form with `R*_D = (1-δ)/δ = 0.75/0.25 = 3`: `D' = 1 + 1·3·(1 - e^{-4/3}) = 1 + 3·(1 - 0.2636) = 1 + 3·0.7364 = 3.21`. So `3.21` vs `3.05` — the approximation overshoots by about 5% in `D'`. But `D'` doesn't enter the loss raw; it enters as `B/D'^β` with `β ≈ 0.35`. The thing that actually matters is `(3.21/3.05)^{0.35} - 1 ≈ (1.052)^{0.35} - 1 ≈ 1.8%`. The power `β < 1` compresses the error. So even at a `δ` four times larger than "small," the smoothed law is within 2% on the loss term. The approximation is safe, and I'll confirm post-hoc that the fitted `δ` really is small. (And the plateau survives the approximation: `R_D = 100`, `D' = 1 + 3(1 - e^{-100/3}) ≈ 1 + 3 = 4`, matching `U + R*_D·U = 4`.)

Now — do I touch only the data term? Let me look at the parameter term `A/N^α` under the same lens. It says every parameter has the same marginal value no matter how much data I have: going from 1B to 10B parameters drops loss by the same absolute amount whether my dataset is one token or a billion tokens. In the extreme that's obviously broken. If I have a single token, the first billion parameters can already memorize everything there is to know about it; the next nine billion parameters cannot possibly extract more, because there *is* no more. So excess parameters — parameters beyond what the available data can justify — must decay in value exactly the way repeated tokens do: the same feature gets relearned, adding nothing. The asymmetry would be arbitrary. So I'll mirror the construction. Define `U_N` as the number of parameters that are "fresh" for this data — the compute-optimal parameter count for the unique pool — and `R_N` as how many times I'm effectively repeating that allocation: `R_N = max(N/U_N - 1, 0)`. Then by the identical geometric-decay argument,

  `N' = U_N + U_N · R*_N · (1 - e^{-R_N/R*_N})`,

with its own learned plateau constant `R*_N`. The `max(·, 0)` is because `N` can legitimately be *below* the data-justified count (then there are no excess parameters, `R_N = 0`, `N' = N`), whereas `D` can never be below `U` — you always do at least one pass.

I need `U_N`, the data-justified parameter count for a pool of `U_D` unique tokens. That's exactly what the compute-optimal frontier gives: find the compute budget at which `U_D` is the optimal token count, and read off the optimal parameter count there. Invert `D_opt(C) = G^{-1}(C/6)^b` to get the budget at which `D_opt = U_D`, then feed that budget into `N_opt(C) = G(C/6)^a`. Doing the algebra — from `D_opt = G^{-1}(C/6)^b` and `N_opt = G(C/6)^a`, eliminate `C/6` to get `N_opt = (D_opt · G)^{a/b} · G`, and with `a/b = β/α`,

  `U_N = min{ (U_D · G)^{β/α} · G, N },   G = (αA/βB)^{1/(α+β)}`,

capped at `N` because if the model is smaller than the data justifies, all its parameters are fresh. So the full law, replacing `N` with `N'` and `D` with `D'` in the single-epoch form:

  `L(U_N, U_D, R_N, R_D) = A / (U_N + U_N R*_N (1 - e^{-R_N/R*_N}))^α + B / (U_D + U_D R*_D (1 - e^{-R_D/R*_D}))^β + E`.

This is a strict generalization of `E + A/N^α + B/D^β`: when there are no data repetitions and no excess parameters it reduces to the single-epoch law, and if the two decay constants go to infinity it becomes exactly the old law for all `N` and `D`. So it inherits everything that law already got right, and adds only the two decay constants. That's the cleanest possible extension — two new parameters, both interpretable.

Before I fit, let me think about whether I've left out a behavior I should model. The empirical contours, when you push parameters or epochs far past optimal, eventually turn *upward* — too much of either can hurt, not just plateau. My law can't represent that: `D'` and `N'` are monotone increasing and saturating, so loss only ever decreases toward a floor. Should I build the upturn in? I could — e.g. let the *exponents* `α, β` decay with repetition toward zero instead of decaying `N, D`, because `lim_{α→0} N^α = 1` pushes `N` back down to 1 and makes loss climb again. Let me sketch why that's tempting and then why I'll reject it. The exponent-decay form would fit the upturn, but it has no clean information-theoretic story — there's no "each pass learns a fraction of the remaining value" behind decaying an exponent, it's just a curve that bends the right way. And when I imagine fitting it, the very high repetition-budget it would need to keep early epochs benign sits oddly against the data. More to the point: the upturn isn't a phenomenon I actually need to *predict*. If excess epochs start to hurt, the fix is to stop training; if excess parameters hurt, remove them. Nobody wants an accurate forecast of *how much* extra loss they'd eat by overtraining — they want to know when returns plateau so they can quit. So I'll keep the monotone-saturating form, declare that it models plateau-not-upturn, and drop the runs where loss demonstrably went back up (and the double-descent cases where it dips, climbs, then dips again) from the fit, since the form can't represent them and they'd just inject noise. That's a modeling choice with a clear rationale, not a fudge.

Now to fit `R*_N` and `R*_D`. I want to follow the same methodology as the single-epoch fit so the two are comparable. I do *not* refit `A, B, E, α, β` — those are pinned to the single-epoch values, because the new law must agree with the validated single-epoch behavior; the only new degrees of freedom are the two decay constants. The objective: minimize, over `R*_N, R*_D`, the Huber loss between predicted and observed log-loss across all runs,

  `min_{R*_N, R*_D} Σ_i Huber_δ( LSE(a - α log N'_i, b - β log D'_i, e) - log L_i )`,

where `a = log A`, `b = log B`, `e = log E`, and `LSE` is log-sum-exp. Why this shape? I fit in *log* space because loss is power-law/multiplicative, so log-errors are the natural homoscedastic residual. `LSE(a - α log N', b - β log D', e)` is exactly `log(A/N'^α + B/D'^β + E)` — the log-sum-exp evaluates the log of the sum of the three additive terms without ever forming the sum directly, which keeps it numerically stable when the terms span many orders of magnitude. And Huber instead of plain squared error with a small `δ` (`δ = 1e-3`) because the low-compute and double-descent runs are noisy outliers, and Huber stops them from dominating the fit. I'll minimize with L-BFGS — the surface in `(R*_N, R*_D)` is smooth but non-convex, so I run it from a grid of initializations (each of `R*_N, R*_D` swept over `{0, 4, 8, ..., 20}`) and keep the best local optimum.

When I run this on the body of runs — order a couple hundred, parameters from millions up to billions, epochs from one into the hundreds, with the upturn/double-descent outliers removed — I get `R*_N ≈ 5.31` and `R*_D ≈ 15.39`. First, the consistency check I promised: from `1/R*_D ≈ δ`, `R*_D ≈ 15.4` means `δ ≈ 0.06`, comfortably small; for `R*_N ≈ 5.3`, `δ ≈ 0.19`, still in the regime where the 2%-error bound holds. The small-`δ` approximation was self-consistent. Now the *content* of those two numbers, which is the whole payoff. `R*_D > R*_N` — the data decays *slower* than the parameters. Repeated tokens stay useful for around fifteen epochs before sharply diminishing; excess parameters lose their value about three times faster. So in the data-constrained regime, where I'm forced to repeat, the right move is to pour additional compute into *more epochs* rather than *more parameters* — the exact opposite of the single-epoch prescription, where you scale them together. The reason is now mechanical, not empirical hand-waving: at the point where I've matched epochs and parameters up to roughly `R*_N`, each new parameter is worth only `e^{-R_N/R*_N}` of its nominal value while each new repeated token is still worth `e^{-R_D/R*_D}` of its nominal value, and since `R*_N < R*_D` the parameter has decayed further for the same number of "repeats." Equal computational cost, unequal remaining value — so tilt toward the data axis.

Let me also pin the single-epoch constants on the actual corpus, because the prior work reported, for this web corpus, only that the optimal allocation is exactly `a = b = 0.5`, i.e. `α = β`, without publishing the coefficients `A, B, E` themselves. So I refit the single-epoch form on the corpus's runs with `α` and `β` *tied* equal (forced by `a = b = 0.5`), minimizing the same LSE-Huber objective over a grid. That yields `a = 6.255`, `b = 7.305`, `e = 0.625`, `α = β = 0.3527`, i.e. `A = e^a ≈ 521`, `B = e^b ≈ 1488`, `E = e^e ≈ 1.87`:

  `L(N, D) = 1.87 + 521/N^{0.353} + 1488/D^{0.353}`.

I validate this is a good fit the same way the original work did — at the Gopher compute budget `C = 5.76e23` FLOPs my `N_opt = G(C/6)^a` and `D_opt = G^{-1}(C/6)^b` give about 70B parameters and 1.37T tokens, right on top of the ~73B / ~1.3T the IsoFLOP curves on this corpus reported. Good — so these are the constants I freeze before fitting the two decay terms.

Substituting the frozen single-epoch constants and the two fitted decay constants, the concrete law I'll predict with is

  `L = 521 / (U_N + 5.3 U_N (1 - e^{-R_N/5.3}))^{0.35} + 1488 / (U_D + 15.4 U_D (1 - e^{-R_D/15.4}))^{0.35} + 1.87`,   with `U_N = min(N, 0.051 · U_D)`.

(The `0.051` is just `(U_D·G)^{β/α}·G / U_D` collapsed with `α = β` and the fitted `G`; the `min` keeps under-sized models from being treated as over-parameterized.)

Two more sanity passes against alternatives, to be sure the smoothed exponential form earns its place. What if I'd skipped the `e^{-x}` smoothing and used the raw geometric `D'` (and the symmetric raw `N'`) directly, fitting `δ` rather than `R*`? That fits marginally *better* — the smoothed form introduces the ~2% approximation, so the raw one lands a hair higher in `R²`. But the raw form's learned constant is `δ`, which is harder to read than "the number of bonus epochs you can extract." Given the fit difference is within noise, interpretability wins. What if I decay only `D` and not `N`? The fit collapses badly — decaying data alone is far better than decaying neither, but leaving the parameter term un-decayed loses most of the structure, because excess parameters in the heavily-repeated runs genuinely stop helping. Decaying both is necessary. And the exponent-decay variant I considered for the upturn? It fits the overall surface clearly worse and gets the *return* curve wrong — it predicts repetition value diminishing far too slowly — confirming that bending the exponents has no honest mechanism behind it. So: decay both `N` and `D`, via the smoothed exponential-of-geometric form, fitting two interpretable plateau constants.

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

Let me retrace the chain. I had a single-epoch loss law that won the allocation argument but assumed unlimited fresh data, and I'm running out of it. Feeding repeated tokens into it as if they were fresh predicts the absurd — that looping over one sentence drives loss to the floor — because the law cannot see a re-read token. So I introduced effective data `D'`: the fresh-token-equivalent of repeating `U` tokens `R_D` times, under the assumption that each pass extracts a fixed fraction of the *remaining* value, which is geometric decay. Summing the geometric series and reparameterizing around the saturation point gave `D' = U + U R*_D (1 - e^{-R_D/R*_D})`, which equals `U` at one epoch (recovering the old law), equals total tokens for a few epochs (matching the deep-bootstrap intuition), and plateaus at `U(1 + R*_D)` — a hard ceiling on what a fixed pool can yield. The small-`δ` smoothing that turned the integer geometric sum into a clean exponential costs under 2% on the loss term because of the `β`-power, and the fitted `δ ≈ 0.06` confirmed the regime. The same diminishing-value logic applied symmetrically to excess parameters gave `N'`, with the capped data-justified count `U_N = min(N, 0.051 U_D)` read off the compute-optimal frontier. Fitting the two decay constants in log space with an LSE-Huber objective and grid-started L-BFGS — freezing the single-epoch coefficients so the law stays pinned to the old law when decay is removed — gave `R*_D ≈ 15.4 > R*_N ≈ 5.3`, which says repeated data outlasts excess parameters, so in the data-constrained regime additional compute should go to more epochs faster than to more parameters. The monotone-saturating form deliberately models plateau, not upturn, because the upturn isn't worth predicting — you just stop. And the whole thing drops into the per-group curve-fitting harness as one loss law: compute the repeats, the effective quantities, evaluate `E + A/N'^α + B/D'^β`.
