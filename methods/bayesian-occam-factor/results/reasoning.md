I keep coming back to the same nagging thing. I have a noisy data set and a pile of candidate models to interpolate it with — polynomials of different degrees, radial basis functions, splines, even a little network — and fitting any one of them is no trouble at all. Pick the model, find the parameters that fit best, put error bars on them, done. That's the easy level. The level that keeps defeating me is choosing *between* the models. And the naive thing, the thing everybody reaches for first, is to just look at how well each fits and take the winner. I keep doing it and it keeps being wrong, and I want to understand exactly why it's wrong before I try to fix it.

The wrongness is plain. Take a polynomial of degree d and one of degree d+1. The degree-(d+1) family *contains* the degree-d family as a special case — set the top coefficient to zero. So whatever fit degree d achieves, degree d+1 can match and then do a little better by wiggling that extra coefficient. The best achievable likelihood is non-decreasing as I add parameters. Always. So if I rank by best fit, I am guaranteed to climb to the most flexible model on the table, and that model fits the noise and generalises horribly. I've watched this happen: the maximum-likelihood interpolant through noisy points oscillates wildly, threading every point, chasing scatter that isn't signal. Best fit is not a ranking; it's a one-way ratchet toward overfitting.

So the field bolts on penalties. Add 2k to minus-twice-log-likelihood, or k log N, to punish parameter count. Or measure a combinatorial capacity of the function class. Or count the bits in a two-part code and minimise that. Or hold out a test set, or cross-validate. They all *work* after a fashion, but every one of them feels like an admission of defeat: I couldn't get a complexity penalty out of the inference itself, so I'm grafting one on from outside, with a coefficient I chose. The capacity measures don't even look at the data I have or the prior I hold — they're worst-case properties of the function class. And the penalty's exact form is folklore: why 2k and not 3k, why k log N? I don't want a graft. I want the penalty to *be there already*, falling out of probability theory, so that I never have to write down a complexity term at all.

Let me go back to the very bottom and write Bayes' theorem for the two things I'm doing, and watch where the pieces go. For one model H with parameters w, fitting is

P(w | D, H) = P(D | w, H) P(w | H) / P(D | H).

Posterior equals likelihood times prior over that denominator. And the denominator — I always throw it away. When I'm fitting, P(D | H) is just a constant in w; it doesn't move the location of the peak, doesn't affect a single parameter estimate, so I divide it out and forget it. Now write the *other* level, comparing models:

P(H | D) ∝ P(D | H) P(H).

And there it is, the same quantity I just threw away, sitting in the load-bearing position. The thing that's irrelevant to fitting parameters is the *entire* data-dependent term for comparing models. If I give the models equal prior probability P(H) — I have no reason to prefer one a priori — then comparing models is nothing but comparing this denominator. It deserves a name; call it the evidence for H. Rank models by their evidence. Suddenly the question isn't "which fits best" at all; it's "which has the largest P(D | H)."

What is this evidence, concretely? It's the normaliser of the fitting equation, so

P(D | H) = ∫ P(D | w, H) P(w | H) dw.

The probability the model assigns to the data I actually saw, averaged over all the parameter values the model thought possible before the data came, weighted by the prior. Stare at that as a function of the *data* for a fixed model. It's a normalised probability distribution over the space of all possible data sets — it has to integrate to one over D. So a model cannot assign high probability to everything. If a flexible model spreads its predictive probability over a huge variety of conceivable data sets, then for any particular data set it must assign *less*, because the total is fixed at one. A rigid model bets its probability on a narrow range of data sets; if the truth lands in that range it wins big, and if it lands outside it loses. A flexible model hedges over everything and so backs each outcome only weakly.

That's the whole thing, isn't it. Picture data space along a horizontal axis. A simple model puts a tall narrow hump of predictive probability over a small region; a complex model puts a low broad smear over a wide region. Suppose the observed data fall in the region the simple model concentrated on. Then even though the complex model *could* have fit those data — could even fit them better once its parameters are tuned — it assigned them less probability *in advance*, because it had to save probability for all the other data sets it's also prepared to explain. With equal model priors, the simple model is more probable. The penalty for complexity isn't a term I added; it's the price a model pays, through normalisation, for the data it does *not* predict. The more datasets you're ready to account for, the less you can claim for the one that happened. That is exactly Occam's razor, and it came for free out of the integral.

I should pressure-test this before I trust it, because "spreads probability and so loses" sounds slippery. An adversarial case keeps bothering me. Let the "Sure Thing" model be the hypothesis that the data will be precisely D, the data set that did occur. Its evidence is enormous — it predicted D and nothing else, so P(D | Sure Thing) ≈ 1. Doesn't that break my story, an absurdly complex pre-knowledge model winning? No — and seeing why sharpens the point. Sure Thing is one of an immense family of equally specific hypotheses (one for every possible D), and any honest prior over that family assigns each member a correspondingly tiny prior probability P(H). So its posterior is negligible. The evidence is the right ranking only when the model priors are sensibly equal; when someone smuggles in a model whose very definition encodes the answer, the prior over models is where that gets paid for. Fine. For the ordinary case — models I genuinely proposed before seeing the data, with comparable priors — the evidence is the objective razor, and in the large-data limit it dominates whatever mild differences there are in model priors. I'll proceed with equal P(H) and let the evidence do the work.

Now I have to actually *compute* P(D | H) = ∫ P(D | w, H) P(w | H) dw, and I want to do it in a way that doesn't just spit out a number but shows me *why* it penalises complexity — I want the complexity penalty to be legible in the formula. The integrand is the unnormalised posterior. For the problems I care about it has a single strong peak at the most probable parameters, w_MP — the same w_MP I find when I fit. An integral dominated by a sharp peak: this is exactly the situation Laplace's method handles. Approximate the integral by the *height* of the peak times its *accessible width*. In one dimension, with one parameter w,

P(D | H) ≈ P(D | w_MP, H) · P(w_MP | H) · σ_{w|D},

where σ_{w|D} is the posterior accessible width, not a new normalisation constant hiding offstage. If the local curvature is A in one dimension, that width is √(2π) A^{-1/2}; if I call A^{-1/2} the posterior standard deviation, the √(2π) belongs to the Gaussian volume. Look at how that factored. The first factor, P(D | w_MP, H), is the best-fit likelihood — the very thing best-fit model selection would use, with w_MP coinciding with the maximum-likelihood fit when the prior is flat across the peak. The remaining two factors are the correction. Let me make them transparent. Suppose the prior P(w | H) is roughly uniform over some range σ_w of values the parameter was allowed before the data — the prior width measured in the same accessible-volume convention. Then its height is P(w_MP | H) ≈ 1/σ_w, and

P(D | H) ≈ P(D | w_MP, H) · (σ_{w|D} / σ_w).

So the evidence is the best-fit likelihood times a factor σ_{w|D}/σ_w — posterior width over prior width. That ratio is less than one (the data shrink the plausible range), so it's a penalty. And I can read it. It's the factor by which the model's parameter space collapsed when the data arrived: the prior said w could be anywhere in σ_w, the posterior says w is pinned to within σ_{w|D}. Think of the model as built from σ_w/σ_{w|D} distinguishable sub-models — equivalent settings of its parameter — of which exactly one survives contact with the data. The Occam factor is one over that number. Its logarithm is the information, in nats, that the data gave me about the parameter. A complex model with many parameters, each free to roam over a wide prior range, accumulates a product of such small ratios and is penalised hard. And — a bonus I didn't expect — a model that has to be *finely tuned* to fit, one whose posterior σ_{w|D} is forced to be tiny, is also penalised; the razor cuts against precarious models, not just many-parameter ones. The trade-off is now explicit: the best-fit likelihood rewards fitting the data, the Occam factor rewards not having to spread or fine-tune your prior to do it, and the evidence is their product. Best fit is the special case where I keep only the first factor and discard the very thing that does the regularising.

Now lift it to k parameters, because real models have many. The posterior near its peak, with Δw = w − w_MP, is a Gaussian: expand the log posterior to second order, P(w | D, H) ≈ P(w_MP | D, H) exp(−½ Δwᵀ A Δw), where A = −∇∇ log P(w | D, H) is the Hessian at the peak — and note this is *exactly* the matrix I already compute to get parameter error bars, since the posterior covariance is A^{−1}. The width of a k-dimensional Gaussian is its covariance volume: the standard Gaussian integral ∫ exp(−½ Δwᵀ A Δw) dᵏw = (2π)^{k/2} |A|^{−1/2}. So

P(D | H) ≈ P(D | w_MP, H) · P(w_MP | H) · (2π)^{k/2} |A|^{−1/2}.

There's the multi-parameter Occam factor: P(w_MP | H) (2π)^{k/2} |A|^{−1/2}, the ratio of the posterior accessible volume to the prior accessible volume in parameter space. Same story, now with a determinant doing the bookkeeping for correlated parameters. Let me sanity-check the constant against Laplace's method directly: for ∫ h(w) e^{−g(w)} dᵏw with g minimised at w*, the asymptotic value is h(w*) e^{−g(w*)} (2π)^{k/2} |∇∇g|^{−1/2}, and here g is minus the log integrand so its Hessian is A — the factors match, (2π)^{k/2} and |A|^{−1/2}, no stray powers of two. Good. And the cost is nothing: to get the Occam factor I need only the Hessian A, which I already formed for the error bars. Bayesian model comparison is just maximum-likelihood model comparison with one extra multiplicative factor, and that factor is sitting in my fitting code already. For a linear model with Gaussian noise and a quadratic regulariser the Gaussian is exact, so this isn't even an approximation there — it's the answer for any N. Elsewhere the central limit theorem makes it better as data accumulate.

Good — I have the razor for choosing a model. But the interpolation problem hides a second razor inside the first model, and I can't compare basis sets honestly until I've dealt with it. To interpolate at all I needed a regulariser, because maximum-likelihood interpolation is ill-posed — the misfit-minimising w is underdetermined and oscillates to fit noise. So I write the likelihood P(D | w, β) = exp(−β E_n)/Z_n with E_n = ½ Σ_m (y(x_m) − t_m)² the data misfit and β = 1/σ_ν² the inverse noise variance, Z_n = (2π/β)^{N/2}. And I impose a prior P(w | α) = exp(−α E_w)/Z_w with E_w a quadratic smoothness penalty like ½ Σ w_h², Z_w = ∫ dᵏw exp(−α E_w). The posterior is exp(−M)/Z_M with M = α E_w + β E_n — minimising this combined misfit is finding the most probable interpolant, the familiar Bayesian reading of regularisation. But now α and β are dangling. They control complexity: crank α up and the interpolant goes stiff and flat and underfits; crank α down and it oscillates and overfits. Setting α is itself an Occam problem. The orthodox tools — force χ² to N, or to N−k, or minimise test error, or cross-validate — are either, I suspect, plain wrong or too noisy to trust on modest data. I want Bayes to set α and β.

Once the evidence is the ranking object, α and β become hypotheses too. I can compare one pair of scales against another by the probability it assigns to the data after w has been integrated out:

P(α, β | D, H) = P(D | α, β, H) P(α, β) / P(D | H).

The data-dependent term P(D | α, β, H) already appeared — it was the normaliser of the posterior over w at fixed α, β. So I want it as a function of α, β, and with a locally flat density over log α and log β, since they're scale parameters, the hyperprior only multiplies this by a slowly varying factor. To find the peak I can maximise this evidence itself. And it's beautifully cheap because the three normalisers I've already named *are* the evidence:

P(D | α, β, H) = Z_M(α, β) / (Z_w(α) Z_n(β)).

The Occam's razor for α is sitting right in that ratio. Make α small — let the prior allow wild weights — and the prior's accessible volume balloons, Z_w blows up, and it divides the evidence down. Make α large and the interpolant can't fit, so Z_M collapses. The evidence for α is maximised at the compromise: small enough to fit the data, not so small that it overfits. No misfit target needed; the partition functions do it.

But the moment I write "find the most probable α and then use that prior," something feels off, and I want to stop and confront it rather than paper over it, because if I've made an error here the whole edifice is suspect. It sounds like I'm choosing my *prior* after seeing the data — fitting α to D and then turning around and using the α-prior to infer w. That's circular, almost cheating. Let me work out what the honest thing is. The honest thing is that it was never a single prior; it's an *ensemble* of priors indexed by α, and proper inference integrates over the whole ensemble. The true posterior over w is

P(w | D, H) = ∫ P(w | D, α, β, H) P(α, β | D, H) dα dβ,

a mixture of the fixed-α posteriors weighted by how probable each α is given the data. So I'm not picking a prior post hoc; I'm averaging over all of them with data-determined weights. Now — *if* P(α, β | D, H) has a single sharp peak at some α̂, β̂, and the w-posterior doesn't change much as α, β wander near that peak, then the mixture is dominated by the single term P(w | D, α̂, β̂, H), and using the most probable α is a *good approximation* to the full integral, not a cheat. The discontent dissolves: the most-probable-α recipe is the peak approximation to a perfectly proper marginalisation. Laplace, by the way, very nearly did exactly this in 1774 when he inferred a nuisance scale parameter and then tried to integrate it out.

And this clarifies a deeper point I want to nail down, because it's the difference between Bayes and maximum likelihood at this level. Why integrate over α, β rather than just jointly maximise the likelihood over w, α and β all at once? Because the joint likelihood has a *skew* peak: the maximum-likelihood point is not where the bulk of the posterior probability sits. The cleanest illustration is the oldest one. Take N samples from a Gaussian with unknown mean μ and standard deviation σ. The maximum-likelihood estimate of σ is σ_N — divide by N. But the *most probable* σ, obtained by integrating μ out first and then looking at the marginal, is σ_{N−1} — divide by N−1. And this has nothing to do with the prior on σ; the prior is flat here. It's the *act of marginalisation* that corrects the bias of maximum likelihood and MAP. Fitting μ used up a degree of freedom — the fitted μ absorbed some of the noise — and integrating it out instead of maximising over it accounts for that automatically. The same correction is what I want for α and β: don't maximise jointly, marginalise the parameters and let α, β be set by *their* evidence. Maximisation gives the biased σ_N; marginalisation gives the honest σ_{N−1}.

So now I need the three integrals Z_M, Z_w, Z_n to actually evaluate the evidence for α, β, and for a quadratic regulariser — which the good ones are — I can do Z_M exactly, no approximation. E_n and E_w are quadratic in w, so M is quadratic. Let C = ∇∇ E_w and B = ∇∇ E_n; then A = ∇∇ M = α C + β B, and Taylor-expanding around the minimiser w_MP,

M(w) = M(w_MP) + ½ (w − w_MP)ᵀ A (w − w_MP),

with the minimiser at w_MP = β A^{−1} B w_ML (the regulariser pulls the unregularised least-squares solution back toward zero). The integral Z_M = ∫ dᵏw exp(−M) is then the Gaussian integral again, Z_M = exp(−M(w_MP)) (2π)^{k/2} |A|^{−1/2}. Put the pieces together and the log evidence for α, β is

log P(D | α, β, H) = −α E_w^MP − β E_n^MP − ½ log det A − log Z_w(α) − log Z_n(β) + (k/2) log 2π.

The term β E_n^MP is the data misfit. The log Occam contribution for α is −α E_w^MP − log Z_w(α) + (k/2) log 2π − ½ log det A: the ratio (2π)^{k/2}|A|^{−1/2} / Z_w(α) is posterior-volume-over-prior-volume in parameter space — the same collapse-ratio as before — and the −α E_w^MP measures how far the fitted weights had to move from their null value. Same razor, now pricing the regulariser strength.

I could just grind this numerically over a grid of α, β. But I want the *structure* of where the optimum sits, because I suspect there's something cleaner hiding. Let me find the condition at the maximum. First rotate into the natural basis of the prior — the basis where ∇∇ E_w = I (for quadratic E_w just diagonalise C and rescale the axes), so E_w = ½ Σ w_h² and A = α I + β B. In this basis the log evidence is

log P(D | α, β, H) = −α E_w^MP − β E_n^MP − ½ log det A + (k/2) log α + (N/2) log β − (N/2) log 2π.

When I differentiate with respect to α, the hidden motion of w_MP does not add a term because w_MP is already the stationary point of M. The only awkward piece is d/dα log det A. The derivative of a log-determinant is d/dα log det A = Trace(A^{−1} dA/dα) = Trace(A^{−1} · I) = Trace A^{−1}. Setting ∂/∂α (log evidence) = 0 gives

2 α E_w^MP = k − α Trace A^{−1}.

Look at the two sides. The left side, 2α E_w^MP = Σ w_MP² / σ_w² with σ_w² = 1/α, is a χ² of the parameters — how far the data dragged the weights from zero, measured in prior units. The right side, call it γ = k − α Trace A^{−1}, has a meaning I have to extract. Write it in the eigenbasis of β B with eigenvalues λ_a; then the eigenvalues of A are λ_a + α, and

γ = k − α Σ_a 1/(λ_a + α) = Σ_a (λ_a + α − α)/(λ_a + α) = Σ_a λ_a/(λ_a + α).

Every term λ_a/(λ_a + α) lives between 0 and 1. In a direction where the data are strong, λ_a ≫ α, the term is ≈ 1 — that parameter is well measured by the data. In a direction where the data are weak, λ_a ≪ α, the term is ≈ 0 — that parameter is pinned by the prior, not the data. So γ is the *effective number of well-measured parameters*, somewhere between 0 and k, and the fitted weights are w_MP,a = (λ_a/(λ_a+α)) w_ML,a, each maximum-likelihood component shrunk by exactly its own data-strength fraction. The condition for the best α now reads: estimate the variance of the distribution the weights are drawn from, 1/α = σ_w² = Σ w² / γ, using γ effective samples rather than k. It's the σ_N-versus-σ_{N−1} correction again, generalised — γ degrees of freedom were genuinely used, not k.

And γ is exactly what I need to settle β, the noise level, which is the other place the orthodox criteria fight. The expected χ² between the *true* interpolant and the data is N. But I don't have the true interpolant, only the fitted one, and fitting suppresses some noise. The discrepancy principle says set χ²_D = 2 β E_n = N; the textbook least-squares correction says N − k. Differentiate the same log evidence with respect to β. Again the w_MP motion cancels at the stationary point of M, and now dA/dβ = B:

0 = −E_n^MP − ½ Trace(A^{−1} B) + N/(2β).

Multiplying by 2β gives 2βE_n^MP = N − β Trace(A^{−1}B). In the eigenbasis above, βB has eigenvalues λ_a, so β Trace(A^{−1}B) = Σ_a λ_a/(λ_a+α) = γ. Therefore

2 β E_n^MP = N − γ.

Not N, not N − k, but N − γ. The reason is mechanical now: each *well-measured* parameter soaks up about one unit of χ² by fitting a bit of noise it can't distinguish from signal, while the *poorly* measured parameters are held by the prior and soak up nothing — so the data lose exactly γ units of χ², not k. The σ_{N−1} calculator-button correction was the one-parameter case of precisely this. At the joint optimum, then, χ²_w = γ and χ²_D = N − γ, and the total misfit obeys the clean 2M = N. Misfit criteria play no role at all in setting α — only β cares about the value of the misfit. This also hands me a fast fixed-point scheme: re-estimate α := γ / (2 E_w) and β := (N − γ) / (2 E_n) and iterate, replacing an expensive determinant search with a trace.

Now the top level, comparing whole models — different basis sets, different regularisers, different noise models. The evidence for a model H is the α, β evidence integrated over α, β:

P(D | H) = ∫ P(D | H, α, β) P(α, β) dα dβ.

A single sharp maximum at α̂, β̂ for quadratic problems, well approximated by a separable Gaussian, and differentiating the optimum condition twice gives its width: (Δ log α)² ≈ 2/γ and (Δ log β)² ≈ 2/(N−1). So the integral is the peak value times those widths,

P(D | H) ≈ P(D | α̂, β̂, H) · P(α̂, β̂ | H) · 2π Δ log α Δ log β,

and I rank models — splines against radial basis functions against polynomials against networks — by this number. The α̂, β̂ prior is the one genuinely subjective ingredient, but it's mild: a scale unknown over a couple of orders of magnitude contributes only about ±1 to the log evidence, usually negligible against the differences between models, and for interpolation models that all share α and β it cancels outright.

Before running any demonstration I can already see the shape this score must have for a family — Legendre polynomials, say — as I increase the number of basis functions. On the left, with too few functions, the models can't fit and the evidence is low: the misfit penalty there scales as N, the number of data points, so the climb is steep. As I add functions the fit improves and the evidence rises. Then it turns over: past the point where the extra functions describe anything real, each new parameter brings its own Occam factor, its own little volume-collapse ratio less than one, and the product of those ratios drags the evidence back down. A maximum in the middle — the "Occam hill," steep up the misfit side, gently down the complexity side. The maximum-likelihood fit term by itself would keep climbing as I add functions, exactly the ratchet I started out distrusting; it's the Occam factor, and *only* the Occam factor, that bends the evidence over and picks out the adequate model. And I never wrote down a complexity penalty. I wrote down a prior and an integral, and the penalty was the width of that integral all along.

So the chain is: best-fit model selection fails because likelihood only ever rises with complexity; the cure is to rank by the evidence P(D | H) = ∫ P(D | w, H) P(w | H) dw, which as a normalised distribution over data sets automatically charges a model for the data it doesn't predict; Laplace's method factors that integral into best-fit likelihood times an Occam factor — posterior width over prior width, σ_{w|D}/σ_w in one dimension and P(w_MP|H)(2π)^{k/2}|A|^{−1/2} in k dimensions, computed from the very Hessian the error bars already need; the same evidence, applied one level down, sets the regularisation strength and noise level by marginalising rather than maximising, which is what corrects the σ_N-to-σ_{N−1} bias and reveals γ = Σ λ/(λ+α) effective parameters and χ²_D = N − γ; and ranking whole models by the integrated evidence traces an Occam hill whose downslope is nothing but the accumulated Occam factors. No ad-hoc term anywhere. The razor was a consequence of integrating over parameters instead of maximising over them.

```python
import numpy as np

def design_matrix(x, centres, width):
    # linear-in-parameters interpolation model y(x) = sum_h w_h phi_h(x)
    return np.exp(-0.5 * ((x[:, None] - centres[None, :]) / width) ** 2)

def log_evidence(Phi, t, alpha, beta):
    # quadratic regulariser E_w = (1/2)||w||^2, Gaussian noise E_n = (1/2)||Phi w - t||^2
    N, k = Phi.shape
    A = alpha * np.eye(k) + beta * (Phi.T @ Phi)          # A = alpha C + beta B, the Hessian of M
    w_mp = beta * np.linalg.solve(A, Phi.T @ t)           # most probable interpolant, w_MP = beta A^-1 B w_ML
    E_w = 0.5 * w_mp @ w_mp                               # weight penalty at w_MP
    E_n = 0.5 * np.sum((Phi @ w_mp - t) ** 2)            # data misfit at w_MP
    M_mp = alpha * E_w + beta * E_n
    sign, logdetA = np.linalg.slogdet(A)
    if sign <= 0:
        raise ValueError("Hessian must be positive definite")

    log_Zm = -M_mp + 0.5 * k * np.log(2 * np.pi) - 0.5 * logdetA
    log_Zw = 0.5 * k * np.log(2 * np.pi / alpha)
    log_Zn = 0.5 * N * np.log(2 * np.pi / beta)
    log_ev = log_Zm - log_Zw - log_Zn                 # = log[Z_M / (Z_w Z_n)]

    # evidence = likelihood at the posterior peak x Occam factor; with a flat
    # local prior this is the usual best-fit-likelihood x Occam-factor split
    fit_loglik = -beta * E_n - 0.5 * N * np.log(2 * np.pi / beta)              # log P(D | w_MP, beta)
    log_occam = 0.5 * k * np.log(2 * np.pi) - 0.5 * logdetA - alpha * E_w - log_Zw
    assert np.isclose(log_ev, fit_loglik + log_occam)

    gamma = k - alpha * np.trace(np.linalg.inv(A))        # effective # of well-measured parameters
    return log_ev, fit_loglik, log_occam, gamma

# Sweeping k lets the fit term and accumulated Occam factors be inspected separately;
# their sum is the evidence used to rank the candidate bases.
```
