OK, let me think this through from scratch. I have a function f: R^n → R that I can only probe — hand it a point, get back a number. No gradient, no formula, nothing but the value. And the value costs me, so my real currency is the number of evaluations to reach a decent f. The landscapes I actually care about are non-linear, non-convex, and the two things that make them nasty: ill-conditioned and non-separable. Ill-conditioned means the curvature is wildly different along different directions — a tiny move one way changes f a lot, a big move another way barely registers. Non-separable means the variables are coupled, so the directions that matter are some tilted combination of coordinates, not the axes themselves. Geometrically: the contour lines of f are long, thin, tilted ellipsoids.

So how do I search such a thing with only ranked function values? The robust thing to do in a rugged, possibly-discontinuous landscape is to sample stochastically rather than trust any local model. Carry a probability distribution over R^n, draw a handful of candidate points from it, evaluate them, keep the good ones, and move the distribution toward where the good ones were. That's the evolution-strategy idea — Rechenberg in '73, Schwefel in '81 built exactly this loop: sample λ offspring, select the best μ, recombine into the next center. Now which distribution? Given a mean and a covariance, the multivariate normal has the largest entropy of any distribution on R^n, and it distinguishes no coordinate direction over another. So if all I'm willing to commit to is a center and some second moments, the Gaussian is the honest, least-presumptuous choice. Good — the search distribution is N(m, C), and a step is x = m + (a normal vector).

The naive version is isotropic: N(m, σ²I). One scalar σ, and the equal-density surfaces are spheres. Let me stress-test that on an ill-conditioned quadratic, f(x) = Σ h_i x_i² with the h_i spread over, say, six orders of magnitude. A spherical cloud of samples pokes out equally in every direction. But the only σ that doesn't overshoot the steep direction (the big h_i) is tiny — and that same tiny σ makes progress along the flat direction (the small h_i) glacial. The convergence rate ends up tied to the condition number; crank the conditioning and the sphere chokes. And non-separability is even more damning: the productive direction is a rotated mix of coordinates, and a sphere — having no preferred direction at all — can never concentrate its samples along it. The shape of my sampling cloud has to match the shape of the landscape, and a sphere matches almost nothing.

What shape *should* it be, then? Take the convex-quadratic f(x) = ½xᵀHx. If I sample from N(m, C) with C = H⁻¹, what happens? The inverse Hessian is exactly the linear map that turns the tilted ellipsoid back into a circle: ½xᵀHx under the change of variables that whitens H becomes ½ a plain sum of squares. So sampling with C = H⁻¹ on the ellipsoid behaves like sampling with C = I on the sphere. The optimal covariance is the inverse Hessian, up to a scalar. That's the whole target in one line: I want C to track H⁻¹. This is the gradient-free, rank-only cousin of a quasi-Newton method — it learns the metric of the landscape instead of being told it.

And there's a cleaner way to say why this is the right object. For any full-rank matrix A, ½(Ax)ᵀ(Ax) = ½xᵀ(AᵀA)x. So choosing a covariance is choosing an affine transformation of the search space — the two are the same act. If I can adapt an arbitrary normal distribution, I'm implicitly learning a general linear re-encoding of the problem, and if I learn it well I get invariance to that linear transformation for free. That invariance is the prize: identical behavior across a whole class of rotated, rescaled problems, so any success generalizes.

So the question collapses to: how do I adapt C (and the scale, and the center) toward the right ellipsoid, using nothing but which points ranked well?

Let me look at what people already tried, because I want to know exactly where it breaks. Schwefel's approach was *mutative* self-adaptation: don't just mutate the object variables, also mutate the strategy parameters — the step-sizes, the orientation — generate each offspring using its mutated strategy parameters, and let selection on the object variables implicitly favor the strategy parameters that produced winners. Elegant in spirit. But stare at the mechanism and three things go wrong. First, the selection of a strategy-parameter setting is *indirect*: a setting gets credited only because the one individual it happened to spawn got selected, and that individual's success is mostly about *where* it landed, not *how* it was sampled — so the signal actually steering the covariance is buried in noise. Second, there's a built-in conflict in the mutation strength on the strategy level: to make two competing settings produce a *visible selection difference* you need a fairly large mutation on the strategy parameters, but the strength that gives an *optimal change rate* of those parameters is smaller — and the discrepancy widens with dimension and with the number of strategy parameters, which for a full covariance is O(n²). Third, the only knob that tunes the change rate down to something reliable is the parent number: more parents, heavier recombination, smaller per-generation change. So a working mutative scheme needs a population that scales linearly with the number of strategy parameters. That's expensive, and it's fatal if I want small populations and fast convergence. There's even the chronic symptom that mutative global-step control adapts σ *too small*, because maximizing the probability of being selected is not the same as maximizing progress.

So mutation-then-selection on the strategy parameters is the wrong lever — it's noisy, it tangles the change rate with the mutation strength, and it forces big populations. What if I drop the indirection entirely? The point of the whole exercise is: I want to *reproduce the mutation steps that just got selected*. Mutative control pursues that goal through a fog of indirect selection. Let me pursue it directly — take the steps that selection just favored and bend the distribution so those exact steps become more likely next time. Cut out the mutate-and-hope. That single move — directly raising the likelihood of the selected steps rather than indirectly selecting strategy parameters — is what I'll build on.

Before C, the easy piece: the center. Each generation I rank the offspring x_{1:λ}, …, x_{λ:λ} by f and move the mean to a weighted average of the best μ:

  m ← Σ_{i=1}^μ w_i x_{i:λ},  w_1 ≥ … ≥ w_μ > 0, Σ w_i = 1.

Truncation selection (μ < λ) plus weighting; equal weights w_i = 1/μ is just the mean of the best μ. Why weight at all rather than take the single best? Because averaging independent good samples cancels the noise in their positions — the more samples I pool, the more the recombined center sits at the genuine trend rather than at one lucky draw. How much noise reduction do the weights buy me? Averaging μ independent samples cuts variance by μ for equal weights; for general weights the factor is

  μ_eff = (Σ w_i)² / Σ w_i² = 1 / Σ w_i²,

which sits between 1 and μ and equals μ for equal weights. So μ_eff is the *effective* number of samples I'm actually pooling — the amount of information in the recombination. I'll keep this number around; it's going to set every timescale in the algorithm, because everything downstream is limited by how much information one generation supplies. I'll also write the mean update as a displacement, m ← m + c_m Σ w_i (x_{i:λ} − m), with c_m = 1 by default; that form will be convenient when I build the paths.

Now C. Start from the dream case: suppose one generation's sample is rich enough to just estimate the covariance I want and resample from it. The selected best μ give me the steps y_i = (x_{i:λ} − m)/σ, and I can form

  C_μ = Σ_{i=1}^μ w_i (x_{i:λ} − m)(x_{i:λ} − m)ᵀ / σ² = Σ w_i y_i y_iᵀ.

The reference mean decides the behavior. I subtracted the *old* center m — the actual mean the offspring were sampled around. So C_μ is the covariance of the selected *steps*. Compare the obvious alternative, the one the estimation-of-distribution crowd uses (EMNA, Larrañaga et al. 2001/2002, and the continuous cross-entropy method): fit a Gaussian to the μ best points around *their own* sample mean. That estimates the spread *within* the selected cloud. And its reference mean is by construction the minimizer of that spread, so it systematically *shrinks* the variance — and on a slope it shrinks exactly along the gradient direction, geometrically fast, every generation. That's a recipe for premature convergence: the cloud collapses in the very direction I want to keep moving. Subtract the old mean instead and the opposite happens — on a slope, the selected steps point downhill and C_μ *grows* the variance along the productive direction. Same μ points, opposite fate, decided entirely by the reference. I want the old mean. Reproduce successful *steps*, not the shape of the selected blob.

But "rich enough to estimate C from one generation" is a fantasy for the populations I want. To re-estimate C reliably on the sphere — say condition number under 10 for the selected-step covariance estimate — one generation needs μ_eff on the order of 10n. That is already a population linear in dimension, while fast local search wants λ only logarithmic in n, hence μ_eff ≈ λ/4 and small. With such small μ_eff, C_μ from a single generation is too noisy; its rank is only min(μ, n), and it cannot be trusted as the whole covariance.

The fix is the same one that rescues any noisy per-step estimate: don't trust one generation, accumulate. Average the C_μ's over generations. But weight recent generations more, because the landscape's local geometry drifts as I move — so exponential smoothing with a learning rate c_μ:

  C ← (1 − c_μ) C + c_μ Σ_{i=1}^μ w_i y_i y_iᵀ.

This is the rank-μ update. Each generation it nudges C toward the cloud of selected steps; the sum of outer products has rank min(μ, n), hence the name. The 1/c_μ is a backward time horizon — roughly the number of past generations contributing about 63% of C (expand the recursion: C is (1−c_μ)^{g+1} C⁰ + c_μ Σ (1−c_μ)^{g−i} C_μ^{(i)}, and (1−c_μ)^{Δg} ≈ 1/e gives Δg ≈ 1/c_μ). For this average to be reliable I want the horizon to be about n²/μ_eff generations' worth of information, which pins c_μ ≈ μ_eff/n². Small enough to be stable, large enough to track. Good for large populations, where each generation already carries a lot.

But for *small* populations — the regime I actually care about for speed — even rank-μ is starved: with μ_eff ≈ 1, each generation hands C essentially a single noisy outer product. Can I squeeze more out of a single step? Take the extreme, μ = 1, and just keep the best step y each generation:

  C ← (1 − c_1) C + c_1 y yᵀ.

Adding y yᵀ tilts the ellipsoid to make y more likely — y yᵀ is precisely the rank-one "line distribution" that generates the vector y with maximum likelihood among all zero-mean normals. Iterate it and the distribution aligns to the recent run of selected steps. Fine. But something is being thrown away, and it took me a second to see it: y yᵀ = (−y)(−y)ᵀ. The outer product is blind to the *sign* of the step. If my last six steps all marched in the same direction — a clear signal that there's a long productive axis there — the rank-one update treats that the same as six steps ricocheting back and forth along that axis. The sign, the *correlation between consecutive steps*, is exactly the information that tells me there's a persistent direction, and squaring kills it.

So I need to feed the covariance update something that *remembers* the sign and the cross-generation correlation. Don't update from a single step — update from the *path* the mean has been tracing. Cumulate the successive mean-displacements into a running vector, the evolution path:

  p_c ← (1 − c_c) p_c + (normalization) · (m_new − m_old)/σ.

Now if consecutive steps line up, p_c grows long and points along the persistent axis; if they cancel, p_c stays short. Update C from p_c instead of from a bare step:

  C ← (1 − c_1) C + c_1 p_c p_cᵀ.

This is the rank-one update with cumulation. What's the right normalization on the new term? I want a sanity anchor: under *random* selection (no real signal), the path should be a stationary, unbiased object — it shouldn't drift or inflate. If each incoming step is distributed N(0, C) and I want p_c to also be distributed N(0, C) in steady state, I need the two mixing coefficients to combine like a unit-variance convex blend: (1 − c_c)² + (coef)² = 1, so coef = √(c_c(2 − c_c)). And the weighted average of μ_eff independent steps has covariance C/μ_eff, so to restore unit scale I multiply by √μ_eff. Hence the normalization √(c_c(2 − c_c) μ_eff), giving p_c ~ N(0, C) under random selection — stationary, exactly as I wanted. (Check the degenerate case: c_c = 1, μ_eff = 1 makes the factor 1 and p_c = (x_{1:λ} − m)/σ, the single step. Consistent.)

Suppose there really is a long axis and consecutive steps are positively correlated. The cumulation sum Σ(1−c_c)^i → 1/c_c, while perfectly anti-correlated steps give an alternating sum → 1/(2−c_c). Multiply by the √(c_c(2−c_c)) applied to each input and the path length gets modulated by up to √((2−c_c)/c_c) ≈ 1/√c_c for correlated steps. So even though c_1 ≈ 2/n² is a glacial learning rate, the path being up-to-1/√c_c times longer acts like a learning-rate boost of the same factor along the true axis — and with the horizon 1/c_c chosen between √n and n, the number of evaluations to learn a long cigar-like axis comes out O(n) instead of O(n²). That's the whole reason cumulation exists: it lets a tiny, safe learning rate still adapt a dominant direction fast, precisely in the small-population regime where rank-μ alone is too data-starved.

Now combine the two. Rank-μ shines in large populations (lots of per-generation information, fills out the whole matrix); rank-one with the path shines in small populations (exploits cross-generation correlation to nail the dominant direction). They're complementary, so add them, each with its own learning rate:

  C ← (1 − c_1 − c_μ Σw_i) C + c_1 p_c p_cᵀ + c_μ Σ w_i y_i y_iᵀ,

with c_1 ≈ 2/n² and c_μ ≈ μ_eff/n². The decay coefficient is just whatever keeps the total weight at one. Setting c_1 = 0 recovers pure rank-μ, c_μ = 0 recovers pure rank-one — they nest cleanly.

In the coordinate system where the current C is the identity — substitute z_i = C^{−1/2} y_i — the rank-μ update reads C^{1/2}(I + c_μ Σ w_i (z_i z_iᵀ − I)) C^{1/2}. So I'm pushing the identity in the direction Σ w_i (z_i z_iᵀ − I): grow variance where the *whitened* selected steps landed, shrink it elsewhere, all measured against the current metric. That is exactly an ascent step in the space of distributions, measured in the distribution's own (Fisher) metric — a natural-gradient step on the expected fitness with respect to the Gaussian's parameters. I didn't set out to do information geometry; I set out to reproduce selected steps with the least-presumptuous estimator, and out fell the natural gradient. Which is reassuring: it means rank-μ isn't a heuristic, it's the steepest-ascent direction under the only metric that's invariant to how I happen to parameterize the Gaussian. The whitening C^{−1/2} sandwiching the update is the inverse-Fisher factor.

I deliberately split off the overall scale σ from C — sampling x = m + σ·N(0,C) with C kept at, say, unit determinant-ish scale, σ carrying the magnitude. Why not just let C absorb the scale too? Two reasons, and they're about *timescales*. First, the optimal overall step length depends on μ (or μ_eff) — roughly the optimal σ on the sphere is proportional to μ_eff·√f/n — and the rank-μ/rank-one update simply cannot produce that μ_eff-dependence; it adapts directions, not a population-tied global magnitude. Second, and worse, even if it could, C's largest safe learning rate (∼1/n²) is far too slow: to stay competitive on the sphere the overall step length has to be able to change on a timescale proportional to n, but C adapts on a timescale of n²/μ_eff. If I yoke the scale to the matrix, the scale can't keep up and the search either stalls or diverges. So σ needs its own, fast, dedicated controller.

How do I control σ from ranked values alone? I have the evolution path idea already, so reach for it again — but I have to be more careful this time, because I'm going to compare a *length* against a fixed threshold, and the plain path p_c is distributed N(0, C), whose expected length depends on direction. I need a path whose expected length is the *same in every direction* so one threshold works. Whiten it: build a *conjugate* path using C^{−1/2} on the mean step,

  p_σ ← (1 − c_σ) p_σ + √(c_σ(2 − c_σ) μ_eff) · C^{−1/2} (m_new − m_old)/σ.

The transformation C^{−1/2} = B D^{−1} Bᵀ (from the eigendecomposition C = B D² Bᵀ) does exactly the right thing read right-to-left: Bᵀ rotates the distribution's principal axes onto the coordinate axes, D^{−1} rescales every axis to unit length, B rotates back so the principal axes themselves aren't permanently rotated and successive whitened steps stay comparable. After this, under random selection p_σ ~ N(0, I) regardless of the sequence of C's, so its expected length is a direction-independent constant, χ_n = E‖N(0,I)‖ = √2 Γ((n+1)/2)/Γ(n/2) ≈ √n.

Now the decision rule writes itself from the geometry of the path. If the cumulated path is *long*, the recent steps were pointing the same way — they were correlated, I was making consistent progress and could have covered the same ground with fewer, larger steps, so σ is too small: increase it. If the path is *short*, the steps were cancelling each other — anti-correlated, oscillating, overshooting — so σ is too big: decrease it. The neutral reference is random selection, where successive steps are independent, roughly perpendicular, and the path has its expected length χ_n. So I compare ‖p_σ‖ to χ_n: longer → grow σ, shorter → shrink σ, equal → leave it. Multiplicatively, to keep σ positive and the update symmetric on the log scale:

  σ ← σ · exp( (c_σ/d_σ)(‖p_σ‖/χ_n − 1) ).

Damping d_σ ≈ 1 controls the magnitude. I can also compare the squared length ‖p_σ‖² to its expectation n, giving σ ← σ·exp((c_σ/2d_σ)(‖p_σ‖²/n − 1)); that behaves almost the same and has the exact neutral expectation E‖p_σ‖² = n under random selection. Either way, with no real signal the scale neither inflates nor collapses, which is the stationarity property I insisted on for the path. A bias either way would risk divergence or premature convergence whenever selection pressure drops, so this neutrality is load-bearing, not cosmetic. The time constant 1/c_σ I keep between √n and n so σ can move fast enough even with small populations. As a bonus, this whole construction makes successive mean-steps approximately C⁻¹-conjugate — the search marches in conjugate directions, like a conjugate-gradient method falling out of the bookkeeping.

The mean and the σ update and the selection all touch f only through the *ranking* of the offspring — never the magnitudes — so the whole thing is invariant to any strictly monotonic transformation of the f-value. And because C is adapting toward the inverse Hessian / the right affine encoding, with C initialized to match a given linear transformation the search behaves identically on f(x) and on f(Ax): rotation invariance, scale invariance, full affine invariance of the search space. That is exactly the property that lets it handle an ill-conditioned, non-separable problem that an isotropic ES can't touch — it learns the transformation that turns the tilted ellipsoid back into a sphere, and then it's just optimizing a sphere.

Now the constants, so the thing actually runs. Population and weights, tuned (on the sphere, but they transfer): λ = 4 + ⌊3 ln n⌋, μ = ⌊λ/2⌋, raw weights w_i' = ln((λ+1)/2) − ln i (a gently decreasing, log-shaped profile — best gets most, with diminishing emphasis), the positive ones normalized to sum to 1 so μ_eff comes out around λ/4. The timescales all read off as 1/c ~ horizon: c_σ = (μ_eff + 2)/(n + μ_eff + 5); c_c = (4 + μ_eff/n)/(n + 4 + 2μ_eff/n) (so 1/c_c lands between √n and n); c_1 = 2/((n+1.3)² + μ_eff) ≈ 2/n²; c_μ = min(1 − c_1, 2(μ_eff − 2 + 1/μ_eff)/((n+2)² + μ_eff)) ≈ μ_eff/n², capped so the two C-learning-rates don't exceed 1. In the minimal implementation, damping is d_σ = 2μ_eff/λ + 0.3 + c_σ and the step-size update uses the squared path length with the exponent capped above by 1. A Heaviside switch h_σ *stalls* the rank-one path update when ‖p_σ‖ is unusually large, which happens when σ is far too small and the path inflates; otherwise the axes of C would balloon in a near-linear regime. In the squared form it checks (‖p_σ‖²/n)/(1−(1−c_σ)^{2(g+1)}) < 2 + 4/(n+1). There's also an "active" refinement where the worst-ranked steps enter the rank-μ sum with *negative* weights — shrinking variance along directions selection disfavored — with the negative weights rescaled so C stays positive-definite and the net decay on C is zero; it sharpens adaptation but the positive-weight version above is the core.

So the algorithm, every generation: sample x_k = m + σ B D z_k with z_k ~ N(0,I) (since B D z gives N(0,C)); evaluate and rank; move the mean to the weighted best; update the conjugate path p_σ and from its squared length update σ; update the path p_c and combine rank-one (p_c p_cᵀ) with rank-μ (Σ w_i y_i y_iᵀ) into C; repeat. Let me write it grounded in a clean implementation.

```python
import numpy as np


class CMAES:
    """(mu/mu_w, lambda)-CMA-ES for minimizing a black-box f: R^n -> R."""

    def __init__(self, x0, sigma0, max_evals=None, target=None):
        self.xmean = np.asarray(x0, dtype=float)
        self.sigma = float(sigma0)
        n = self.N = len(self.xmean)
        self.max_evals = (int(max_evals) if max_evals is not None
                          else int(100 * (4 + int(3 * np.log(n)))
                                   + 150 * (n + 3)**2 * np.sqrt(4 + int(3 * np.log(n)))))
        self.target = target

        # --- strategy parameters (tuned defaults; all are ~1/timescale) ---
        self.lam = 4 + int(3 * np.log(n))          # population size
        self.mu = self.lam // 2                      # number of parents

        weights = np.zeros(self.lam)
        weights[:self.mu] = np.log(self.lam / 2 + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights[:self.mu] /= np.sum(weights[:self.mu])      # positive weights sum to 1
        self.weights = weights
        self.mueff = 1.0 / np.sum(weights[:self.mu]**2)     # effective sample size

        # learning rates / time constants
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)          # for sigma path
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)  # for C path
        self.c1 = 2 / ((n + 1.3)**2 + self.mueff)                  # rank-one rate
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff)
                         / ((n + 2)**2 + self.mueff))               # rank-mu rate
        self.damps = 2 * self.mueff / self.lam + 0.3 + self.cs     # sigma damping

        # --- dynamic state ---
        self.pc = np.zeros(n)        # evolution path for C  (carries the sign)
        self.ps = np.zeros(n)        # conjugate (whitened) path for sigma
        self.C = np.eye(n)           # covariance: the learned metric ~ inverse Hessian
        self.counteval = 0
        self.fitvals = np.array([])
        self.best_x = self.xmean.copy()
        self.best_f = np.inf

    def ask(self):
        # eigendecomposition C = B D^2 B^T; sample x = m + sigma * B D z, z~N(0,I)
        D2, self.B = np.linalg.eigh(self.C)
        self.D = np.sqrt(D2)
        self.invsqrtC = (self.B * (1 / self.D)) @ self.B.T   # C^{-1/2}
        Z = np.random.randn(self.lam, self.N)
        Y = Z @ (self.B * self.D).T               # Y[k] ~ N(0, C)
        return self.xmean + self.sigma * Y        # X[k] ~ N(m, sigma^2 C)

    def tell(self, X, fitnesses):
        self.counteval += len(fitnesses)
        order = np.argsort(fitnesses)             # rank by f -> rank invariance
        X = np.asarray(X, dtype=float)[order]
        self.fitvals = np.asarray(fitnesses, dtype=float)[order]
        if self.fitvals[0] < self.best_f:
            self.best_f = float(self.fitvals[0])
            self.best_x = X[0].copy()
        xold = self.xmean.copy()

        # move the mean: weighted average of the best mu steps
        self.xmean = self.weights[:self.mu] @ X[:self.mu]
        ymean = (self.xmean - xold) / self.sigma
        Y = (X - xold) / self.sigma                # selected steps around the old mean

        # conjugate path for sigma: whiten the mean step with C^{-1/2}
        self.ps = ((1 - self.cs) * self.ps
                   + np.sqrt(self.cs * (2 - self.cs) * self.mueff)
                     * (self.invsqrtC @ ymean))
        # stall the C-path if ps is abnormally long (sigma far too small)
        ps2 = float(self.ps @ self.ps)
        hsig = ((ps2 / self.N)
                / (1 - (1 - self.cs)**(2 * self.counteval / self.lam))
                < 2 + 4 / (self.N + 1))
        # evolution path for C: cumulate signed mean steps
        self.pc = ((1 - self.cc) * self.pc
                   + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * ymean)

        # covariance update = rank-one (path) + rank-mu (selected steps)
        c1a = self.c1 * (1 - (1 - float(hsig)**2) * self.cc * (2 - self.cc))
        rank_mu = (Y * self.weights[:, None]).T @ Y
        self.C = ((1 - c1a - self.cmu * np.sum(self.weights)) * self.C
                  + self.c1 * np.outer(self.pc, self.pc)   # rank-one
                  + self.cmu * rank_mu)                    # rank-mu

        # step-size: squared path length is compared to E||N(0,I)||^2 = N
        exponent = min(1.0, (self.cs / self.damps) * (ps2 / self.N - 1) / 2)
        self.sigma *= np.exp(exponent)

    def stop(self):
        reasons = {}
        if self.counteval >= self.max_evals:
            reasons["max_evals"] = self.max_evals
        if self.target is not None and self.best_f <= self.target:
            reasons["target"] = self.target
        if self.fitvals.size and self.fitvals[-1] - self.fitvals[0] < 1e-12:
            reasons["tolfun"] = 1e-12
        if np.linalg.cond(self.C) > 1e14:
            reasons["condition"] = np.linalg.cond(self.C)
        if self.sigma * np.sqrt(np.max(np.diag(self.C))) < 1e-11:
            reasons["tolx"] = 1e-11
        return reasons


def optimize(f, x0, sigma0, max_evals=None, target=None):
    es = CMAES(x0, sigma0, max_evals=max_evals, target=target)
    while not es.stop():
        X = es.ask()
        es.tell(X, [f(x) for x in X])
    return es.best_x if es.best_f < f(es.xmean) else es.xmean
```

Causal chain, start to finish: I need to minimize an ill-conditioned, non-separable black box from ranked values, so I sample a Gaussian whose ideal shape is the inverse Hessian; I move its mean to the weighted-best to follow the descent; I learn its covariance toward the successful *steps* (old-mean reference, so variance grows in good directions instead of collapsing) via rank-μ for population information and a sign-preserving evolution path for rank-one, so the metric is learned fast even with small populations; and I control the overall scale separately and faster by whitening the path and comparing its squared length to what pure chance would produce — long means increase, short means decrease. The covariance learning is a natural-gradient step on the Gaussian; the scale control keeps successive steps conjugate and stationary under no signal; and the rank-only, affine-adapting design buys invariance to monotonic f-transforms and to rotation and scaling of the search space — which is exactly what turns a hopeless tilted ellipsoid into an easy sphere.
