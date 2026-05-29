OK, let me think this through from scratch. I have two generative models on my desk that, when I squint, are doing the same thing. One is the noise-conditional score network: take data, add Gaussian noise at a ladder of standard deviations σ_1 < σ_2 < ... < σ_N, train a single network to estimate the score ∇ log p_σ at every level, and sample by annealed Langevin dynamics walking down the ladder. The other is the diffusion probabilistic model: a discrete Markov chain that nibbles the signal down step by step, x_i = sqrt(1-β_i) x_{i-1} + sqrt(β_i) z, with a learned variational reverse chain trained by a reweighted ELBO, sampled ancestrally. Different forward process, different loss, different sampler. And yet — both losses, when you write them out, are weighted sums of denoising score matching terms, and the optimum of each is the score of the noised data. The first uses weight σ_i², the second uses 1-α_i, and in both cases that weight is proportional to one over the expected squared norm of the conditional score. That's too much coincidence. They're learning the same object through two costumes.

So the real question isn't "which costume is better." It's: what is the thing underneath, and is the discreteness — the hand-picked ladder of scales, the per-scale loss, the bespoke sampler — actually load-bearing, or is it an artifact of being forced to write everything as a finite chain?

Let me poke at the discreteness first, because that's where the arbitrariness lives. Why N scales? Why geometric spacing for one and a near-arithmetic β schedule for the other? If I want twice as many sampling steps I have to interpolate between scales in some ad-hoc way. Every one of these is a knob I'd rather not have to justify. The cleanest thing I can imagine is: don't pick a number of scales at all. Let the noise grow *continuously*. Index everything by a real time t, and let the perturbed distribution be a continuum p_t that slides from the data distribution at t=0 to some simple prior at t=T as I keep injecting noise. If the discrete ladders are really discretizations of one smooth process, then the smooth process is the honest object and the ladders are just two ways of sampling it on a grid.

A continuous-time process driven by noise — that's a stochastic differential equation. Write it in the most general Itô form I can get away with for now,

  dx = f(x, t) dt + g(t) dw,

where w is standard Brownian motion, f is a drift, g is a scalar diffusion coefficient. (I'll keep g a scalar for cleanliness; I can generalize to a matrix G(x,t) later and the algebra survives.) The whole forward process has *no trainable parameters* — it's a fixed prescription that diffuses any data point toward noise. Let p_t(x) be the marginal density of x(t), and p_{0t}(x(t)|x(0)) the transition kernel. Pick f and g so that p_T is something I can sample trivially, like a fixed Gaussian. Good. That's the "easy direction": creating noise from data is easy.

Now the hard direction. I have x(T) from the prior; I want x(0) from the data. I want to *run the SDE backwards*. My instinct is that reversing a stochastic process is hopeless — noise is irreversible, you can't un-stir. But let me not trust the instinct; let me ask what's actually known.

And here's the thing I half-remembered and now go check: a diffusion run backwards in time is still a diffusion. Anderson worked this out in 1982. If the forward process is dx = f dt + g dw, then the time-reversed process satisfies

  dx = [f(x, t) - g(t)² ∇_x log p_t(x)] dt + g(t) dw̄,

with w̄ a Wiener process flowing backward from T to 0 and dt an infinitesimal *negative* step. Stare at this. The diffusion coefficient is unchanged — same g. The drift picks up exactly one new term, -g² ∇_x log p_t(x). And that new term is the *score* of the marginal at time t. That is the only thing in the reverse process that depends on the data distribution. Everything else (f, g) I chose myself and know in closed form.

So the entire problem collapses to: estimate ∇_x log p_t(x) for all x and all t. If I have that function, I plug it into the reverse-time SDE and integrate from T back to 0 with any numerical solver, and out comes a sample from p_0 = p_data. The discrete ladder was never the point. The point was always: learn the time-indexed score field, then reverse a diffusion. The score is the bridge.

Why does that -g²∇log p term do the un-stirring, intuitively? The forward SDE has the diffusion term g dw smearing probability mass outward; the marginal p_t spreads and flattens. Running backward, I need to *concentrate* mass back toward where the data was. The score ∇log p_t points uphill, toward higher density — toward the data manifold. The notation is easy to misread: when I sample backward, dt is negative, so the contribution (-g²∇log p_t)dt is actually +g²|dt|∇log p_t, an uphill push whose strength matches the smearing coefficient. It's not magic; it's the gradient of the log-density doing exactly the job its name promises.

Now I have to actually get the score. I can't compute ∇log p_t directly — p_t is a giant integral over the data convolved with the transition kernel, no closed form. But I don't need it in closed form; I need a network s_θ(x, t) that matches it, and I have a way to train such a thing: score matching. The naive Fisher-divergence objective E_{p_t(x)} ||s_θ(x,t) - ∇log p_t(x)||² still has the intractable ∇log p_t inside. The trick that rescued the discrete models is Vincent's denoising identity, and it generalizes verbatim to continuous time. Minimizing the divergence against the *marginal* score is equivalent, up to a constant independent of θ, to regressing against the *conditional* score of the transition kernel:

  θ* = argmin_θ E_t { λ(t) · E_{x(0)} E_{x(t)|x(0)} [ ||s_θ(x(t), t) - ∇_{x(t)} log p_{0t}(x(t)|x(0))||² ] },

with t drawn uniformly on [0,T], x(0) from the data, x(t) from the transition kernel, and λ(t) a positive weight I'll pin down in a moment. Why is this legitimate — why can I swap the marginal score for the conditional one? Because of how the cross term works out. Expand the marginal objective: the only θ-dependent pieces are ||s_θ||² and the inner product -2⟨s_θ, ∇log p_t⟩. Write ∇log p_t(x̃) = ∇log ∫ p_data(x) p_{0t}(x̃|x) dx = [∫ p_data(x) ∇_x̃ p_{0t}(x̃|x) dx]/p_t(x̃) = ∫ p_data(x) p_{0t}(x̃|x)/p_t(x̃) · ∇log p_{0t}(x̃|x) dx, which is exactly E_{x(0)|x̃}[∇log p_{0t}(x̃|x(0))]. So when I take E_{p_t(x̃)} of the cross term, the p_t(x̃) cancels and the inner product against the marginal score becomes the inner product against the conditional score averaged over the joint. The two objectives differ only by a θ-free constant. The conditional score I *can* compute, because I get to choose the transition kernel.

Why is the conditional score trivial? If the kernel is Gaussian, p_{0t}(x(t)|x(0)) = N(x(t); μ_t x(0), σ_t² I), then x(t) = μ_t x(0) + σ_t z with z ~ N(0,I), and ∇_{x(t)} log p_{0t} = -(x(t) - μ_t x(0))/σ_t² = -z/σ_t. The regression target is just minus the noise I added, scaled. So as long as my SDE produces a Gaussian transition kernel, training is: sample t, sample data x(0), sample z, form x(t) = μ_t x(0) + σ_t z, and push s_θ(x(t),t) toward -z/σ_t. That is the whole training loop.

When *is* the kernel Gaussian? When the drift f is affine in x. Then the mean and covariance of p_{0t} obey linear ODEs with closed-form solutions — standard linear-SDE theory. So I'll restrict my SDEs to affine drift, get Gaussian kernels for free, and never have to simulate the forward process to train. (For non-affine drift I'd lose the closed form; then I'd fall back to sliced score matching, which only needs s_θ and a random projection v: minimize E[½||s_θ||² + vᵀ∇s_θ v] with x(t) obtained by actually simulating the SDE. Good to know there's an escape hatch, but affine is where I want to live.)

Now, λ(t). Across times t the conditional score -z/σ_t has wildly different magnitudes — at small noise σ_t is tiny so the target is huge, at large noise it's small. If I weight all t equally the small-noise terms dominate the loss and the network ignores the rest. The fix is the same one both discrete models stumbled onto: weight inversely to the expected squared target, λ(t) ∝ 1/E||∇log p_{0t}(x(t)|x(0))||² = 1/E||z/σ_t||² ∝ σ_t². With λ(t) = σ_t² the per-time loss becomes E||σ_t s_θ + z||², which is order-one at every t. And — there it is — σ_i² and (1-α_i) were *this* weight, sampled on the two grids. The discrete models were doing the right thing for a reason neither stated cleanly; in continuous time the reason is just "make the loss scale-invariant in t."

So the framework is complete in outline: pick an affine-drift SDE, train s_θ by denoising score matching with the σ_t² weight, then integrate the reverse-time SDE. But "pick an SDE" is doing a lot of work. Which SDE? Let me derive the two I already have in costume and see what they become.

Start with the additive-noise ladder. The score-network forward process is x perturbed by N(0, σ²I). I can realize the marginal at scale σ_i as a Markov chain that only ever *adds* noise: x_i = x_{i-1} + sqrt(σ_i² - σ_{i-1}²) z_{i-1}, with σ_0 = 0, because then Var(x_i - x_0) telescopes to σ_i². Now take N→∞. Let the index become continuous, x(i/N) = x_i, σ(i/N) = σ_i, step Δt = 1/N. The increment is x(t+Δt) = x(t) + sqrt(σ²(t+Δt) - σ²(t)) z ≈ x(t) + sqrt( (d[σ²(t)]/dt) Δt ) z, the approximation holding for small Δt. Matching dx = f dt + g dw, there's no drift, and the noise has variance (d[σ²]/dt) dt per step, so

  dx = sqrt( d[σ²(t)]/dt ) dw.

No drift at all. What does that do to the perturbation variance? With f=0 the mean is frozen at x(0), and the conditional variance is σ²(t)-σ²(0); under the σ_0=0 ladder convention that is σ²(t), and it grows without bound as the largest noise scale grows. The prior at the end is therefore N(0, σ_max² I) after the data scale is negligible. The variance *explodes*. So I'll call this the Variance Exploding SDE. If I want the standard geometric schedule σ(t) = σ_min (σ_max/σ_min)^t on t>0, then d[σ²]/dt = σ²(t) · 2 log(σ_max/σ_min), so g(t) = σ(t) sqrt(2 log(σ_max/σ_min)). (One wrinkle: σ(0) = 0 by the σ_0 = 0 convention, but the geometric limit gives σ(0⁺) = σ_min ≠ 0, so σ(t) isn't differentiable at 0 and the SDE is undefined exactly at t=0. I'll just integrate on [ε, 1] for a tiny ε and use σ(t) itself as the training std, which matches the small-σ_min implementation convention.)

Now the signal-decaying chain. x_i = sqrt(1-β_i) x_{i-1} + sqrt(β_i) z_{i-1}. To take a clean limit I need the β's to scale with the step size, so define β̄_i = N β_i and rewrite x_i = sqrt(1 - β̄_i/N) x_{i-1} + sqrt(β̄_i/N) z_{i-1}. Let β(t) be the continuous limit of β̄_i, Δt = 1/N. Then x(t+Δt) = sqrt(1 - β(t+Δt)Δt) x(t) + sqrt(β(t+Δt)Δt) z. Taylor-expand the square root: sqrt(1 - β Δt) ≈ 1 - ½ β Δt, so x(t+Δt) ≈ x(t) - ½ β(t) Δt x(t) + sqrt(β(t) Δt) z (dropping a β(t+Δt)→β(t) error that's higher order). Matching dx = f dt + g dw,

  dx = -½ β(t) x dt + sqrt(β(t)) dw.

Now there's a drift, -½ β x, pulling x toward the origin — that's the signal decay the sqrt(1-β) scaling was doing. What about the variance now? The drift is affine, so I can use the variance ODE for linear SDEs: dΣ(t)/dt = -2·(½β)Σ + β I = β(t)(I - Σ(t)). Solve it: Σ(t) = I + e^{-∫_0^t β(s)ds}(Σ(0) - I). This is *bounded* for all t, and if I start at Σ(0) = I it stays exactly I forever. The variance is preserved. So this is the Variance Preserving SDE, and its natural prior is N(0, I). For the standard schedule I take β(t) = β_min + t(β_max - β_min). And the closed-form kernel from the linear-SDE solutions is N(x(t); x(0) e^{-½∫β}, (1 - e^{-∫β}) I).

Let me sanity-check the two against their discrete originals numerically — does the continuous kernel's variance actually track the discrete ladder's at N=1000? It does, essentially exactly, for both, and for the VP mean too. So these really are the same processes, just one sampled on a grid and one not. The unification isn't a hand-wave; it's an identity in the limit.

Two SDEs, two priors. The VE one explodes the variance; the VP one pins it. Now I have room to *invent* a third, because I'm no longer constrained to match an existing discrete model — I just need affine drift and a Gaussian kernel that goes to a fixed prior. Is there something between or below VP that might behave better? The thing I'd want to control is how much noise I actually inject, because every bit of injected noise is error the reverse process has to undo. The VP variance is Σ(t) = I + e^{-∫β}(Σ0 - I); the noise it accumulates is governed by that. What if I keep the *same* drift -½βx (so the mean decays identically, e^{-½∫β}), but shrink the diffusion so the variance is provably smaller at every t while still reaching I at the end? Try

  dx = -½ β(t) x dt + sqrt( β(t) (1 - e^{-2∫_0^t β(s)ds}) ) dw.

The extra factor (1 - e^{-2∫β}) is zero at t=0 (so almost no noise injected early, when the signal is intact) and rises to 1 for large t (so asymptotically it behaves like VP). What I care about for training is the perturbation kernel's variance — the spread of x(t) given a fixed clean x(0), i.e. the conditional covariance Σ(t) with Σ(0)=0. With this diffusion that covariance obeys dΣ/dt = -β(t)Σ + β(t)(1 - e^{-2∫β}). Let h(t) = ∫_0^t β. Multiply by the integrating factor e^h: d(Σ e^h)/dt = e^h β (1 - e^{-2h}) = β e^h - β e^{-h}, whose antiderivative is e^h + e^{-h}, so Σ e^h = e^h + e^{-h} + C; the condition Σ(0)=0 fixes C = -2, giving Σ(t) = 1 + e^{-2h} - 2 e^{-h} = [1 - e^{-h}]². Compare to the VP perturbation-kernel variance, which from dΣ/dt = β(1-Σ), Σ(0)=0 is just [1 - e^{-h}]. Since 1 - e^{-h} ∈ [0,1], squaring shrinks it: this process has perturbation variance [1-e^{-h}]² ≤ [1-e^{-h}] = VP's at every single t, with equality only at the endpoints. For an initial data covariance Σ0, the full covariance is e^{-h}Σ0 + [1-e^{-h}]²I, which is still below e^{-h}Σ0 + [1-e^{-h}]I for the VP process and still tends to I as h grows. Less injected noise, same mean dynamics, same endpoint. And this settles the code convention: the marginal routine should return a standard deviation, not a variance. VP returns sqrt(1-e^{-h}) because its kernel variance is 1-e^{-h}; this one returns 1-e^{-h} with no square root because its kernel variance is already [1-e^{-h}]². I'll call it the sub-VP SDE. I expect it to be easier on the reverse process, since the model has less noise to account for.

Now sampling. The lazy answer is: I have a reverse-time SDE, throw a black-box solver at it — Euler–Maruyama, stochastic Runge–Kutta, whatever. That works and is fully general. But I have more information than a generic SDE solver does, and I should use it. Specifically, two extra things.

First extra thing: I don't just have the drift of the reverse SDE; I have the *score itself*, s_θ ≈ ∇log p_t, at every t. That means at any time slice I can run score-based MCMC — Langevin dynamics — to sample from p_t directly, x ← x + ε s_θ(x,t) + sqrt(2ε) z, which converges to p_t. So the solver can use more than the drift: at each time step, take a numerical reverse-SDE step (call it the *predictor*) to move from t to t-Δt, then run a few Langevin steps (call it the *corrector*) to nudge the sample back onto the correct marginal p_{t-Δt}. The predictor advances time; the corrector fixes whatever discretization error the predictor introduced, using information the solver can't see. This is exactly the structure of predictor–corrector methods in numerical continuation, so I'll call it a PC sampler.

And look what falls out of this: the original samplers are the two degenerate corners of PC. The score-network sampler — annealed Langevin with no time-advancing step — is corrector-only (identity predictor). The diffusion sampler — ancestral, no MCMC — is predictor-only (identity corrector). PC isn't a new competitor; it *contains* both and lets me spend compute on whichever side helps. The natural thing to expect is that splitting a fixed compute budget between predictor and corrector beats spending it all on either — which is the comparison I'd want to run.

Two practical details for the corrector. The Langevin step size ε needs to be set across very different noise scales without hand-tuning per scale; I'll fix a target ratio r between the size of the score-driven move and the size of the injected noise — a "signal-to-noise ratio" — and back out ε from it: ε = 2α(r ||z|| / ||s_θ||)², with α=1 for VE and α=1-β_i for the VP-style discrete grid. That keeps the corrector stable as the scale changes. And the predictor: rather than re-derive an ancestral rule for each new SDE, I'll discretize the reverse SDE the *same way* I discretized the forward one — if the forward grid step is x_{i+1} = x_i + f_i + G_i z, the mirrored reverse step is x_i = x_{i+1} - f_{i+1}(x_{i+1}) + G_{i+1}G_{i+1}ᵀ s_θ(x_{i+1}, i+1) + G_{i+1} z. Call that the reverse diffusion predictor; it's automatic for any SDE. And the old ancestral sampler? If I Taylor-expand the diffusion ancestral update 1/sqrt(1-β) = 1 + ½β + o(β) and collect terms, it equals (2 - sqrt(1-β)) x_{i+1} + β s_θ(x_{i+1},i+1) + sqrt(β) z up to o(β) — which is exactly a particular discretization of the reverse VP SDE. So ancestral sampling was a reverse-SDE solver all along; my framework just names it.

Second extra thing, and this is the one that surprises me. I assumed reversing the process *had* to be stochastic — it's a diffusion, you put noise in, you sample noise out. But is there a deterministic process with the same marginals? Let me go to the master equation for how p_t evolves. For dx = f dt + G dw the Fokker–Planck (Kolmogorov forward) equation is

  ∂_t p_t = -Σ_i ∂_{x_i}[f_i p_t] + ½ Σ_{i,j} ∂²_{x_i x_j}[ Σ_k G_{ik}G_{jk} p_t ].

That second-order term is what makes the dynamics stochastic. But I can try to fold it into the first-order term. Pull one derivative out: ½ Σ_{ij} ∂_{x_i} ∂_{x_j}[ (GGᵀ)_{ij} p_t ] = ½ Σ_i ∂_{x_i} [ Σ_j ∂_{x_j}[(GGᵀ)_{ij} p_t] ]. Now work on the inner bracket, Σ_j ∂_{x_j}[(GGᵀ)_{ij} p_t]. Product rule: it's Σ_j (∂_{x_j}(GGᵀ)_{ij}) p_t + Σ_j (GGᵀ)_{ij} ∂_{x_j} p_t. The first piece is p_t times the divergence (∇·[GGᵀ])_i. The second piece is Σ_j (GGᵀ)_{ij} p_t ∂_{x_j} log p_t = p_t (GGᵀ ∇log p_t)_i, using ∂_{x_j} p_t = p_t ∂_{x_j} log p_t. So the inner bracket equals p_t ( ∇·[GGᵀ] + GGᵀ ∇log p_t )_i. Substitute back:

  ∂_t p_t = -Σ_i ∂_{x_i}[f_i p_t] + ½ Σ_i ∂_{x_i}[ p_t (∇·[GGᵀ] + GGᵀ∇log p_t)_i ]
          = -Σ_i ∂_{x_i}{ [ f_i - ½(∇·[GGᵀ] + GGᵀ∇log p_t)_i ] p_t }.

The entire right-hand side is now a single divergence of (drift × p_t). That's a continuity equation — the Liouville equation for a *deterministic* flow with no diffusion term at all. Define

  f̃(x,t) = f(x,t) - ½ ∇·[GGᵀ] - ½ GGᵀ ∇log p_t(x),

and the marginals p_t are reproduced exactly by the ODE

  dx = f̃(x,t) dt.

For my scalar-g case GGᵀ = g²I, its divergence vanishes, and this is just

  dx = [ f(x,t) - ½ g(t)² ∇log p_t(x) ] dt.

Compare to the reverse SDE drift, f - g²∇log p_t. *Same form, half the score coefficient, and no noise.* The factor of ½ isn't a guess — it's exactly what the Fokker-Planck-to-continuity algebra produced. I'll call this the probability flow ODE. It samples from the same p_0 as the SDE (same marginals at every t), but deterministically: fix the initial x(T) and the trajectory is determined.

A deterministic, invertible flow whose only learned ingredient is the score — that's a neural ODE. And neural ODEs come with a gift I couldn't get from the SDE: exact likelihood. The instantaneous change-of-variables formula says along a trajectory of dx = f̃_θ dt, d log p_t(x(t))/dt = -∇·f̃_θ(x(t),t). Integrate from 0 to T:

  log p_0(x(0)) = log p_T(x(T)) + ∫_0^T ∇·f̃_θ(x(t),t) dt,

with x(t) obtained by solving the ODE. log p_T is a known Gaussian; the integral is the only work. The divergence ∇·f̃_θ is a trace of a Jacobian — naively O(d) backprops, too expensive for images. But I can estimate it: ∇·f̃_θ = E_v[ vᵀ (∇f̃_θ) v ] for any v with mean 0 and identity covariance (Skilling–Hutchinson). The quantity vᵀ∇f̃_θ is one vector-Jacobian product, the cost of a single backward pass, and the estimator is unbiased so averaging drives the error to zero. Now I can compute exact bits/dim by running an adaptive ODE solver (RK45) on the augmented system [x; log-density] from data to prior. The score-based model just became likelihood-capable — and I didn't train it by maximum likelihood.

The probability flow ODE buys me three more things almost for free. Because the forward SDE has no trainable parameters and the ODE gives the same trajectory for any model that learns the true score, the encoding x(0) ↦ x(T) is *uniquely identifiable* — two networks with different architectures, trained separately, should map the same image to nearly the same latent. (Two flow models trained the usual way have no reason to agree; this one does, because the latent is defined by the data-independent forward process, not by the network. Worth verifying empirically.) I can also manipulate latents — interpolate, temperature-scale — like any invertible model. And I can sample with a black-box adaptive ODE solver that chooses its own step count, giving me an explicit tolerance dial to trade numerical accuracy against function evaluations.

One more sample-quality detail I shouldn't forget, because it explains an old discrepancy. Samples carry a thin layer of residual noise that's invisible to the eye but poison to FID. A single denoising step at the very end — one predictor step without re-adding noise, i.e. Tweedie — removes it. The additive-noise family had been measuring worse than the diffusion family partly because it skipped this step, so the sampler should include that final mean step.

Finally, conditioning, because the continuous-time view makes it almost trivial. Suppose I want to sample from p_0(x | y) for some side information y — a class label, the known pixels of an image. I need the *conditional* reverse SDE, which by Anderson is the same as before but with the conditional score ∇log p_t(x | y) in place of ∇log p_t(x). And by Bayes on the time-t marginals, p_t(x | y) ∝ p_t(x) p_t(y | x), so

  ∇log p_t(x | y) = ∇log p_t(x) + ∇log p_t(y | x).

The first term is my already-trained unconditional score s_θ. The second is the gradient of a likelihood of y given the noised x — for a class label, train a small time-dependent classifier p_t(y | x(t)) (easy, since I can generate (x(t), y) pairs by noising labeled data and minimizing a sum of cross-entropies across t) and differentiate it; for inpainting/colorization, y is a set of known dimensions and the conditional score can be approximated using only the unconditional model by substituting a forward-noised sample of the known part. The point is I get controllable generation, inverse problems, all of it, from *one* unconditional model plus a cheap conditional term — no retraining of the generator per task.

So the causal chain, start to finish: the discrete noise ladders of both existing model families are discretizations of one continuous forward SDE that has no trainable parameters; reversing that SDE (Anderson) needs only the time-dependent score; the score is learnable by continuous denoising score matching with a σ_t² weight, tractable because affine drift makes the transition kernel Gaussian; the additive-noise ladder is the Variance Exploding SDE and the signal-decay ladder is the Variance Preserving SDE, with a new sub-VP SDE squeezed below VP for less injected noise; sampling is a predictor–corrector loop that contains both old samplers as corners; and the same marginals are carried by a deterministic probability flow ODE that hands me exact likelihood, identifiable latents, and adaptive sampling; with conditioning dropping out of Bayes on the marginals.

```python
import abc
import numpy as np
import torch
from scipy import integrate

class SDE(abc.ABC):
    def __init__(self, N):
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        pass

    def discretize(self, x, t):
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        return drift * dt, diffusion * torch.sqrt(torch.tensor(dt, device=t.device))

    def reverse(self, score_fn, probability_flow=False):
        N, T = self.N, self.T
        sde_fn, discretize_fn = self.sde, self.discretize

        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (
                    0.5 if self.probability_flow else 1.0
                )
                diffusion = 0.0 if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                f, G = discretize_fn(x, t)
                score = score_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * score * (
                    0.5 if self.probability_flow else 1.0
                )
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__(N)
        self.beta_0, self.beta_1 = beta_min, beta_max
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        return -0.5 * beta_t[:, None, None, None] * x, torch.sqrt(beta_t)

    def marginal_prob(self, x, t):
        log_mean = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        n = np.prod(z.shape[1:])
        return -n / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        return torch.sqrt(alpha)[:, None, None, None] * x - x, torch.sqrt(beta)

class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        super().__init__(N)
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), N))

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * torch.sqrt(torch.tensor(
            2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device))
        return torch.zeros_like(x), diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return x, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        n = np.prod(z.shape[1:])
        return -n / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(
            z**2, dim=(1, 2, 3)) / (2 * self.sigma_max**2)

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent = torch.where(
            timestep == 0, torch.zeros_like(t), self.discrete_sigmas.to(t.device)[timestep - 1])
        return torch.zeros_like(x), torch.sqrt(sigma**2 - adjacent**2)

class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__(N)
        self.beta_0, self.beta_1 = beta_min, beta_max
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        discount = 1.0 - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t**2)
        return -0.5 * beta_t[:, None, None, None] * x, torch.sqrt(beta_t * discount)

    def marginal_prob(self, x, t):
        log_mean = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean)[:, None, None, None] * x
        # The kernel variance is (1 - exp(-int beta))^2, so this is the std.
        std = 1.0 - torch.exp(2.0 * log_mean)
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        n = np.prod(z.shape[1:])
        return -n / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0

def get_loss_fn(sde, eps=1e-5, likelihood_weighting=False):
    def loss_fn(model, batch):
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        x_t = mean + std[:, None, None, None] * z
        score = model(x_t, t)
        if likelihood_weighting:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = losses.reshape(losses.shape[0], -1).sum(dim=-1) * g2
        else:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = losses.reshape(losses.shape[0], -1).sum(dim=-1)
        return torch.mean(losses)
    return loss_fn

def reverse_diffusion_predictor(rsde, x, t):
    f, G = rsde.discretize(x, t)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * torch.randn_like(x)
    return x, x_mean

def langevin_corrector(score_fn, sde, x, t, snr, n_steps):
    if hasattr(sde, "alphas"):
        timestep = (t * (sde.N - 1) / sde.T).long()
        alpha = sde.alphas.to(t.device)[timestep]
    else:
        alpha = torch.ones_like(t)
    for _ in range(n_steps):
        grad, noise = score_fn(x, t), torch.randn_like(x)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + step[:, None, None, None] * grad
        x = x_mean + torch.sqrt(step * 2)[:, None, None, None] * noise
    return x, x_mean

def pc_sample(model, sde, shape, snr=0.16, n_steps=1, eps=1e-3, device='cpu'):
    score_fn = lambda x, t: model(x, t)
    rsde = sde.reverse(score_fn)
    x = sde.prior_sampling(shape).to(device)
    timesteps = torch.linspace(1, eps, sde.N, device=device)                # T -> eps
    for i in range(sde.N):
        vec_t = torch.ones(shape[0], device=device) * timesteps[i]
        x, x_mean = langevin_corrector(score_fn, sde, x, vec_t, snr, n_steps)
        x, x_mean = reverse_diffusion_predictor(rsde, x, vec_t)            # predictor
    return x_mean                                                           # final denoising step

def ode_sample(model, sde, shape, eps=1e-3, device='cpu'):
    score_fn = lambda x, t: model(x, t)
    rsde = sde.reverse(score_fn, probability_flow=True)                     # half-score drift, no noise
    def ode_func(t, x_flat):
        x = torch.tensor(x_flat.reshape(shape), dtype=torch.float32, device=device)
        vec_t = torch.ones(shape[0], device=device) * t
        drift, _ = rsde.sde(x, vec_t)
        return drift.detach().cpu().numpy().reshape(-1)
    x0 = sde.prior_sampling(shape).cpu().numpy().reshape(-1)
    sol = integrate.solve_ivp(ode_func, (1, eps), x0, rtol=1e-5, atol=1e-5, method='RK45')
    return torch.tensor(sol.y[:, -1].reshape(shape))

def divergence_fn(fn):
    def div(x, t, noise):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = torch.sum(fn(x, t) * noise)
            grad = torch.autograd.grad(y, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad * noise, dim=tuple(range(1, len(x.shape))))
    return div
```
