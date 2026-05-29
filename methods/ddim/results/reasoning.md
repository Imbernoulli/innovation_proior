The thing that is killing me is the wall clock. I have a denoising model that draws images as good as a GAN, trained by a clean variational objective with none of the adversarial instability, and it is essentially unusable because pulling one image out of it means running a thousand sequential network passes. Twenty hours for fifty thousand small images; a GAN does the same in under a minute with a single pass. At 256×256 it is hundreds of hours. So the question I actually care about is narrow and concrete: can I get a sample out of this exact trained model in twenty or fifty steps instead of a thousand, ideally without retraining anything, because retraining a 256×256 model is itself days of compute.

Let me first be honest about *why* it is a thousand steps, because if the slowness is load-bearing I am wasting my time. The forward process slowly turns an image into Gaussian noise over T steps; the generative process is trained to invert it step by step. The reason T has to be large is that each true reverse conditional q(x_{t-1}|x_t) is only close to Gaussian when the step is small — when very little noise was added between x_{t-1} and x_t. If I take big steps, the reverse conditional is some complicated multimodal thing and a Gaussian model of it is wrong. So large T is not laziness, it is what makes the Gaussian generative model valid. That seems to nail the door shut: the generative chain approximates the reverse of the forward chain, the forward chain needs many steps to stay Gaussian-reversible, therefore the generative chain needs many steps. Slow by construction.

But wait. Let me look at what training actually pinned down, because the chain of "approximates the reverse of" is doing a lot of unexamined work. The objective, after all the variance-reduction algebra, is

  L_γ(ε_θ) = Σ_{t=1}^T γ_t · E_{x_0, ε} [ ‖ ε_θ^{(t)}(√α_t x_0 + √(1−α_t) ε) − ε ‖² ].

Stare at this. Two things jump out and they are the whole game. First, it is a sum of independent per-t terms. If the network does not share weights across t — if ε_θ^{(t)} is its own little function for each t — then minimizing the sum is just minimizing each term separately, and the minimizer of each term does not care what γ_t multiplies it. The optimal ε_θ is *the same function* no matter how I reweight the terms. Second, and this is the one I keep walking past: each term feeds the network √α_t x_0 + √(1−α_t) ε. That is a draw from the *marginal* q(x_t|x_0) = N(√α_t x_0, (1−α_t)I). The loss never once references the *joint* q(x_{1:T}|x_0). It never asks how x_t and x_{t-1} are correlated. It only ever asks the network to denoise a single marginal draw.

So what has the trained network actually committed to? Only the marginals. It is a noise predictor for q(x_t|x_0), for each t, and nothing else. The Markov forward chain — the specific story about x_t depending only on x_{t-1} — was one particular joint that happens to have these marginals. It is an arbitrary choice of how to thread the latents together. The training did not see it. Any *other* joint with the same marginals q(x_t|x_0) = N(√α_t x_0, (1−α_t)I) is, as far as the loss is concerned, an equally valid story, and the same trained network is its solution too.

That reframes everything. The slowness came from "the generative process approximates the reverse of *the Markov chain*." But the network is not wedded to the Markov chain; it is wedded to the marginals. If I can build a *different* inference process — same marginals, different joint — whose corresponding reverse/generative process happens to be short, or deterministic, or both, then the same trained network already optimizes it. No retraining. The constraint I have to respect is exactly one: preserve q(x_t|x_0) = N(√α_t x_0, (1−α_t)I) for every t. Everything else is mine to design.

OK so let me try to build such a family. I want a joint q(x_{1:T}|x_0) whose every marginal is the fixed Gaussian. The cleanest way to control marginals is to specify the process *backwards*, conditioned on x_0, because then I can write down the reverse conditional q(x_{t-1}|x_t,x_0) directly — which is also exactly the object the generative process will mimic. So factor

  q(x_{1:T}|x_0) = q(x_T|x_0) · ∏_{t=2}^T q(x_{t-1}|x_t,x_0),

set q(x_T|x_0) = N(√α_T x_0, (1−α_T)I) to nail the last marginal, and now I need to choose the reverse conditionals q(x_{t-1}|x_t,x_0) so that the *other* marginals come out right too. Make them Gaussian — I want closed forms and I want the eventual KL terms to be tractable. The natural ansatz: the mean is affine in x_0 and x_t, the covariance is some σ_t² I. Write a free coefficient and pin it down by the marginal constraint. Let

  q(x_{t-1}|x_t,x_0) = N( √α_{t-1} x_0 + k_t·(x_t − √α_t x_0)/√(1−α_t) , σ_t² I ),

where I have already guessed the shape: a √α_{t-1} x_0 "signal" piece plus something proportional to the *residual* (x_t − √α_t x_0), which in distribution is √(1−α_t) ε, the noise that took x_0 to x_t. Dividing that residual by √(1−α_t) normalizes it to a unit-ish direction; k_t is the unknown scale I will solve for, and σ_t² is whatever variance I want to leave free.

Now impose the constraint by induction downward from t=T, where the marginal already holds. Suppose q(x_t|x_0) = N(√α_t x_0, (1−α_t)I). Marginalize x_t out of the product:

  q(x_{t-1}|x_0) = ∫ q(x_t|x_0) q(x_{t-1}|x_t,x_0) dx_t.

This is a linear-Gaussian marginalization: x_t is Gaussian, x_{t-1} is Gaussian-given-x_t with a mean affine in x_t, so x_{t-1} is Gaussian and I just propagate. For the mean, substitute E[x_t] = √α_t x_0 into the mean of the conditional:

  mean = √α_{t-1} x_0 + k_t·(√α_t x_0 − √α_t x_0)/√(1−α_t) = √α_{t-1} x_0.

The residual term vanishes outright — because at the mean of x_t the residual is zero — so the mean is √α_{t-1} x_0 regardless of k_t. Good, the mean constraint is automatically satisfied by my choice of writing the conditional mean around (x_t − √α_t x_0). The variance is where k_t earns its keep. Var of x_{t-1} = (conditional variance) + (squared map coefficient)·(variance of x_t):

  Var = σ_t² + (k_t/√(1−α_t))²·(1−α_t) = σ_t² + k_t².

I need this to equal 1−α_{t-1}. So k_t² = 1 − α_{t-1} − σ_t², i.e.

  k_t = √(1 − α_{t-1} − σ_t²).

That is forced. The reverse conditional that preserves the marginals is

  q_σ(x_{t-1}|x_t,x_0) = N( √α_{t-1} x_0 + √(1 − α_{t-1} − σ_t²)·(x_t − √α_t x_0)/√(1−α_t) , σ_t² I ),   (★)

and the induction closes: every marginal is N(√α_t x_0, (1−α_t)I), exactly the fixed ones, with each σ_t free subject to 0 ≤ σ_t² ≤ 1−α_{t-1} so the square root is real. One whole degree of freedom per step survives the marginal-matching, and it is precisely the stochasticity of the reverse conditional. This is a *family* of inference processes, indexed by σ, all sharing the marginals my network was trained on. (And note what σ_t does to the *forward* direction: by Bayes, q_σ(x_t|x_{t-1},x_0) ∝ q_σ(x_{t-1}|x_t,x_0)q_σ(x_t|x_0)/q_σ(x_{t-1}|x_0), which is Gaussian but in general depends on x_0 as well as x_{t-1} — so for σ < the special value the forward process is no longer Markovian. x_t depends on both x_{t-1} and x_0. That is fine; I never needed it to be Markovian, I only needed the marginals.)

Now turn the family into a *generative* process. At sample time I have x_t but not x_0, and (★) needs x_0. But that is exactly what the network gives me. The network predicts the noise ε in x_t = √α_t x_0 + √(1−α_t) ε, and inverting,

  f_θ(x_t) := (x_t − √(1−α_t)·ε_θ(x_t)) / √α_t

is a prediction of x_0 from x_t. So define the generative reverse step by plugging this predicted x_0 into (★): p_θ(x_{t-1}|x_t) = q_σ(x_{t-1}|x_t, f_θ(x_t)). Sampling from that Gaussian, and noticing that (x_t − √α_t f_θ)/√(1−α_t) = ε_θ(x_t) by the very definition of f_θ, the update is

  x_{t-1} = √α_{t-1}·f_θ(x_t)  +  √(1 − α_{t-1} − σ_t²)·ε_θ(x_t)  +  σ_t ε.   (◇)

Three pieces, and each has a reading. √α_{t-1}·f_θ is "jump to where the predicted clean image would sit at level t−1." √(1 − α_{t-1} − σ_t²)·ε_θ is a "direction pointing back toward x_t" — it re-injects, deterministically, exactly the amount of the predicted noise that the marginal at t−1 still wants to carry. And σ_t ε is fresh randomness. The density model can give the t=1 decoder a tiny Gaussian support term N(f_θ(x_1), σ_1² I); the sampler itself takes α_0 = 1, so the final implemented step returns the predicted clean image as σ_1 goes to zero.

Before I get excited, I have to check the thing I've been assuming: that the network trained by L_1 actually optimizes *this* generative process, for any σ I pick. If a different σ needed a different network, the whole no-retraining premise collapses. So write down the variational objective for this generative process and see what it reduces to:

  J_σ(ε_θ) = E_{q_σ}[ log q_σ(x_{1:T}|x_0) − log p_θ(x_{0:T}) ].

Expand using the backward factorization of q_σ and the chain factorization of p_θ. The log q_σ(x_T|x_0) and the prior log p_θ(x_T) terms do not depend on θ, drop them. What is left, grouping the per-step pieces, is — up to θ-independent constants — a sum of KL divergences between the inference reverse conditional and the generative reverse conditional:

  J_σ ≡ Σ_{t=2}^T E_{x_0,x_t}[ KL( q_σ(x_{t-1}|x_t,x_0) ‖ p_θ(x_{t-1}|x_t) ) ] − E[ log p_θ(x_0|x_1) ].

Now the payoff of building p_θ out of the *same* family (★): both arguments of each KL are Gaussians with the *same* covariance σ_t² I. They differ only in their means, and the means differ only in their x_0-slot — q_σ uses the true x_0, p_θ uses f_θ(x_t). A KL between two Gaussians with equal covariance σ²I is just ‖mean difference‖²/(2σ²), so I need the mean difference exactly, not just its direction. The x_t part cancels, but x_0 appears twice in the mean, once in the clean term and once inside the residual. For t>1 the difference is

  μ_q − μ_p = ( √α_{t-1} − √α_t·√(1−α_{t-1}−σ_t²)/√(1−α_t) ) · (x_0 − f_θ(x_t)).

Call that scalar λ_t. It is positive for the variance choices I can use, and even if the reweighting argument only needs positivity, this is the factor I must not drop. The KL term is therefore

  E[ λ_t² · ‖ x_0 − f_θ(x_t) ‖² / (2 σ_t²) ].

For t=1, where the decoder is N(f_θ(x_1), σ_1²I), the same form holds with λ_1 = 1. Now convert x_0 − f_θ to noise space. Both x_0 and f_θ are (x_t − √(1−α_t)·{ε or ε_θ})/√α_t, so

  x_0 − f_θ(x_t) = ( √(1−α_t)·ε_θ(x_t) − √(1−α_t)·ε ) / √α_t = (√(1−α_t)/√α_t)·(ε_θ − ε),

and therefore ‖x_0 − f_θ‖² = ((1−α_t)/α_t)·‖ε − ε_θ‖². Each term of J_σ becomes

  γ_t · E[ ‖ ε − ε_θ^{(t)}(√α_t x_0 + √(1−α_t) ε) ‖² ],

with

  γ_t = λ_t²(1−α_t)/(2 α_t σ_t²),     λ_t = √α_{t-1} − √α_t·√(1−α_{t-1}−σ_t²)/√(1−α_t)   for t>1,

and γ_1 = (1−α_1)/(2α_1σ_1²). If the implementation writes the loss as an average per dimension instead of a summed norm, every γ_t is divided by that dimension; nothing else changes.

That is a per-t noise-prediction MSE. Summing over t,

  J_σ(ε_θ) = L_γ(ε_θ) + C,

with C collecting the θ-independent constants. So J_σ is L_γ for *some* positive γ — and the exact value of γ does not need to match the old Markov-chain variational weights, because of the first structural fact I noticed at the very start: when the per-t parameters are not shared, the optimum of L_γ is the same for every positive weighting. For every strictly positive σ, the minimizer of J_σ is the minimizer of L_1; the deterministic σ=0 endpoint is reached as a limit. The plain unweighted noise-prediction loss that I already trained is a valid surrogate for every member of the family I can sample from. One trained network; a whole continuum of generative processes. This is exactly what I needed.

Now spend the free knob σ. Two endpoints are interesting. Pick σ_t = √((1−α_{t-1})/(1−α_t))·√(1 − α_t/α_{t-1}) and the forward process becomes Markovian again and (◇) collapses to the original ancestral sampler — so the old model is just the σ = this-value member, nothing special. The endpoint I actually want is the opposite extreme: σ_t = 0 for all t. Then the random-noise term in (◇) disappears entirely:

  x_{t-1} = √α_{t-1}·( (x_t − √(1−α_t) ε_θ(x_t)) / √α_t )  +  √(1 − α_{t-1})·ε_θ(x_t).

No randomness injected anywhere (down to a vanishing supporting noise at t=1). x_0 is now a *deterministic* function of x_T. This is an implicit generative model in the GAN/flow sense: samples are a fixed pushforward of the latent x_T through a deterministic procedure. And that single change buys three things at once. The terminal noise x_T is now a real latent code — fix it and you get one reproducible image — so I can interpolate two images by interpolating their x_T (spherically, since these are high-dimensional Gaussian samples), and I can hope to *encode* a real image by running the procedure backward. None of that is possible when every step injects fresh noise, because then the same x_T fans out to many different images and x_T carries almost no information about the result.

But the reason I started all this was speed, and determinism alone does not shorten the chain — (◇) at σ=0 is still written for all T steps. So now use the marginal-only fact a second time, more aggressively. The loss depends on the inference process only through the marginals q(x_t|x_0). Nothing forces the generative process to visit *every* t. I can define the whole construction on a *sub-sequence* τ = (τ_1 < τ_2 < … < τ_S) of [1..T], reusing (★) with the index pairs (τ_i, τ_{i-1}) in place of (t, t−1). The marginal-consistency lemma still holds for those indices — the induction only ever used the marginals at the two endpoints of a step, so it does not care whether the steps are adjacent integers or jumps. And the J_σ = L_γ + C argument still goes through on the sub-sequence (the leftover indices not in τ enter only as star-graph terms in the objective, never in the sampler). So I can train at T = 1000 and *sample* on S ≪ T steps with the same network and no retraining. The per-step update on the trajectory is just (◇) with τ-indices:

  x_{τ_{i-1}} = √α_{τ_{i-1}}·( (x_{τ_i} − √(1−α_{τ_i}) ε_θ(x_{τ_i})) / √α_{τ_i} )  +  √(1 − α_{τ_{i-1}} − σ_{τ_i}²)·ε_θ(x_{τ_i})  +  σ_{τ_i} ε.

This is the actual cure for the wall clock: S = 20 or 50 instead of 1000, a 20×–50× cut, by a free choice that training never constrained. The headache I thought was structural was an artifact of identifying the generative chain length with the forward chain length, and that identification was never in the loss.

Why should deterministic sampling, in particular, be the safer few-step choice? With σ_t > 0 every step injects fresh noise; on a short trajectory there are too few remaining denoising steps to average that perturbation back down, so stochastic error can remain visible. The larger the per-step σ, the more I ask the short chain to clean up. The σ = 0 process has no injected noise to clean up, so cutting steps only coarsens an otherwise smooth map rather than leaving residual noise behind. To slide between the two regimes cleanly, write σ_{τ_i}(η) = η·√((1−α_{τ_{i-1}})/(1−α_{τ_i}))·√(1 − α_{τ_i}/α_{τ_{i-1}}) with a single dial η ≥ 0: η = 1 is the stochastic ancestral endpoint for that trajectory, recovering the original ancestral sampler on the full adjacent grid; η = 0 is the deterministic one, and intermediate η interpolates.

Take the σ = 0 update with a small step — adjacent levels t and t−Δt — and divide (◇) through by √α_{t-1}:

  x_{t-Δt}/√α_{t-Δt} = x_t/√α_t + ( √((1−α_{t-Δt})/α_{t-Δt}) − √((1−α_t)/α_t) )·ε_θ(x_t).

This is begging for a change of variables. Let x̄ = x/√α and σ = √((1−α)/α). Then α = 1/(1+σ²) and x = x̄/√(σ²+1), and the update reads

  x̄(t−Δt) = x̄(t) + ( σ(t−Δt) − σ(t) )·ε_θ( x̄(t)/√(σ²+1) ).

Divide by −Δt and send Δt → 0:

  dx̄/dt = (dσ/dt)·ε_θ( x̄/√(σ²+1) ),   i.e.   dx̄ = ε_θ( x̄/√(σ²+1) ) dσ.

The deterministic sampler is an Euler integration of an ODE, with σ as the integration variable. In the rescaled coordinate the noisy endpoint is x̄(T) ∼ N(0, σ(T)²I) at large σ(T), while the sampler's stored latent x_T = x̄(T)/√(1+σ(T)²) is approximately N(0,I) at the α ≈ 0 end. That explains the consistency: the generated image is the ODE's solution from the initial condition x̄(T), and the number of sampling steps is just the fineness of the Euler grid. Same x_T, different S, nearly the same image — the high-level content is fixed by the ODE trajectory and only fine detail moves with the discretization. It also explains the encoding: run the ODE the other way, from t = 0 up to T, and I deterministically map x_0 to its latent x_T; decode by running back. That is exactly a flow's invertibility, which is why reconstruction error should fall as S grows.

And there is a sanity check I can run on this ODE against the score-matching world. The optimal ε_θ is the minimizer of E‖ε_θ(x_t) − ε‖² where x_t = √α_t x_0 + √(1−α_t)ε; by denoising score matching, predicting the added noise is, up to a known scale, predicting the score of the noised data. Concretely the σ(t)-perturbed score satisfies ∇_{x̄} log p_t = −ε_θ(x̄/√(σ²+1))/σ. The continuous score-based view samples by the probability-flow ODE dx̄ = −½ g(t)² ∇_{x̄} log p_t dt with g(t)² = dσ²/dt. Substitute the score:

  dx̄ = −½ (dσ²/dt)·( −ε_θ(x̄/√(σ²+1))/σ ) dt = ½ (dσ²/dt)/σ · ε_θ dt = (dσ/dt)·ε_θ dt = ε_θ dσ,

since ½(dσ²/dt)/σ = (σ dσ/dt)/σ = dσ/dt. Identical ODE. So the deterministic sampler is the probability-flow ODE of the variance-exploding diffusion, reached from a purely variational starting point with no Langevin or SDE machinery — which is reassuring, and also clarifies that the two are the *same ODE* but *different discretizations*: I take Euler steps in dσ, the score-based Euler step is in dt, and they coincide only as the levels get close. In few steps, stepping in dσ — which does not hinge on the arbitrary parameterization of "time" — is the better-behaved discretization.

One more thing to confirm the contribution is the marginal-preserving non-Markovian construction and not anything Gaussian-specific. Replace the Gaussian marginal by a categorical one for one-hot data: q(x_t|x_0) = Cat(α_t x_0 + (1−α_t)1_K) with 1_K the uniform vector. The same backward construction gives a mixture reverse conditional q(x_{t-1}|x_t,x_0) = Cat(σ_t x_t + (α_{t-1} − σ_t α_t) x_0 + ((1−α_{t-1}) − (1−α_t)σ_t) 1_K), with σ_t chosen so all three mixture weights are nonnegative. Marginalizing x_t gives σ_t[α_t x_0 + (1−α_t)1_K] + (α_{t-1}−σ_tα_t)x_0 + ((1−α_{t-1})−(1−α_t)σ_t)1_K = α_{t-1}x_0 + (1−α_{t-1})1_K, so the coefficients are pinned exactly as before — and as the uniform coefficient goes to zero the step becomes a choice between copying x_t and using the predicted x_0 instead of drawing from the uniform background. The generative process replaces x_0 by f_θ; each objective term is a KL between two categoricals, and by convexity it is upper-bounded by (α_{t-1} − σ_t α_t)·KL(Cat(x_0)‖Cat(f_θ)), a plain multi-class classification loss whose optimum is again independent of σ up to reweighting. Same skeleton, no Gaussian needed: the idea is the marginal-preserving family, the Gaussian case is one instance.

So the picture closes. Training never constrained the joint, only the marginals and (with unshared per-t parameters) only up to reweighting; I built the full family of marginal-preserving non-Markovian inference processes (★) with one free variance σ per step; its variational objective is L_γ for some γ and therefore shares the optimum of the plain L_1 I already trained; the σ = 0 member is a deterministic implicit model whose update is (◇) without the noise term; and because the objective is blind to forward chain length, I run that update on a short sub-sequence τ for the 20×–50× speedup, with η dialing stochasticity back in if I want it. Now the code. It is the generative `sample` slot, filled in: build a trajectory τ, walk it in reverse, and at each step form the predicted x_0, the direction term, and the (η-scaled) noise, exactly as in (◇).

```python
import torch

def alpha_at(alphas, t):
    # cumulative coefficient with alpha_0 := 1 at the t = -1 sentinel
    a = torch.cat([alphas.new_ones(1), alphas], dim=0)
    return a.index_select(0, t + 1).view(-1, 1, 1, 1)

def q_sample(x0, t, alphas, eps):
    # unchanged one-shot corruption used by training
    a = alpha_at(alphas, t)
    return a.sqrt() * x0 + (1 - a).sqrt() * eps

def training_loss(model, x0, alphas):
    # unchanged unweighted noise-prediction MSE
    t = torch.randint(0, len(alphas), (x0.size(0),), device=x0.device)
    eps = torch.randn_like(x0)
    xt = q_sample(x0, t, alphas, eps)
    return ((model(xt, t.float()) - eps) ** 2).mean()

@torch.no_grad()
def sample(model, alphas, x, seq, eta=0.0):
    # Same loop shape as the existing sequential sampler: keep the whole path and x0 estimates.
    # x is the initial latent at the largest selected noise level; seq is increasing zero-based indices.
    # If seq reaches len(alphas)-1, this is the usual x_T ~ N(0, I) initialization.
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    xs = [x]
    x0_preds = []
    for i, j in zip(reversed(seq), reversed(seq_next)):            # walk tau backwards
        t = torch.ones(n, device=x.device) * i
        t_next = torch.ones(n, device=x.device) * j
        a = alpha_at(alphas, t.long())                             # alpha_{tau_i}
        a_next = alpha_at(alphas, t_next.long())                   # alpha_{tau_{i-1}}

        xt = xs[-1].to(x.device)
        eps = model(xt, t)                                         # predicted noise
        x0 = (xt - eps * (1 - a).sqrt()) / a.sqrt()                # predicted x_0 = f_theta
        x0_preds.append(x0)

        c1 = eta * (((1 - a / a_next) * (1 - a_next) / (1 - a)).sqrt())
        c2 = ((1 - a_next) - c1 ** 2).sqrt()
        xt_next = a_next.sqrt() * x0 + c1 * torch.randn_like(xt) + c2 * eps
        xs.append(xt_next)                                         # equation (◇) on tau
    return xs, x0_preds

def make_seq(num_timesteps, num_sampling_steps, kind="uniform"):
    if kind == "uniform":
        skip = max(num_timesteps // num_sampling_steps, 1)
        return list(range(0, num_timesteps, skip))
    return [int(s) for s in (
        torch.linspace(0, (0.8 * num_timesteps) ** 0.5, num_sampling_steps) ** 2
    ).tolist()]
```

Training is untouched — the existing unweighted ε-MSE — and `sample` is the only thing that changed. Causal recap: the loss only ever saw the marginals q(x_t|x_0) and, with unshared per-t weights, only up to a reweighting, so a trained noise predictor is committed to nothing but those marginals; I built every non-Markovian inference process that keeps them (★), found one free per-step variance σ, showed its variational objective is L_γ and hence shares the L_1 optimum already trained, took σ→0 to get a deterministic implicit model whose update (◇) loses its noise term, recognized that update as an Euler step of an ODE in σ, and — because the objective is indifferent to forward chain length — ran it on a short sub-sequence τ, turning a thousand sequential passes into twenty without retraining.
