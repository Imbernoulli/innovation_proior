OK, let me think this through from scratch. The thing that bothers me about score-based diffusion is purely the cost of sampling. The model itself is wonderful: I train a network s_φ(x,t) to approximate the score ∇ log p_t of the noise-blurred data, and I have this beautiful deterministic object, the probability-flow ODE,

    dx_t/dt = μ(x_t,t) − ½ σ(t)² ∇ log p_t(x_t),

whose trajectories carry the same marginals p_t as the forward noising SDE. So if I sit a sample at t=T where p_T is essentially a Gaussian, and integrate this ODE backward to t≈0, I land on a sample from p_data. Exact in principle. But to integrate it I have to evaluate s_φ over and over, once (or twice, with Heun) per step, and even a good solver needs more than ten steps. That sequential chain of network calls is the whole bottleneck. GANs and VAEs spit out a sample in one forward pass; I want that, but without adversarial training and without giving up the things iterative sampling buys me — spending more compute for better samples, and zero-shot editing.

Let me adopt the cleanest version of the ODE so I'm not carrying μ and σ around. With μ=0 and σ(t)=√(2t), the perturbation kernel is just p_t = p_data ⊗ N(0, t² I) — adding noise at time t means adding Gaussian noise with standard deviation exactly t. So t literally is the noise level. The terminal is π = N(0, T² I), and the empirical PF ODE collapses to

    dx_t/dt = − t · s_φ(x_t, t).

I'll fix T=80 and stop at a small ε=0.002 because the score blows up as t→0, and call x_ε the sample. Good. Now the slow part is solving this thing.

What does "solving it" actually give me? For each starting noise vector at t=T, the ODE is deterministic, so there is exactly one trajectory {x_t}, and it ends at exactly one point x_ε. The entire sampling procedure is: follow this one trajectory from its noisy end to its data end. The reason it's slow is that I crawl along the trajectory in many little steps. But I don't actually care about the intermediate points — I only want the endpoint x_ε. So why am I tracing the whole path? What if I train a network to jump from any point on the trajectory straight to its origin, in one shot?

Let me make that precise. Suppose I had a function f that, given any point x_t on a PF-ODE trajectory together with its time t, returns the origin of that trajectory:

    f(x_t, t) = x_ε.

Then sampling is one call: draw x_T ~ N(0, T² I), return f(x_T, T). Done, single step. What must such a function satisfy? Two points x_t and x_{t'} that lie on the *same* trajectory share the same origin, so

    f(x_t, t) = f(x_{t'}, t')   for all t, t' on that trajectory.

That's a strong internal constraint — call it self-consistency: the function's output is constant along each trajectory. And there's a boundary fact that comes for free: the origin of the trajectory through x_ε is x_ε itself, so

    f(x_ε, ε) = x_ε,

i.e. at the smallest time f is the identity. So the object I want is a map that is (a) constant along trajectories and (b) the identity at t=ε. If I can learn such an f, I get one-step generation, and because it's defined on *every* (x_t, t), I keep the option of using it at multiple noise levels later for a quality/compute tradeoff.

Now — can I just parameterize a network F_θ(x,t) and call it f? Not quite, because the boundary condition f(x_ε,ε)=x_ε is exact, not approximate, and I'll see in a moment that it's load-bearing. If I let the network be free at t=ε, nothing stops it from collapsing. In fact stare at the self-consistency requirement alone: f(x_t,t)=f(x_{t'},t') is satisfied perfectly by the constant function f≡0 (or f≡any constant). That's the degenerate solution. The thing that rules it out is precisely the boundary: at t=ε the function is *pinned* to the identity, x_ε ↦ x_ε, which is not constant in x_ε. So the boundary condition isn't a cosmetic detail — it's what makes the problem non-trivial. I must build it into the architecture so it holds *by construction*, not just hope the loss enforces it.

How to bake in f(x,ε)=x? The blunt way:

    f_θ(x,t) = x if t=ε, else F_θ(x,t).

This is correct at the boundary but it's a discontinuity glued on by hand; the function isn't differentiable across t=ε, and I have a feeling I'll want differentiability later (if I ever push the time discretization to a continuum, I'll be taking ∂f/∂t). Let me find something smoother. I want a parameterization where the identity at ε emerges from continuous coefficients. Try a skip connection:

    f_θ(x,t) = c_skip(t) · x + c_out(t) · F_θ(x,t),

with c_skip, c_out differentiable. For the boundary I need f_θ(x,ε)=x for all x, which forces

    c_skip(ε) = 1   and   c_out(ε) = 0.

Then at t=ε the network's contribution is multiplied by zero and the input passes straight through. Differentiable everywhere if c_skip, c_out, F_θ are. And it has a bonus I didn't go looking for: this is exactly the shape of the EDM denoiser preconditioning, D = c_skip·x + c_out·F. So if I can find c_skip, c_out that hit c_skip(ε)=1, c_out(ε)=0 while otherwise behaving like EDM's, I get the boundary by construction and an off-the-shelf backbone at the same time.

Let me pin down concrete scalings. EDM uses

    c_skip(t) = σ_data² / (t² + σ_data²),   c_out(t) = σ_data · t / √(σ_data² + t²),

with σ_data=0.5. Do these already satisfy my boundary? At t=ε they give c_skip(ε) = σ_data²/(ε²+σ_data²), which with σ_data=0.5, ε=0.002 is 0.25/(0.000004+0.25)=0.9999840…, not 1; and c_out(ε)=σ_data·ε/√(σ_data²+ε²)=0.5·0.002/√(0.250004)=0.001999…, not 0. So EDM's scalings hit the boundary only in the limit ε→0, but I stop at ε>0, and "almost 1, almost 0" is not the exact identity the induction below is going to lean on. I need to move the point where these vanish from 0 to ε. The clean way is to replace t by (t−ε) exactly where each coefficient has to hit its boundary value:

    c_skip(t) = σ_data² / ( (t−ε)² + σ_data² ),   c_out(t) = σ_data (t−ε) / √(σ_data² + t²).

Now evaluate at t=ε: the (t−ε)² in c_skip's denominator is 0, so c_skip(ε)=σ_data²/(0+σ_data²)=1 on the nose; and the (t−ε) factor in c_out's numerator is 0, so c_out(ε)=0 on the nose. Both exact, not merely close. Away from ε the (t−ε)² shift is negligible — at t=2, c_skip is 0.25/((1.998)²+0.25)=0.05893 versus EDM's 0.25/(4+0.25)=0.05882, a 0.2% difference — so I keep EDM's well-tuned behavior everywhere except the immediate neighborhood of ε, where I have deliberately bent it to satisfy the boundary. (I keep t² rather than (t−ε)² inside c_out's square root because ε is tiny and t² is the magnitude that controls the output scale; the (t−ε) factor in the numerator is the only thing that needs to vanish.) So: boundary exact, differentiable in t, and the architecture is an EDM denoiser with retouched coefficients, so I can lift a powerful diffusion backbone wholesale.

Now the real question: how do I *train* f_θ to be self-consistent? I want f_θ(x_t,t) to be the same for all points on a trajectory. I don't have the trajectories in closed form, but I do have a teacher score model s_φ that defines the empirical PF ODE, and I can use it to relate two *adjacent* points on a trajectory cheaply. I discretize [ε,T] into t_1=ε < t_2 < … < t_N=T using the EDM grid t_i=(ε^{1/ρ} + (i−1)/(N−1)(T^{1/ρ}−ε^{1/ρ}))^ρ with ρ=7. Pick a data point x ~ p_data, and a level t_{n+1}; since p_t = p_data ⊗ N(0,t²I), I can land a point on a trajectory by

    x_{t_{n+1}} = x + t_{n+1} z,   z ~ N(0, I),

which is an exact draw from p_{t_{n+1}}. Now I take *one* numerical ODE step backward from t_{n+1} to the adjacent t_n using the teacher, to get an estimate of the neighboring point on the *same* trajectory:

    x̂^φ_{t_n} = x_{t_{n+1}} + (t_n − t_{n+1}) Φ(x_{t_{n+1}}, t_{n+1}; φ),

where Φ is the solver's update. For Euler on dx/dt=−t s_φ, Φ = −t s_φ, so

    x̂^φ_{t_n} = x_{t_{n+1}} − (t_n − t_{n+1}) t_{n+1} s_φ(x_{t_{n+1}}, t_{n+1}).

Now (x_{t_{n+1}}, t_{n+1}) and (x̂^φ_{t_n}, t_n) are, to the solver's accuracy, two adjacent points on one trajectory, so a self-consistent f must map them to the same origin. So I just minimize the distance between the model's outputs at the two points:

    L_CD = E[ λ(t_n) · d( f_θ(x_{t_{n+1}}, t_{n+1}), f_θ(x̂^φ_{t_n}, t_n) ) ].

The expectation is over x ~ p_data, n uniform in {1,…,N−1}, x_{t_{n+1}} ~ N(x, t_{n+1}² I). The metric d ≥ 0 with equality iff equal — I'll try squared ℓ2, ℓ1, and a perceptual distance (LPIPS). And λ(t_n) is a positive weighting; empirically λ≡1 is fine.

Wait — there's a subtlety in how the two ends are treated. If I let gradients flow through *both* f_θ(x_{t_{n+1}}) and f_θ(x̂_{t_n}), the network can cheat by dragging both outputs toward each other in lazy ways, and the target keeps moving. This is the same instability people hit when a network regresses toward its own moving output. The fix that's standard in that situation — bootstrapping a value function toward a slowly-updated copy of itself in Q-learning, or matching an online encoder to a momentum target in self-supervised learning — is to make the *target* end a stop-gradient copy of the network whose weights track θ slowly. So I introduce θ⁻, an exponential moving average of θ, and write the target end with f_{θ⁻} and no gradient:

    L_CD = E[ λ(t_n) · d( f_θ(x_{t_{n+1}}, t_{n+1}), f_{θ⁻}(x̂^φ_{t_n}, t_n) ) ],
    θ⁻ ← stopgrad( μ θ⁻ + (1−μ) θ ).

The online network f_θ at the noisier point t_{n+1} is pulled toward the target network f_{θ⁻} at the cleaner point t_n. Because the cleaner end is, recursively, anchored ultimately at t=ε where f is pinned to the identity, the signal "the right answer is the data" propagates inward from the boundary. The EMA also just stabilizes training a lot compared to setting θ⁻=θ. Once optimization converges, θ⁻=θ anyway, so I haven't changed the fixed point, only the dynamics.

Does this actually recover the true consistency function? Let me convince myself, because I worried the constant solution lurks. Suppose I drive the loss to zero with θ⁻=θ. Since p_{t_n}>0 everywhere for t_n≥ε>0, zero expected loss with a positive weight and a genuine metric forces the integrand to vanish pointwise:

    f_θ(x_{t_{n+1}}, t_{n+1}) = f_θ(x̂^φ_{t_n}, t_n)   for all n, all points.

Let f(·,·;φ) be the *true* consistency function of the empirical PF ODE (the one s_φ defines), and define the error at level t_n:

    e_n := f_θ(x_{t_n}, t_n) − f(x_{t_n}, t_n; φ).

I want to bound e_n. Start from e_{n+1} and use the zero-loss identity plus the fact that the true f is constant along the trajectory, f(x_{t_{n+1}},t_{n+1};φ)=f(x_{t_n},t_n;φ):

    e_{n+1} = f_θ(x_{t_{n+1}}, t_{n+1}) − f(x_{t_{n+1}}, t_{n+1}; φ)
            = f_θ(x̂^φ_{t_n}, t_n) − f(x_{t_n}, t_n; φ)        [zero-loss; true-f constant]
            = [ f_θ(x̂^φ_{t_n}, t_n) − f_θ(x_{t_n}, t_n) ] + [ f_θ(x_{t_n}, t_n) − f(x_{t_n}, t_n; φ) ]
            = [ f_θ(x̂^φ_{t_n}, t_n) − f_θ(x_{t_n}, t_n) ] + e_n.

If f_θ is L-Lipschitz in its first argument,

    ‖e_{n+1}‖ ≤ ‖e_n‖ + L · ‖x̂^φ_{t_n} − x_{t_n}‖.

And ‖x̂^φ_{t_n} − x_{t_n}‖ is exactly the *local truncation error* of the ODE solver over one step, which for a method of order p is O((t_{n+1}−t_n)^{p+1}). So ‖e_{n+1}‖ ≤ ‖e_n‖ + O((t_{n+1}−t_n)^{p+1}). What's the base case? At n=1, t_1=ε, and here the boundary saves me:

    e_1 = f_θ(x_{t_1}, ε) − f(x_{t_1}, ε; φ) = x_{t_1} − x_{t_1} = 0,

because my parameterization forces f_θ(·,ε)=identity and the true consistency function is also the identity at ε. So e_1=0 — there's the payoff of the boundary condition, a second time: it gives the induction a clean zero start, and it kills the f≡0 escape because f≡0 has e_1 = 0 − x_{t_1} = −x_{t_1} ≠ 0, contradicting the zero-loss conclusion. Telescoping,

    ‖e_n‖ ≤ Σ_{k=1}^{n−1} O((t_{k+1}−t_k)^{p+1}) = Σ (t_{k+1}−t_k) · O((t_{k+1}−t_k)^p)
          ≤ O((Δt)^p) · Σ (t_{k+1}−t_k) = O((Δt)^p)·(t_n − ε) ≤ O((Δt)^p)(T−ε) = O((Δt)^p),

with Δt the largest step. So as the discretization tightens, the learned model converges to the true consistency function at the solver's order. That's reassuring: the scheme is consistent, and the boundary is doing real work both in the induction and in excluding the trivial solution.

To sample once I'm trained: draw x̂_T ~ N(0, T² I), return f_θ(x̂_T, T). One evaluation. And if I want to spend more compute for quality, I can chain: denoise to an estimate of the data, re-noise it to some intermediate level τ, denoise again, and so on. Concretely, x ← f_θ(x̂_T, T); then for a decreasing sequence τ_1 > τ_2 > …: sample z ~ N(0,I), set x̂_{τ_n} = x + √(τ_n² − ε²) z (re-noise the current estimate up to level τ_n — the √(τ_n²−ε²) is so the total noise variance is τ_n², since x already carries ε-level structure), and set x ← f_θ(x̂_{τ_n}, τ_n). Each extra step is one more evaluation, and I can pick the τ's by a greedy search on sample quality. This is the same add-noise/denoise loop diffusion uses for editing, so inpainting, colorization, super-resolution, interpolation all come along for free by masking in the appropriate space and only injecting the model's output where the data is unknown.

So far I needed a pretrained score model s_φ — this is distillation. But that feels like a crutch. The only place s_φ enters is in the one ODE step that produces the neighbor x̂^φ_{t_n}. What is s_φ giving me there? It's an estimate of ∇ log p_{t_{n+1}}(x_{t_{n+1}}). Do I actually need a trained network for that, or can I estimate the score directly from data? Here's a fact about Gaussian-perturbed densities. With p_t = p_data ⊗ N(0,t²I) and the kernel p(x_t|x)=N(x_t; x, t²I),

    ∇ log p_t(x_t) = ∇_{x_t} log ∫ p_data(x) p(x_t|x) dx
                   = [ ∫ p_data(x) ∇_{x_t} p(x_t|x) dx ] / p_t(x_t)
                   = ∫ [ p_data(x) p(x_t|x) / p_t(x_t) ] ∇_{x_t} log p(x_t|x) dx
                   = ∫ p(x | x_t) ∇_{x_t} log p(x_t|x) dx          [Bayes]
                   = E[ ∇_{x_t} log p(x_t|x) | x_t ].

And for the Gaussian kernel, ∇_{x_t} log p(x_t|x) = −(x_t − x)/t². So

    ∇ log p_t(x_t) = − E[ (x_t − x)/t² | x_t ].

It says: given a noisy x_t, the score is the negative posterior-mean of (x_t−x)/t². Before I build anything on this, let me check it on a case where I can compute both sides in closed form, because every step above relied on the Gaussian kernel and a Bayes flip and I want to be sure I didn't drop a factor. Take p_data = N(0, σ_data²) in one dimension with σ_data=0.5, so σ_data²=0.25. Then p_t = N(0, σ_data² + t²), and its score is known directly: ∇ log p_t(x_t) = −x_t/(σ_data²+t²). Pick t=1.3, x_t=0.7. Left side: −0.7/(0.25+1.69) = −0.7/1.94 = −0.36082. Now the right side. For jointly-Gaussian x~N(0,σ_data²) and x_t=x+tz, the posterior mean is E[x|x_t] = (σ_data²/(σ_data²+t²)) x_t = (0.25/1.94)·0.7 = 0.090206. So −E[(x_t−x)/t²|x_t] = −(x_t − E[x|x_t])/t² = −(0.7 − 0.090206)/1.69 = −0.609794/1.69 = −0.36082. The two agree to all the digits I carried — the identity holds, factors and all.

Now I don't have the posterior in general, but I have something better for a stochastic objective. If I draw the *pair* (x, x_t) jointly by sampling x ~ p_data and x_t = x + t z, then for that single sample, −(x_t − x)/t² is an *unbiased* one-sample estimate of the conditional expectation E[−(x_t−x)/t²|x_t], hence of the score. It is not the score for that particular x_t — it's a high-variance draw whose mean over the posterior is the score — but inside an expectation that distinction can wash out. I can plug this directly into the Euler step in place of s_φ. No pretrained model at all, if the algebra cooperates.

Let me check that swapping the true score (or an exact s_φ) for this unbiased estimate gives me the *same* training objective in the limit, not just something heuristically similar — otherwise I'm not actually distilling the right thing. Start from the distillation loss with the Euler solver, with an exact teacher s_φ = ∇ log p_t. The Euler target point is

    x̂^φ_{t_n} = x_{t_{n+1}} + (t_n − t_{n+1}) (− t_{n+1}) s_φ(x_{t_{n+1}}, t_{n+1})
              = x_{t_{n+1}} + (t_{n+1} − t_n) t_{n+1} ∇ log p_{t_{n+1}}(x_{t_{n+1}}).

Write Δ = t_n − t_{n+1} (negative). Taylor-expand the target end f_{θ⁻}(x̂^φ_{t_n}, t_n) around (x_{t_{n+1}}, t_{n+1}) to first order, and likewise expand d to first order in its second argument (using ∂₂d at the diagonal). The first-order term of f_{θ⁻} has two pieces: the spatial displacement x̂ − x_{t_{n+1}} = −Δ t_{n+1} ∇log p, contracted with ∂₁f_{θ⁻}, and the time displacement Δ, contracted with ∂₂f_{θ⁻}. So

    L_CD = E[ λ d( f_θ(x_{t_{n+1}}), f_{θ⁻}(x_{t_{n+1}}) ) ]
         + E{ λ ∂₂d(...) [ ∂₁f_{θ⁻}(x_{t_{n+1}}) · (−Δ) t_{n+1} ∇log p_{t_{n+1}}(x_{t_{n+1}}) ] }
         + E{ λ ∂₂d(...) [ ∂₂f_{θ⁻}(x_{t_{n+1}}) · Δ ] }
         + E[ o(|Δ|) ].

Now the only place the score appears is that middle term, sitting *inside* the outer expectation, multiplied by quantities that depend on x_{t_{n+1}} but not on the clean x. By the law of total expectation I can condition on x_{t_{n+1}} and replace ∇log p_{t_{n+1}}(x_{t_{n+1}}) by its conditional-expectation expression — but the unbiased estimator says E[ −(x_{t_{n+1}}−x)/t_{n+1}² | x_{t_{n+1}} ] *is* that score, so inside the full expectation I may substitute the single-sample quantity −(x_{t_{n+1}}−x)/t_{n+1}² without changing the value. The displacement fed into f_{θ⁻} is x̂ − x_{t_{n+1}} = (t_{n+1} − t_n) t_{n+1} ∇log p = −Δ t_{n+1} ∇log p, with Δ = t_n − t_{n+1}. Substituting ∇log p → −(x_{t_{n+1}}−x)/t_{n+1}²:

    x̂ − x_{t_{n+1}}  ⟶  (t_{n+1} − t_n) t_{n+1} · ( −(x_{t_{n+1}}−x)/t_{n+1}² )
                       = −(t_{n+1} − t_n)(x_{t_{n+1}}−x)/t_{n+1}
                       = (t_n − t_{n+1})(x_{t_{n+1}}−x)/t_{n+1}.

The t_{n+1}/t_{n+1}² leaves a single 1/t_{n+1}, and the leading minus flips (t_{n+1}−t_n) into (t_n−t_{n+1}).

Now fold the first-order terms back up with a *reverse* Taylor expansion: the loss equals, up to o(Δt),

    L_CD = E[ λ d( f_θ(x_{t_{n+1}}, t_{n+1}), f_{θ⁻}( x_{t_{n+1}} + (t_n − t_{n+1})(x_{t_{n+1}}−x)/t_{n+1}, t_n ) ) ] + o(Δt).

Recall x_{t_{n+1}} = x + t_{n+1} z with z = (x_{t_{n+1}} − x)/t_{n+1} ~ N(0,I). Then (x_{t_{n+1}} − x)/t_{n+1} = z, so the target argument is

    x_{t_{n+1}} + (t_n − t_{n+1}) z = x + t_{n+1} z + (t_n − t_{n+1}) z = x + t_n z.

So the whole target end is just f_{θ⁻}(x + t_n z, t_n), and the online end is f_θ(x + t_{n+1} z, t_{n+1}). Therefore

    L_CD = E[ λ(t_n) d( f_θ(x + t_{n+1} z, t_{n+1}), f_{θ⁻}(x + t_n z, t_n) ) ] + o(Δt)
         =: L_CT + o(Δt),   z ~ N(0,I).

The two ends are now just the *same data point x noised by the same z to two adjacent levels* — no score model, no ODE solver, no φ anywhere. This is a clean objective I can train from scratch; call it L_CT.

But I should be careful about what exactly I just proved, because the substitution "replace ∇log p by −(x−x_t)/t² without changing the value" was justified only *inside the expectation*, by the law of total expectation. For a single sample those two are different numbers, and I want to know how different — if they were wildly off per-sample the o(Δt) story would be shaky. Let me put numbers on it, again with p_data=N(0,σ_data²), σ_data=0.5. Take one draw x=0.3, z=−1.1, and the level t_{n+1}=1.7, so x_{t_{n+1}}=0.3+1.7·(−1.1)=−1.57. The *true* score there is −x_{t_{n+1}}/(σ_data²+t_{n+1}²)=1.57/(0.25+2.89)=0.5. The *one-sample* estimate is −(x_{t_{n+1}}−x)/t_{n+1}²=−(−1.87)/2.89=0.6471. Already different for this draw, as expected. Now run one Euler step each, with Δt = t_n−t_{n+1} = −0.5 (so t_n=1.2): velocity is −t·score, step is x_{t_{n+1}} + Δt·(−t_{n+1}·score). With the true score I get −1.57 + (−0.5)·(−1.7·0.5) = −1.57+0.425 = −1.145. With the one-sample estimate I get −1.57 + (−0.5)·(−1.7·0.6471) = −1.57+0.55 = −1.02. And −1.02 is exactly x + t_n z = 0.3+1.2·(−1.1) = −1.02 — so the algebra that collapsed the one-sample Euler target to x+t_n z checks out on this concrete draw. The two targets differ by 0.125 = |Δt|/... — and that gap is O(Δt): halving Δt to 0.25, 0.125, 0.0625 gives gaps 0.0625, 0.03125, 0.015625, halving each time. So per-sample the one-sample target is off from the true-score target by O(Δt), but the offsets average to zero over the posterior of x given x_{t_{n+1}}, which is exactly the condition the law-of-total-expectation step needed. That is the honest content of "L_CD = L_CT + o(Δt)": not that the two losses are equal sample-by-sample, but that the per-sample O(Δt) discrepancy is mean-zero in the right conditional, so it contributes only at higher order once integrated.

For the o(Δt) to actually be subleading I also need L_CT itself not to vanish faster than Δt. Heuristically L_CT ≥ O(Δt) whenever the distillation loss stays bounded away from zero: if L_CT were smaller than O(Δt) then L_CD = L_CT + o(Δt) would force L_CD → 0, contradicting that assumption. I haven't proven the constant in that lower bound here, so I'd flag it as the one step I'm taking on faith; the per-sample check above is what makes me believe the leading terms genuinely match. Granting it, minimizing L_CT trains the same consistency function I would have distilled — the pretrained diffusion model was never essential; it only ever supplied a score I can get from data.

Two practical wrinkles for training-in-isolation. First, the estimator −(x_t−x)/t² is unbiased but noisy, and the bias↔variance tradeoff depends on N (the number of discretization levels, equivalently Δt). With small N (large Δt) the discrete objective is a biased proxy for the true consistency loss but has low variance — good early on, fast convergence. With large N (small Δt) the bias shrinks but variance grows — better late. So I anneal N upward over training with a schedule N(k), increasing the number of levels as training proceeds. Second, I tie the EMA rate μ to N: μ(k) = exp( s_0 ln μ_0 / N(k) ) so the target tracks at a rate consistent with the changing grid; a convenient N(k) = ⌈√( k/K ((s_1+1)² − s_0²) + s_0² ) − 1⌉ + 1 ramps the level count from s_0 to s_1 over K iterations. The CT loss itself is identical in form to CD — pull the online network at the noisier level toward the EMA target at the cleaner level — only the way I produce the adjacent pair changed.

Let me also see what happens if I refuse to discretize at all and push N → ∞, because my parameterization was deliberately differentiable for exactly this. Take θ⁻ = θ and a smooth metric. Expand the distillation loss to second order: d(f_θ(x_{t_{n+1}}), f_θ(x̂_{t_n})) ≈ ½ (f_θ(x̂_{t_n}) − f_θ(x_{t_{n+1}}))ᵀ G (f_θ(x̂_{t_n}) − f_θ(x_{t_{n+1}})), where G is the Hessian of d with respect to the second argument at the diagonal; for squared ℓ2, G = 2I. The increment, by Taylor expansion of f_θ along the Euler step, is

    f_θ(x̂_{t_n}, t_n) − f_θ(x_{t_{n+1}}, t_{n+1}) = −( ∂f_θ/∂t − t ∂f_θ/∂x · s_φ ) τ'(u) Δu + O(Δu²),

with t_n = τ(u_n) a smooth reparameterization and Δu = 1/(N−1). The squared increment is O(Δu²), so scaling the loss by (N−1)² and letting N → ∞ gives a finite limit,

    L_CD^∞(θ,θ;φ) = ½ E[ (λ/((τ⁻¹)')²) ( ∂f_θ/∂t − t ∂f_θ/∂x · s_φ )ᵀ G ( ∂f_θ/∂t − t ∂f_θ/∂x · s_φ ) ].

For squared ℓ2 this is just E[ (λ/((τ⁻¹)')²) ‖ ∂f_θ/∂t − t ∂f_θ/∂x · s_φ ‖² ]. And this object has a clean meaning: along a trajectory, self-consistency means f_θ(x_t,t) is constant in t, so its total time-derivative is zero,

    d/dt f_θ(x_t,t) = ∂f_θ/∂x · (dx_t/dt) + ∂f_θ/∂t = ∂f_θ/∂x · (−t s_φ) + ∂f_θ/∂t = 0,

i.e. ∂f_θ/∂t − t ∂f_θ/∂x · s_φ ≡ 0 exactly when f_θ is the true consistency function. So the continuous loss is literally the squared violation of "the directional derivative of f along the ODE field is zero" — the infinitesimal form of self-consistency. The discrete adjacent-pair loss was a finite-difference approximation of this all along. If I use ℓ1 instead, the Hessian is zero almost everywhere and that second-order route becomes vacuous, so I scale by N−1 rather than (N−1)² and keep the first-order magnitude:

    L_CD,ℓ1^∞(θ,θ;φ) = E[ λ/(τ⁻¹)' · ‖ t ∂f_θ/∂x · s_φ − ∂f_θ/∂t ‖₁ ].

Same directional-derivative condition, different scaling and norm. The price of these continuous losses is a Jacobian-vector product ∂f/∂x · s_φ, so I need forward-mode autodiff or an equivalent trick.

The practical target-network version has another wrinkle. With θ⁻ = stopgrad(θ), the continuous object should be trusted through its gradient, not as an ordinary loss value. Let H be the Hessian of d(y,x) with respect to y at y=x. Expanding the finite loss, differentiating with respect to θ, and then letting N grow gives a pseudo-objective whose gradient matches the scaled discrete gradient:

    L_CD^∞(θ,θ⁻;φ) = E[ λ/(τ⁻¹)' · f_θ(x_t,t)ᵀ H(f_{θ⁻}(x_t,t)) ( ∂f_{θ⁻}/∂t − t ∂f_{θ⁻}/∂x · s_φ ) ].

For squared ℓ2, H=2I, so the factor 2 remains. If f_θ already matches the true trajectory-origin map, the bracket is zero, the pseudo-objective value is zero, and its θ-gradient is zero; the converse need not hold, so this is not something I should monitor as a scalar training loss. It is a way to get the right gradient when the target branch is stopped.

Now substitute the data-only score estimate into that stop-gradient limit. The ODE velocity is −t s_φ; replacing s_φ by −(x_t−x)/t² makes the velocity +(x_t−x)/t. Therefore the CT pseudo-objective becomes

    L_CT^∞(θ,θ⁻) = E[ λ/(τ⁻¹)' · f_θ(x_t,t)ᵀ H(f_{θ⁻}(x_t,t)) ( ∂f_{θ⁻}/∂t + ∂f_{θ⁻}/∂x · (x_t−x)/t ) ].

The sign is important: the score estimate is negative, and the PF-ODE field has another negative sign, so the spatial term is plus. The gradients of (N−1)L_CD^N and (N−1)L_CT^N match in the limit when the teacher score is exact and Euler is used. This removes the grid and the pretrained model at once, but it inherits the Jacobian-vector-product cost and high variance, which is why the adjacent-pair discrete objective remains the practical route.

Now to code. I keep the EDM `KarrasDenoiser` harness. When `distillation` is false it uses the ordinary EDM scalings; when `distillation` is true it swaps in the boundary-respecting scalings so c_skip(ε)=1 and c_out(ε)=0. The forward pass returns f_θ = c_skip·x + c_out·F_θ(c_in·x, ·). The consistency loss samples a level index, places the online point at the noisier level, builds the cleaner target point with Euler for training-in-isolation or Heun for distillation from a teacher, restores the dropout RNG state so the online and target branches see the same dropout mask, and weights the chosen image-space or perceptual distance. In CT, setting the denoiser output to the clean data point makes the Karras ODE derivative `(x_t - denoised) / t` equal `(x_t - x) / t`, which is the PF-ODE velocity obtained from the unbiased score estimate.

```python
import numpy as np
import torch as th
import torch.nn.functional as F
from piq import LPIPS


def append_dims(x, target_dims):
    return x[(...,) + (None,) * (target_dims - x.ndim)]


def mean_flat(x):
    return x.flatten(1).mean(1)


def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        return snrs
    if weight_schedule == "snr+1":
        return snrs + 1
    if weight_schedule == "karras":
        return snrs + 1.0 / sigma_data**2
    if weight_schedule == "truncated-snr":
        return th.clamp(snrs, min=1.0)
    if weight_schedule == "uniform":
        return th.ones_like(snrs)
    raise NotImplementedError(weight_schedule)


class KarrasDenoiser:
    def __init__(
        self,
        sigma_data=0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        distillation=False,
        loss_norm="lpips",
    ):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.weight_schedule = weight_schedule
        self.distillation = distillation
        self.loss_norm = loss_norm
        self.lpips_loss = LPIPS(replace_pooling=True, reduction="none") if loss_norm == "lpips" else None
        self.num_timesteps = 40

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data**2 / ((sigma - self.sigma_min) ** 2 + self.sigma_data**2)
        c_out = (sigma - self.sigma_min) * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        scalings = self.get_scalings_for_boundary_condition if self.distillation else self.get_scalings
        c_skip, c_out, c_in = [append_dims(v, x_t.ndim) for v in scalings(sigmas)]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, rescaled_t, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised

    def consistency_losses(
        self,
        model,
        x_start,
        num_scales,
        model_kwargs=None,
        target_model=None,
        teacher_model=None,
        teacher_diffusion=None,
        noise=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        if target_model is None:
            raise NotImplementedError("Must have a target model")

        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(model, x, t, **model_kwargs)[1]

        @th.no_grad()
        def target_denoise_fn(x, t):
            return self.denoise(target_model, x, t, **model_kwargs)[1]

        if teacher_model is not None:
            @th.no_grad()
            def teacher_denoise_fn(x, t):
                return teacher_diffusion.denoise(teacher_model, x, t, **model_kwargs)[1]

        @th.no_grad()
        def heun_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0 if teacher_model is None else teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            denoiser = x0 if teacher_model is None else teacher_denoise_fn(samples, next_t)
            next_d = (samples - denoiser) / append_dims(next_t, dims)
            return x + (d + next_d) * append_dims((next_t - t) / 2, dims)

        @th.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0 if teacher_model is None else teacher_denoise_fn(x, t)
            d = (x - denoiser) / append_dims(t, dims)
            return x + d * append_dims(next_t - t, dims)

        indices = th.randint(0, num_scales - 1, (x_start.shape[0],), device=x_start.device)
        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho
        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)
        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t)

        if teacher_model is None:
            x_t2 = euler_solver(x_t, t, t2, x_start).detach()
        else:
            x_t2 = heun_solver(x_t, t, t2, x_start).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2).detach()

        weights = get_weightings(self.weight_schedule, self.get_snr(t), self.sigma_data)
        if self.loss_norm == "l1":
            loss = mean_flat((distiller - distiller_target).abs()) * weights
        elif self.loss_norm == "l2":
            loss = mean_flat((distiller - distiller_target) ** 2) * weights
        elif self.loss_norm == "l2-32":
            distiller = F.interpolate(distiller, size=32, mode="bilinear")
            distiller_target = F.interpolate(distiller_target, size=32, mode="bilinear")
            loss = mean_flat((distiller - distiller_target) ** 2) * weights
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                distiller = F.interpolate(distiller, size=224, mode="bilinear")
                distiller_target = F.interpolate(distiller_target, size=224, mode="bilinear")
            loss = self.lpips_loss((distiller + 1) / 2.0, (distiller_target + 1) / 2.0) * weights
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")
        return {"loss": loss}

@th.no_grad()
def sample_onestep(distiller, x, sigmas, generator=None, progress=False, callback=None):
    s_in = x.new_ones([x.shape[0]])
    return distiller(x, sigmas[0] * s_in)

@th.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x,
    sigmas,
    generator,
    ts,
    progress=False,
    callback=None,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * (next_t**2 - t_min**2) ** 0.5
    return x

def create_ema_and_scales_fn(
    target_ema_mode,
    start_ema,
    scale_mode,
    start_scales,
    end_scales,
    total_steps,
    distill_steps_per_iter,
):
    def ema_and_scales_fn(step):
        if target_ema_mode == "fixed" and scale_mode == "fixed":
            target_ema, scales = start_ema, start_scales
        elif target_ema_mode == "fixed" and scale_mode == "progressive":
            target_ema = start_ema
            scales = np.ceil(
                np.sqrt(
                    step / total_steps * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1) + 1
        elif target_ema_mode == "adaptive" and scale_mode == "progressive":
            scales = np.ceil(
                np.sqrt(
                    step / total_steps * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            target_ema = np.exp(start_scales * np.log(start_ema) / scales)
            scales = scales + 1
        elif target_ema_mode == "fixed" and scale_mode == "progdist":
            distill_stage = step // distill_steps_per_iter
            scales = np.maximum(start_scales // (2**distill_stage), 2)
            sub_stage = np.maximum(
                step - distill_steps_per_iter * (np.log2(start_scales) - 1), 0
            )
            sub_stage = sub_stage // (distill_steps_per_iter * 2)
            sub_scales = np.maximum(2 // (2**sub_stage), 1)
            scales = np.where(scales == 2, sub_scales, scales)
            target_ema = 1.0
        else:
            raise NotImplementedError
        return float(target_ema), int(scales)
    return ema_and_scales_fn


def ema_update(target_params, params, mu):
    for tp, p in zip(target_params, params):
        tp.detach().mul_(mu).add_(p, alpha=1 - mu)
```

The causal chain, start to finish: sampling a diffusion model is slow because I integrate the PF ODE step by step, even though I only want the trajectory's endpoint; so I posit a function that jumps from any point on a trajectory straight to its origin; constancy along trajectories (self-consistency) plus a pinned identity at t=ε defines that function and rules out the degenerate constant; I bake the boundary into the architecture with a skip-connection parameterization c_skip·x + c_out·F whose coefficients satisfy c_skip(ε)=1, c_out(ε)=0; I train it by taking adjacent points on a trajectory — one numerical ODE step apart using a teacher score — and matching the network's outputs, with an EMA stop-gradient target to stabilize the moving fixed point; an induction off the boundary proves this recovers the true consistency function as the step shrinks; then noticing the teacher only ever supplied a score, and the score of a Gaussian-blurred density equals the negative posterior mean −E[(x_t−x)/t²|x_t], I replace it by its unbiased single-sample estimate, and a Taylor expansion shows the distilled and the model-free objectives coincide up to o(Δt) — so the same consistency model can be trained from scratch, no diffusion teacher required; finally one network call turns noise into a sample, and a re-noise/denoise loop trades compute for quality and enables zero-shot editing.
