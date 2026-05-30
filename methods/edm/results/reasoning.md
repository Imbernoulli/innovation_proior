The thing that actually bugs me is that I have a half-dozen diffusion models in front of me and they all look like separate machines. One is a discrete Markov chain trained by a variational bound. One is a ladder of score estimates trained by score matching and sampled with annealed Langevin. One is a pair of stochastic differential equations with their own reverse-time formula. Each comes with its own noise schedule, its own way of scaling the image, its own discretization into steps, its own solver, its own input/output normalization for the network, its own loss weighting. And in every write-up these choices arrive welded together — change the schedule and you seem to have to re-derive the sampler; change how the network is normalized and the loss weighting comes along for the ride. I don't believe they're actually coupled. I think the coupling is an artifact of deriving each model "theory-first," and that underneath there's one object with a fistful of independent knobs. So let me try to strip everything down to a single continuous-time process and then, knob by knob, ask what the optimal setting is — not which setting some particular theory hands me.

Start with what's genuinely shared. Whatever the model, there's a clean data distribution p_data(x) with per-coordinate standard deviation σ_data, and the whole game is built on blurring it with Gaussian noise. Convolve p_data with N(0, σ²I) and call the result p(x; σ). At σ ≫ σ_data this is indistinguishable from pure noise; at σ → 0 it's the data. Generation is a walk down σ: start near pure noise, end at the data. The vector field that knows which way "toward data" points is the score ∇_x log p(x; σ), and the lovely thing about it — this is Hyvärinen's point — is that it doesn't care about the normalizing constant of p(x; σ), which is hopeless to compute.

How do I get the score without the normalizer? Vincent's denoising identity. Consider the L2 denoising problem: minimize E_{y∼p_data} E_{n∼N(0,σ²I)} ‖D(y+n; σ) − y‖² over functions D, separately at each σ. This is just regression of the clean signal onto the noisy input, and the minimizer is the conditional mean, D(x; σ) = E[y | x]. And then the score is ∇_x log p(x; σ) = (D(x; σ) − x)/σ². Let me actually convince myself of the connection for a finite dataset {y_i}, because I'll want the closed form later. With a finite set, p_data is a sum of deltas, so p(x; σ) = (1/Y) Σ_i N(x; y_i, σ²I). The denoising loss splits over x: writing it as ∫ [ (1/Y) Σ_i N(x; y_i, σ²I) ‖D(x;σ) − y_i‖² ] dx, I can minimize the bracket pointwise in x; setting the gradient w.r.t. D(x;σ) to zero gives Σ_i N(x; y_i, σ²I)(D − y_i) = 0, i.e. D(x; σ) = Σ_i N(x; y_i, σ²I) y_i / Σ_i N(x; y_i, σ²I), a softmax-weighted average of data points. Now differentiate log p directly: ∇_x N(x; y_i, σ²I) = N(x; y_i, σ²I)·(y_i − x)/σ², so ∇_x log p = Σ_i N(·)(y_i − x)/σ² / Σ_i N(·) = ( Σ_i N(·)y_i / Σ_i N(·) − x )/σ². The fraction is exactly the optimal denoiser. So ∇_x log p(x; σ) = (D(x; σ) − x)/σ². Denoiser and score are the same object. Good — I'll model the denoiser with a network and read off the score whenever I need it.

Now the process. The continuous formulation I've seen writes a forward SDE dx = f(t) x dt + g(t) dω, and then either reverses it or uses its deterministic "probability-flow" twin dx = [f(t) x − ½ g(t)² ∇_x log p_t(x)] dt, which has the same marginals p_t at every t. I'm going to start from the deterministic ODE, not the SDE, because it's the cleaner thing to analyze: the only randomness is the starting noise, the trajectories are honest curves I can reason about geometrically, and I can talk about truncation error of a solver. I'll bring stochasticity back later as a generalization if it earns its place.

But f and g bother me. They're the "first-class citizens" of the SDE derivation, and yet they're not what I care about. What I care about is the family of marginals p(x; σ) — those are what the network has to learn, what bootstraps sampling, what determines how the walk behaves. f and g are bookkeeping for how some particular SDE happens to realize those marginals. Let me try to throw them out and write the ODE directly in terms of the marginals.

The perturbation kernel is N(s(t) x_0, s(t)² σ(t)² I), where s(t) = exp ∫₀ᵗ f and σ(t) = √(∫₀ᵗ g²/s²); s is an overall rescaling of the image and σ is the effective noise level. The marginal is p_t(x) = ∫ N(x; s x_0, s²σ²I) p_data(x_0) dx_0. Pull the scale out: N(x; s x_0, s²σ²I) = s^{−d} N(x/s; x_0, σ²I), so p_t(x) = s^{−d} ∫ N(x/s; x_0, σ²I) p_data(x_0) dx_0 = s^{−d} p(x/s; σ). There it is — the time-dependent marginal is just the σ-mollified data density p(·; σ), evaluated at the de-scaled point x/s, times a constant. The f-and-g machinery has collapsed into two interpretable functions: σ(t), the noise level, and s(t), the rescaling.

Push this through the ODE. The drift uses ∇_x log p_t(x) = ∇_x [ log s^{−d} + log p(x/s; σ) ]. The first term is a constant in x, so its gradient is zero — the s^{−d} prefactor vanishes. So the score term is just ∇_x log p(x/s; σ). Now I still have f and g sitting in front; rewrite them via their definitions. From s = exp ∫f, take a log and differentiate: ∫₀ᵗ f = log s, so f = ṡ/s. From σ² = ∫ g²/s², differentiate: g²/s² = d(σ²)/dt = 2 σ̇ σ, so g = s √(2 σ̇ σ). Substitute both into [f x − ½ g² ∇log]:

dx = [ (ṡ/s) x − ½ · 2 s² σ̇ σ · ∇_x log p(x/s; σ) ] dt = [ (ṡ/s) x − s² σ̇ σ ∇_x log p(x/s; σ) ] dt.

That's the whole probability-flow ODE, written entirely in σ(t) and s(t). Every diffusion model I started with is now a *reparameterization* of this single ODE: choosing σ(t) reparameterizes time, choosing s(t) reparameterizes the image. No SDE needed. And if I set s(t) = 1 — no rescaling — it shrinks to

dx = − σ̇ σ ∇_x log p(x; σ) dt,

which I can immediately rewrite with the denoiser, since ∇log p = (D − x)/σ²: dx = − σ̇ σ (D(x;σ) − x)/σ² dt = σ̇/σ · (x − D(x;σ)) dt.

Now I get to pick σ(t). The √t schedule is "mathematically natural" because it's constant-speed heat diffusion, but natural-for-the-physics is not the same as good-for-the-solver, and I'm choosing this to minimize sampling cost, not to honor a heat equation. Let me look at what σ(t) = t does. Then σ̇ = 1 and the ODE becomes

dx/dt = (x − D(x; t))/t,

so σ and t are literally the same variable. Stare at the right-hand side. At any point x at time t, the tangent points along x − D(x; t) — straight away from the current denoiser output. A single Euler step all the way to t = 0 would land exactly on D(x; t): x + (0 − t)·(x − D)/t = D. So at every moment the trajectory aims directly at "what the denoiser thinks the clean image is." How fast does that aim swing around as t changes? The denoiser output D(x; t) is the posterior-mean estimate of the clean image; for large t it's essentially the dataset mean (the noise drowns everything), for small t it's essentially x itself, and it only transitions meaningfully across a fairly narrow band of intermediate t. So the tangent direction is nearly constant at both ends and only bends in the middle. Nearly-constant tangent means nearly-straight trajectory means small curvature, and curvature is exactly what a numerical solver pays for. So σ(t) = t, s(t) = 1 should give the straightest trajectories and the cheapest sampling. I'll take it. (And it's the choice DDIM was implicitly making, which is reassuring — that sampler was already cheap.)

Why does this matter so much? Because the reason diffusion sampling is slow is purely numerical: solving the ODE means taking finite steps, each step drifts off the true trajectory by a local truncation error, and these accumulate. For a first-order method like Euler the local error is O(h²) in the step size h; the global error is bounded by a constant times the worst local error (assuming the denoiser is Lipschitz, which the architectures are). So if I want fewer steps, I want to crush the worst local error, and that means two things: a better solver, and a step schedule that puts the error where it can afford to be.

Take the schedule first. I measured — at least in my head, from the structure — that an Euler step's local error is large at low σ and small at high σ, and barely depends on which sample I'm at. The sample-independence is great news: I don't need a per-sample adaptive schedule, one fixed sequence {σ_i} suffices. And large-error-at-low-σ tells me steps should shrink as σ decreases. Let me parameterize that. I want {σ_i} to be a warp of a uniform grid; a polynomial warp σ_i = (A i + B)^ρ is the simplest knob, where ρ ≥ 1 bunches the steps toward low σ. Pin the endpoints: σ_0 = σ_max and σ_{N−1} = σ_min. From σ_0 = B^ρ and σ_{N−1} = (A(N−1) + B)^ρ I get B = σ_max^{1/ρ} and A(N−1) = σ_min^{1/ρ} − σ_max^{1/ρ}, so

σ_i = ( σ_max^{1/ρ} + (i/(N−1)) (σ_min^{1/ρ} − σ_max^{1/ρ}) )^ρ for i < N, and σ_N = 0.

ρ = 1 is uniform-in-σ; as ρ → ∞ it becomes a geometric sequence (which is what the VE model used, so that's a special case again). What ρ equalizes the per-step error? With Euler, pushing the measured local error through this warp points near ρ ≈ 2 for a raw-error balance, but the remaining error is still too large. If I switch to the second-order solver I am about to add, the same diagnostic becomes much flatter around ρ ≈ 3. But equal raw error per step is not the same as equal *perceptual* damage. Errors at low σ shape the fine detail you actually see; σ_max is somewhat arbitrary — I picked some large cutoff — so error spent up there barely changes the final distribution. So I should deliberately tilt the schedule to spend more accuracy at low σ at the cost of high σ, i.e. push ρ above 3. A broad sweep over generated-image quality keeps improving past 3 and settles into a stable good region; ρ = 7 is a safe pick across datasets. Endpoints σ_min ≈ 0.002, σ_max ≈ 80.

Now the solver. Euler assumes dx/dt is constant across the step, which is exactly wrong on a curving trajectory — and I've just argued the curvature, while small, is concentrated in the middle band where it matters. The cheapest fix that actually corrects for the change in slope is Heun's method (the trapezoidal rule, improved Euler): take the Euler step to get a predictor at the endpoint, evaluate the derivative *there*, average the two slopes, and redo the step. Where Euler has O(h²) local error, this trapezoidal averaging gives O(h³), and it costs exactly one extra network evaluation per step. At a fixed budget of network evaluations that trade is overwhelmingly worth it — one extra eval per step buys an order in the error. One caveat: the final step lands at σ = 0, and the derivative (x − D)/σ blows up there, so I revert to plain Euler on that last step.

Let me check I'm not leaving a free parameter on the table. Heun is one member of a family of two-stage second-order Runge–Kutta schemes, x_{i+1} = x_i + h[(1 − 1/2α) d_i + (1/2α) f(x_i + α h d_i)], where α says where to probe the correction slope: α = 1 is Heun (probe at the far endpoint), α = ½ is the midpoint method, α = ⅔ is Ralston. They're all second order and all cost the same. Is α = 1 special? Two reasons it is. First, in experiments it's essentially optimal anyway (a hair past 1 sometimes edges it, but values above 1 overshoot the target time and I can't justify them, so I won't add a knob). Second, and this is the load-bearing reason: α = 1 evaluates the correction at *exactly* t_{i+1}, the next scheduled noise level. That means I never query the network at an off-grid σ — which lets me run this solver on a network that was only ever trained at a discrete set of σ values. The other α's probe at intermediate σ's the network may not know. So fix α = 1, i.e. Heun.

So deterministic sampling is settled: ρ = 7 schedule, Heun with an Euler fallback at the end. Before I add stochasticity, I want to nail the part I think is actually the heart — how the network is parameterized — because right now "D(x; σ)" is a fiction; a network can't just *be* the denoiser.

Here's why it can't. The network's input is x = y + n, whose variance is σ_data² + σ² — it ranges over orders of magnitude as σ sweeps from 0.002 to 80. Feeding that straight into a network violates the basic hygiene of keeping inputs near unit variance. The standard patch normalizes the input and trains the network to predict the *unit-variance noise vector*, reconstructing the clean signal as D = x − σ·F(·). But look at what that does at large σ: the network's job is to nail the noise so precisely that x − σF lands on the clean image, and any error ε in F becomes σε in the output — error amplification that grows linearly in σ, right where the problem is hardest. At large σ it's obviously easier to just predict the expected clean image directly, since the input carries almost no signal anyway. And at small σ the opposite is true: the input is almost the clean image, so predicting the noise (a small correction) is easier and predicting the whole image is silly. So neither "predict the signal" nor "predict the noise" is right everywhere; the right choice slides with σ. Let me build that slide in explicitly rather than commit to either end. Give the denoiser a σ-dependent skip connection so the network output can be blended with the raw input:

D_θ(x; σ) = c_skip(σ) x + c_out(σ) F_θ( c_in(σ) x ; c_noise(σ) ),

where c_in scales the input, c_out scales the output, c_skip mixes in the input directly, c_noise turns σ into the network's conditioning input, and F_θ is the actual network. Now I get to *derive* the c's from principles instead of inheriting them.

The training objective, over noise levels, is E_{σ,y,n}[ λ(σ) ‖D_θ(y+n; σ) − y‖² ] with σ drawn from some p_train, y ∼ p_data, n ∼ N(0, σ²I). I want to know what the network F_θ is really being asked to do, so rewrite the loss in terms of F_θ. Substitute D_θ = c_skip x + c_out F (writing x = y + n):

λ ‖ c_skip(y+n) + c_out F − y ‖² = λ c_out² ‖ F − (1/c_out)( y − c_skip(y+n) ) ‖².

So F_θ is being regressed, with effective weight λ(σ) c_out(σ)², onto the effective target F_target = (1/c_out)( y − c_skip (y+n) ). Three requirements, in order.

First, the network's input should have unit variance. The input is c_in(y + n). Since y and n are independent with variances σ_data² and σ²,

Var[ c_in(y+n) ] = c_in² (σ_data² + σ²) = 1 ⟹ c_in(σ) = 1/√(σ² + σ_data²).

Second, the network's *target* should have unit variance, so the regression is well-scaled. F_target = (1/c_out)[ y − c_skip(y+n) ] = (1/c_out)[ (1 − c_skip) y − c_skip n ]. (Sign on the noise term: y − c_skip y − c_skip n = (1−c_skip)y − c_skip n.) Its variance:

Var[F_target] = (1/c_out²)[ (1 − c_skip)² σ_data² + c_skip² σ² ] = 1,

so c_out(σ)² = (1 − c_skip)² σ_data² + c_skip² σ². That fixes c_out *given* c_skip, but c_skip is still free — which is the third requirement's job.

Third, pick c_skip to amplify the network's errors as little as possible. An error δ in F shows up in D as c_out δ, so minimizing error amplification means minimizing c_out, equivalently minimizing c_out² (it's nonnegative). It's a clean convex problem in c_skip:

d/dc_skip [ (1 − c_skip)² σ_data² + c_skip² σ² ] = 0
⟹ σ_data² · 2(c_skip − 1) + σ² · 2 c_skip = 0
⟹ (σ² + σ_data²) c_skip = σ_data²
⟹ c_skip(σ) = σ_data² / (σ² + σ_data²).

Now back-substitute to finish c_out. With c_skip = σ_data²/(σ²+σ_data²), I have 1 − c_skip = σ²/(σ²+σ_data²), so

c_out² = ( σ²/(σ²+σ_data²) )² σ_data² + ( σ_data²/(σ²+σ_data²) )² σ²
       = [ σ⁴ σ_data² + σ_data⁴ σ² ] / (σ²+σ_data²)²
       = σ² σ_data² (σ² + σ_data²) / (σ²+σ_data²)²
       = σ² σ_data² / (σ² + σ_data²),

so

c_out(σ) = σ · σ_data / √(σ² + σ_data²).

Let me sanity-check the two limits, because this is exactly the place a sign or a factor slips. As σ → 0: c_skip → σ_data²/σ_data² = 1, c_out → 0, c_in → 1/σ_data. So D_θ → 1·x + 0 = x — the denoiser returns its input, which is correct, since there's no noise to remove and the network's contribution is scaled to zero (so its errors can't hurt). As σ → ∞: c_skip → 0, c_out → σ_data, c_in → 1/σ. So D_θ → c_out F — the skip is off and the output is the network's prediction at a sane unit scale, which is right because the input is pure noise and the answer must be built from scratch, not from x. And crucially c_out stays bounded (→ σ_data) instead of blowing up like the σ-multiplier of the noise-prediction parameterization. The skip is automatically doing "predict-the-noise-ish" at low σ and "predict-the-signal-ish" at high σ, sliding between them exactly where the variance algebra says to.

Fourth, the loss weighting. The effective weight on the network's regression is λ(σ) c_out(σ)². If I want every noise level to contribute equally to the gradient at the start — no level dominating because its target happens to be large — I should make that effective weight uniform:

λ(σ) c_out(σ)² = 1 ⟹ λ(σ) = 1/c_out(σ)² = (σ² + σ_data²)/(σ · σ_data)².

Let me verify this actually flattens the initial loss. Standard practice initializes the output layer to zero, so at init F_θ ≡ 0 and D_θ = c_skip x. Plug into the per-σ loss with this λ:

E[ λ ‖ c_skip(y+n) − y ‖² ] = E[ λ ‖ (c_skip − 1) y + c_skip n ‖² ].

With c_skip = σ_data²/(σ²+σ_data²) and c_skip − 1 = −σ²/(σ²+σ_data²), the bracket is ‖ (σ_data² n − σ² y)/(σ²+σ_data²) ‖². Multiply by λ = (σ²+σ_data²)/(σ σ_data)²:

= 1/(σ σ_data)² · 1/(σ²+σ_data²) · ‖ σ_data² n − σ² y ‖²,

and per coordinate E[σ_data² n − σ² y]² = σ_data⁴ Var(n) + σ⁴ Var(y) = σ_data⁴ σ² + σ⁴ σ_data² = σ² σ_data² (σ_data² + σ²) (the cross term vanishes, y ⊥ n). So the per-coordinate value is σ² σ_data² (σ²+σ_data²) / [ (σ σ_data)² (σ²+σ_data²) ] = 1 at every σ, and the usual mean reduction preserves that equality. The weighting genuinely equalizes the starting loss across noise levels.

That leaves c_noise, the map from σ to the network's conditioning scalar. There's no variance principle pinning this — it's just how the network reads off "which noise level am I at," and the network can learn around any smooth monotone encoding. So I set it empirically to a convenient compressing transform, c_noise(σ) = ¼ ln σ, and move on.

One more training knob: which σ to actually train on, p_train. The weighting λ flattens the loss *at initialization*, but after training the per-σ loss is still U-shaped — and for a real reason. At very low σ the noise is so faint that discerning it is both trivial and pointless (it carries no signal worth a gradient step). At very high σ the best possible answer collapses toward the dataset average no matter the input, so there's nothing to learn either. The action is all at intermediate σ. So don't spend training samples uniformly; concentrate them in the middle. The cleanest way is to draw ln σ from a normal: ln σ ∼ N(P_mean, P_std²). Centered a bit below σ_data with a wide-ish spread does it — P_mean = −1.2, P_std = 1.2, with σ_data = 0.5 for these images. That's the training noise distribution.

There is also a plain overfitting issue on smaller image sets. The GAN toolbox already has a useful answer: apply geometric augmentations during training, and prevent leakage by giving the augmentation parameters to the network as conditioning information. Here that fits cleanly. I apply the augmentation before adding noise, because the denoising target should be the augmented clean image paired with its noisy version; I pass the augmentation labels through the same raw network call; and at sampling time those labels are zero, so the sampler is not asked to generate augmented images. This is not part of the score-matching identity, but it is part of the training recipe that keeps the denoiser from memorizing.

Now stochasticity, which I deferred. Deterministic sampling is clean and few-step, but it tends to give slightly worse quality than re-injecting fresh noise each step. Since the ODE and the noise-injecting SDE share marginals in the continuum, the benefit must be a *discretization* effect. Write the general process keeping the same marginals as the probability-flow ODE plus a Langevin term: dx = −σ̇σ ∇log p dt ± [ β σ² ∇log p dt + √(2β) σ dω ]. The bracket is a Langevin diffusion — a deterministic score-driven decay of noise plus a fresh-noise injection — whose two pieces have canceling net effect on the marginal, so β(t) just sets the *rate at which old noise is swapped for new* without changing where the distribution should be. The point of that exchange: the Langevin part actively drags a sample back onto the correct marginal at each level, scrubbing out errors the earlier steps made. The SDE-derivation's particular β = σ̇/σ (which makes the score drop out of the forward equation) has no special virtue here — it's one arbitrary choice. So treat β as a free amount-of-stochasticity to tune.

But churn isn't free. If I add and remove a lot of noise, detail gradually washes out and colors drift toward oversaturation. The likely cause: a learned denoiser isn't a perfectly conservative field, and being L2-trained it regresses toward the mean — it removes a touch too much noise each cycle. So I add stochasticity carefully, as a tailored procedure rather than a generic SDE solver. Each step: from x_i at level t_i, first nudge *up* to a higher level t̂_i = t_i + γ_i t_i by adding fresh noise, then take one deterministic Heun step from t̂_i down to t_{i+1}. The added-noise variance to climb from t_i to t̂_i is t̂_i² − t_i², so the perturbation is √(t̂_i² − t_i²)·(fresh standard normal). Two heuristics to fight the degradation: only enable churn within a σ-window [S_min, S_max] (kill it at the extreme levels where oversaturation appears), with γ_i = S_churn/N clamped so I never inject more noise than is already there (cap at √2 − 1); and inflate the fresh-noise standard deviation slightly above 1, S_noise a hair over 1, to compensate for the denoiser eating a bit too much noise. S_churn = 0 recovers pure deterministic sampling — which, as the network gets better, is often what wins anyway, since a more accurate denoiser needs less error-correcting churn.

Let me write it down, mirroring the structure I'd actually run.

```python
import numpy as np
import torch

# --- the network parameterization: wrap a raw U-Net F_theta into a denoiser D(x; sigma) ---
class EDMPrecond(torch.nn.Module):
    def __init__(self, model, sigma_data=0.5):
        super().__init__()
        self.model = model            # raw network F_theta(scaled_input, noise_cond, labels)
        self.sigma_data = sigma_data

    def forward(self, x, sigma, class_labels=None):
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)              # minimizes error amplification
        c_out  = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()    # unit-variance target
        c_in   = 1 / (sd ** 2 + sigma ** 2).sqrt()             # unit-variance input
        c_noise = sigma.log() / 4                              # empirical sigma -> conditioning
        F_x = self.model(c_in * x, c_noise.flatten(), class_labels=class_labels)
        return c_skip * x + c_out * F_x                        # D = c_skip x + c_out F

# --- training: log-normal sigma, weight lambda = 1/c_out^2 ---
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean, self.P_std, self.sigma_data = P_mean, P_std, sigma_data

    def __call__(self, denoiser, images, class_labels=None):
        # ln sigma ~ N(P_mean, P_std^2): concentrate training at intermediate noise levels
        rnd = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2  # 1/c_out^2
        n = torch.randn_like(images) * sigma
        D_yn = denoiser(images + n, sigma, class_labels)
        return weight * (D_yn - images) ** 2                   # equal effective weight at every sigma

# --- sampling: rho=7 schedule, Heun (with Euler fallback at the end), optional churn ---
@torch.no_grad()
def edm_sampler(denoiser, latents, class_labels=None, num_steps=18,
                sigma_min=0.002, sigma_max=80, rho=7,
                S_churn=0, S_min=0, S_max=float('inf'), S_noise=1):
    # sigma_i = (sigma_max^{1/rho} + i/(N-1)(sigma_min^{1/rho} - sigma_max^{1/rho}))^rho, sigma_N = 0
    i = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1/rho) + i/(num_steps-1) * (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # append sigma_N = 0

    x_next = latents.to(torch.float64) * t_steps[0]               # start from noise at sigma_max
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        # churn up to t_hat = t_cur + gamma t_cur by injecting fresh noise (variance t_hat^2 - t_cur^2)
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        # Euler step using sigma = t: dx/dt = (x - D(x;t))/t
        d_cur = (x_hat - denoiser(x_hat, t_hat, class_labels)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        # Heun 2nd-order correction: average the slope at the endpoint (skip on the final step to 0)
        if i < num_steps - 1:
            d_prime = (x_next - denoiser(x_next, t_next, class_labels)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next
```

The chain, end to end: I refused to take the half-dozen diffusion models as separate animals and instead wrote the probability-flow ODE directly in terms of the marginals, dx = [(ṡ/s)x − s²σ̇σ ∇log p(x/s;σ)] dt, so σ(t) and s(t) became free reparameterizations of one canonical process. Demanding the straightest, cheapest-to-integrate trajectories picked σ(t)=t, s(t)=1, collapsing the ODE to dx/dt=(x−D(x;t))/t whose tangent always points at the denoiser output; a curvature/truncation-error argument then picked Heun over Euler (one extra eval buys an order in local error) and the ρ=7 warped schedule (spend accuracy at low σ where it's perceptually load-bearing). The network can't be the denoiser directly because its input variance σ²+σ_data² explodes and naive noise-prediction amplifies errors by σ, so I wrapped it as D = c_skip x + c_out F(c_in x; c_noise) and *derived* c_in=1/√(σ²+σ_data²) and c_out, c_skip from unit-variance-input, unit-variance-target, and minimal-error-amplification, getting c_skip=σ_data²/(σ²+σ_data²), c_out=σσ_data/√(σ²+σ_data²); setting λ=1/c_out² equalizes the per-σ loss (provably 1 at init), and a log-normal p_train aims training at the intermediate σ where there's anything to learn. Stochasticity re-enters as a tunable Langevin churn — add noise up a level, then one Heun step down — fenced in by a σ-window and a slight noise inflation to fight the detail loss a regress-to-mean denoiser causes.
