# Denoising Diffusion Implicit Models (DDIM)

## Problem

Denoising diffusion models match GAN sample quality with stable, likelihood-based, non-adversarial training, but sampling is prohibitively slow: the generative chain reverses a forward noising chain of ~1000 steps, each step a full network pass run sequentially (≈20 h for 50k 32×32 images vs <1 min for a GAN; ~1000 h at 256×256). DDIM accelerates sampling 10×–50× **using an already-trained model, with no retraining**, and as a bonus makes the generative process deterministic so the initial noise becomes a usable latent code.

## Key idea

The diffusion training objective, after reduction to noise prediction,

  L_γ(ε_θ) = Σ_t γ_t · E_{x_0,ε}[ ‖ ε_θ(√α_t x_0 + √(1−α_t) ε, t) − ε ‖² ]   (α_t is the *cumulative* signal coefficient),

depends on the inference process **only through the marginals** q(x_t|x_0) = N(√α_t x_0, (1−α_t)I), never through the joint q(x_{1:T}|x_0); and with per-t parameters its optimum is the same for any positive weighting γ. So a trained noise predictor is committed only to those marginals. DDIM constructs the whole family of (generally non-Markovian) inference processes that keep the same marginals, indexed by a free per-step variance σ with 0 ≤ σ_t² ≤ 1−α_{t-1}:

  q_σ(x_{t-1}|x_t,x_0) = N( √α_{t-1} x_0 + √(1−α_{t-1}−σ_t²)·(x_t − √α_t x_0)/√(1−α_t) , σ_t² I ),

where the mean coefficients are forced by requiring q_σ(x_t|x_0) to stay N(√α_t x_0, (1−α_t)I) for all t (proved by downward induction; the residual term vanishes at the mean of x_t, and σ_t² + (1−α_{t-1}−σ_t²) = 1−α_{t-1}). For t>1 the equal-covariance Gaussian KL has mean-difference scalar λ_t = √α_{t-1} − √α_t√(1−α_{t-1}−σ_t²)/√(1−α_t), so its ε-MSE weight is γ_t = λ_t²(1−α_t)/(2α_tσ_t²), with γ_1 = (1−α_1)/(2α_1σ_1²) for the decoder term (divide by dimension when the loss uses mean rather than summed squared error). Thus, for positive σ, J_σ = L_γ + C for some positive γ, and it is solved by the same unweighted L_1 already trained.

The generative process predicts x_0 from x_t via f_θ(x_t) = (x_t − √(1−α_t) ε_θ(x_t,t))/√α_t and plugs it into q_σ, giving the sampling step

  x_{t-1} = √α_{t-1}·f_θ(x_t)  +  √(1−α_{t-1}−σ_t²)·ε_θ(x_t,t)  +  σ_t ε.   (★)

Two free choices, both applied to a fixed network:

- **Stochasticity σ.** Write σ_t(η) = η·√((1−α_{t-1})/(1−α_t))·√(1−α_t/α_{t-1}). η = 1 is the stochastic ancestral endpoint for the chosen trajectory and recovers the original stochastic ancestral sampler on the full adjacent grid; **η = 0 is the deterministic limiting endpoint**, making x_0 a function of x_T — an implicit model. Deterministic sampling injects no fresh noise for a short chain to clean up, and it makes x_T a latent code supporting spherical interpolation and image encoding.
- **Trajectory length.** Because the objective is blind to the forward chain length, run (★) on a **sub-sequence** τ = (τ_1<…<τ_S) of [1..T] with S ≪ T (same marginal-consistency proof on the index pairs (τ_i, τ_{i-1})). This is the 10×–50× speedup.

At η = 0 with small steps, (★) is Euler integration of the ODE dx̄ = ε_θ(x̄/√(σ²+1)) dσ in coordinates x̄ = x/√α, σ = √((1−α)/α); fewer steps = coarser discretization (hence "consistency": same x_T → same high-level image at any S), and running it backward encodes x_0 → x_T. With the optimal ε_θ this ODE is the probability-flow ODE of the variance-exploding diffusion, with score ∇_{x̄} log p_t = -ε_θ/σ and g(t)^2 = dσ^2/dt, differing from the score-based sampler only in taking Euler steps in dσ rather than dt.

The same marginal-preserving construction also works for one-hot categorical data. With q(x_t|x_0)=Cat(α_t x_0+(1−α_t)1_K), choose q(x_{t-1}|x_t,x_0)=Cat(σ_t x_t+(α_{t-1}−σ_tα_t)x_0+((1−α_{t-1})−(1−α_t)σ_t)1_K), with nonnegative mixture weights. Marginalizing x_t recovers Cat(α_{t-1}x_0+(1−α_{t-1})1_K), and replacing x_0 by f_θ gives categorical KL terms upper-bounded by (α_{t-1}−σ_tα_t)·KL(Cat(x_0)‖Cat(f_θ)), a reweighted classification loss.

## Algorithm

Training: unchanged — the unweighted ε-MSE at T = 1000. Sampling: pick τ (length S) and η; set x_{τ_S} ∼ N(0,I); for i = S down to 1 apply (★) with index pair (τ_i, τ_{i-1}), where σ_{τ_i} = σ_{τ_i}(η) and the noise term is dropped when η = 0; the final state is x_{τ_0} = x_0.

## Code

The sampler below keeps the cumulative-α convention while matching the `generalized_steps` loop: build `seq_next`, predict ε, form x_0, then combine the x_0 term, the direction term, and the η-scaled noise. Training is the standard diffusion ε-MSE.

```python
import torch

def alpha_at(alphas, t):
    # cumulative coefficient with alpha_0 := 1 (the t = -1 sentinel maps to alpha_0)
    a = torch.cat([alphas.new_ones(1), alphas], dim=0)
    return a.index_select(0, t + 1).view(-1, 1, 1, 1)

def q_sample(x0, t, alphas, eps):
    a = alpha_at(alphas, t)
    return a.sqrt() * x0 + (1 - a).sqrt() * eps

def training_loss(model, x0, alphas):
    t = torch.randint(0, len(alphas), (x0.size(0),), device=x0.device)
    eps = torch.randn_like(x0)
    xt = q_sample(x0, t, alphas, eps)
    return ((model(xt, t.float()) - eps) ** 2).mean()

@torch.no_grad()
def sample(model, alphas, x, seq, eta=0.0):
    """
    model(x_t, t) -> predicted noise eps.
    alphas: cumulative signal coefficients alpha_1..alpha_T (decreasing in (0, 1]).
    x:   initial latent at the largest selected noise level; usually x_T ~ N(0, I).
    seq: increasing zero-based indices into alphas; len(seq) = S sampling steps.
    eta: 0 -> deterministic implicit sampler; 1 -> DDPM-style stochastic endpoint.
    """
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    xs = [x]
    x0_preds = []
    for i, j in zip(reversed(seq), reversed(seq_next)):          # walk tau backwards
        t = torch.ones(n, device=x.device) * i
        t_next = torch.ones(n, device=x.device) * j
        a = alpha_at(alphas, t.long())                           # alpha_{tau_i}
        a_next = alpha_at(alphas, t_next.long())                  # alpha_{tau_{i-1}}

        xt = xs[-1].to(x.device)
        eps = model(xt, t)                                       # predicted noise
        x0 = (xt - eps * (1 - a).sqrt()) / a.sqrt()              # predicted x_0 = f_theta
        x0_preds.append(x0)

        c1 = eta * (((1 - a / a_next) * (1 - a_next) / (1 - a)).sqrt())
        c2 = ((1 - a_next) - c1 ** 2).sqrt()
        xt_next = a_next.sqrt() * x0 + c1 * torch.randn_like(xt) + c2 * eps
        xs.append(xt_next)
    return xs, x0_preds

def make_seq(num_timesteps, num_sampling_steps, kind="uniform"):
    if kind == "uniform":
        skip = max(num_timesteps // num_sampling_steps, 1)
        return list(range(0, num_timesteps, skip))
    return [int(s) for s in (
        torch.linspace(0, (0.8 * num_timesteps) ** 0.5, num_sampling_steps) ** 2
    ).tolist()]

# x_T = torch.randn(batch, C, H, W)
# xs, x0_preds = sample(model, alphas, x_T, make_seq(1000, 50), eta=0.0)
# x_0 = xs[-1]
```

Notes: `eta=0.0` is DDIM (deterministic, fast, latent code); `eta=1.0` is the ancestral/noisy endpoint, exactly the original stochastic sampler when `seq` is the full adjacent grid. `S` trades compute for quality; the same trained `model` serves all settings.
