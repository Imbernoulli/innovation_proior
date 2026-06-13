Let me start from what actually hurts. I have a generative idea I genuinely believe in — fix a forward process that slowly turns data into pure Gaussian noise, then learn the reverse chain that turns noise back into data — and it has every theoretical virtue: it is a proper latent-variable model, each step is a tractable Gaussian, I can train it by a variational bound, and I can make it arbitrarily expressive by taking many small steps. And yet when people build it, the samples come out blurry and weak. Nobody has shown this family can make a sharp image. So the architecture is not obviously the problem; the U-Net I have for image-to-image work is the same kind of net that powers the best autoregressive and score models. The problem must be in the two choices I haven't pinned down: *what exactly the reverse network should output*, and *how I weight each timestep's loss*. Those are the two free knobs, they are entangled, and the obvious settings of them are the ones that have already failed. So let me re-derive the whole thing from the bound and watch where a different choice could fall out.

The forward process is fixed. Each step is `q(x_t|x_{t-1}) = N(x_t; sqrt(1-beta_t) x_{t-1}, beta_t I)` — scale the signal down by `sqrt(1-beta_t)`, inject variance `beta_t`. The scaling matters: it keeps the total variance bounded so the chain converges to a standard normal instead of blowing up, and it is why a standard-normal prior is the right endpoint. Write `alpha_t = 1 - beta_t`. The first thing I want is to be able to jump to any timestep without simulating the whole chain, because I am going to train on random timesteps and I do not want a thousand sequential steps per gradient. Composing Gaussians: `x_1 = sqrt(alpha_1) x_0 + sqrt(beta_1) eps_1`, then `x_2 = sqrt(alpha_2) x_1 + sqrt(beta_2) eps_2 = sqrt(alpha_2 alpha_1) x_0 + sqrt(alpha_2 beta_1) eps_1 + sqrt(beta_2) eps_2`. Those last two terms are independent zero-mean Gaussians; their variances add, `alpha_2 beta_1 + beta_2 = alpha_2(1-alpha_1) + (1-alpha_2) = 1 - alpha_1 alpha_2`. So with `alpha_bar_t = prod_{s<=t} alpha_s`,

  q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I),

and in reparameterized form `x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps` for a single `eps ~ N(0,I)`. Good — one Gaussian sample takes me to any `t`. Notice `alpha_bar_t` runs from near 1 (almost clean) down to near 0 (almost pure noise) as `t` grows, and `1 - alpha_bar_t` is the noise variance climbing the other way. With a small linear `beta` schedule, `alpha_bar_T` is tiny, so `x_T` is essentially `N(0,I)` and the prior matches the chain's endpoint — `KL(q(x_T|x_0) || N(0,I))` is on the order of `1e-5` bits per dimension, negligible.

Now the training objective. I maximize a lower bound on `log p_theta(x_0)`, equivalently minimize

  L = E_q[ -log p(x_T) - sum_t log( p_theta(x_{t-1}|x_t) / q(x_t|x_{t-1}) ) ].

If I leave it like this it is a high-variance Monte-Carlo mess, because each `log` ratio is a single sample and the cross-step correlations are nasty. Let me do the standard regrouping that turns it into KL divergences between Gaussians I can compute in closed form. Condition the forward posterior on `x_0`: while `q(x_{t-1}|x_t)` alone is intractable, `q(x_{t-1}|x_t, x_0)` is a tractable Gaussian. Insert `q(x_t|x_{t-1}) = q(x_t|x_{t-1},x_0)` (Markov) and apply Bayes step by step. I rewrite each `q(x_t|x_{t-1})` as `q(x_{t-1}|x_t,x_0) · q(x_t|x_0) / q(x_{t-1}|x_0)`; the `q(x_t|x_0)/q(x_{t-1}|x_0)` factors telescope across the sum, leaving the `q(x_T|x_0)` at the top and `q(x_1|x_0)` at the bottom, and I get

  L = E_q[ KL(q(x_T|x_0) || p(x_T))  +  sum_{t>1} KL(q(x_{t-1}|x_t,x_0) || p_theta(x_{t-1}|x_t))  -  log p_theta(x_0|x_1) ].

Call them `L_T`, the `L_{t-1}` terms, and `L_0`. This is the form I want: every middle term is a KL between two Gaussians. And `L_T` has no learnable parameters — `q` is fixed and `p(x_T)=N(0,I)` is fixed — so `L_T` is just a constant I can drop from training. That already tells me the prior end of the chain is free; all the learning is in the `L_{t-1}` denoising terms and the final decoder `L_0`.

I need the forward posterior `q(x_{t-1}|x_t,x_0)` explicitly. It is Gaussian; complete the square on the product `q(x_t|x_{t-1}) q(x_{t-1}|x_0)` in the exponent. The mean and variance come out as

  q(x_{t-1}|x_t,x_0) = N(x_{t-1}; mu_tilde_t(x_t,x_0), beta_tilde_t I),
  mu_tilde_t(x_t,x_0) = ( sqrt(alpha_bar_{t-1}) beta_t / (1-alpha_bar_t) ) x_0 + ( sqrt(alpha_t) (1-alpha_bar_{t-1}) / (1-alpha_bar_t) ) x_t,
  beta_tilde_t = ( (1-alpha_bar_{t-1}) / (1-alpha_bar_t) ) beta_t.

Good. Now the reverse step I am learning is `p_theta(x_{t-1}|x_t) = N(x_{t-1}; mu_theta(x_t,t), Sigma_theta(x_t,t))`. What should `Sigma_theta` be? Sohl-Dickstein's entropy analysis gives me two natural fixed choices for the per-step variance: `sigma_t^2 = beta_t` (optimal when the data is itself standard normal) and `sigma_t^2 = beta_tilde_t` (optimal when the data is a single point); these are the upper and lower bounds on the reverse entropy for unit-variance data. They are the two extremes, and in practice they behave similarly. Let me just *fix* `Sigma_theta = sigma_t^2 I` to one of these, untrained. The alternative is to learn a diagonal `Sigma_theta` jointly with the bound, but a learned variance is one more thing to estimate from noisy signal in a term I am about to divide by, and there is no reason to expect it to be better than the two principled fixed endpoints. Fixing it also has a clean payoff: with both `q` and `p_theta` Gaussian and the variance fixed and equal, the KL between them collapses to a scaled squared distance between means,

  L_{t-1} = E_q[ (1/(2 sigma_t^2)) || mu_tilde_t(x_t,x_0) - mu_theta(x_t,t) ||^2 ] + C,

`C` independent of `theta`. So the per-step learning problem is: regress the network's mean onto the forward posterior mean. The most literal parameterization is to have the network output `mu_theta` directly and predict `mu_tilde_t`. Before I commit to that coordinate system, let me actually look at what `mu_tilde_t` *is* as a function of the network's input.

Here is the thing I keep circling: at training time the network sees `x_t`, not `x_0`. But `mu_tilde_t` is written in terms of `x_0` and `x_t` both. And `x_0` and `x_t` are not independent — they are tied by the very corruption I used to make `x_t`: `x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps`, so `x_0 = (x_t - sqrt(1-alpha_bar_t) eps) / sqrt(alpha_bar_t)`. The only thing the network does not know, given `x_t`, is `eps`. So `mu_tilde_t` is really a function of `x_t` and `eps`. Let me substitute that expression for `x_0` into `mu_tilde_t` and see what survives. Plug `x_0 = (x_t - sqrt(1-alpha_bar_t) eps)/sqrt(alpha_bar_t)` into the coefficient form:

  mu_tilde_t = coef1 · x_0 + coef2 · x_t,  coef1 = sqrt(alpha_bar_{t-1}) beta_t/(1-alpha_bar_t),  coef2 = sqrt(alpha_t)(1-alpha_bar_{t-1})/(1-alpha_bar_t).

The `x_t` terms: `coef1/sqrt(alpha_bar_t) + coef2`. Use `alpha_bar_{t-1} = alpha_bar_t/alpha_t`, so `sqrt(alpha_bar_{t-1})/sqrt(alpha_bar_t) = 1/sqrt(alpha_t)`. Then `coef1/sqrt(alpha_bar_t) = beta_t/((1-alpha_bar_t) sqrt(alpha_t))` and `coef2 = sqrt(alpha_t)(1-alpha_bar_{t-1})/(1-alpha_bar_t)`. Sum over the common denominator `(1-alpha_bar_t) sqrt(alpha_t)`: numerator `beta_t + alpha_t(1-alpha_bar_{t-1}) = beta_t + alpha_t - alpha_t alpha_bar_{t-1} = beta_t + alpha_t - alpha_bar_t = (1) - alpha_bar_t` since `beta_t + alpha_t = 1`. So the `x_t` coefficient collapses to `(1-alpha_bar_t)/((1-alpha_bar_t) sqrt(alpha_t)) = 1/sqrt(alpha_t)`. Clean. The `eps` terms: only `coef1` carries `eps`, with factor `-sqrt(1-alpha_bar_t)/sqrt(alpha_bar_t)`, giving `coef1·(-sqrt(1-alpha_bar_t)/sqrt(alpha_bar_t)) = -beta_t/((1-alpha_bar_t)sqrt(alpha_t)) · sqrt(1-alpha_bar_t) = -beta_t/(sqrt(1-alpha_bar_t) sqrt(alpha_t))`. Putting it together,

  mu_tilde_t = (1/sqrt(alpha_t)) ( x_t - ( beta_t / sqrt(1-alpha_bar_t) ) eps ).

Stare at this. The forward posterior mean, the thing my network is supposed to predict, is *the input `x_t` minus a specific multiple of the noise `eps` that was added*. Everything multiplying `x_t` is a known constant; the only unknown the network needs to supply is `eps`. So predicting `mu_tilde_t` directly is making the network re-learn how to reconstruct `x_t` from `x_t` plus a correction — it has to output something whose dominant part is just `x_t` again, with the real signal buried in a small `eps`-dependent correction that is scaled differently at every `t`. That is a poorly conditioned regression target: the target's overall scale and its dependence on the input both wander with `t`, so the net spends capacity tracking the trivial `x_t` part and the loss is dominated by it.

So let me change what the network outputs. Since `x_t` is already an input, let me bake the known `x_t`-dependence into the parameterization and have the network predict only the genuinely unknown piece — the noise. Define a function approximator `eps_theta(x_t, t)` and *set*

  mu_theta(x_t,t) = (1/sqrt(alpha_t)) ( x_t - ( beta_t / sqrt(1-alpha_bar_t) ) eps_theta(x_t,t) ),

the exact same shape as `mu_tilde_t` but with the true `eps` replaced by the network's guess `eps_theta`. Now substitute both into the squared-mean loss. The `x_t/sqrt(alpha_t)` parts are identical and cancel; only the `eps` difference survives:

  || mu_tilde_t - mu_theta ||^2 = (1/alpha_t)(beta_t/sqrt(1-alpha_bar_t))^2 || eps - eps_theta ||^2 = ( beta_t^2 / (alpha_t (1-alpha_bar_t)) ) || eps - eps_theta(x_t,t) ||^2.

So with this parameterization the per-step loss is

  L_{t-1} - C = E_{x_0, eps}[ ( beta_t^2 / (2 sigma_t^2 alpha_t (1-alpha_bar_t)) ) || eps - eps_theta( sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps, t ) ||^2 ].

The regression target is now just `eps`, a standard normal vector — zero mean, unit variance, *the same scale at every timestep*. That is the well-posed problem I wanted: the network looks at a noisy image and predicts the noise in it, a target whose statistics do not drift with `t`, and the messy `t`-dependent geometry has been pushed entirely into a known scalar weight out front. This has to be a better-conditioned thing to fit than a mean whose scale wanders.

Wait — predict the noise. Let me make sure I am not just shuffling symbols. I could also have the network predict `x_0` directly: from `x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps`, predicting `x_0` and predicting `eps` are linearly interchangeable, `eps = (x_t - sqrt(alpha_bar_t) x_0)/sqrt(1-alpha_bar_t)`. So all three — predict `mu_tilde`, predict `x_0`, predict `eps` — are *the same model* up to an invertible, `t`-dependent linear reparameterization of the output. They are not different models; they are different coordinate systems for the same map. But the *loss* is not invariant under reparameterizing the output, because I am measuring squared error in whichever coordinate the network outputs, and the implicit per-`t` weighting differs. If I train in `x_0` coordinates, then `x0_theta - x_0 = -sqrt(1-alpha_bar_t)/sqrt(alpha_bar_t) · (eps_theta - eps)`, so `||x0_theta - x_0||^2 = ((1-alpha_bar_t)/alpha_bar_t) ||eps_theta - eps||^2`. That factor is near zero when `t` is small and explodes when `alpha_bar_t` goes to zero. So `x_0`-prediction over-weights the high-noise end where `x_0` is least determined by `x_t`. The `eps` target, by contrast, has constant scale and the cleanest conditioning.

Now look harder at the reverse/sampling step this gives me, because something is poking at me. To sample `x_{t-1}` I draw from `p_theta(x_{t-1}|x_t) = N(mu_theta, sigma_t^2 I)`:

  x_{t-1} = (1/sqrt(alpha_t)) ( x_t - ( beta_t / sqrt(1-alpha_bar_t) ) eps_theta(x_t,t) ) + sigma_t z,  z ~ N(0,I).

This is `x_t`, minus a step in the direction of `eps_theta`, plus a fresh shot of Gaussian noise. A step against the predicted noise, plus noise. That is the shape of one **Langevin** step — move along an estimated vector field, then jitter. And what vector field? Recall the score of the forward marginal: `q(x_t|x_0) = N(sqrt(alpha_bar_t) x_0, (1-alpha_bar_t) I)`, so `∇_{x_t} log q(x_t|x_0) = -(x_t - sqrt(alpha_bar_t) x_0)/(1-alpha_bar_t)`. But `x_t - sqrt(alpha_bar_t) x_0 = sqrt(1-alpha_bar_t) eps`, so

  ∇_{x_t} log q(x_t|x_0) = - eps / sqrt(1-alpha_bar_t).

The score *is* the noise, up to the constant `-1/sqrt(1-alpha_bar_t)`. So a network trained to predict `eps` is, up to that scaling, a network trained to predict the score of the noised data density, `s_theta(x_t,t) ≈ -eps_theta(x_t,t)/sqrt(1-alpha_bar_t)`. My reverse step is annealed Langevin dynamics where the noise level is indexed by `t`, and the "learned gradient of the data density" is exactly `eps_theta`. The two worlds I thought were separate — variational inference on a fixed diffusion, and denoising score matching sampled by annealed Langevin — are the same construction seen in two coordinate systems. I did not impose this; it fell out of choosing to predict the noise.

Let me push the connection all the way, because it is going to tell me how to weight the loss. Denoising score matching (this is Vincent's identity) says: corrupt `x` to `x̃` with `N(x̃; x, sigma^2 I)`, and matching `s_theta(x̃)` to `∇_{x̃} log q_sigma(x̃|x) = -(x̃-x)/sigma^2` trains the network on the score of the noised density. NCSN trains one network across a ladder of noise levels and, at level `sigma`, minimizes `(1/2) E || s_theta(x̃,sigma) + (x̃-x)/sigma^2 ||^2`, then combines levels with a weight `lambda(sigma)`. NCSN's empirical observation is that at optimum `|| s_theta || ∝ 1/sigma`, so the per-level loss magnitude scales like `1/sigma^2`; to make every level contribute equally they pick `lambda(sigma) = sigma^2`, and then the weighted per-level loss becomes

  lambda(sigma) · (1/2) E || s_theta + (x̃-x)/sigma^2 ||^2 = (1/2) E || sigma s_theta(x̃,sigma) + (x̃-x)/sigma ||^2.

But `(x̃-x)/sigma` is exactly the unit-variance noise that produced `x̃` — call it the noise. So NCSN's `sigma^2`-weighted score-matching objective is, term by term, *an unweighted mean-squared error between the added noise and a (rescaled) network output*. That is precisely my `eps`-prediction loss with the per-`t` weight thrown away. The thermodynamics derivation and the score-matching derivation hand me the same loss, and the score-matching side has already discovered, empirically, that the *unweighted* version — equal magnitude across noise levels — is the one that trains well.

So now the second free knob, the loss weighting, is staring at me. My principled per-step weight from the bound is `w_t = beta_t^2 / (2 sigma_t^2 alpha_t (1-alpha_bar_t))`. Let me see what it actually does across `t`. For early middle terms, `beta_t` is tiny but `(1-alpha_bar_t)` is also tiny, so the ratio stays large; with the linear schedule and `sigma_t^2 = beta_t`, the first non-decoder KL term is around `0.27`, while a middle-chain term is around `0.005`. Early `t` means `x_t` is barely corrupted, almost the clean image with a whisper of noise. Denoising a whisper of noise is the *easiest* sub-problem — and the true bound weights it the heaviest. Meanwhile the genuinely hard sub-problems, the large-`t` denoising where the image is mostly noise and the network has to recover global structure, are weighted much less. That is backwards for sample quality: I am pouring gradient into perfecting imperceptible high-frequency details and starving the terms that decide whether the picture has any coherent content. The bound is the right objective for *codelength* — those tiny-`t` terms really do carry the bits — but codelength is not what I am optimizing for; I want sharp samples.

So drop the weight. Set every per-`t` weight to a constant 1:

  L_simple(theta) = E_{t, x_0, eps}[ || eps - eps_theta( sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps, t ) ||^2 ],  t ~ Uniform{1,...,T}.

This is exactly NCSN's equal-magnitude-per-level objective, arrived at from the variational side. Relative to the true bound it down-weights the small-`t` terms and up-weights the large-`t` ones — it tells the network to spend its capacity on the difficult, high-noise denoising tasks that actually shape the image, rather than on the easy near-clean ones. And it is dramatically simpler to implement: sample a timestep, sample noise, corrupt, predict the noise, take the MSE. No per-term weight to compute, no variance to track in a denominator. The `t=1` term plays the role of the discrete decoder `L_0` (the Gaussian density times the bin width, ignoring edge effects and `sigma_1^2`), so the single MSE covers the whole chain including the final decode. `L_T` never appears because the forward variances are fixed.

Let me also fix the loose ends so the chain is consistent end to end. The reverse variance `sigma_t^2`: with the simplified loss it no longer appears in training, so I just pick one of the two principled fixed values for sampling. The data I scale to `[-1,1]` so the network always sees inputs on the same scale as the `N(0,I)` prior. The schedule: `T=1000`, `beta` linear from `1e-4` to `0.02`, chosen small relative to `[-1,1]` so each reverse step is well-approximated by a Gaussian (the Sohl-Dickstein small-step condition) and so `alpha_bar_T` is small enough that `x_T` is essentially standard normal. And the sampler's coefficients are not hand-tuned the way a bolt-on Langevin sampler's would be — they are `beta_t`, `alpha_t`, `alpha_bar_t` straight from the forward process, so training the bound *is* training the sampler.

Now I want the two functions that actually live in the code, because the whole method reduces to a matched training-target / inversion pair. Training: the target the network regresses on is just the noise, `target = eps`. Inversion, used at sampling time to turn the network output back into a prediction of the clean image: invert `x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps_theta` for `x_0`,

  x0_hat = ( x_t - sqrt(1-alpha_bar_t) eps_theta ) / sqrt(alpha_bar_t).

These two are exact inverses by construction — the inversion is literally solving the corruption equation for `x_0` given the predicted noise — so the sampler is guaranteed consistent with what the network was trained to do. (Equivalently `x0_hat = sqrt(1/alpha_bar_t) x_t - sqrt(1/alpha_bar_t - 1) eps_theta`, the same thing with the constants precomputed.) From `x0_hat` the reverse step plugs back through `mu_tilde_t(x_t, x0_hat)` to take one denoising move, which is exactly the `mu_theta` step above.

So let me write it as the canonical training routine — sample `t`, sample noise, corrupt, regress the noise — together with the noise-to-`x_0` inversion that the sampler calls:

```python
import torch


  # fixed forward-diffusion tensors (no learnable parameters)
def get_schedule(betas):
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alpha": alphas_cumprod.sqrt(),                    # sqrt(alpha_bar_t)
        "sqrt_one_minus_alpha": (1.0 - alphas_cumprod).sqrt(),  # sqrt(1 - alpha_bar_t)
    }


def q_sample(x_0, noise, t, schedule):
    # forward process: x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps  (jump straight to step t)
    sa = schedule["sqrt_alpha"][t].view(-1, 1, 1, 1)
    soma = schedule["sqrt_one_minus_alpha"][t].view(-1, 1, 1, 1)
    return sa * x_0 + soma * noise


  # the parameterization: a matched (target, inverse) pair

def compute_training_target(x_0, noise, timesteps, schedule):
    # epsilon prediction: the network regresses on the noise itself.
    # target scale is N(0, I) at every t -> well-conditioned, unweighted MSE.
    return noise


def predict_x0(model_output, x_t, timesteps, schedule):
    # invert x_t = sqrt(abar) x_0 + sqrt(1-abar) eps for x_0, using eps = model_output:
    #   x_0 = (x_t - sqrt(1-abar) eps) / sqrt(abar)
    sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
    return (x_t - sqrt_one_minus_alpha * model_output) / sqrt_alpha.clamp(min=1e-8)


  # training loop: L_simple = E || eps - eps_theta(x_t, t) ||^2

def train_step(model, x_0, schedule, T):
    B = x_0.shape[0]
    t = torch.randint(0, T, (B,), device=x_0.device)        # t ~ Uniform{0,...,T-1}
    noise = torch.randn_like(x_0)                           # eps ~ N(0, I)
    x_t = q_sample(x_0, noise, t, schedule)                 # corrupt to level t
    target = compute_training_target(x_0, noise, t, schedule)  # = eps
    pred = model(x_t, t)                                    # time-conditioned U-Net
    return ((pred - target) ** 2).mean()                   # unweighted MSE  (= L_simple)
```

And the same logic in the textbook training/sampling pair, to show it is one self-consistent chain — train by predicting the noise, sample by stepping against the predicted noise as a learned score:

```python
  # Training
  #   repeat:
  #     x_0 ~ data
  #     t   ~ Uniform({1, ..., T})
  #     eps ~ N(0, I)
  #     x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps
  #     gradient step on  || eps - eps_theta(x_t, t) ||^2

  # Sampling  (eps_theta as a learned score; one Langevin-like step per t)
  #   x_T ~ N(0, I)
  #   for t = T, ..., 1:
  #     z = N(0, I) if t > 1 else 0
  #     x_{t-1} = (1/sqrt(alpha_t)) ( x_t - (1-alpha_t)/sqrt(1-abar_t) eps_theta(x_t, t) ) + sigma_t z
  #   return x_0
```

Let me lay out the causal chain so I trust it. I believed in fix-the-forward-diffusion, learn-the-reverse, but it made blurry images, and the two unsettled knobs were *what the reverse net outputs* and *how the timesteps are weighted*. I wrote the variational bound, regrouped it into closed-form Gaussian KL terms by conditioning the forward posterior on `x_0`, and saw the per-step loss is a squared distance to the posterior mean `mu_tilde_t`. Substituting `x_0 = (x_t - sqrt(1-alpha_bar_t) eps)/sqrt(alpha_bar_t)` showed `mu_tilde_t = (1/sqrt(alpha_t))(x_t - (beta_t/sqrt(1-alpha_bar_t)) eps)` — i.e. the only unknown given `x_t` is the noise `eps`, and predicting the mean directly forces the net to relearn the trivial `x_t` part. Re-parameterizing the network to predict `eps` instead reduced the loss to `|| eps - eps_theta ||^2` with a clean per-`t` weight, a target with constant `N(0,I)` scale — better conditioned than predicting `x_0` (whose MSE equals `(1-alpha_bar_t)/alpha_bar_t` times the corresponding epsilon error) or the mean. The resulting reverse step turned out to be one Langevin step driven by `eps_theta`, and since the forward marginal's score is `-eps/sqrt(1-alpha_bar_t)`, predicting noise is predicting the score: the variational diffusion and denoising-score-matching-with-annealed-Langevin are the same construction. That equivalence brings in NCSN's equal-magnitude-per-level weighting, and inspecting the bound's own weight `beta_t^2/(2 sigma_t^2 alpha_t (1-alpha_bar_t))` confirms it is largest near the clean end, lavishing gradient on the easy near-clean denoising and starving the hard high-noise terms; so dropping the weight gives `L_simple`, which emphasizes the difficult terms that decide image content. Fixing the reverse variance to a principled constant (rather than learning it), scaling data to `[-1,1]`, and a small linear `beta` schedule with `T=1000` close the chain, and the method collapses to a matched pair: train the net to predict the added noise, and invert the corruption equation to recover `x_0` for sampling.
