In front of me is a trained, guided diffusion model that makes gorgeous, prompt-faithful images when I push the guidance scale up, and a sampler, DDIM, that needs a couple hundred network calls to get there. I want fifteen or twenty. Fast high-order ODE solvers already exist and already get *unconditional* sampling down to ten or twenty steps, so the tempting move is "just point DPM-Solver at the guided model too." The diagnostic finding on guided models says that move is not harmless.

At guidance scale eight, around fifteen function evaluations, the second- and third-order noise-prediction solvers — DPM-Solver-2, DPM-Solver-3, DEIS, PNDM — give visibly worse images than plain first-order DDIM, and the degradation grows as the order goes up. That's backwards. Higher order is supposed to mean smaller error. So before I design anything I need to understand why high order *hurts* here, because whatever I build cannot repeat that failure.

I need to reconstruct the mechanism. Guided sampling replaces the plain noise prediction with a steered one. Classifier guidance is `eps_tilde = eps_theta - s * sigma_t * grad_x log p_phi(c | x_t)`; classifier-free guidance is `eps_tilde = s * eps_theta(x,t,c) + (1-s) * eps_theta(x,t,empty)`; either way there is a scalar `s`, and large `s` is exactly the regime that gives the sharp, aligned images people want. A high-order solver works by estimating derivatives of the model output along the trajectory and using them to take a bigger, more accurate step. But `s` multiplies the steered prediction, so it multiplies its total derivatives too. A solver's stability — the largest step it can take before the local Taylor model stops being a good local model — is set by the size of those derivatives and of the derivatives that appear in the remainder. Amplify them and the radius shrinks; to stay inside it the solver would need smaller steps. At a fixed tiny budget I cannot shrink the steps, so the high-order correction starts extrapolating outside the region where the derivative estimates are trustworthy. Order three is especially exposed: it uses a second-derivative correction and its local error is governed by the next derivative. That's the inversion: with guidance-amplified derivatives, higher order can become a liability. The wall is useful. It says I cannot just bolt on more order; I have to make the per-step error constant smaller, and I have to find a source of stability the pure deterministic high-order march does not have.

There's a second failure, a different disease, I should pin down before moving. Image pixels live in `[-1, 1]`. A large guidance scale pushes the steered prediction away from the true noise, and integrating that all the way down, the converged clean image `x_0` lands *outside* `[-1, 1]` — and an out-of-range image renders saturated and garish. People already know the fix for *that* one: thresholding, clip the predicted clean image back into the bound at every step, statically to `[-1,1]` or dynamically by a percentile. But notice the precondition hiding in "clip the predicted clean image": to clip `x_0` I have to be *holding* an estimate of `x_0` at each step. A noise-prediction solver never forms one — it carries `eps`, and `eps` lives on the whole real line, there's nothing to clip into a data bound. So thresholding and the noise-prediction formulation don't compose. A second, independent reason the existing solvers fight me on guided sampling.

Both problems point at the same suspect: the *parameterization*. The existing fast solvers are all written on `eps_theta`. Let me ask what the network even is, because the two readings are related by an exact algebraic identity and I keep treating `eps` as the only one. Training gives me a network that, from a noisy `x_t`, reads equally as predicting the noise `eps` or the clean datum `x_0`: `x_theta(x_t, t) = (x_t - sigma_t eps_theta(x_t, t)) / alpha_t`. Same network, same information, two faces. The whole solver machinery — semi-linear ODE, variation of constants, change of variable to log-SNR, Taylor expansion, exponential integrator — is parameterization-agnostic in principle; nobody made me build it on `eps`. So rebuild it on `x_theta` and see what changes. If the data-prediction face fixes the bound problem (it literally hands me an `x_0` estimate to clip) and shrinks the error constant, that's both walls at once.

Redo the derivation from the ODE in data-prediction form. The probability-flow ODE in the noise face is `dx/dt = f(t) x + (g^2/(2 sigma_t)) eps_theta`. Substitute the identity `eps_theta = (x_t - alpha_t x_theta)/sigma_t` and collect:

  dx/dt = f x + (g^2/(2 sigma_t)) * (x - alpha_t x_theta)/sigma_t = (f + g^2/(2 sigma^2)) x - (alpha g^2/(2 sigma^2)) x_theta.

So in the data face the ODE is `dx/dt = (f + g^2/(2 sigma^2)) x - (alpha g^2/(2 sigma^2)) x_theta`. Still semi-linear: a linear-in-`x` term plus the network term, just with a different linear coefficient. Solve the linear part exactly the same way, by variation of constants. The exponential-integrator route from the noise face gave "(linear factor) times `x_s` minus `alpha_t` times an integral of `e^{-lambda} eps`"; I want the data-face analogue. I claim the exact solution between `s` and `t` is

  x_t = (sigma_t / sigma_s) x_s + sigma_t * integral_{lambda_s}^{lambda_t} e^{lambda} x_theta(x_lambda, lambda) d lambda,

and rather than re-run the integrating-factor manipulation let me verify it by differentiating, which is cleaner and catches sign errors. Differentiate the right side with respect to `t`. The first term gives `(d sigma_t/dt)(x_s/sigma_s)`. The second is a product: `(d sigma_t/dt) * integral + sigma_t * (d/dt of the integral)`. The derivative of the integral with respect to its upper limit `lambda_t` is `e^{lambda_t} x_theta(x_{lambda_t}, lambda_t)`, times `d lambda_t/dt` by the chain rule. So

  dx_t/dt = (d sigma_t/dt)(x_s/sigma_s) + (d sigma_t/dt) * integral + sigma_t (d lambda_t/dt) e^{lambda_t} x_theta.

The first two terms reassemble: `(d sigma_t/dt)[ x_s/sigma_s + integral ] = (d sigma_t/dt) * (x_t/sigma_t)`, because the bracket is exactly `x_t/sigma_t` by my claimed solution. And `(d sigma_t/dt)/sigma_t = d log sigma_t/dt`, while `sigma_t e^{lambda_t} = sigma_t (alpha_t/sigma_t) = alpha_t`. So

  dx_t/dt = (d log sigma_t/dt) x_t + alpha_t (d lambda_t/dt) x_theta.

I need this to equal `(f + g^2/(2 sigma^2)) x - (alpha g^2/(2 sigma^2)) x_theta`. The data-face coefficient identities give `d log sigma_t/dt = f + g^2/(2 sigma^2)` and `alpha_t d lambda_t/dt = - alpha g^2/(2 sigma^2)` — the second because `d lambda/dt = d log alpha/dt - d log sigma/dt = f - (f + g^2/(2 sigma^2)) = -g^2/(2 sigma^2)`. Both match. So the claimed data-prediction exact solution is correct:

  x_t = (sigma_t/sigma_s) x_s + sigma_t * integral_{lambda_s}^{lambda_t} e^{lambda} x_theta(x_lambda, lambda) d lambda.

Look hard at how this differs from the noise face. There the linear-carry is `alpha_t/alpha_s` and the integral weight is `e^{-lambda}` against `eps`; here the linear-carry is `sigma_t/sigma_s` and the integral weight is `e^{+lambda}` against `x_theta`. These are genuinely different discretizations of the same trajectory — they exactly-compute different halves of the dynamics and approximate different integrands. The data face exactly carries the `sigma_t/sigma_s` part and only ever approximates `integral e^{lambda} x_theta`. So every approximation lives in an integral of `x_theta` — a clean-image estimate that can be clipped or dynamically thresholded — instead of an integral of unbounded `eps`. The bound problem is solved structurally, before I write a single solver step.

Get the first-order case and confirm it's sane. Take `x_theta` constant over the interval, equal to its left-endpoint value `x_theta(x_s, s)`, and pull it out:

  x_t ≈ (sigma_t/sigma_s) x_s + sigma_t x_theta(x_s,s) * integral_{lambda_s}^{lambda_t} e^{lambda} d lambda = (sigma_t/sigma_s) x_s + sigma_t x_theta(x_s,s) (e^{lambda_t} - e^{lambda_s}).

With `h = lambda_t - lambda_s`, `e^{lambda_t} - e^{lambda_s} = e^{lambda_t}(1 - e^{-h})`, and `sigma_t e^{lambda_t} = alpha_t`, so

  x_t = (sigma_t/sigma_s) x_s - alpha_t (e^{-h} - 1) x_theta(x_s,s).

That's the first-order data-prediction step. Is it DDIM? DDIM with `eta=0` written on the data face is `x_t = alpha_t x_theta + sigma_t eps_theta`, and `eps_theta = (x_s - alpha_s x_theta)/sigma_s`, so the coefficient on the old latent is `sigma_t/sigma_s` and the coefficient on `x_theta` is `alpha_t - sigma_t alpha_s/sigma_s = alpha_t(1 - e^{-h}) = -alpha_t(e^{-h} - 1)`. Same formula. So I lose nothing at order one; the data-prediction generalization of DDIM is the right base to climb from.

Now climb. To approximate the integral better, Taylor-expand `x_theta` in `lambda` around the left endpoint and integrate term by term. Write `x_theta^{(n)}` for the `n`-th total derivative of `x_theta` with respect to `lambda`. Expanding to order `k-1`,

  x_t = (sigma_t/sigma_s) x_s + sigma_t * sum_{n=0}^{k-1} x_theta^{(n)}(lambda_s) * integral_{lambda_s}^{lambda_t} e^{lambda} (lambda - lambda_s)^n / n! d lambda + O(h^{k+1}).

Each scalar integral is elementary — integrate by parts `n` times. So an order-`k` solver needs estimates of the first `k-1` derivatives of `x_theta` in `lambda`. That's the standard structure, and there are two textbook ways to get those derivatives: insert fresh intermediate network evaluations *within* the step (singlestep, like Runge–Kutta), or reuse the network values from *previous* steps (multistep, like Adams–Bashforth).

My first-wall lesson decides the choice. The instability comes from high-order derivatives being amplified by guidance; one defense is simply smaller steps, so the same order extrapolates over a shorter `lambda`-interval where the truncation series still converges. Compare the two derivative-estimation styles at a fixed budget `N` of network calls. A singlestep order-`k` method spends `k` calls per step, so it affords only `M = N/k` steps — big steps. A multistep method spends *one* call per step and recycles the last few, so it affords `M = N` steps — steps `k` times smaller. Smaller `h` directly shrinks the `O(h^k)` error and keeps me inside the convergence radius guidance narrows. And the singlestep method *throws away* its intermediate evaluations after one use, wasteful when every evaluation is a forward pass through a giant network. So multistep wins twice: smaller steps for free, and full reuse of expensive evaluations. I build multistep.

That decision puts real weight on a piece I have not examined yet: a multistep method gets its derivatives from past network outputs sitting at unequal `lambda`-spacings. The Karras grid I will use is a power-warped schedule, so the gaps between consecutive noise levels are not constant in `lambda`; the previous step might be `h_1` behind and the one before that a further, different `h_2` behind. A finite-difference estimate of a derivative on an unequal grid is a divided difference, and every difference I form is an estimate of a derivative times the length of the interval it spans. If I combine differences taken over different-length intervals without first putting them on a common footing, the combination silently mis-scales the high-order correction. This is exactly the kind of bookkeeping that runs fine while quietly being wrong.

Before higher order, I still want a *stability source* beyond just small steps — and there's a known one I keep circling: re-injected noise. Why would adding noise help? Deterministic high-order extrapolation accumulates error along the trajectory with nothing to wash it out; the trajectory can drift off the data manifold and stay off. If instead I add a controlled bit of Gaussian noise each step and let the *next* denoising step remove it, that's a Langevin correction — the noise-and-denoise cycle pulls the state back toward where the model thinks data lives, cancelling accumulated discretization error. It's documented that fully-deterministic sampling can be perceptually *worse* than sampling with such re-injection. So I don't want to abandon the SDE — I want to solve it with the same exponential-integrator machinery, on the same data face, and keep a knob for how much noise to put back.

Derive the SDE solver. The reverse diffusion SDE is `dx = [f x + (g^2/sigma_t) eps_theta] dt + g dw_bar`. Notice already that the drift's network coefficient is `g^2/sigma_t`, *twice* the ODE's `g^2/(2 sigma_t)` — the SDE pushes twice as hard on the drift and pays with the `g dw` noise, a relationship I have to keep straight. Move to the `lambda` variable. For a VP schedule, `f = d log alpha/dt`, `g = sigma_t sqrt(-2 d lambda_t/dt)`, and using `d log alpha/d lambda = sigma^2`, `d log sigma/d lambda = -alpha^2` (which follow from `alpha^2 + sigma^2 = 1` differentiated), the SDE in `lambda`, after carrying the change of variable through the Itô term, becomes

  dx_lambda = [sigma_lambda^2 x_lambda - 2 sigma_lambda eps_theta] d lambda + sqrt(2) sigma_lambda dw_lambda,

and in the data face, substituting `eps = (x - alpha x_theta)/sigma`,

  dx_lambda = [-(1 + alpha_lambda^2) x_lambda + 2 alpha_lambda x_theta] d lambda + sqrt(2) sigma_lambda dw_lambda.

Variation of constants again, now with a stochastic forcing. The homogeneous part is `dx = -(1+alpha^2) x d lambda`, integrating factor `exp(integral (1+alpha^2) d lambda)`. Using `d log sigma/d lambda = -alpha^2`, `integral (1 + alpha^2) d lambda = (lambda_t - lambda_s) - (log sigma_t - log sigma_s)`, so the homogeneous solution carries `x_s` with factor `(sigma_t/sigma_s) e^{-(lambda_t - lambda_s)}`. Pushing the data term and the noise term through the integrating factor — the algebra is the same shape as the deterministic case, just keeping the Itô integral — gives the exact SDE solution

  x_t = (sigma_t/sigma_s) e^{-h} x_s + 2 alpha_t integral_{lambda_s}^{lambda_t} e^{-2(lambda_t - lambda)} x_theta(x_lambda, lambda) d lambda + sqrt(2) sigma_t integral_{lambda_s}^{lambda_t} e^{-(lambda_t - lambda)} dw_lambda.

Three pieces: a *more* strongly decayed signal carry `(sigma_t/sigma_s)e^{-h}` (the extra `e^{-h}` is the SDE pushing harder), the data integral weighted `e^{-2(lambda_t - lambda)}` (note the `-2`, double the ODE's effective contraction), and an Itô noise integral. The Itô integral is the only new object; I only need its variance, since it's a deterministic-coefficient times a Gaussian. By Itô isometry, `integral_{lambda_s}^{lambda_t} e^{-(lambda_t - lambda)} dw_lambda` is Gaussian with variance `integral e^{-2(lambda_t - lambda)} d lambda = (1 - e^{-2h})/2`. So `sqrt(2) sigma_t` times it is Gaussian with standard deviation `sqrt(2) sigma_t sqrt((1-e^{-2h})/2) = sigma_t sqrt(1 - e^{-2h})`. The noise term is `sigma_t sqrt(1 - e^{-2h}) z`, `z ~ N(0,I)`. Clean.

First-order SDE step: hold `x_theta` constant. The data integral becomes `2 alpha_t e^{-2 lambda_t} (integral e^{2 lambda} d lambda) x_theta(s) = 2 alpha_t e^{-2 lambda_t} ((e^{2 lambda_t} - e^{2 lambda_s})/2) x_theta = alpha_t (1 - e^{-2h}) x_theta(s)`. So

  x_t = (sigma_t/sigma_s) e^{-h} x_s + alpha_t (1 - e^{-2h}) x_theta(x_s,s) + sigma_t sqrt(1 - e^{-2h}) z.

This is the first-order stochastic data-prediction step. Sanity hook: a known sampler in disguise? DDIM with stochasticity parameter `eta_DDIM` is `x_t = alpha_t x_theta + sqrt(sigma_t^2 - eta_DDIM^2) eps_theta + eta_DDIM z`. If I set `eta_DDIM = sigma_t sqrt(1 - e^{-2h})`, the DDIM noise level matches mine exactly, and substituting `eps_theta = (x_s - alpha_s x_theta)/sigma_s` gives the same deterministic coefficients. So my first-order SDE step is stochastic DDIM with that specific DDIM noise level. Good — it sits on top of the thing everyone already trusts.

Now I want a *knob*, not the two extremes "deterministic ODE" or "full SDE." Look at what distinguishes them in the `alpha = 1` noise-level convention I'll implement in: the deterministic data-prediction step contracts by `e^{-h}` and adds no noise; the full SDE contracts by `e^{-2h}` and adds noise `sqrt(1 - e^{-2h})`. The "2" in the SDE is exactly the doubled drift coefficient. So introduce `eta >= 0` and let it interpolate the *renoising rate*: replace the contraction exponent by `h_eta = h (eta + 1)`, so `eta = 0` gives exponent `h` and `eta = 1` gives `2h`. The constant data-prediction part keeps the two weights `e^{-h_eta}` and `1 - e^{-h_eta}`, so that part is a *convex* combination of the current latent and the clean prediction — the weights sum to one, `e^{-h_eta} + (1 - e^{-h_eta}) = 1`, which is precisely the boundedness property I wanted: the update never leaves the convex hull of the latent and the clippable clean prediction. And the noise: at `eta = 1` it must be `sigma sqrt(1 - e^{-2h})`; the extra contraction beyond the ODE is by `e^{-h eta}` in amplitude, hence `e^{-2 h eta}` in variance, so the noise variance that restores the marginal is `1 - e^{-2 h eta}` and its standard deviation is `sqrt(1 - e^{-2 h eta})`. Check the endpoints. `eta=0`: `h_eta=h`, signal `e^{-h}`, data `1-e^{-h}`, noise `sqrt(1 - e^0) = 0` — exactly the deterministic ODE, no noise. `eta=1`: `h_eta=2h`, signal `e^{-2h}`, data `1-e^{-2h}`, noise `sqrt(1-e^{-2h})` — exactly the full SDE above. So one parameter slides continuously from the deterministic march to the noisy one, and I can dial in more stochasticity than the plain SDE by taking `eta` past one when guidance and big steps need extra Langevin self-correction. I leave `eta` exposed; `eta=1` is the full-SDE default, and larger values are an available robustness tradeoff rather than a new derivation.

The per-step skeleton, with `denoised` standing for `x_theta(x, sigma)` and the VE convention `alpha = 1` (so `lambda = -log sigma`, and `sigma_t/sigma_s = e^{-lambda_t}/e^{-lambda_s} = e^{-h}`): the signal carry is `(sigma_t/sigma_s)` for the ODE part with an extra `e^{-h eta}`-type contraction for the SDE part, and in this convention `sigma_t/sigma_s = e^{-h}` is itself an exponential of the `lambda`-step, so the whole signal factor folds into a single `e^{-h_eta}` (at `eta = 1`, `(sigma_t/sigma_s) e^{-h} = e^{-h} e^{-h} = e^{-2h} = e^{-h_eta}`; at `eta = 0` it is `e^{-h}` with no renoise). The skeleton is

  x = e^{-h_eta} x + (1 - e^{-h_eta}) denoised,    then if eta > 0:  x += sigma_next sqrt(1 - e^{-2 h eta}) z.

Now lift it to higher order with the multistep correction, using the data integral's Taylor expansion rather than just holding `x_theta` constant. I keep the last two `denoised` values and the last two step sizes. Expand `x_theta(lambda_s + u)` around the left endpoint: `x_theta = D_0 + D_1 u + (1/2) D_2 u^2 + ...`, where `D_1 = x_theta^{(1)}(lambda_s)`, `D_2 = x_theta^{(2)}(lambda_s)` are the first and second `lambda`-derivatives. The constant part of the generalized data term, with `kappa = eta + 1` and `H = h_eta = kappa h`, is `kappa integral_0^h e^{-kappa(h-u)} D_0 du = (1 - e^{-H}) D_0`, which is the base step I already have. After factoring the first derivative as `h D_1`, the scalar coefficient is

  phi_2(h_eta) = (e^{-h_eta} - 1)/h_eta + 1,

which for small `h_eta` is about `h_eta/2` (Taylor: `(e^{-h_eta}-1)/h_eta = -1 + h_eta/2 - h_eta^2/6 + ...`, plus 1, gives `h_eta/2 - h_eta^2/6 + ...`). The second is

  phi_3(h_eta) = phi_2(h_eta)/h_eta - 0.5,

which for small `h_eta` is about `-h_eta/6` (`phi_2/h_eta = 1/2 - h_eta/6 + ...`, minus `1/2`, gives `-h_eta/6 + ...`) — note it is negative. That sign is the piece the implementation has to preserve. A fully tuned coefficient on `(h^2/2)D_2` for this generalized integral would be positive and equals `1 - 2/H + 2(1 - e^{-H})/H^2`, which is `2(-phi_3)`. The canonical k-diffusion 3M SDE uses the same-sign practical half-weight, `-phi_3`, so I should not pretend this is a clean `O(h^4)` third-order proof. I should make the divided differences recover `h D_1` and `(h^2/2)D_2`, then use the canonical positive multiplier `-phi_3` on the curvature estimate.

Now the derivative estimates from past values, and this is where I have to be careful about the unit problem I flagged earlier. I have three `lambda`-points: the current `denoised` at `lambda_s` (the left endpoint of the step I'm taking), the previous `denoised_1` one step back at `lambda`-distance `h_1`, and `denoised_2` two steps back at a *further* distance `h_2` beyond that. I want `D_1` and `D_2` at `lambda_s`, built from a quadratic through these three points — a Newton divided difference. The naive first thing I'd reach for is to scale each backward difference by its own interval: `(denoised - denoised_1)` over `h_1`, and `(denoised_1 - denoised_2)` over `h_2`, then combine the second one relative to the first by `h_2/h_1`. Let me actually try that and see if it gives me consistent derivative estimates — because if I get this wrong the third-order term is quietly mis-scaled.

Here's the catch. The `phi` weights I just derived are functions of `h_eta = (eta+1) h` — they're attached to the *current* step `h`. The Taylor expansion was `x_theta = D_0 + D_1 u + (1/2) D_2 u^2`, and when I integrated `u^1` against the exponential I got `phi_2 * h * D_1` and integrating `u^2/2` I got something proportional to `h^2 * D_2`. So the quantities I actually need to multiply the `phi`s by are `h * D_1` and `(h^2/2) * D_2` — derivative *times powers of the current step `h`*, not times `h_1` or `h_2`. That tells me every finite difference I form must be expressed in the same unit: as a multiple of the current step `h`. So I should scale *both* backward differences to the current step. Define

  r0 = h_1 / h,    r1 = h_2 / h,

and form

  d1_0 = (denoised - denoised_1) / r0,    d1_1 = (denoised_1 - denoised_2) / r1.

Let me see what these are. `(denoised - denoised_1)` is approximately `D_1 * h_1 = D_1 * r0 * h`, so dividing by `r0` gives `d1_0 ≈ h * D_1` — a near-interval estimate of the first derivative *scaled by the current step*. Likewise `(denoised_1 - denoised_2) ≈ D_1 * h_2 = D_1 * r1 * h` (to leading order), so `d1_1 ≈ h * D_1` over the next interval back — *also* in units of `h * D_1`. Both differences now live in the same unit. The second difference is then dimensionally honest:

  d2 = (d1_0 - d1_1) / (r0 + r1),

and the first-derivative estimate corrected from the interval-average value to the *endpoint* `lambda_s` by extrapolating the slope forward is

  d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1).

I want to be sure these recover the right things, so I test on a pure quadratic around the current endpoint. Write the three values as `y0 = D_0`, `y1 = D_0 - h_1 D_1 + (h_1^2/2) D_2`, and `y2 = D_0 - (h_1+h_2)D_1 + ((h_1+h_2)^2/2)D_2`. With `r0 = h_1/h` and `r1 = h_2/h`,

  d1_0 = (y0-y1)/r0 = h D_1 - (h h_1/2)D_2,
  d1_1 = (y1-y2)/r1 = h D_1 - h(h_1 + h_2/2)D_2.

Subtracting gives `d1_0 - d1_1 = h(h_1+h_2)D_2/2 = (r0+r1)(h^2/2)D_2`, so

  d2 = (d1_0-d1_1)/(r0+r1) = (h^2/2)D_2.

Then the endpoint first-derivative estimate cancels the interval-average bias:

  d1 = d1_0 + (d1_0-d1_1) r0/(r0+r1)
     = h D_1 - (h h_1/2)D_2 + (h^2/2)D_2 * (h_1/h)
     = h D_1.

So the divided difference is exact on a quadratic for any unequal `h_1` and `h_2`: `d1` is `h * x_theta^{(1)}(lambda_s)` and `d2` is `(h^2/2) * x_theta^{(2)}(lambda_s)`.

Now let me deliberately re-run the easiest case with the naive scaling I almost used — `r1 = h_2 / h_1` instead of `h_2 / h`. If the model output is only linear, `D_2 = 0`, the correct second difference must vanish. But with `r1 = h_2/h_1`, the older difference becomes `d1_1 = (y1-y2)/(h_2/h_1) = h_1 D_1 = r0 h D_1`, while `d1_0 = h D_1`. Unless `h_1 = h`, `d1_0 - d1_1` is nonzero even though the function is linear. That creates fake curvature before I even get to a quadratic. For unequal spacings the curvature correction would be mis-scaled, the 3M term would be wrong, and the sampler would still "run." That settles it: both finite differences must be scaled to the current step, `r0 = h_1/h` and `r1 = h_2/h`. The common unit is what makes the `phi` weights — functions of `h_eta`, attached to `h` — line up with the derivative estimates dimensionally.

Now the step has the constant term plus the two multistep corrections:

  x = e^{-h_eta} x + (1 - e^{-h_eta}) denoised + phi_2 * d1 + (coefficient) * d2.

The `phi_2 * d1` term is unambiguous: `phi_2` is the integrated weight after I factor out `h D_1`, and `d1` is `h D_1`, so `phi_2 * d1` is the exact first-derivative correction. The `d2` term needs its sign nailed down. The exact Taylor coefficient multiplying `D_2` is `kappa integral_0^h e^{-kappa(h-u)} (u^2/2) du`, which is positive; since `d2` estimates `+(h^2/2)D_2`, any curvature correction with the right sign must apply a positive multiplier to `d2`. The canonical 3M SDE multiplier is `-phi_3`. Because `phi_3 < 0`, `-phi_3 > 0`, so `-phi_3 * d2` applies the positive multiplier. If I instead wrote `+ phi_3 * d2`, it would subtract the curvature estimate, flipping the sign of the 3M correction. So the assembled step is

  x = e^{-h_eta} x + (1 - e^{-h_eta}) denoised + phi_2 * d1 - phi_3 * d2,

with the `- phi_3 * d2` adding the curvature estimate because `phi_3 < 0`. The magnitude is conservative but in the right asymptotic slot: `phi_2 ≈ h_eta/2`, `d1 ≈ h D_1`, so the first correction is order `h^2 D_1`; `-phi_3 ≈ h_eta/6`, `d2 ≈ (h^2/2)D_2`, so the curvature correction is order `h^3 D_2`. The sign and scaling check out, while the coefficient remains the canonical practical one rather than the fully tuned `2(-phi_3)` coefficient.

When I only have one past value — the very first correction step, after one full step has been taken — I drop `d2` and use a two-point first-derivative estimate `d = (denoised - denoised_1)/r`, `r = h_1/h`, giving the second-order step `x += phi_2 * d`. When I have no past value yet, the first step is just the constant data-prediction step with the noise. The history ramps naturally: first step constant, second step first-derivative correction, then the 3M curvature correction.

I keep the order claim honest, because this is not a clean "provably third order" derivation. The `phi_2 * d1` first-derivative term matches the exact integral coefficient. The `-phi_3 * d2` second-derivative term is a practical curvature correction with the correct units and sign, but its coefficient is half of the fully tuned coefficient in this normalization, so I should not claim a local `O(h^4)` multistep proof. This is the same kind of simplification the second-order SDE derivation already makes: there, the exact coefficient on the first-derivative term is `(e^{-2h} - 1 + 2h)/(2h)`, and it is deliberately replaced by the simpler `(1 - e^{-2h})/2`, which agrees to leading order (`≈ h`) but trades the exactly tuned constant for a cleaner, robust form. So I am landing on a practical 3M multistep correction. The thing that absolutely must be right is the bookkeeping — `r1 = h_2/h` so the differences share a unit, and `-phi_3 * d2` so the curvature estimate is added rather than subtracted — because those errors do not reduce the order gracefully; they corrupt the correction outright.

A few implementation realities before I write it. The exponentials all show up as `e^{x} - 1` with small `x` — `1 - e^{-h_eta}`, `1 - e^{-2 h eta}` — where naive `exp(x) - 1` cancels catastrophically; use `expm1`, which is built for it, so `(1 - e^{-h_eta})` is `(-h_eta).expm1().neg()` and the noise standard deviation `sqrt(1 - e^{-2 h eta})` is `(-2 * h * eta).expm1().neg().sqrt()`. The noise itself wants to be Brownian-consistent and reproducible across steps, not a fresh independent `randn` each time, so I draw it from a Brownian-tree noise sampler keyed to the noise levels — it returns increments scaled correctly and the same seed gives the same trajectory. The very last step lands at `sigma = 0`, where there's nothing left to denoise toward and no noise to add: I just return `denoised`, the clean-image estimate itself (Tweedie's clean prediction at zero noise). And the time grid is the Karras power schedule, `sigma_i = (sigma_max^{1/rho} + (i/(N-1))(sigma_min^{1/rho} - sigma_max^{1/rho}))^rho` with `rho = 7`, appending a final `sigma = 0` — the power warp puts more steps at low noise where the per-step truncation error is largest, so the tight budget is spent where it matters. That warp is exactly *why* the `lambda`-spacings `h_1`, `h_2` are unequal, which is why the divided-difference unit-consistency I worked out is not academic. With `alpha = 1` the half-log-SNR is just `lambda = -log sigma`, so the change of variable is `t_fn = lambda sigma: -log sigma` and `h = lambda_next - lambda = (-log sigma_next) - (-log sigma)`, positive as `sigma` decreases. Each step keeps an optional small `s_noise` multiplier on the injected noise (default one) so the noise scale can be tuned without touching `eta`.

Now write the per-step solver — the data-prediction, multistep, `eta`-stochastic, Karras-scheduled march I derived — so it drops into the generic sampling loop, with each block tied back to the step that motivated it:

```python
import torch
from tqdm.auto import trange


@torch.no_grad()
def sample_dpmpp_3m_sde(model, x, sigmas, extra_args=None, callback=None,
                        disable=None, eta=1., s_noise=1., noise_sampler=None):
    """DPM-Solver++(3M) SDE: multistep, stochastic, data-prediction exponential-integrator sampler.
    `model(x, sigma)` returns the clean-image estimate x_theta ('denoised')."""
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    # Brownian-tree noise: reproducible, correctly-scaled Brownian increments per step.
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # multistep history: last two data-predictions and last two lambda-step-sizes
    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)   # x_theta at this noise level
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            # last step: sigma -> 0, return the clean-image estimate itself, no renoise
            x = denoised
        else:
            # half-log-SNR (alpha = 1, so lambda = -log sigma); h is the lambda-step
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)                              # eta interpolates ODE (0) <-> SDE (1)

            # constant data-prediction step: convex combination, weights sum to 1
            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                # 3M: Newton divided difference for D_1, D_2 in lambda.
                # BOTH differences scaled to the *current* step h, so they share a unit
                # and line up with the phi-weights (functions of h_eta, attached to h).
                r0 = h_1 / h
                r1 = h_2 / h                                  # NOT h_2/h_1: keep the common unit
                d1_0 = (denoised - denoised_1) / r0           # ~ h * x_theta^(1) (near interval)
                d1_1 = (denoised_1 - denoised_2) / r1         # ~ h * x_theta^(1) (next interval back)
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)    # endpoint first-derivative est ~ h*D_1
                d2 = (d1_0 - d1_1) / (r0 + r1)                # second-difference ~ (h^2/2) * D_2
                phi_2 = h_eta.neg().expm1() / h_eta + 1        # first-derivative weight (>0)
                phi_3 = phi_2 / h_eta - 0.5                    # curvature helper (<0)
                x = x + phi_2 * d1 - phi_3 * d2               # -phi_3 (>0) adds the curvature estimate
            elif h_1 is not None:
                # second order: two-point first-derivative estimate only
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                # Langevin renoise: restores the variance removed by the extra SDE contraction
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] \
                    * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        # shift the multistep history forward
        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x
```

And the time grid it marches on, the Karras power schedule that concentrates steps at low noise (and thereby makes the `lambda`-spacings unequal):

```python
def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho   # warp, interp, unwarp
    return torch.cat([sigmas, sigmas.new_zeros([1])]).to(device)          # append final sigma = 0
```

The causal chain end to end: I want fast guided sampling, and the existing high-order solvers get *worse* than first-order DDIM at large guidance because the guidance scale amplifies the model's derivatives and shrinks the convergence radius the high-order terms need, and because a noise-prediction solver carries an unbounded quantity that thresholding cannot fix. Both pathologies trace to building on `eps`; re-deriving the same exponential-integrator solver on the data-prediction face — variation of constants, change of variable to log-SNR — gives an exact solution that carries `sigma_t/sigma_s` exactly and only ever approximates an integral of `x_theta`, a clean-image estimate I can clip or threshold, and whose per-step update is a convex combination of the latent and that bounded prediction. To regain the stability guidance steals I take small steps via a multistep scheme, one network call per step with past predictions reused, and I add a Langevin self-correction by solving the diffusion SDE with the same machinery, introducing a single `eta` knob that interpolates the renoising rate from the deterministic ODE (`eta=0`) through the full SDE (`eta=1`) and beyond. First order recovers deterministic DDIM at `eta=0` and the first-order stochastic SDE step at `eta=1`; the first history value gives a two-point first-derivative correction; two history values give the 3M Newton divided-difference curvature correction. The two pieces of that curvature term that must be exactly right are the unit-consistency of the finite differences — both scaled to the current step, `r0 = h_1/h` and `r1 = h_2/h`, so they line up with the `phi`-weights attached to `h` — and the sign of the second-derivative term, `-phi_3 * d2`, which applies a positive multiplier because `phi_3` itself is negative. Karras's power schedule spends the tight budget where truncation error lives, `expm1` keeps the small-argument exponentials honest, and a Brownian-tree sampler injects reproducible, correctly scaled noise. That lands on the canonical `sample_dpmpp_3m_sde` update.
