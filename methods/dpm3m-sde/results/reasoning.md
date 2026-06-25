The thing in front of me is a guided diffusion model that makes gorgeous, prompt-faithful images when I crank the guidance scale up, and a sampler — DDIM — that needs two hundred network calls to get there. I want fifteen or twenty. The fast high-order ODE solvers already exist and already get unconditional sampling down to ten or twenty steps, so the lazy answer is "just use DPM-Solver for guided sampling too." Let me actually try that in my head at a large guidance scale, because I have a nagging memory that it doesn't just work.

And it doesn't. At guidance scale eight, fifteen function evaluations, the second- and third-order noise-prediction solvers — DPM-Solver-2, DPM-Solver-3, DEIS, PNDM — give me visibly worse images than plain first-order DDIM. Worse, and *worse still* as the order goes up. That's backwards. Higher order is supposed to mean smaller error. So before I design anything I have to understand why high order *hurts* here, because whatever I build has to not do that.

Let me reconstruct the mechanism. Guided sampling replaces the plain noise prediction with a steered one. Classifier guidance is `eps_tilde = eps_theta - s * sigma_t * grad_x log p_phi(c | x_t)`, classifier-free is `eps_tilde = s * eps_theta(x,t,c) + (1-s) * eps_theta(x,t,empty)` — either way there's a scalar `s`, and large `s` is exactly the regime that gives the sharp, aligned images people want. Now a high-order solver works by estimating *derivatives* of the model function along the trajectory and using them to take a bigger, more accurate step. But `s` multiplies the whole steered prediction, and therefore it multiplies its derivatives too — and the higher the derivative, the more it gets amplified, because each derivative picks up another factor of however fast the guidance term is varying. A solver's stability — its convergence radius, the largest step it can take before the truncation series stops converging — is set by the size of those derivatives. Amplify the derivatives by `s` and you shrink the radius by roughly `s`; to stay inside it the solver would need steps `s` times smaller. At a fixed tiny budget I can't shrink the steps, so the high-order solver runs *outside* its convergence radius and the high-order terms, instead of correcting the step, blow it up. The third-order term, carrying the third derivative, is the most amplified, so order three should be the most unstable — and that matches the observed ordering, where the degradation grew with the solver order. So the mechanism predicts the very pattern I'm staring at, which makes me trust it. The uncomfortable consequence is that the obvious move — bolt on more order — is exactly the wrong one here: with derivatives this large, higher order is a liability. Whatever I build has to drive the per-step error down by a smaller *constant* and over a shorter interval, not by a higher order, and it probably needs some source of stability the deterministic high-order march doesn't have. I don't yet know what that source is; I'm just noting that pure ODE extrapolation won't be enough.

There's a second failure I should pin down before moving, because it's a different disease. Image pixels live in `[-1, 1]`. A large guidance scale pushes the steered prediction away from the true noise, and when I integrate that all the way down, the converged clean image `x_0` lands *outside* `[-1, 1]` — and an out-of-range image renders saturated and garish. People already know the fix for *that* one: thresholding, clip the predicted clean image back into the bound at every step, either statically to `[-1,1]` or dynamically by some percentile. But notice the precondition hiding in "clip the predicted clean image": to clip `x_0` I need to be *holding* an estimate of `x_0` at each step. A noise-prediction solver never forms one — it carries `eps`, and `eps` lives on the whole real line, there's nothing to clip into a data bound. So thresholding and the noise-prediction formulation don't compose. That's a second, independent reason the existing solvers fight me on guided sampling.

Both problems point at the same suspect: the *parameterization*. The existing fast solvers are all written on `eps_theta`. Let me ask what the network even is, because the two readings of it are related by an exact algebraic identity and I keep treating `eps` as the only one. Training gives me a network that, from a noisy `x_t`, can equally be read as predicting the noise `eps` or as predicting the clean datum `x_0`: `x_theta(x_t, t) = (x_t - sigma_t eps_theta(x_t, t)) / alpha_t`. Same network, same information, two faces. The whole solver machinery — semi-linear ODE, variation of constants, change of variable to log-SNR, Taylor expansion, exponential integrator — is parameterization-agnostic in principle; nobody made me build it on `eps`. So let me rebuild it on `x_theta` and see what changes. If the data-prediction face fixes the bound problem (it literally hands me an `x_0` estimate to clip) and shrinks the error constant, that's both walls at once.

I redo the derivation from the ODE in data-prediction form. The probability-flow ODE in the noise face is `dx/dt = f(t) x + (g^2/(2 sigma_t)) eps_theta`. Substitute the identity `eps_theta = (x_t - alpha_t x_theta)/sigma_t` and collect:

  dx/dt = f x + (g^2/(2 sigma_t)) * (x - alpha_t x_theta)/sigma_t = (f + g^2/(2 sigma_t^2)) x - (alpha_t g^2/(2 sigma_t^2)) x_theta.

So in the data face the ODE is `dx/dt = (f + g^2/(2 sigma^2)) x - (alpha g^2/(2 sigma^2)) x_theta`. Still semi-linear: a linear-in-`x` term plus the network term, just with a different linear coefficient. I solve the linear part exactly the same way, by variation of constants. The exponential-integrator route from the noise face gave a solution of the form "(linear factor) times `x_s` minus `alpha_t` times an integral of `e^{-lambda} eps`"; let me find the data-face analogue. I claim the exact solution between `s` and `t` is

  x_t = (sigma_t / sigma_s) x_s + sigma_t * integral_{lambda_s}^{lambda_t} e^{lambda} x_theta(x_lambda, lambda) d lambda,

and rather than re-run the whole integrating-factor manipulation let me verify it by differentiating, which is cleaner and catches sign errors. Differentiate the right side with respect to `t`. The first term gives `(d sigma_t/dt)(x_s/sigma_s)`. The second is a product: `(d sigma_t/dt) * integral + sigma_t * (d/dt of the integral)`. The derivative of the integral with respect to its upper limit `lambda_t` is `e^{lambda_t} x_theta(x_{lambda_t}, lambda_t)`, times `d lambda_t/dt` by the chain rule. So

  dx_t/dt = (d sigma_t/dt)(x_s/sigma_s) + (d sigma_t/dt) * integral + sigma_t (d lambda_t/dt) e^{lambda_t} x_theta.

Now the first two terms reassemble: `(d sigma_t/dt)[ x_s/sigma_s + integral ] = (d sigma_t/dt) * (x_t/sigma_t)`, because the bracket is exactly `x_t/sigma_t` by my claimed solution. And `(d sigma_t/dt)/sigma_t = d log sigma_t/dt`, while `sigma_t e^{lambda_t} = sigma_t (alpha_t/sigma_t) = alpha_t`. So

  dx_t/dt = (d log sigma_t/dt) x_t + alpha_t (d lambda_t/dt) x_theta.

I need this to equal `(f + g^2/(2 sigma^2)) x - (alpha g^2/(2 sigma^2)) x_theta`. The data-face coefficient identities give `d log sigma_t/dt = f + g^2/(2 sigma^2)` and `alpha_t d lambda_t/dt = - alpha g^2/(2 sigma^2)` — the second because `d lambda/dt = d log alpha/dt - d log sigma/dt = f - (f + g^2/(2 sigma^2)) = -g^2/(2 sigma^2)`. Both match. So the claimed data-prediction exact solution is correct:

  x_t = (sigma_t/sigma_s) x_s + sigma_t * integral_{lambda_s}^{lambda_t} e^{lambda} x_theta(x_lambda, lambda) d lambda.

Look hard at how this differs from the noise face. There the linear-carry is `alpha_t/alpha_s` and the integral weight is `e^{-lambda}` against `eps`; here the linear-carry is `sigma_t/sigma_s` and the integral weight is `e^{+lambda}` against `x_theta`. These are genuinely different discretizations of the same trajectory — they exactly-compute different halves of the dynamics and approximate different integrands. The data face exactly carries the `sigma_t/sigma_s` part and only ever approximates `integral e^{lambda} x_theta`. So when I build a solver here, every approximation lives in an integral of `x_theta` — a clean-image estimate that can be clipped or dynamically thresholded — instead of an integral of unbounded `eps`. That is the precondition the threshold trick needed and the noise face could not supply: each step now holds an `x_0` estimate to clip. Whether clipping it actually keeps the bound problem at bay at large `s` is an empirical question I can only settle by running it; but at least the operation is now *well-defined* at every step, which it never was on `eps`.

Before going to higher order, let me get the first-order case and confirm it's sane. Take `x_theta` constant over the interval, equal to its left-endpoint value `x_theta(x_s, s)`, and pull it out:

  x_t ≈ (sigma_t/sigma_s) x_s + sigma_t x_theta(x_s,s) * integral_{lambda_s}^{lambda_t} e^{lambda} d lambda = (sigma_t/sigma_s) x_s + sigma_t x_theta(x_s,s) (e^{lambda_t} - e^{lambda_s}).

With `h = lambda_t - lambda_s`, `e^{lambda_t} - e^{lambda_s} = e^{lambda_t}(1 - e^{-h})`, and `sigma_t e^{lambda_t} = alpha_t`, so

  x_t = (sigma_t/sigma_s) x_s - alpha_t (e^{-h} - 1) x_theta(x_s,s).

That's the first-order data-prediction step. Is it DDIM? DDIM with `eta=0` written on the data face is `x_t = alpha_t x_theta + sigma_t eps_theta`, and `eps_theta = (x_s - alpha_s x_theta)/sigma_s`, so the coefficient on the old latent is `sigma_t/sigma_s` and the coefficient on `x_theta` is `alpha_t - sigma_t alpha_s/sigma_s = alpha_t(1 - e^{-h}) = -alpha_t(e^{-h} - 1)`. It is exactly the same formula. So I lose nothing at order one; I find the data-prediction generalization of DDIM, which is the right base to climb from.

Now climb. To approximate the integral better, Taylor-expand `x_theta` in `lambda` around the left endpoint and integrate term by term. Write `x_theta^{(n)}` for the `n`-th total derivative of `x_theta` with respect to `lambda`. Expanding to order `k-1`,

  x_t = (sigma_t/sigma_s) x_s + sigma_t * sum_{n=0}^{k-1} x_theta^{(n)}(lambda_s) * integral_{lambda_s}^{lambda_t} e^{lambda} (lambda - lambda_s)^n / n! d lambda + O(h^{k+1}).

Each scalar integral is elementary — integrate by parts `n` times. So an order-`k` solver just needs estimates of the first `k-1` derivatives of `x_theta` in `lambda`. That's the standard structure, and there are two textbook ways to get the derivatives: insert fresh intermediate network evaluations *within* the step (singlestep, like Runge–Kutta), or reuse the network values from *previous* steps (multistep, like Adams–Bashforth).

Here my first-wall lesson decides the choice for me. The instability comes from high-order derivatives being amplified by guidance; one defense is simply to take *smaller steps*, so the same order extrapolates over a shorter `lambda`-interval where the truncation series still converges. Compare the two derivative-estimation styles at a fixed budget `N` of network calls. A singlestep order-`k` method spends `k` calls per step, so it can afford only `M = N/k` steps — big steps. A multistep method spends *one* call per step and recycles the last few, so it affords `M = N` steps — steps `k` times smaller. Smaller `h` directly shrinks the `O(h^k)` error and keeps me inside the convergence radius that guidance narrows. And the singlestep method *throws away* its intermediate evaluations after one use, which is wasteful when every evaluation is a forward pass through a giant network. So multistep wins twice over for this regime: smaller steps for free, and full reuse of expensive evaluations. I build multistep.

Let me write the second-order multistep step explicitly to make sure I have the bookkeeping, then push to third. I have the current network value `x_theta(x_{t_{i-1}}, t_{i-1})` and the one from the previous step `x_theta(x_{t_{i-2}}, t_{i-2})`, sitting at `lambda`-spacing `h_{i-1} = lambda_{t_{i-1}} - lambda_{t_{i-2}}` behind. A backward finite difference estimates the first `lambda`-derivative: `x_theta^{(1)}(lambda_{t_{i-1}}) ≈ (x_theta(t_{i-1}) - x_theta(t_{i-2}))/h_{i-1}`. Keep the `n=0` and `n=1` terms of the expansion. The `n=0` integral is `(e^{lambda_t}-e^{lambda_s})`, giving the DDIM term; the `n=1` integral is `integral e^{lambda}(lambda - lambda_s) d lambda`, which by parts is `e^{lambda_t}(lambda_t - lambda_s) - (e^{lambda_t}-e^{lambda_s}) = e^{lambda_t}(h - 1 + e^{-h})`. Folding `sigma_t e^{lambda_t} = alpha_t`, the cleanest way the second-order method is written reuses the same `(e^{-h}-1)` prefactor and a linear extrapolation of the data prediction onto the interval: with `r = h_{i-1}/h_i`,

  D = (1 + 1/(2 r)) x_theta(t_{i-1}) - (1/(2 r)) x_theta(t_{i-2}),
  x_{t_i} = (sigma_{t_i}/sigma_{t_{i-1}}) x_{t_{i-1}} - alpha_{t_i}(e^{-h_i} - 1) D.

`D` is just the value of `x_theta` linearly extrapolated to the *midpoint in lambda* of the interval — `(1 + 1/(2r))` weight on the near point, `-1/(2r)` on the far point — which is the right thing to multiply the leading integral by to pick up the first-derivative correction. That's the deterministic second-order multistep solver on the data face. It is more stable than the noise-face counterpart and it threshold-composes because the model quantity being approximated is the clean prediction. Good. But it's still a *deterministic* ODE march, and my first wall tells me I also want a *stability source* beyond just small steps — and there is a known one I keep circling: re-injected noise.

Why would adding noise help? Deterministic high-order extrapolation accumulates error along the trajectory with nothing to wash it out; the trajectory can drift off the data manifold and stay off. If instead I add a controlled bit of Gaussian noise each step and let the *next* denoising step remove it, that's a Langevin correction — the noise-and-denoise cycle pulls the state back toward where the model thinks data lives, cancelling accumulated discretization error. It's documented that fully-deterministic sampling can be perceptually *worse* than sampling with such re-injection. So I don't want to throw away the SDE — I want to solve it with the same exponential-integrator machinery, on the same data face, and keep a knob for how much noise to put back.

So derive the SDE solver. The reverse diffusion SDE is `dx = [f x + (g^2/sigma_t) eps_theta] dt + g dw_bar`. Notice already that the drift's network coefficient is `g^2/sigma_t`, *twice* the ODE's `g^2/(2 sigma_t)` — the SDE pushes twice as hard on the drift and pays for it with the `g dw` noise, which is the relationship I need to keep straight. Move to the `lambda` variable. For a VP schedule, `f = d log alpha/dt`, `g = sigma_t sqrt(-2 d lambda_t/dt)`, and using `d log alpha/d lambda = sigma^2`, `d log sigma/d lambda = -alpha^2` (which follow from `alpha^2 + sigma^2 = 1` differentiated), the SDE in `lambda` becomes, after carrying the change of variable through the Itô term,

  dx_lambda = [sigma_lambda^2 x_lambda - 2 sigma_lambda eps_theta] d lambda + sqrt(2) sigma_lambda dw_lambda,

and in the data face, substituting `eps = (x - alpha x_theta)/sigma`,

  dx_lambda = [-(1 + alpha_lambda^2) x_lambda + 2 alpha_lambda x_theta] d lambda + sqrt(2) sigma_lambda dw_lambda.

Variation of constants again, now with a stochastic forcing. The homogeneous part is `dx = -(1+alpha^2) x d lambda`, integrating factor `exp(integral (1+alpha^2) d lambda)`. Using `d log sigma/d lambda = -alpha^2`, `integral (1 + alpha^2) d lambda = (lambda_t - lambda_s) - (log sigma_t - log sigma_s)`, so the homogeneous solution carries `x_s` with factor `(sigma_t/sigma_s) e^{-(lambda_t - lambda_s)}`. Pushing the data term and the noise term through the integrating factor — the algebra is the same shape as the deterministic case, just keeping the Itô integral — gives the exact SDE solution

  x_t = (sigma_t/sigma_s) e^{-h} x_s + 2 alpha_t integral_{lambda_s}^{lambda_t} e^{-2(lambda_t - lambda)} x_theta(x_lambda, lambda) d lambda + sqrt(2) sigma_t integral_{lambda_s}^{lambda_t} e^{-(lambda_t - lambda)} dw_lambda.

Three pieces: a *more* strongly decayed signal carry `(sigma_t/sigma_s)e^{-h}` (the extra `e^{-h}` is the SDE pushing harder), the data integral now weighted `e^{-2(lambda_t - lambda)}` (note the `-2`, double the ODE's effective contraction), and an Itô noise integral. The Itô integral is the only new object; let me compute its variance, because that's all I need — it's a deterministic-coefficient times a Gaussian. By Itô isometry, `integral_{lambda_s}^{lambda_t} e^{-(lambda_t - lambda)} dw_lambda` is Gaussian with variance `integral e^{-2(lambda_t - lambda)} d lambda = (1 - e^{-2h})/2`. So `sqrt(2) sigma_t` times it is Gaussian with standard deviation `sqrt(2) sigma_t sqrt((1-e^{-2h})/2) = sigma_t sqrt(1 - e^{-2h})`. The noise term is `sigma_t sqrt(1 - e^{-2h}) z`, `z ~ N(0,I)`. Clean.

First-order SDE step: hold `x_theta` constant. The data integral becomes `2 alpha_t e^{-2 lambda_t} (integral e^{2 lambda} d lambda) x_theta(s) = 2 alpha_t e^{-2 lambda_t} ((e^{2 lambda_t} - e^{2 lambda_s})/2) x_theta = alpha_t (1 - e^{-2h}) x_theta(s)`. So

  x_t = (sigma_t/sigma_s) e^{-h} x_s + alpha_t (1 - e^{-2h}) x_theta(x_s,s) + sigma_t sqrt(1 - e^{-2h}) z.

This is the first-order stochastic data-prediction step. Sanity hook: is it a known sampler in disguise? DDIM with stochasticity parameter `eta_DDIM` is `x_t = alpha_t x_theta + sqrt(sigma_t^2 - eta_DDIM^2) eps_theta + eta_DDIM z`. Set `eta_DDIM = sigma_t sqrt(1 - e^{-2h})`: then `eta_DDIM^2 = sigma_t^2(1 - e^{-2h})`, the DDIM noise level matches mine on the nose, and `sqrt(sigma_t^2 - eta_DDIM^2) = sigma_t sqrt(e^{-2h}) = sigma_t e^{-h}`. Now I should actually check the deterministic parts agree, not assume it. Work in the `alpha = 1` convention (so `sigma_t/sigma_s = e^{-h}`, `alpha_t = 1`, `eps_theta = (x_s - x_theta)/sigma_s`). My step's deterministic part is `(sigma_t/sigma_s) e^{-h} x_s + (1 - e^{-2h}) x_theta`. DDIM's is `x_theta + sigma_t e^{-h} (x_s - x_theta)/sigma_s`. Substitute `sigma_s = sigma_t e^{h}` everywhere: mine becomes `e^{-2h} x_s + (1 - e^{-2h}) x_theta = e^{-2h}(x_s - x_theta) + x_theta`; DDIM's becomes `x_theta + e^{-2h}(x_s - x_theta)`. They are the same expression — the difference is zero. So my first-order SDE step is exactly stochastic DDIM at this `eta_DDIM`. Good — it sits on top of the thing everyone already trusts, and the fact that an independent re-derivation lands back on it is the kind of evidence I want that I haven't wandered off.

Now I want a *knob*, not the two extremes "deterministic ODE" or "full SDE." Look at what distinguishes them in the `alpha = 1` noise-level convention used by the implementation: the deterministic data-prediction step contracts by `e^{-h}` and adds no noise; the full SDE contracts by `e^{-2h}` and adds noise `sqrt(1 - e^{-2h})`. The "2" in the SDE is exactly the doubled drift coefficient. So introduce `eta >= 0` and let it interpolate the *renoising rate*: replace the contraction exponent by `h_eta = h (eta + 1)`, so `eta = 0` gives exponent `h` and `eta = 1` gives `2h`. The constant data-prediction part keeps the two weights `e^{-h_eta}` and `1 - e^{-h_eta}`, so that part is a convex interpolation between the current latent and the clean prediction, while boundedness still comes from clipping or thresholding the clean prediction itself. And the noise: at `eta = 1` it must be `sigma sqrt(1 - e^{-2h})`; the extra contraction beyond the ODE is by `e^{-h eta}` in amplitude, hence `e^{-2 h eta}` in variance, so the noise variance that restores the marginal is `1 - e^{-2 h eta}` and its standard deviation is `sqrt(1 - e^{-2 h eta})`. Check the endpoints: `eta=0` gives `h_eta=h`, signal `e^{-h}`, data `1-e^{-h}`, noise `sqrt(1 - e^0) = 0` — exactly the deterministic ODE, no noise. `eta=1` gives `h_eta=2h`, signal `e^{-2h}`, data `1-e^{-2h}`, noise `sqrt(1-e^{-2h})` — exactly the full SDE above. So one parameter slides continuously from the deterministic march to the noisy one, and I can dial in *more* stochasticity than the plain SDE by taking `eta` past one — useful precisely when guidance and big steps need extra Langevin self-correction. I keep `eta` a touch above one as the working default; the extra noise buys robustness at this budget.

The per-step skeleton, with `denoised` standing for `x_theta(x, sigma)` and the VE convention `alpha = 1` (so `lambda = -log sigma`, and `sigma_t/sigma_s = e^{-lambda_t}/e^{-lambda_s} = e^{-h}`): the signal carry in the general step is `(sigma_t/sigma_s)` for the ODE part and an extra `e^{-h(eta)}`-type contraction for the SDE part, and in this convention `sigma_t/sigma_s = e^{-h}` is itself an exponential of the `lambda`-step, so the whole signal factor folds into a single `e^{-h_eta}` (at `eta = 1`, `(sigma_t/sigma_s) e^{-h} = e^{-h} e^{-h} = e^{-2h} = e^{-h_eta}`; at `eta = 0` it is `e^{-h}` with no renoise). The skeleton is:

  x = e^{-h_eta} x + (1 - e^{-h_eta}) denoised,    then if eta > 0:  x += sigma_next sqrt(1 - e^{-2 h eta}) z.

Now lift it to higher order with the multistep correction, using the data integral's own Taylor expansion rather than just holding `x_theta` constant. I keep the last two `denoised` values and the last two step sizes. Expand `x_theta(lambda_s + u)` around the left endpoint: `x_theta = D_0 + D_1 u + (1/2) D_2 u^2 + ...`, where `D_1 = x_theta^{(1)}`, `D_2 = x_theta^{(2)}` are the first and second `lambda`-derivatives. The exact data term, with the general contraction rate `kappa = eta + 1`, is `kappa integral_0^h e^{-kappa(h-u)} x_theta(lambda_s + u) du` (this reproduces `1 - e^{-h_eta}` for constant `x_theta`, as it must). Integrating the `u^1` monomial against that exponential gives a coefficient on `D_1`; integrating `u^2/2` gives a coefficient on `D_2`. Those coefficients are the exponential-integrator `phi` functions evaluated at `h_eta`. The first comes out of the `u^1` integral:

  phi_2(h_eta) = (e^{-h_eta} - 1)/h_eta + 1.

I should not just trust that this is the right coefficient — let me actually equate it to the integral I derived. With contraction rate `kappa = eta + 1` and `h_eta = kappa h`, the exact coefficient multiplying `D_1` is `kappa * integral_0^h e^{-kappa(h-u)} u du`. Integrating by parts: `integral_0^h e^{-kappa(h-u)} u du = [u e^{-kappa(h-u)}/kappa]_0^h - (1/kappa) integral_0^h e^{-kappa(h-u)} du = h/kappa - (1 - e^{-kappa h})/kappa^2`. Multiply by `kappa`: `h - 1/kappa + e^{-kappa h}/kappa = h - (1 - e^{-h_eta})/(kappa)`. And `phi_2(h_eta) * h = h[(e^{-h_eta}-1)/h_eta + 1] = h + (e^{-h_eta}-1)/kappa = h - (1 - e^{-h_eta})/kappa`. The two expressions are identical. So `phi_2` is exactly the coefficient that carries the first-derivative term when the estimate is scaled by `h` — not an approximation, an equality. The second coefficient, from the `u^2/2` integral, is

  phi_3(h_eta) = phi_2(h_eta)/h_eta - 0.5.

The sign of this one matters for the sign of its term, so let me get its small-`h_eta` behavior honestly rather than eyeballing it. Taylor-expand `phi_2`: `(e^{-z}-1)/z + 1 = (-z + z^2/2 - z^3/6 + ...)/z + 1 = -1 + z/2 - z^2/6 + ... + 1 = z/2 - z^2/6 + z^3/24 - ...`, so `phi_2(h_eta) ≈ h_eta/2` for small `h_eta`. Then `phi_3 = phi_2/h_eta - 1/2 = (1/2 - h_eta/6 + h_eta^2/24 - ...) - 1/2 = -h_eta/6 + h_eta^2/24 - ...`, so `phi_3(h_eta) ≈ -h_eta/6` — and it is indeed *negative* at leading order, which I'll need to track through the sign of its term. These come from the recurrence `phi_{k+1}(z) = (phi_k(z) - phi_k(0))/z` with `phi_1(z) = (e^z - 1)/z`, adapted to the `e^{-h_eta}` weight; I read them off the integrals, not from memory.

Now the derivative estimates from past values. I have three `lambda`-points: the current `denoised` at `lambda_s`, the previous `denoised_1` one step back at `lambda`-distance `h_1`, and `denoised_2` two steps back at a further `h_2`. I want `D_1` and `D_2` at `lambda_s`, built from a quadratic through these three points — a Newton divided-difference. Scale spacings to the current step: `r0 = h_1/h`, `r1 = h_2/h`. Form the two first differences, each scaled to the current step:

  d1_0 = (denoised - denoised_1)/r0,    d1_1 = (denoised_1 - denoised_2)/r1.

`d1_0` is a backward estimate of `h * D_1` over the nearest interval; `d1_1` the same over the next interval back. Their difference is a second-difference, so the second-derivative estimate (scaled by `h^2/2`) is

  d2 = (d1_0 - d1_1)/(r0 + r1),

and the first-derivative estimate corrected to the endpoint by extrapolating the slope forward is

  d1 = d1_0 + (d1_0 - d1_1) * r0/(r0 + r1).

I want to be sure these formulas actually recover the scaled derivatives and that I haven't fumbled the `r0/(r0+r1)` weighting, so I run a quadratic through the bookkeeping by hand. Let `x_theta(L) = a + b L + c L^2` in the `lambda`-coordinate `L` measured from the current endpoint `L = 0`. Then `denoised = a`, `denoised_1 = a - b h_1 + c h_1^2` (the previous point sits at `L = -h_1`), `denoised_2 = a - b(h_1+h_2) + c(h_1+h_2)^2`. The true derivatives at the endpoint are `D_1 = b`, `D_2 = 2c`, so I'm hoping `d1 = h b` and `d2 = c h^2 = (h^2/2)(2c) = (h^2/2) D_2`. Compute `d1_0 = (denoised - denoised_1)/r0 = (b h_1 - c h_1^2)/(h_1/h) = h(b - c h_1)`. Likewise `d1_1 = (denoised_1 - denoised_2)/r1 = (b h_2 + c(h_1^2 - (h_1+h_2)^2))/(h_2/h)`; expanding `h_1^2 - (h_1+h_2)^2 = -2 h_1 h_2 - h_2^2`, this is `(b h_2 - c h_2(2 h_1 + h_2))/(h_2/h) = h(b - c(2 h_1 + h_2))`. Their difference `d1_0 - d1_1 = h c(h_1 + h_2)` (a *positive* multiple of `c`), so `d2 = (d1_0 - d1_1)/(r0 + r1) = h c(h_1+h_2)/((h_1+h_2)/h) = c h^2` — exactly `(h^2/2) D_2`. And `d1 = d1_0 + (d1_0 - d1_1) r0/(r0+r1) = h(b - c h_1) + h c(h_1+h_2) (h_1/h)/((h_1+h_2)/h) = h(b - c h_1) + h c h_1 = h b` — exactly `h D_1`, the `c h_1` bias from the backward difference cancelling against the slope extrapolation. So both estimates land where they should, with the quadratic part of `x_theta` reproduced exactly. Assemble the third-order step: the constant term plus the two correction terms weighted by the `phi`s,

  x = e^{-h_eta} x + (1 - e^{-h_eta}) denoised + phi_2 * d1 - phi_3 * d2,

and since `phi_3` is negative, the `- phi_3 * d2` term actually *adds* a positive multiple of the second-difference, which is the right curvature correction. When I only have one past value (the very first correction step), drop `d2` and use the two-point first-derivative estimate `d = (denoised - denoised_1)/r`, `r = h_1/h`, giving the second-order step `x += phi_2 * d`. When I have no past value yet (the first step), it's just the constant data-prediction step with the noise — first order.

I keep the order claim honest here, because this is not a clean "provably third order" derivation. The `phi_2` first-derivative term matches the exact integral coefficient exactly; the `phi_3` second-derivative term carries a constant that makes the local truncation error scale like `h^3` rather than the `h^4` a perfectly-tuned third-order multistep would give. That's the same kind of simplification the second-order derivation makes when it replaces an exact `phi`-coefficient by a convenient `(1 - e^{-h_eta})/2`-style midpoint factor: a deliberate trade of an exactly-tuned constant for a simpler, robust form. So this is a *practical* third-order multistep correction — it brings in the curvature of `x_theta` and improves on the second-order step, without pretending it's an exact `O(h^4)` scheme. Given that my whole reason for going multistep is robustness under guidance, a slightly-conservative but stable curvature correction is exactly the right posture; chasing an exact `O(h^4)` constant buys precision I lose to instability anyway.

A few implementation realities before I write it. The exponentials all show up as `e^{x} - 1` with small `x` — `1 - e^{-h_eta}`, `1 - e^{-2 h eta}` — where naive `exp(x) - 1` cancels catastrophically; use `expm1`, which is built for it, so `(1 - e^{-h_eta})` is `(-h_eta).expm1().neg()` and the noise standard deviation `sqrt(1 - e^{-2 h eta})` is `(-2 * h * eta).expm1().neg().sqrt()`. The noise itself wants to be Brownian-consistent and reproducible across steps, not a fresh independent `randn` each time, so I draw it from a Brownian-tree noise sampler keyed to the noise levels — it returns increments scaled by `sqrt(|delta t|)` so the injected noise has the right scale and the same seed gives the same trajectory. The very last step lands at `sigma = 0`, where there's nothing left to denoise toward and no noise to add: I just return `denoised`, the clean-image estimate itself (Tweedie's clean prediction at zero noise). And the time grid is the Karras power schedule, `sigma_i = (sigma_max^{1/rho} + (i/(N-1))(sigma_min^{1/rho} - sigma_max^{1/rho}))^rho` with `rho = 7`, appending a final `sigma = 0` — the power warp puts more steps at low noise where the per-step truncation error is largest, so the tight budget is spent where it matters. With `alpha = 1` the half-log-SNR is just `lambda = -log sigma`, so the change of variable is `t_fn = lambda sigma: -log sigma` and `h = lambda_next - lambda = (-log sigma_next) - (-log sigma)`, positive as `sigma` decreases. Each step keeps an optional small `s_noise` multiplier on the injected noise (default one) so the noise scale can be tuned without touching `eta`.

Now I write the per-step solver — the data-prediction, multistep, `eta`-stochastic, Karras-scheduled march I derived — so it drops into the generic sampling loop, with each block tied back to the step that motivated it:

```python
import torch
from tqdm.auto import trange


@torch.no_grad()
def sample_dpmpp_3m_sde(model, x, sigmas, extra_args=None, callback=None,
                        disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Third-order multistep, stochastic, data-prediction exponential-integrator sampler.
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
                # third-order: quadratic divided-difference for D_1, D_2 in lambda
                r0 = h_1 / h
                r1 = h_2 / h                                  # older interval scaled to current step
                d1_0 = (denoised - denoised_1) / r0           # ~ h * x_theta^(1) (near interval)
                d1_1 = (denoised_1 - denoised_2) / r1         # ~ same, next interval back
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)    # endpoint first-derivative estimate
                d2 = (d1_0 - d1_1) / (r0 + r1)                # ~ (h^2/2) * x_theta^(2)
                phi_2 = h_eta.neg().expm1() / h_eta + 1        # first-derivative weight
                phi_3 = phi_2 / h_eta - 0.5                    # second-derivative weight (negative)
                x = x + phi_2 * d1 - phi_3 * d2               # -phi_3 (>0) adds the curvature term
            elif h_1 is not None:
                # second-order: two-point first-derivative estimate only
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                # Langevin renoise: restores variance removed by the extra SDE contraction
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] \
                    * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        # shift the multistep history forward
        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x
```

And the time grid it marches on, the Karras power schedule that concentrates steps at low noise:

```python
def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho   # warp, interp, unwarp
    return torch.cat([sigmas, sigmas.new_zeros([1])]).to(device)          # append final sigma = 0
```

The causal chain end to end: I want fast guided sampling, and the existing high-order solvers get *worse* than first-order DDIM at large guidance because the guidance scale amplifies the model's derivatives and so shrinks the convergence radius the high-order terms need, and because a noise-prediction solver carries an unbounded quantity that thresholding cannot fix. Both pathologies trace to building on `eps`; re-deriving the same exponential-integrator solver on the *data*-prediction face — variation of constants, change of variable to log-SNR — gives an exact solution that exactly-carries `sigma_t/sigma_s` and only ever approximates an integral of `x_theta`, a clean-image estimate I can clip or threshold. To regain stability lost to guidance I take small steps via a multistep scheme, one network call per step with past predictions reused, and I add a Langevin self-correction by solving the diffusion *SDE* with the same machinery, introducing a single `eta` knob that interpolates the renoising rate from the deterministic ODE (`eta=0`) through the full SDE (`eta=1`) and beyond. First order recovers deterministic or stochastic DDIM depending on `eta`; second order adds a two-point first-derivative correction; third order adds a quadratic divided-difference curvature correction weighted by `phi_2` and `phi_3`, using the canonical `r1 = h_2/h` scaling and the canonical `- phi_3*d2` sign. Karras's power schedule spends the tight budget where truncation error lives, `expm1` keeps the small-argument exponentials honest, and a Brownian-tree sampler injects reproducible, correctly-scaled noise — landing on a training-free guided sampler that converges in fifteen to twenty network calls.
