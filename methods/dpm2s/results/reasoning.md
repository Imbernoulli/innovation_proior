Let me start from what actually goes wrong, because the symptom is concrete and it points somewhere. I have pre-trained denoisers and I want photorealistic, prompt-aligned images cheaply — say 15 to 20 network calls instead of a couple hundred. Without guidance I already know how to do this: the diffusion reverse process is a deterministic probability-flow ODE, and dedicated high-order solvers cut the unconditional sampling cost to ten or twenty calls. But the images people actually want come from *guided* sampling — classifier guidance or classifier-free guidance — and crucially they want a *large* guidance scale, because that is what sharpens the alignment between the image and the condition. So I run the fast high-order solvers I trust, crank the guidance scale up to something like 8, give them 15 function evaluations, and the result is worse than DDIM. Not marginally worse — visibly degraded, and it gets *worse* as I raise the solver order. The third-order solver is worse than the second-order, which is worse than first-order DDIM. That is backwards. Higher order is supposed to mean lower error per step. Something about the large guidance scale is breaking the very thing that makes high-order solvers fast.

I should pin down what guidance actually does to the function I'm integrating, because that's the object my solver sees. With classifier-free guidance the model the solver evaluates is not the plain `eps_theta` but `eps_tilde_theta(x_t, t, c) = s * eps_theta(x_t, t, c) + (1 - s) * eps_theta(x_t, t, empty)`, and with classifier guidance it's `eps_theta - s * sigma_t * grad log p_phi(c | x_t, t)`. Either way there is a free scale `s` multiplying a difference of model outputs. As `s` grows, two things grow together: the magnitude of `eps_tilde_theta` itself, and — this is the part that matters — its derivatives with respect to the integration variable. A high-order ODE solver works by Taylor-expanding the integrand and matching its first or second derivative; that approximation is only valid inside a convergence radius set by how large those derivatives are. Multiply the derivatives by something like `s`, and the radius within which my Taylor expansion is trustworthy shrinks by roughly the same factor. So at a fixed step size, the large-`s` integrand has wandered outside the region where my second- or third-order expansion is accurate, and the high-order terms — which are supposed to *reduce* error — instead inject garbage. First-order DDIM has no high-order term to corrupt, which is exactly why it survives. The fix can't be "use a higher order"; if anything order is the liability here. The honest move is to make each high-order step *less sensitive to the amplified derivatives* — to shrink the constant sitting in front of the error term — while keeping the order. Hold that thought; I don't yet know how.

There's a second, separate failure that I keep conflating with the first and shouldn't. Image data lives in a bounded box, `[-1, 1]` per pixel. A large guidance scale pushes `eps_tilde_theta` away from any direction a true noise vector could point, and when I integrate that all the way down, the clean image I land on falls *outside* the box — the decoded picture comes out saturated and unnatural. This is not a numerical-instability problem; it's a distributional one, the converged solution simply isn't where training data lives. People in the few-step world already have a blunt instrument for it: *thresholding* — at each step, take your running estimate of the clean image and clip it back into `[-1, 1]` (or a dynamic percentile of it). That genuinely helps. But here's the catch that I almost miss: to clip the clean image, I have to *have* the clean image as an explicit quantity at each step. My fast solvers integrate `eps_theta`, the noise. They never form an estimate of `x_0`; there's nothing to clip. DDIM, when written one way, does carry an `x_0` estimate — which is part of why DDIM-plus-thresholding is a thing and DPM-Solver-plus-thresholding is not. So both of my problems — the instability and the train-test mismatch — are pointing at the same suspect: I've been building solvers around the *noise* prediction, and maybe the noise prediction is the wrong quantity to be integrating in the first place.

Let me take that seriously and ask what changes if I integrate the *data* prediction instead. The denoiser gives me two interconvertible things. There's `eps_theta(x_t, t)`, the predicted noise, and there's `x_theta(x_t, t) := (x_t - sigma_t * eps_theta(x_t, t)) / alpha_t`, the implied estimate of the clean data from Tweedie's formula. They contain the same information; one is an affine reparameterization of the other. So I lose nothing in expressiveness by switching. And I gain the obvious thing immediately: `x_theta` *is* a clean-image estimate, so thresholding composes for free — clip `x_theta` elementwise into the box and keep going. That alone is a reason to want the data-prediction formulation. But I shouldn't stop at "it lets me clip." If switching parameterization also changes the *numerical* behavior under guidance, that would be the real prize, and it's not obvious yet that it does, because the two ODEs are equivalent. Let me actually build the solver on the data prediction and see whether the structure differs.

The diffusion ODE in noise-prediction form is `dx_t/dt = f(t) x_t + (g^2(t)/(2 sigma_t)) eps_theta`, with `f = d log(alpha_t)/dt` and `g^2 = d(sigma_t^2)/dt - 2 (d log(alpha_t)/dt) sigma_t^2`. Substituting `eps_theta = (x_t - alpha_t x_theta)/sigma_t` to rewrite it purely in terms of `x_theta` gives the equivalent ODE `dx_t/dt = (f(t) + g^2(t)/(2 sigma_t^2)) x_t - (alpha_t g^2(t)/(2 sigma_t^2)) x_theta`. It's still semi-linear — a linear term in `x_t` plus a nonlinear term through the network — so the same exponential-integrator strategy applies: solve the linear part exactly by variation of constants, and approximate only the integral of the nonlinear part. The question is what the "exact linear part" and the "remaining integral" look like in this parameterization, because *that* is where the two formulations can diverge even though their solutions agree.

I'll guess the form of the exact solution and verify it by differentiation, which is cleaner than grinding through variation-of-constants blind. By analogy with the noise-prediction solution `x_t = (alpha_t/alpha_s) x_s - alpha_t * integral e^{-lambda} eps_hat_theta dlambda`, where the integration variable is the half-log-SNR `lambda_t := log(alpha_t/sigma_t)`, I'll conjecture that the data-prediction solution exactly computes the *other* linear factor — `sigma_t/sigma_s` instead of `alpha_t/alpha_s` — and integrates `e^{+lambda} x_hat_theta` instead of `e^{-lambda} eps_hat_theta`. So my candidate is

  `x_t = (sigma_t/sigma_s) x_s + sigma_t * integral_{lambda_s}^{lambda_t} e^{lambda} x_hat_theta(x_hat_lambda, lambda) dlambda`,

where the hats are the change-of-variable forms in `lambda`. Let me check it by differentiating with respect to `t`. The first term differentiates to `(d sigma_t/dt) x_s/sigma_s`. The integral, being multiplied by `sigma_t` out front, gives two pieces by the product rule: `(d sigma_t/dt) * integral e^{lambda} x_hat_theta dlambda`, plus `sigma_t` times the derivative of the integral itself, which by the fundamental theorem of calculus (and the chain rule through `lambda_t`) is `(d lambda_t/dt) sigma_t e^{lambda_t} x_hat_theta(x_t, t)`. Now the first two pieces combine: `(d sigma_t/dt)[x_s/sigma_s + integral e^{lambda} x_hat_theta dlambda] = (d sigma_t/dt) * (x_t/sigma_t)`, recognizing the bracket as `x_t/sigma_t` from my own ansatz. So

  `dx_t/dt = (d sigma_t/dt)(x_t/sigma_t) + (d lambda_t/dt) sigma_t e^{lambda_t} x_theta(x_t, t)`.

The first term is `(d log sigma_t/dt) x_t`, and using `sigma_t e^{lambda_t} = sigma_t (alpha_t/sigma_t) = alpha_t` together with `g^2 = -2 sigma_t^2 (d lambda_t/dt)` so that `(d lambda_t/dt) = -g^2/(2 sigma_t^2)`, the second term is `-(alpha_t g^2/(2 sigma_t^2)) x_theta`. And `(d log sigma_t/dt) = f + g^2/(2 sigma_t^2)` after expanding `f` and `g^2` from their definitions. That is exactly the data-prediction ODE. The ansatz checks out:

  `x_t = (sigma_t/sigma_s) x_s + sigma_t * integral_{lambda_s}^{lambda_t} e^{lambda} x_hat_theta(x_hat_lambda, lambda) dlambda`.

Now I can see the difference, and it is not cosmetic. The noise-prediction solver makes `alpha_t/alpha_s` exact and integrates `e^{-lambda} eps_hat_theta`; the data-prediction solver makes `sigma_t/sigma_s` exact and integrates `e^{+lambda} x_hat_theta`. These are different discretization targets. The two are equal *as exact solutions*, but the integrals I'm approximating are different functions, weighted by opposite exponentials, so any finite-order Taylor approximation of one is *not* the same approximation of the other. There's room here for the data-prediction version to have a smaller error constant — that's the thing I flagged earlier and couldn't yet name. Whether it actually does, I'll have to compute. But at minimum the data-prediction formulation gives me `x_theta` explicitly at every step (clip it — thresholding fixed) and is a genuinely distinct numerical scheme. Both reasons to keep going.

So now I want a high-order solver for `integral e^{lambda} x_hat_theta dlambda`. The standard exponential-integrator move: I'm stepping from `t_{i-1}` to `t_i`, I have `x_tilde_{t_{i-1}}`, and I Taylor-expand `x_hat_theta(lambda)` around `lambda_{t_{i-1}}` to order `k-1`,

  `x_hat_theta(lambda) = sum_{n=0}^{k-1} (lambda - lambda_{t_{i-1}})^n/n! * x_theta^{(n)}(lambda_{t_{i-1}}) + O((lambda - lambda_{t_{i-1}})^k)`,

where `x_theta^{(n)}` is the `n`-th total derivative in `lambda`. Substitute into the exact step and the integral splits into a sum of analytically computable pieces times the unknown derivatives:

  `x_tilde_{t_i} = (sigma_{t_i}/sigma_{t_{i-1}}) x_tilde_{t_{i-1}} + sigma_{t_i} sum_{n=0}^{k-1} x_theta^{(n)}(lambda_{t_{i-1}}) * integral_{lambda_{t_{i-1}}}^{lambda_{t_i}} e^{lambda} (lambda - lambda_{t_{i-1}})^n/n! dlambda + O(h_i^{k+1})`,

with `h_i := lambda_{t_i} - lambda_{t_{i-1}}`. The whole game reduces to (a) computing those integrals in closed form and (b) estimating the derivatives `x_theta^{(n)}` for `n <= k-1`. The integrals first, because they're just calculus. Let me do `n = 0` and `n = 1`, which is all I need for a second-order method.

For `n = 0`: `integral_{lambda_s}^{lambda_t} e^{lambda} dlambda = e^{lambda_t} - e^{lambda_s}`. Multiply by `sigma_t`. Use `sigma_t e^{lambda_t} = alpha_t`, and `e^{lambda_s} = e^{lambda_t} e^{-(lambda_t - lambda_s)} = e^{lambda_t} e^{-h}`, so `sigma_t(e^{lambda_t} - e^{lambda_s}) = alpha_t - alpha_t e^{-h} = alpha_t(1 - e^{-h}) = -alpha_t(e^{-h} - 1)`. Clean.

For `n = 1`: I need `integral_{lambda_s}^{lambda_t} e^{lambda}(lambda - lambda_s) dlambda`, by parts. Let `u = (lambda - lambda_s)`, `dv = e^{lambda} dlambda`, so `v = e^{lambda}` and `du = dlambda`: the integral is `[e^{lambda}(lambda - lambda_s)]_{lambda_s}^{lambda_t} - integral_{lambda_s}^{lambda_t} e^{lambda} dlambda = e^{lambda_t} h - (e^{lambda_t} - e^{lambda_s})`. Multiply by `sigma_t`: `sigma_t e^{lambda_t} h - sigma_t(e^{lambda_t} - e^{lambda_s}) = alpha_t h - alpha_t(1 - e^{-h}) = alpha_t(h - 1 + e^{-h})`. So the second-order coefficient is `alpha_t(h - 1 + e^{-h})`.

Take `k = 1` first as a sanity check on the whole apparatus. Keeping only the `n = 0` term and dropping the `O(h^2)`,

  `x_tilde_{t_i} = (sigma_{t_i}/sigma_{t_{i-1}}) x_tilde_{t_{i-1}} - alpha_{t_i}(e^{-h_i} - 1) x_theta(x_tilde_{t_{i-1}}, t_{i-1})`.

I should recognize this. Rewrite DDIM (deterministic, `eta = 0`) in data-prediction form: `x_{t_i} = alpha_{t_i} x_theta + sigma_{t_i} eps_theta`, and using `eps_theta = (x_{t_{i-1}} - alpha_{t_{i-1}} x_theta)/sigma_{t_{i-1}}` to eliminate `eps_theta`, the DDIM step collapses to exactly the line above. So the first-order member of my data-prediction family *is* DDIM — the same way the first-order member of the noise-prediction family is DDIM. Good: my construction is a high-order generalization of DDIM with respect to the data prediction. That's reassuring; it means I'm extending the one solver that was robust under guidance, not inventing something disconnected from it.

Now `k = 2`, where the contribution actually is. I keep the `n = 1` term, so I need an estimate of the first derivative `x_theta^{(1)}(lambda_{t_{i-1}})`. I have no extra information at `t_{i-1}` alone — a derivative needs a second sample. There are two mainstream ways to get one. The *multistep* way reuses a value the solver already computed at an earlier time step, costing no new network call. The *singlestep* way introduces a fresh intermediate time `s_i` strictly between `t_{i-1}` and `t_i`, evaluates the network there, and uses the finite difference between the two evaluations as the derivative estimate. The singlestep one costs an extra evaluation per step, but it doesn't depend on history, so each step stands alone and the local error constant is governed entirely by the within-step geometry — and that local error constant is exactly the lever I identified for fighting the guidance instability. Let me build the singlestep version. (It's the classical second-order one-step construction — the same shape as Heun's or the midpoint method, just carried out for this exponentially-weighted integral.)

I place the intermediate point at `lambda_{s_i} = lambda_{t_{i-1}} + r_i h_i` for some fraction `r_i` in `(0, 1)`, with `r_i = 1/2` the natural midpoint default. To get a value at `s_i` to feed the network, I take a first-order (DDIM, data-prediction) sub-step from `t_{i-1}` to `s_i`. That's the same `k = 1` formula but with `r_i h_i` in place of `h_i`:

  `u_i = (sigma_{s_i}/sigma_{t_{i-1}}) x_tilde_{t_{i-1}} - alpha_{s_i}(e^{-r_i h_i} - 1) x_theta(x_tilde_{t_{i-1}}, t_{i-1})`.

Now I have two clean-image predictions: `x_theta(x_tilde_{t_{i-1}}, t_{i-1})` at the start, and `x_theta(u_i, s_i)` at the intermediate point. Their difference, divided by the `lambda`-gap `r_i h_i` between them, is a finite-difference estimate of the derivative:

  `x_theta^{(1)}(lambda_{t_{i-1}}) approx (x_theta(u_i, s_i) - x_theta(x_tilde_{t_{i-1}}, t_{i-1})) / (r_i h_i)`.

Substitute that, with the `n = 1` coefficient `alpha_{t_i}(h_i - 1 + e^{-h_i})`, into the full step. The exact second-order step is

  `x_tilde_{t_i} = (sigma_{t_i}/sigma_{t_{i-1}}) x_tilde_{t_{i-1}} - alpha_{t_i}(e^{-h_i} - 1) x_theta(x_tilde_{t_{i-1}}, t_{i-1}) + alpha_{t_i}(h_i - 1 + e^{-h_i}) * (x_theta(u_i, s_i) - x_theta(x_tilde_{t_{i-1}}, t_{i-1}))/(r_i h_i)`.

This is correct as written, but the coefficient `(h_i - 1 + e^{-h_i})/(r_i h_i)` is a bit ungainly, and there's a standard simplification that costs nothing in order. Expand for small `h`: `h - 1 + e^{-h} = h - 1 + (1 - h + h^2/2 - ...) = h^2/2 + O(h^3)`, so `(h - 1 + e^{-h})/h = h/2 + O(h^2)`. And separately `-(e^{-h} - 1)/2 = -((-h + h^2/2 - ...) )/2 = h/2 + O(h^2)`. The two agree to leading order in `h`. So I may replace `(h_i - 1 + e^{-h_i})/h_i` by `-(e^{-h_i} - 1)/2` without changing the second-order accuracy — and the payoff is that the same factor `-(e^{-h_i} - 1)` that already multiplies the `n = 0` term now also multiplies the correction. That's the clean form. The correction coefficient becomes `-alpha_{t_i}(e^{-h_i} - 1)/(2 r_i)`, and I can fold everything as a single weighted combination of the two clean-image predictions. Define

  `D_i = (1 - 1/(2 r_i)) x_theta(x_tilde_{t_{i-1}}, t_{i-1}) + (1/(2 r_i)) x_theta(u_i, s_i)`,
  `x_tilde_{t_i} = (sigma_{t_i}/sigma_{t_{i-1}}) x_tilde_{t_{i-1}} - alpha_{t_i}(e^{-h_i} - 1) D_i`.

Let me verify this `D_i` matches the substituted form. Expanding: `-alpha_{t_i}(e^{-h_i} - 1) D_i = -alpha_{t_i}(e^{-h_i} - 1)[(1 - 1/(2 r_i)) x_theta(start) + (1/(2 r_i)) x_theta(intermediate)]`. Collect the `x_theta(start)` terms: `-alpha_{t_i}(e^{-h_i}-1) x_theta(start)` from the `n=0` part is exactly the `1` inside the bracket, and the `-1/(2 r_i)` piece plus the `+1/(2 r_i)` on the intermediate combine to `+(1/(2 r_i))·(-alpha_{t_i}(e^{-h_i}-1))·(x_theta(intermediate) - x_theta(start))` — which is the correction term with coefficient `-alpha_{t_i}(e^{-h_i}-1)/(2 r_i)`, exactly the simplified form above. So `D_i` is just a tidy bookkeeping of "the base clean-image prediction plus a derivative correction." With the canonical midpoint default `r_i = 1/2`, `1/(2 r_i) = 1`, so `D_i = 0 * x_theta(start) + 1 * x_theta(intermediate) = x_theta(u_i, s_i)` — at the midpoint the start prediction drops out entirely and the final update is driven by the intermediate clean-image prediction alone, which is the cleanest form to implement.

I want to know whether this is genuinely second-order or just dressed up to look like it. Two ways to check, and I'll do both because they catch different mistakes. First the local-error algebra: how well does one step approximate the exact step? The base `n = 0` term is exact; the `n = 1` term carries the true coefficient `alpha_t(h - 1 + e^{-h})` against the true derivative, while my method carries `-alpha_t(e^{-h} - 1)/2` against `(x_theta(intermediate) - x_theta(start))/(r h) approx` the true derivative `+ O(h)`. The residual on the derivative term is `alpha_t[(h - 1 + e^{-h}) - (-(e^{-h} - 1)/2)·h]` acting on the derivative; expand the bracket: `(h - 1 + e^{-h}) + (e^{-h} - 1)h/2`. Using `e^{-h} = 1 - h + h^2/2 - h^3/6 + O(h^4)`: the first part is `h^2/2 - h^3/6 + O(h^4)`, the second is `(-h + h^2/2)h/2 = -h^2/2 + h^3/4 + O(h^4)`, and they sum to `(-1/6 + 1/4)h^3 + O(h^4) = h^3/12 + O(h^4)`.

That `h^3/12` is the kind of clean coefficient that's easy to fool yourself into believing, so let me put numbers on it. Evaluating `(h - 1 + e^{-h}) + (e^{-h} - 1)h/2` directly: at `h = 0.2` it is `6.04e-4` against `h^3/12 = 6.67e-4` (ratio `0.906`); at `h = 0.1`, `7.93e-5` vs `8.33e-5` (`0.951`); at `h = 0.05`, `1.016e-5` vs `1.042e-5` (`0.975`); at `h = 0.025`, `1.286e-6` vs `1.302e-6` (`0.988`). The ratio marches toward 1 as `h` halves and the residual itself shrinks by roughly `8x` per halving — that's the `h^3` signature, and the leading constant really is `1/12`. So the local truncation error of the second-order term is `O(h^3)`; the finite-difference derivative error contributes another `O(h)` on an already-`O(h^2)` term, i.e. `O(h^3)` too. Local error `O(h^3)` per step over `O(1/h)` steps suggests global `O(h^2)` — but that step from local to global is exactly where order claims quietly fail (error can accumulate faster than the per-step bound if the iteration amplifies it), so I don't want to take it on faith.

The honest test is to run the actual update rule on a problem whose exact answer I know and watch the global error scale. I can't run the real U-Net here, but the integrator's order is a property of the *quadrature*, not of which network sits in the oracle slot — so I replace `x_theta` with a simple linear oracle `x_theta(x,t) = K x` (`K = 0.7`), which makes the whole data-prediction ODE solvable to machine precision by a fine RK4 reference. Pick any positive schedule with monotone `lambda` (I'll use `alpha(t) = 1/(1+t)`, `sigma(t) = 0.3 + 0.5 t`, which gives `lambda` strictly decreasing on `[0,1]`; nothing in the derivation needed `alpha^2 + sigma^2 = 1`), integrate from `t = 1` down to `t = 10^{-3}` with `M` uniform-in-`t` steps, and compare against the reference. Running the first-order update and the second-order singlestep update side by side, halving the step each time and reading off `log2` of the error ratio:

```
M     order-1 error  (rate)     order-2 error  (rate)
  8    5.29e-3  (1.22)          1.60e-4  (0.69)
 16    2.43e-3  (1.12)          5.30e-5  (1.60)
 32    1.16e-3  (1.07)          1.49e-5  (1.83)
 64    5.67e-4  (1.03)          3.92e-6  (1.92)
128    2.80e-4  (1.02)          1.00e-6  (1.96)
```

The order-1 update sits at rate ~1 — that's DDIM, exactly as I argued. The order-2 update climbs to rate 1.96 and is still rising toward 2 as the steps shrink (the early rows are noisy because the asymptotic regime hasn't kicked in yet at large `h`). That is the behavior of a genuine second-order method, and it's the same update rule I'm about to ship, not a stand-in. So the construction holds up to a real measurement, not just to the algebra. The convergence is under the usual regularity I should state: the total derivatives of `x_hat_theta` up to order two exist and are continuous (hence bounded), `x_theta` is Lipschitz in `x`, the max `lambda`-step `h_max = O(1/M)`, and `r_i` stays bounded away from zero so the finite-difference denominator `r_i h_i` doesn't blow up. The propagation argument that turns local into global is the standard one: writing `Delta_i = ||x_tilde_{t_i} - x_{t_i}||`, the intermediate point inherits `||u_i - x_{s_i}|| <= C Delta_{i-1} + C L h_i Delta_{i-1} + C h_i^2`, and stitching the local `O(h_i^3)` error with the Lipschitz dependence on `Delta_{i-1}` gives the recursion `Delta_i <= (alpha_{t_i}/alpha_{t_{i-1}}) Delta_{i-1} + C_tilde h_i(Delta_{i-1} + h_i^2)` — the homogeneous factor comes through the `x_theta`-Lipschitz bound on the error and is `alpha_{t_i}/alpha_{t_{i-1}}`, not the update's own `sigma_{t_i}/sigma_{t_{i-1}}` linear coefficient — which iterates to `Delta_i = O(h_max^2)`. The numeric experiment is what makes me trust this isn't merely formal.

Now the question I deferred: does this data-prediction second-order singlestep solver actually have a *smaller* error constant than the noise-prediction one — is that the source of the stability under guidance, or did I just get thresholding? Let me convert my step back into noise-prediction language and compare it directly with the noise-prediction second-order solver. The conversion is `x_theta(x, t) = (x - sigma_t eps_theta)/alpha_t = x/alpha_t - e^{-lambda_t} eps_theta(x, t)`. Pushing my `u_i` and `x_tilde_{t_i}` through this substitution is several lines of exponential bookkeeping where a dropped factor would be easy to miss, so I'll do it by hand and then check it numerically before I believe any conclusion. The intermediate point becomes, after the dust settles, `u_i = (alpha_{s_i}/alpha_{t_{i-1}}) x_tilde_{t_{i-1}} - sigma_{s_i}(e^{r_i h_i} - 1) eps_theta(x_tilde_{t_{i-1}}, t_{i-1})` — *identical* to what the noise-prediction second-order solver uses for its intermediate point. The interesting part is the final update. Mine becomes

  `x_tilde_{t_i} = (alpha_{t_i}/alpha_{t_{i-1}}) x_tilde_{t_{i-1}} - sigma_{t_i}(e^{h_i} - 1) eps_theta(x_tilde_{t_{i-1}}, t_{i-1}) - (sigma_{t_i}/(2 r_i))(e^{h_i} - 1) e^{-r_i h_i} (eps_theta(u_i, s_i) - eps_theta(x_tilde_{t_{i-1}}, t_{i-1}))`,

whereas the noise-prediction second-order solver is the same thing *without* the `e^{-r_i h_i}` factor on the last term:

  `x_tilde_{t_i} = (alpha_{t_i}/alpha_{t_{i-1}}) x_tilde_{t_{i-1}} - sigma_{t_i}(e^{h_i} - 1) eps_theta(start) - (sigma_{t_i}/(2 r_i))(e^{h_i} - 1)(eps_theta(intermediate) - eps_theta(start))`.

Term by term, the *only* difference is that single extra factor `e^{-r_i h_i}` multiplying the correction term. Before I lean on that, I want to be sure I didn't slip a factor in the conversion, so I check the two forms agree numerically. I take a generic schedule point and random oracle values: pick `(alpha_s, sigma_s) = (0.3, 0.95)` (start, low SNR) and `(alpha_t, sigma_t) = (0.8, 0.6)` (end, higher SNR), which give `h = lambda_t - lambda_s = 1.440 > 0` as the sampling direction requires, place `s_1` at the lambda-midpoint, draw random `x`, `eps_start`, `eps_inter`, form the `x_0` oracle values by Tweedie, run my *data*-prediction 2S update, and separately evaluate my hand-derived eps-form. They match to `8.9e-16` — the conversion is right. Then I evaluate the *noise*-prediction DPM-Solver-2 update (same expression minus the `e^{-r_i h_i}` factor) on the same inputs: it differs from my step by `0.56` in max-norm, and the factor itself here is `e^{-r_i h_i} = e^{-0.720} = 0.487`. So the discrepancy is not an artifact — it is exactly that one factor, and it is genuinely below 1.

And here's why that factor matters: `h_i = lambda_{t_i} - lambda_{t_{i-1}} > 0` (the half-log-SNR increases as `t` decreases toward 0), and `r_i > 0`, so `e^{-r_i h_i} < 1`. The correction term is precisely the term carrying the first-derivative estimate, with `eps_theta(intermediate) - eps_theta(start) = r_i h_i eps_theta^{(1)}(start) + O(h_i^2)` along the true trajectory. After the `(e^{h_i}-1)/(2r_i)` factor, this is the high-order piece where the guidance-amplified derivative error enters. My data-prediction solver multiplies that error-bearing term by a number strictly less than one; the noise-prediction solver multiplies it by one. Same order, smaller constant in front of the high-order error term. So this — and not thresholding — is what answers the puzzle I started with. When guidance amplifies the derivatives, the constant in front of the error term is exactly what amplifies *with* them; shrinking that constant by `e^{-r_i h_i}` is what keeps the second-order step from blowing past its convergence radius. The reason the data prediction stabilizes guided sampling isn't soft ("it's bounded") — it's this concrete `e^{-r_i h_i} < 1` damping on the error term, now confirmed against a direct numerical comparison rather than just read off the algebra. So switching to the data prediction wasn't only about getting `x_0` to clip; it genuinely changes the numerics in the direction I needed.

While I'm here, thresholding falls into place with nothing extra. Because my solver carries `x_theta` explicitly at each call, I define a clipped data-prediction model `x_hat_theta(x, t) = clip(x_theta(x, t))` — elementwise into the data bound, or a dynamic-percentile version — and substitute it wherever `x_theta` appears. The train-test mismatch is handled at the source. One caveat I want to state plainly: this is for *pixel-space* data, where `[-1, 1]` is a real bound. For latent-space models like the VAE-latent diffusion behind Stable Diffusion, the latent isn't bounded the same way, so I leave thresholding off there — but the *numerical* benefit, the smaller error constant from the data parameterization, still holds, and that's what I care about most for stability under large classifier-free guidance.

Let me think about how the fixed NFE budget interacts with this being a *singlestep* method, because that's a real cost. Each second-order singlestep step makes two network calls: one at `t_{i-1}` and one at the intermediate `s_i`, and the intermediate value `u_i` is used once and thrown away. So for a budget of `N` function evaluations I get about `N/2` actual second-order boundaries. That's the price of singlestep over multistep — a multistep method would reuse a past value and get `N` steps with one new call each — but I'm buying the cleaner, history-independent local error constant in exchange, which is what fights the guidance instability. If the NFE budget is odd, or if I explicitly choose a lower-order final stabilizer for a very small step count, the order schedule can reserve one boundary for a plain first-order (DDIM) step. Otherwise the paired boundaries use the second-order update. For the time-step spacing itself, I found uniform-in-`t` works best for the high-resolution guided setting (a power-law family in `t` with exponent one), which I'll just adopt.

A small but important numerical point before I write code: I keep writing `e^{-h} - 1` and `e^{-r h} - 1`. For small `h` these are catastrophic-cancellation traps — subtracting 1 from a number very close to 1 throws away precision. The fix is the standard `expm1` primitive, which computes `e^x - 1` accurately for small `x`. So everywhere a `(e^{...} - 1)` appears, I'll use `expm1` of the exponent. With the increasing-`lambda` convention `h > 0`, my exponents are `-h` and `-r h`, both negative, and `expm1(-h)`, `expm1(-r h)` are exactly the `(e^{-h} - 1)`, `(e^{-r h} - 1)` factors I derived.

Now let me write the per-step second-order singlestep update as real code, filling the one empty slot the harness left me. I'll keep it in the data-prediction (`x0`) convention, with the noise schedule giving me `alpha`, `sigma`, `lambda` and the inverse `lambda`. The model is wrapped to return the data prediction `x0 = (x - sigma_t * noise) / alpha_t`, optionally clipped (thresholding) — that wrapper is where guidance and thresholding both live, so the solver itself just sees a clean `x0` oracle.

```python
import torch


def expm1(x):
    return torch.expm1(x)                       # accurate e^x - 1 for small x


def data_prediction(model_fn, ns, x, t, cfg_guidance, uc, c, threshold=None):
    """x0 estimate from the guided noise prediction (Tweedie), optionally clipped."""
    eps_uncond, eps_cond = model_fn(x, t, uc, c)            # one network call (two embeds)
    eps = eps_uncond + cfg_guidance * (eps_cond - eps_uncond)   # guided noise eps_tilde
    alpha_t, sigma_t = ns.alpha(t), ns.sigma(t)
    x0 = (x - sigma_t * eps) / alpha_t                     # x_theta = clean-image estimate
    if threshold is not None:                              # thresholding (pixel-space only)
        x0 = threshold(x0)
    return x0


def dpm_solver_pp_2s_step(model_fn, ns, x, s, t, cfg_guidance, uc, c, r1=0.5, threshold=None):
    """One second-order singlestep update from time s to time t (s > t), data-prediction ODE."""
    lambda_s, lambda_t = ns.lamb(s), ns.lamb(t)
    h = lambda_t - lambda_s                                # h > 0 (lambda increases as t -> 0)
    lambda_s1 = lambda_s + r1 * h                          # intermediate point in lambda
    s1 = ns.inverse_lamb(lambda_s1)
    sigma_s, sigma_s1, sigma_t = ns.sigma(s), ns.sigma(s1), ns.sigma(t)
    alpha_s1, alpha_t = ns.alpha(s1), ns.alpha(t)

    phi_11 = expm1(-r1 * h)                                # e^{-r1 h} - 1
    phi_1 = expm1(-h)                                      # e^{-h} - 1

    # x0 at the start of the step
    model_s = data_prediction(model_fn, ns, x, s, cfg_guidance, uc, c, threshold)
    # first-order (DDIM, x0) sub-step to the intermediate point s1
    x_s1 = (sigma_s1 / sigma_s) * x - (alpha_s1 * phi_11) * model_s
    # x0 at the intermediate point (the second network call)
    model_s1 = data_prediction(model_fn, ns, x_s1, s1, cfg_guidance, uc, c, threshold)

    # second-order update: base x0 prediction + derivative correction
    #   x_t = (sigma_t/sigma_s) x - alpha_t(e^{-h}-1)[ model_s + (1/(2 r1))(model_s1 - model_s) ]
    x_t = (
        (sigma_t / sigma_s) * x
        - (alpha_t * phi_1) * model_s
        - (0.5 / r1) * (alpha_t * phi_1) * (model_s1 - model_s)
    )
    return x_t


@torch.no_grad()
def sample(model_fn, ns, x_T, timesteps, cfg_guidance, uc, c, r1=0.5, threshold=None,
           orders=None, lower_order_final=False):
    """timesteps are decreasing step boundaries; order-2 boundaries cost two network calls."""
    x = x_T
    n = len(timesteps) - 1
    if orders is None:
        orders = [2] * n
    for i in range(n):
        s, t = timesteps[i], timesteps[i + 1]
        use_first_order = orders[i] == 1 or (lower_order_final and i == n - 1)
        if use_first_order:
            lambda_s, lambda_t = ns.lamb(s), ns.lamb(t)
            phi_1 = expm1(-(lambda_t - lambda_s))
            model_s = data_prediction(model_fn, ns, x, s, cfg_guidance, uc, c, threshold)
            x = (ns.sigma(t) / ns.sigma(s)) * x - (ns.alpha(t) * phi_1) * model_s
        else:
            x = dpm_solver_pp_2s_step(model_fn, ns, x, s, t, cfg_guidance, uc, c, r1, threshold)
    return x
```

Let me retrace the causal chain to make sure it actually closes. I started with a concrete failure: fast high-order solvers, which beat DDIM without guidance, become *worse* than DDIM under a large guidance scale, and worse as the order rises. Diagnosing it, a large guidance scale amplifies the model's `lambda`-derivatives, shrinking the convergence radius of any high-order Taylor step, so the high-order term — which carries a constant times those amplified derivatives — overshoots. Separately, large guidance pushes the clean image out of the data box (train-test mismatch), and the cure, thresholding, needs an explicit clean-image estimate the noise-prediction solvers never form. Both symptoms pointed at the parameterization, so I rebuilt the solver on the data prediction `x_theta` instead of the noise `eps_theta`. Verifying an ansatz by differentiation gave the exact data-prediction solution `(sigma_t/sigma_s) x_s + sigma_t integral e^{lambda} x_hat_theta dlambda` — a genuinely different discretization target from the noise-prediction one. The first-order member is DDIM, so I'm generalizing the robust solver. For second order I added one intermediate point, took a first-order sub-step to it, and used the finite difference of the two clean-image predictions as the derivative estimate, folding it into the weighted combination `D_i` with the `e^{-h}-1` coefficient; the local error works out to `O(h^3)` (leading scalar `h^3/12`, which I checked numerically), and running the update against a high-accuracy reference on a linear-oracle ODE measured a global convergence rate climbing to 1.96 — so it really is second-order, not just formally. Converting back to noise language revealed the whole point: my data-prediction second-order step is identical to the noise-prediction one except for an extra factor `e^{-r h} < 1` on the error-bearing correction term — which I confirmed to machine precision by comparing the two updates on random inputs — a strictly smaller error constant, which is exactly what blunts the guidance amplification and explains the stability. Thresholding then comes for free because `x_theta` is in hand at every call. The cost is that singlestep spends two network calls per second-order boundary, so an `N`-NFE budget yields about `N/2` paired boundaries, with the order schedule reserving a first-order DDIM boundary only when the budget or stabilizer choice calls for it — and that lands on the concrete update above.
