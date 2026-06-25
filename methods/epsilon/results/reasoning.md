Let me start from what actually hurts. I have a generative idea I genuinely believe in — fix a forward process that slowly turns data into pure Gaussian noise, then learn the reverse chain that turns noise back into data — and it has every theoretical virtue: it is a proper latent-variable model, each step is a tractable Gaussian, I can train it by a variational bound, and I can make it arbitrarily expressive by taking many small steps. And yet when people build it, the samples come out blurry and weak. Nobody has shown this family can make a sharp image. So the architecture is not obviously the problem; the U-Net I have for image-to-image work is the same kind of net that powers the best autoregressive and score models. The problem must be in the two choices I haven't pinned down: *what exactly the reverse network should output*, and *how I weight each timestep's loss*. Those are the two free knobs, they are entangled, and the obvious settings of them are the ones that have already failed. So let me re-derive the whole thing from the bound and watch where a different choice could fall out.

The forward process is fixed. Each step is `q(x_t|x_{t-1}) = N(x_t; sqrt(1-beta_t) x_{t-1}, beta_t I)` — scale the signal down by `sqrt(1-beta_t)`, inject variance `beta_t`. The scaling matters: it keeps the total variance bounded so the chain converges to a standard normal instead of blowing up, and it is why a standard-normal prior is the right endpoint. Write `alpha_t = 1 - beta_t`. The first thing I want is to be able to jump to any timestep without simulating the whole chain, because I am going to train on random timesteps and I do not want a thousand sequential steps per gradient. Composing Gaussians: `x_1 = sqrt(alpha_1) x_0 + sqrt(beta_1) eps_1`, then `x_2 = sqrt(alpha_2) x_1 + sqrt(beta_2) eps_2 = sqrt(alpha_2 alpha_1) x_0 + sqrt(alpha_2 beta_1) eps_1 + sqrt(beta_2) eps_2`. Those last two terms are independent zero-mean Gaussians; their variances add, `alpha_2 beta_1 + beta_2 = alpha_2(1-alpha_1) + (1-alpha_2) = 1 - alpha_1 alpha_2`. So with `alpha_bar_t = prod_{s<=t} alpha_s`,

  q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) x_0, (1 - alpha_bar_t) I),

and in reparameterized form `x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps` for a single `eps ~ N(0,I)`. Good — one Gaussian sample takes me to any `t`. Notice `alpha_bar_t` runs from near 1 (almost clean) down to near 0 (almost pure noise) as `t` grows, and `1 - alpha_bar_t` is the noise variance climbing the other way. With a small linear `beta` schedule — say `T=1000`, `beta` linear `1e-4` to `0.02` — let me actually compute how close the endpoint is to the prior rather than just assert it. The cumulative product `alpha_bar_T = prod (1-beta_t)`: with these numbers the `log alpha_bar_T = sum log(1-beta_t)` sums to about `-10.1`, so `alpha_bar_T ≈ e^{-10.1} ≈ 4.0e-5`. Then for a unit-variance data coordinate `q(x_T|x_0) = N(sqrt(alpha_bar_T) x_0, 1-alpha_bar_T)`, and `KL` against `N(0,1)`, averaged over `x_0 ~ N(0,1)`, is `½(var_q + alpha_bar_T·E[x_0^2] - 1 - log var_q)` with `var_q = 1-alpha_bar_T ≈ 0.99996`. Plugging in: `½(0.99996 + 4.0e-5 - 1 - log 0.99996) = ½(0 - log(1-4.0e-5)) ≈ ½·4.0e-5 ≈ 2.0e-5` nats, i.e. `~3e-5` bits per dimension. So the prior really does match the chain's endpoint to within a hundredth of a millibit — `L_T` is negligible, not by hope but by that calculation.

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

That was four lines of cancellation; I don't trust it until I've checked it numerically against the un-simplified two-coefficient form. Take `t` with `alpha_t = 0.99`, `alpha_bar_t = 0.5`, so `alpha_bar_{t-1} = alpha_bar_t/alpha_t = 0.50505`, `beta_t = 0.01`. Pick `x_0 = 1`, `eps = 0.6`, so `x_t = sqrt(0.5)·1 + sqrt(0.5)·0.6 = 0.70711·1.6 = 1.13137`. Original form: `coef1 = sqrt(0.50505)·0.01/0.5 = 0.71067·0.02 = 0.014213`, `coef2 = sqrt(0.99)·0.49495/0.5 = 0.99499·0.98990 = 0.98494`, giving `mu_tilde = 0.014213·1 + 0.98494·1.13137 = 0.014213 + 1.11433 = 1.12854`. Collapsed form: `(1/sqrt(0.99))(1.13137 - (0.01/sqrt(0.5))·0.6) = 1.00504·(1.13137 - 0.014142·0.6) = 1.00504·(1.13137 - 0.008485) = 1.00504·1.12289 = 1.12854`. The two land on `1.12854` together, so the four lines of cancellation are right. Stare at it. The forward posterior mean, the thing my network is supposed to predict, is *the input `x_t` minus a specific multiple of the noise `eps` that was added*. Everything multiplying `x_t` is a known constant; the only unknown the network needs to supply is `eps`. So predicting `mu_tilde_t` directly is making the network re-learn how to reconstruct `x_t` from `x_t` plus a correction — it has to output something whose dominant part is just `x_t` again, with the real signal buried in a small `eps`-dependent correction that is scaled differently at every `t`. That is a poorly conditioned regression target: the target's overall scale and its dependence on the input both wander with `t`, so the net spends capacity tracking the trivial `x_t` part and the loss is dominated by it.

So let me change what the network outputs. Since `x_t` is already an input, let me bake the known `x_t`-dependence into the parameterization and have the network predict only the genuinely unknown piece — the noise. Define a function approximator `eps_theta(x_t, t)` and *set*

  mu_theta(x_t,t) = (1/sqrt(alpha_t)) ( x_t - ( beta_t / sqrt(1-alpha_bar_t) ) eps_theta(x_t,t) ),

the exact same shape as `mu_tilde_t` but with the true `eps` replaced by the network's guess `eps_theta`. Now substitute both into the squared-mean loss. The `x_t/sqrt(alpha_t)` parts are identical and cancel; only the `eps` difference survives:

  || mu_tilde_t - mu_theta ||^2 = (1/alpha_t)(beta_t/sqrt(1-alpha_bar_t))^2 || eps - eps_theta ||^2 = ( beta_t^2 / (alpha_t (1-alpha_bar_t)) ) || eps - eps_theta(x_t,t) ||^2.

So with this parameterization the per-step loss is

  L_{t-1} - C = E_{x_0, eps}[ ( beta_t^2 / (2 sigma_t^2 alpha_t (1-alpha_bar_t)) ) || eps - eps_theta( sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps, t ) ||^2 ].

The regression target is now just `eps`, a standard normal vector — zero mean, unit variance, *the same scale at every timestep*. That is the well-posed problem I wanted: the network looks at a noisy image and predicts the noise in it, a target whose statistics do not drift with `t`, and the messy `t`-dependent geometry has been pushed entirely into a known scalar weight out front. This has to be a better-conditioned thing to fit than a mean whose scale wanders.

Wait — predict the noise. Let me make sure I am not just shuffling symbols. I could also have the network predict `x_0` directly: from `x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps`, predicting `x_0` and predicting `eps` are linearly interchangeable, `eps = (x_t - sqrt(alpha_bar_t) x_0)/sqrt(1-alpha_bar_t)`. So all three — predict `mu_tilde`, predict `x_0`, predict `eps` — are *the same model* up to an invertible, `t`-dependent linear reparameterization of the output. They are not different models; they are different coordinate systems for the same map. But the *loss* is not invariant under reparameterizing the output, because I am measuring squared error in whichever coordinate the network outputs, and the implicit per-`t` weighting differs. If I train in `x_0` coordinates, then `x0_theta - x_0 = -sqrt(1-alpha_bar_t)/sqrt(alpha_bar_t) · (eps_theta - eps)`, so `||x0_theta - x_0||^2 = ((1-alpha_bar_t)/alpha_bar_t) ||eps_theta - eps||^2`. Let me put numbers on that factor `(1-alpha_bar_t)/alpha_bar_t` to see how violently it swings. Near the clean end, `alpha_bar_t = 0.99` gives `0.01/0.99 ≈ 0.0101`. At the noisy end, `alpha_bar_t = 4e-5` (the `alpha_bar_T` I computed above) gives `(1-4e-5)/4e-5 ≈ 25000`. So an equal-quality denoiser, measured in `x_0`-MSE, is scored about `25000/0.0101 ≈ 2.5e6` times more heavily at the noisy end than at the clean end — a six-order-of-magnitude swing in implicit per-`t` weight, all of it piled onto the high-noise regime where `x_0` is least determined by `x_t` and the target therefore least learnable. That is exactly the wrong place to concentrate the weight. The `eps` target, by contrast, has constant `N(0,I)` scale at every `t`, so its implicit weight is flat — the cleanest conditioning of the three.

Now look harder at the reverse/sampling step this gives me, because something is poking at me. To sample `x_{t-1}` I draw from `p_theta(x_{t-1}|x_t) = N(mu_theta, sigma_t^2 I)`:

  x_{t-1} = (1/sqrt(alpha_t)) ( x_t - ( beta_t / sqrt(1-alpha_bar_t) ) eps_theta(x_t,t) ) + sigma_t z,  z ~ N(0,I).

This is `x_t`, minus a step in the direction of `eps_theta`, plus a fresh shot of Gaussian noise. A step against the predicted noise, plus noise. That is the shape of one **Langevin** step — move along an estimated vector field, then jitter. And what vector field? Recall the score of the forward marginal: `q(x_t|x_0) = N(sqrt(alpha_bar_t) x_0, (1-alpha_bar_t) I)`, so `∇_{x_t} log q(x_t|x_0) = -(x_t - sqrt(alpha_bar_t) x_0)/(1-alpha_bar_t)`. But `x_t - sqrt(alpha_bar_t) x_0 = sqrt(1-alpha_bar_t) eps`, so

  ∇_{x_t} log q(x_t|x_0) = - eps / sqrt(1-alpha_bar_t).

The score *is* the noise, up to the constant `-1/sqrt(1-alpha_bar_t)` — and that is an algebraic equality I just derived by substitution, not a hoped-for analogy. So a network trained to predict `eps` is, up to that scaling, a network trained to predict the score of the noised data density, `s_theta(x_t,t) ≈ -eps_theta(x_t,t)/sqrt(1-alpha_bar_t)`. My reverse step is then annealed Langevin dynamics where the noise level is indexed by `t`, and the "learned gradient of the data density" is exactly `eps_theta`. So variational inference on a fixed diffusion and denoising score matching sampled by annealed Langevin look like they may be the same construction in two coordinate systems — but a shared score formula is suggestive, not yet a proof that the two *objectives* coincide. Let me check that next rather than declare it, because if it holds it will also tell me how to weight the loss.

Let me push the connection all the way, because it is going to tell me how to weight the loss. Denoising score matching (this is Vincent's identity) says: corrupt `x` to `x̃` with `N(x̃; x, sigma^2 I)`, and matching `s_theta(x̃)` to `∇_{x̃} log q_sigma(x̃|x) = -(x̃-x)/sigma^2` trains the network on the score of the noised density. NCSN trains one network across a ladder of noise levels and, at level `sigma`, minimizes `(1/2) E || s_theta(x̃,sigma) + (x̃-x)/sigma^2 ||^2`, then combines levels with a weight `lambda(sigma)`. NCSN's empirical observation is that at optimum `|| s_theta || ∝ 1/sigma`, so the per-level loss magnitude scales like `1/sigma^2`; to make every level contribute equally they pick `lambda(sigma) = sigma^2`, and then the weighted per-level loss becomes

  lambda(sigma) · (1/2) E || s_theta + (x̃-x)/sigma^2 ||^2 = (1/2) E || sigma s_theta(x̃,sigma) + (x̃-x)/sigma ||^2.

The rescale step deserves its own check, since the whole equivalence hinges on it: is `lambda(sigma)·(1/2)||s + (x̃-x)/sigma^2||^2` really equal to `(1/2)||sigma s + (x̃-x)/sigma||^2` when `lambda = sigma^2`? Take `sigma = 0.5`, an arbitrary `s = -0.8`, and a corruption where `(x̃-x) = sigma·eps = 0.5·0.25 = 0.125`. Left: `0.25·(1/2)·(-0.8 + 0.125/0.25)^2 = 0.125·(-0.8 + 0.5)^2 = 0.125·0.09 = 0.01125`. Right: `(1/2)·(0.5·(-0.8) + 0.125/0.5)^2 = 0.5·(-0.4 + 0.25)^2 = 0.5·0.0225 = 0.01125`. Equal — so the `sigma^2` weight is exactly the factor that turns score-coordinates into noise-coordinates. And `(x̃-x)/sigma = 0.125/0.5 = 0.25 = eps`, the unit-variance noise that produced `x̃`. So NCSN's `sigma^2`-weighted score-matching objective is, term by term, *an unweighted mean-squared error between the added noise and a (rescaled) network output* — which is my `eps`-prediction loss with the per-`t` weight thrown away. So it is not just the score formula that matches: the two objectives coincide term for term. The thermodynamics derivation and the score-matching derivation hand me the same loss, and the score-matching side has already discovered, empirically, that the *unweighted* version — equal magnitude across noise levels — is the one that trains well. The one place the equivalence is loose is the chain's discrete endpoint, the `L_0` decoder term, which has no NCSN counterpart; I'll let the `t=1` MSE term stand in for it and not pretend the match is exact there.

So now the second free knob, the loss weighting, is staring at me. My principled per-step weight from the bound is `w_t = beta_t^2 / (2 sigma_t^2 alpha_t (1-alpha_bar_t))`; with `sigma_t^2 = beta_t` this is `w_t = beta_t / (2 alpha_t (1-alpha_bar_t))`. I do not have an intuition for how this behaves across `t` — `beta_t` grows with `t` but `1-alpha_bar_t` also grows, so the ratio could go either way. Let me just evaluate it on the actual schedule. At `t=2` (the first non-decoder term): `beta_2 ≈ 1.2e-4`, `alpha_2 ≈ 0.99988`, `alpha_bar_2 ≈ 0.99978`, so `1-alpha_bar_2 ≈ 2.2e-4` and `w_2 ≈ 1.2e-4 / (2·0.99988·2.2e-4) ≈ 0.273`. Mid-chain at `t=500`: `beta_500 ≈ 0.0101`, `1-alpha_bar_500 ≈ 0.918`, so `w_500 ≈ 0.0101/(2·0.990·0.918) ≈ 0.0055`. So the early term outweighs the mid term by a factor of about `0.273/0.0055 ≈ 50`. Interesting — and not quite as clean as "monotone decreasing": at the very end `t=999`, `beta` is large (`0.02`) and `1-alpha_bar_t ≈ 1`, giving `w_999 ≈ 0.02/(2·0.98·1.0) ≈ 0.0102`, which has ticked *back up* above the mid-chain value. So the weight is large at the clean end, dips through the middle, and rises modestly again at the noisy end — but the dominant feature is unmistakable: the small-`t` terms carry by far the most weight, ~50x the trough.

Early `t` means `x_t` is barely corrupted, almost the clean image with a whisper of noise. Denoising a whisper of noise is the *easiest* sub-problem — and the true bound weights it the heaviest. Meanwhile the genuinely hard sub-problems in the middle of the chain, where the image is half-dissolved and the network has to recover global structure, sit in the trough. That is backwards for sample quality: I am pouring gradient into perfecting imperceptible high-frequency details and starving the terms that decide whether the picture has any coherent content. The bound is the right objective for *codelength* — those tiny-`t` terms really do carry the bits — but codelength is not what I am optimizing for; I want sharp samples.

So drop the weight. Set every per-`t` weight to a constant 1:

  L_simple(theta) = E_{t, x_0, eps}[ || eps - eps_theta( sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps, t ) ||^2 ],  t ~ Uniform{1,...,T}.

This is exactly NCSN's equal-magnitude-per-level objective, arrived at from the variational side. Relative to the true bound it down-weights the small-`t` terms and up-weights the large-`t` ones — it tells the network to spend its capacity on the difficult, high-noise denoising tasks that actually shape the image, rather than on the easy near-clean ones. And it is dramatically simpler to implement: sample a timestep, sample noise, corrupt, predict the noise, take the MSE. No per-term weight to compute, no variance to track in a denominator. The `t=1` term plays the role of the discrete decoder `L_0` (the Gaussian density times the bin width, ignoring edge effects and `sigma_1^2`), so the single MSE covers the whole chain including the final decode. `L_T` never appears because the forward variances are fixed.

Let me also fix the loose ends so the chain is consistent end to end. The reverse variance `sigma_t^2`: with the simplified loss it no longer appears in training, so I just pick one of the two principled fixed values for sampling. The data I scale to `[-1,1]` so the network always sees inputs on the same scale as the `N(0,I)` prior. The schedule: `T=1000`, `beta` linear from `1e-4` to `0.02`, chosen small relative to `[-1,1]` so each reverse step is well-approximated by a Gaussian (the Sohl-Dickstein small-step condition) and so `alpha_bar_T` is small enough that `x_T` is essentially standard normal. And the sampler's coefficients are not hand-tuned the way a bolt-on Langevin sampler's would be — they are `beta_t`, `alpha_t`, `alpha_bar_t` straight from the forward process, so training the bound *is* training the sampler.

Now I want the two functions that actually live in the code, because the whole method reduces to a matched training-target / inversion pair. Training: the target the network regresses on is just the noise, `target = eps`. Inversion, used at sampling time to turn the network output back into a prediction of the clean image: invert `x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps_theta` for `x_0`,

  x0_hat = ( x_t - sqrt(1-alpha_bar_t) eps_theta ) / sqrt(alpha_bar_t).

Let me confirm these two really invert each other rather than just claim it, by running a round trip on a number. Take `t` with `alpha_bar_t = 0.5`, a clean value `x_0 = 0.37`, and a draw `eps = -1.31`. Corrupt: `x_t = sqrt(0.5)·0.37 + sqrt(0.5)·(-1.31) = 0.70711·(0.37 - 1.31) = 0.70711·(-0.94) = -0.66468`. Now feed the *true* `eps` (the best case the network could hit) back through the inversion: `x0_hat = (-0.66468 - sqrt(0.5)·(-1.31))/sqrt(0.5) = (-0.66468 + 0.92631)/0.70711 = 0.26163/0.70711 = 0.37000`. It returns `0.37`, the original `x_0`, exactly — so when `eps_theta = eps` the recovered clean image is `x_0` with zero error, and the inversion is a genuine inverse of the corruption, not an approximation. (Equivalently `x0_hat = sqrt(1/alpha_bar_t) x_t - sqrt(1/alpha_bar_t - 1) eps_theta`; with `alpha_bar_t = 0.5` that is `sqrt(2)·(-0.66468) - sqrt(1)·(-1.31) = -0.93999 + 1.31 = 0.37001`, the same number, confirming the two algebraic forms of the inversion agree too.) From `x0_hat` the reverse step plugs back through `mu_tilde_t(x_t, x0_hat)` to take one denoising move, which is exactly the `mu_theta` step above.

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

Before I commit, the one thing I want to confirm is that there are no free knobs left dangling — that everything the sampler needs was fixed by the same forward process the training target came from, so training the loss really does train the sampler. Walk the sampler line and account for every constant. `x_{t-1} = (1/sqrt(alpha_t)) ( x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) eps_theta ) + sigma_t z`: the `1/sqrt(alpha_t)`, the `(1-alpha_t)/sqrt(1-alpha_bar_t) = beta_t/sqrt(1-alpha_bar_t)`, and the `alpha_bar_t` inside are all functions of `beta_t` alone — the fixed schedule. `eps_theta` is the trained network. The only thing that is not pinned by the forward process is `sigma_t`, and I deliberately fixed that to one of the two entropy-bound constants `{beta_t, beta_tilde_t}` rather than learning it. So there is nothing left to tune: no Langevin step size, no noise scale, no number-of-steps that a bolt-on MCMC sampler would force me to hand-pick — they are `beta_t`, `alpha_t`, `alpha_bar_t` straight from the forward process. That is the property I most wanted to confirm and it holds. And the loop closes on itself: the training target is `eps`, the sampler recovers `x0_hat` by inverting the corruption (round-trip checked above to return `x_0` exactly when `eps_theta = eps`), then steps through `mu_tilde_t(x_t, x0_hat)` — the same `mu_tilde_t` whose collapse I verified numerically. Train the net to predict the added noise, sample by inverting that prediction; one matched pair, every constant accounted for.
