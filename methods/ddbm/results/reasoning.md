Let me start from what actually hurts. I have diffusion models that generate gorgeous images by reversing a process that turns data into Gaussian noise, and I have a pile of problems that look almost identical — sketch to photo, edges to handbags, day to night, deblurring, inpainting — where the thing I'm handed is *not* noise. It's a structured image from some source distribution, and I want the matching target image. Every time I try to bend a diffusion model to this I end up with a hack: condition the denoiser on the source and pray; or SDEdit, where I add some noise to the source and run an unconditional model, forever tuning how much structure to keep against how much to regenerate; or DDIB, two separate unconditional models with a shared latent. None of these is principled. They map in one direction only, they lose cycle consistency, and I genuinely cannot say what distribution they sample from. So the real task is cleaner than any of these patches: I have *paired* data `(x_0, x_T) ~ p_data(x, y)` — target and source together — and I want a generative process that transports from the source endpoint `x_T = y` back to a sample of `p(x_0 | x_T)`. Two arbitrary distributions, given as the two ends of a path, neither of them forced to be noise.

What do I have to build with? The score-based machinery is excellent and I do not want to throw it away. A forward SDE `dx_t = f(x_t,t) dt + g(t) dw_t` carries data to a prior; Anderson's theorem says its time reversal is again a diffusion, `dx_t = [f - g^2 ∇log p_t(x_t)] dt + g dw_bar`, and there's a deterministic probability-flow ODE `dx_t = [f - ½ g^2 ∇log p_t(x_t)] dt` with the *same* marginals. The only unknown is the score `∇log p_t`, and I get it for free from denoising score matching because the forward kernel is Gaussian, `x_t = α_t x_0 + σ_t ε`, so `∇log p(x_t|x_0)` is closed form and regressing onto it recovers the marginal score. The clock is the signal-to-noise ratio `SNR_t = α_t²/σ_t²`. This is all beautiful — and all of it assumes the prior is a *Gaussian*. The forward SDE has no idea how to drive itself to a *particular* point; it only knows how to forget. That's the wall: the entire formalism is welded to noise at one end.

Is there any way to force a diffusion to land on a specific point? There is, and it's old. Doob's h-transform: take the base diffusion and add a drift,
`dx_t = [f(x_t,t) + g²(t) h(x_t,t,y,T)] dt + g(t) dw_t`, with `h(x_t,t,y,T) = ∇_{x_t} log p(x_T = y | x_t)`, the gradient of the base diffusion's *backward* transition kernel. The textbook statement is that this conditioned process arrives at `x_T = y` almost surely, but I've been burned by quoting textbook limit theorems and getting the constant or the scaling wrong, so before I build anything on it let me actually watch it pin. Take the simplest base — VE with `f=0`, `σ_t² = c²t` so `g² = c²`, `α=1` — fix `x_0 = -0.3`, target `y = 2.5`, and `h = (y - x_t)/(c²(T-t))`. Forward Euler-Maruyama with `c=1`, `T=1`, over 20000 paths, refining the step count:

```
steps=  200   mean(x_T)=2.4998  std(x_T)=0.0708   (target y=2.5)
steps= 2000   mean(x_T)=2.5002  std(x_T)=0.0223
steps=20000   mean(x_T)=2.4999  std(x_T)=0.0070
```

The mean sits on `y` at every resolution, and the terminal spread collapses as the step shrinks — roughly halving when `dt` drops by 10×, i.e. `O(√dt)`, the discretization residual of a process that is genuinely a Dirac at `y` in continuous time. So the drift really does drag every path onto the endpoint; the explosion of `h ~ 1/(T-t)` near `t=T` is what does it. Good — now I trust the construction enough to lean on it. And when the base kernel is Gaussian, `p(x_T|x_t)` is Gaussian, so `h` is closed form. Pin the other end too — fix `x_0` as well — and I have a process tied down at both ends, the kind of thing studied for decades in probability under the name diffusion bridge. One thing to keep in the back of my mind about that extra drift `g²h`: it's a gradient of a log-likelihood of hitting `y`, scaled by `g²`, added to the dynamics — structurally the same move as classifier guidance, where you add `∇log p(class|x_t)` to steer a diffusion. So `h` behaves like a "guidance toward the endpoint," and if that analogy is real it might later give me a knob to turn.

But a bridge with both ends pinned isn't yet a generative model. I have a forward process that, for *fixed* `x_0` and `x_T`, wiggles between them. To *generate*, I need to start at the source `x_T` and walk back to a *sample* of the target, which means I need the reverse-time dynamics of the bridge when `x_0` is unknown and only `x_T` is given — and I need a *learnable* score for it, because the true score of that marginal isn't something I can write down. Let me set up the object precisely. I'll build a process `{x_t}` whose joint endpoints `q(x_0, x_T)` match `p_data(x, y)`, and generating means sampling `q(x_t | x_T)` backward. This `q` is not the diffusion's `p`: under a diffusion, `x_T` given `x_0` is Gaussian; here `x_T` is real data correlated with `x_0`.

First I need a tractable thing to train on. For diffusion the enabling object was a closed-form `p(x_t|x_0)`. The analogue here is the bridge marginal pinned at *both* endpoints, `q(x_t|x_0,x_T)`, and I can just *define* it to equal the diffusion pinned at both ends, which Bayes hands me:
`q(x_t|x_0,x_T) = p(x_T|x_t) p(x_t|x_0) / p(x_T|x_0)`.
Every factor on the right is Gaussian for a VE/VP base, so the product is Gaussian and I can read off its mean and variance by completing the square. Let me actually do it rather than wave at it. Write the three densities for scalar `x` (it's isotropic, so per-coordinate is enough):
`p(x_t|x_0) ∝ exp(-(x_t - α_t x_0)²/(2σ_t²))`,
`p(x_T|x_0) ∝ exp(-(x_T - α_T x_0)²/(2σ_T²))`,
and `p(x_T|x_t)`, the base-diffusion kernel from `t` to `T`, which for `s > t` is `N(α_{s|t} x_t, σ_{s|t}² )` with `α_{s|t} = α_s/α_t` and variance `σ_T² - (α_T²/α_t²)σ_t²`. Rewriting that last one in terms of `x_t` rather than `x_T`,
`p(x_T|x_t) ∝ exp( -((α_T/α_t) x_t - x_T)² / (2(σ_T² - (α_T²/α_t²)σ_t²)) ) = exp( -(x_t - (α_t/α_T) x_T)² / (2σ_t²(SNR_t/SNR_T - 1)) )`,
where I used `SNR_t = α_t²/σ_t²` to fold the algebra. Now collect the exponent of the product divided by `p(x_T|x_0)`:
`-(x_t - α_t x_0)²/(2σ_t²) - (x_t - (α_t/α_T)x_T)²/(2σ_t²(SNR_t/SNR_T - 1)) + (x_T - α_T x_0)²/(2σ_T²)`.
It's a quadratic in `x_t`; the last term has no `x_t`, so it only sets the normalization. The coefficient of `x_t²` is `(1/σ_t²)(1 + 1/(SNR_t/SNR_T - 1)) = (1/σ_t²)·(SNR_t/SNR_T)/(SNR_t/SNR_T - 1)`. Invert that and the variance falls out:
`σ_hat_t² = σ_t²·(SNR_t/SNR_T - 1)/(SNR_t/SNR_T) = σ_t²(1 - SNR_T/SNR_t)`.
The mean is `σ_hat_t²` times the linear coefficient. The linear-in-`x_t` part is `(α_t x_0)/σ_t² + ((α_t/α_T)x_T)/(σ_t²(SNR_t/SNR_T - 1))`, so
`μ_hat_t = σ_hat_t²·[ α_t x_0/σ_t² + (α_t/α_T)x_T/(σ_t²(SNR_t/SNR_T - 1)) ]`.
Multiply through using `σ_hat_t² = σ_t²(1 - SNR_T/SNR_t)`: the `x_0` term becomes `α_t x_0(1 - SNR_T/SNR_t)`, and the `x_T` term, after `(1 - SNR_T/SNR_t)/(SNR_t/SNR_T - 1) = SNR_T/SNR_t` cleans up, becomes `(SNR_T/SNR_t)(α_t/α_T)x_T`. So
`μ_hat_t = (SNR_T/SNR_t)(α_t/α_T) x_T + α_t x_0 (1 - SNR_T/SNR_t)`, `σ_hat_t² = σ_t²(1 - SNR_T/SNR_t)`.
Look at what this says. The mean is a *linear interpolation between the (scaled) endpoints*: weight `SNR_T/SNR_t` on `x_T` and `1 - SNR_T/SNR_t` on `x_0`. At `t = T`, `SNR_T/SNR_t = 1`, so the mean is `(α_T/α_T)x_T = x_T` and the variance is `0` — a Dirac at the source. At `t → 0`, `SNR_T/SNR_t → 0` (the SNR explodes as noise vanishes), so the mean → `α_0 x_0` and variance → `0` — a Dirac at the target.

That was a lot of square-completing in my head and I do not want a sign or an `SNR_t/SNR_T`-vs-`SNR_T/SNR_t` slip propagating into everything downstream. So let me check the closed form against the raw product numerically before I trust it. Pick a concrete base (`α_t = e^{-t/2}`, `σ_t² = 1 - e^{-t}`, `T=1`), endpoints `x_0 = 0.7`, `x_T = -0.4`, time `t = 0.35`. I form the *unnormalized* product `p(x_t|x_0)·p(x_T|x_t)` on a fine grid in `x_t` — that's literally the Bayes numerator before I did any algebra — and read off its mean and variance by quadrature, then compare to `μ_hat_t, σ_hat_t²` from the formula:

```
mean: numeric=0.30928758  derived=0.30928759   diff=1.7e-08
var : numeric=0.22328897  derived=0.22328901   diff=4.6e-08
```

They agree to quadrature precision, so the completed square is right. And checking the endpoints on the same base: at `t→0` the coefficients go `(a_t, b_t/α_t, v_t) → (≈1e-4, 0.99994, ≈1e-4)` and at `t=T` exactly `(1, 0, 0)` — Dirac on the target and on the source respectively, with the middle fat. So this Gaussian is genuinely a *bridge*, tractable to sample from given the endpoints. That clears the training-tractability bar.

Now the generative side: I need the time-reversal of `q(x_t|x_T)`, where I've marginalized out the unknown `x_0`. I can't just quote Anderson's reverse-SDE formula because that's for an *unconditioned* diffusion; my process has the extra h-drift baked in, and I've pinned `x_T`. Let me derive the dynamics from the bottom by tracking how the density evolves. Fix both endpoints first and use `q(x_t|x_0,x_T) = p(x_T|x_t)p(x_t|x_0)/p(x_T|x_0)`. The denominator is constant in `t`, so
`∂_t q(x_t|x_0,x_T) = [p(x_t|x_0)/p(x_T|x_0)] ∂_t p(x_T|x_t) + [p(x_T|x_t)/p(x_T|x_0)] ∂_t p(x_t|x_0)`.
Two pieces. The second factor, `p(x_t|x_0)`, is a marginal of the *forward* diffusion, so it obeys the Kolmogorov forward (Fokker-Planck) equation
`∂_t p(x_t|x_0) = -∇·[f p(x_t|x_0)] + ½ g² ∇·∇ p(x_t|x_0)`.
The first factor, `p(x_T|x_t)` with `x_T` *fixed* and `x_t` the variable, is a transition probability read backward, so it obeys the Kolmogorov *backward* equation
`-∂_t p(x_T|x_t) = f·∇ p(x_T|x_t) + ½ g² ∇·∇ p(x_T|x_t)`.
Substitute both. Call the two assembled terms ① (from `∂_t p(x_T|x_t)`) and ② (from `∂_t p(x_t|x_0)`):
① `= -(p(x_t|x_0)/p(x_T|x_0))·[ f·∇ p(x_T|x_t) + ½ g² ∇·∇ p(x_T|x_t) ]`,
② `= (p(x_T|x_t)/p(x_T|x_0))·[ -∇·(f p(x_t|x_0)) + ½ g² ∇·∇ p(x_t|x_0) ]`.
The two `f` terms want to recombine. `-(p(x_t|x_0)/Z) f·∇ p(x_T|x_t) - (p(x_T|x_t)/Z) ∇·(f p(x_t|x_0))`, with `Z = p(x_T|x_0)` — that's the product rule for `-∇·[f · (p(x_T|x_t)p(x_t|x_0)/Z)] = -∇·[f q(x_t|x_0,x_T)]`. Good, the advection term assembles cleanly. What's left is
`½ g²(③ - ④)`, where
③ `= (p(x_T|x_t)/Z) ∇·∇ p(x_t|x_0)` and
④ `= (p(x_t|x_0)/Z) ∇·∇ p(x_T|x_t)`.
This is *not* yet a single Laplacian of `q`, because `q` is a product and the Laplacian of a product has a cross term. So add and subtract exactly that cross term. The cross term I need is `(1/Z) ∇p(x_T|x_t)·∇p(x_t|x_0)`, and I can write it two ways using `∇p = p ∇log p`:
⑤ `= (1/Z) ∇p(x_t|x_0)·[ p(x_T|x_t) ∇log p(x_T|x_t) ]`,
⑥ `= (1/Z) ∇p(x_T|x_t)·[ p(x_t|x_0) ∇log p(x_t|x_0) ]`,
and ⑤ = ⑥. Now ③ + ⑥ is a product rule going the other way: `∇·( q ∇log p(x_t|x_0) )`, and ④ + ⑤ is `∇·( q ∇log p(x_T|x_t) )`. And by Bayes, `∇log q(x_t|x_0,x_T) = ∇log p(x_T|x_t) + ∇log p(x_t|x_0)` (the `p(x_T|x_0)` term has no `x_t`), so
`③+④+⑤+⑥ = ∇·( q ∇log q(x_t|x_0,x_T) ) = ∇·∇ q`.
But I only have `½ g²(③ - ④)`. Since ⑤ = ⑥, I can write
`½(③ - ④) = ½(③ + ④ + ⑤ + ⑥) - (④ + ⑤)`.
The first term is `½∇·∇q`; the second is `∇·(q ∇log p(x_T|x_t))`. Plugging back, the whole thing collapses to
`∂_t q(x_t|x_0,x_T) = -∇·[ (f + g² ∇log p(x_T|x_t)) q ] + ½ g² ∇·∇ q`.
That's a *Fokker-Planck equation with modified drift* `f + g² ∇log p(x_T|x_t) = f + g² h` — the h-transform drift, exactly. Reassuring: the pinned bridge is itself a diffusion whose drift is the base drift plus Doob's `g²h`. Now marginalize out `x_0` against `p_data(x_0|x_T)`. The drift `f + g²h` doesn't depend on `x_0` and the equation is linear in the density, so the expectation passes through and `E_{x_0}[q(x_t|x_0,x_T)] = q(x_t|x_T)` obeys the *same* Fokker-Planck:
`∂_t q(x_t|x_T) = -∇·[ (f + g²h) q(x_t|x_T) ] + ½ g² ∇·∇ q(x_t|x_T)`.

This is a *forward*-time Fokker-Planck with drift `f + g²h`. To sample I want the *reverse*-time process that produces the same marginals, so I redo Anderson's conversion but for this drift. A forward SDE with drift `b` and diffusion `g` has reverse-time SDE with drift `b - g²∇log q` (and the same `g`). Here `b = f + g²h`, so the reverse-time SDE drift is
`(f + g²h) - g² ∇log q(x_t|x_T) = f - g²( ∇log q(x_t|x_T) - h )`.
Let me name the learned score `s(x_t,t,y,T) = ∇log q(x_t|x_T)` and keep `h = ∇log p(x_T|x_t)`. So
reverse SDE: `dx_t = [ f - g²( s - h ) ] dt + g dw_bar`.
For the deterministic version I use Song's continuity conversion: any Fokker-Planck `∂_t q = -∇·(b q) + ½g²∇·∇q` can be rewritten as a pure continuity equation `∂_t q = -∇·(b̃ q)` with `b̃ = b - ½ g² ∇log q`, which has no diffusion term and hence a deterministic ODE with the *same* marginals. With `b = f + g²h`,
`b̃ = f + g²h - ½ g² s = f - g²( ½ s - h )`,
PF-ODE: `dx_t = [ f - g²( ½ s - h ) ] dt`.
I want to be careful here, because it's the kind of place I'd misremember. Only the *score* `s` gets the ½. The h-transform drift `h` does *not* get halved — it is part of the bridge's defining forward drift, not the thing that the SDE→ODE conversion splits. The ½ comes entirely from the `½ g² ∇log q` I peeled off to kill the diffusion term, and that acts only on `q`'s score `s`. If I'd reflexively halved the whole `(s - h)` bracket I'd have changed the very drift that pins the endpoint, and the ODE would no longer be a valid bridge. So: SDE has `(s - h)`, ODE has `(½ s - h)`. Hold onto that, it's load-bearing for the sampler.

Now the score `s = ∇log q(x_t|x_T)` is unknown, so I have to learn it — and I want the same cheap denoising identity diffusion uses. The target I *can* compute is the conditional score `∇log q(x_t|x_0,x_T)` of the pinned bridge, which is just `-(x_t - μ_hat_t)/σ_hat_t²` from the Gaussian I derived. Claim: regressing a network `s_θ(x_t,x_T,t)` onto that conditional score recovers the marginal score `∇log q(x_t|x_T)`. Let me confirm it's the standard denoising-score-matching identity and not something that breaks under the extra `x_T` conditioning. The loss is `E[ w(t) ‖s_θ - ∇log q(x_t|x_0,x_T)‖² ]` with `(x_0,x_T)~p_data`, `x_t ~ q(x_t|x_0,x_T)`, `t` from any nonzero `p(t)`. Since it's a per-`(x_t,x_T,t)` weighted `L₂` and the weights are nonzero, the minimizer is the conditional expectation of the target given `(x_t,x_T,t)`:
`s* = ∫ [q(x_t|x_0,x_T) p_data(x_0,x_T) / q(x_t,x_T)] ∇log q(x_t|x_0,x_T) dx_0`.
Use `∇log q(x_t|x_0,x_T) = ∇q(x_t|x_0,x_T)/q(x_t|x_0,x_T)` so the `q(x_t|x_0,x_T)` cancels:
`s* = ∫ [p_data(x_0,x_T)/q(x_t,x_T)] ∇_{x_t} q(x_t|x_0,x_T) dx_0 = ∇_{x_t}[ ∫ p_data(x_0,x_T) q(x_t|x_0,x_T) dx_0 ] / q(x_t,x_T) = ∇_{x_t} q(x_t,x_T) / q(x_t,x_T) = ∇log q(x_t|x_T)`,
because the numerator integral is exactly `q(x_t,x_T)` and `∇log q(x_t,x_T) = ∇log q(x_t|x_T)` (the `x_T`-only factor has zero `x_t`-gradient). So the network does recover the marginal bridge score; the conditioning on `x_T` rides along harmlessly. Training is: sample a pair `(x_0,x_T)`, sample `x_t` from the closed-form Gaussian bridge, regress onto its closed-form conditional score. Cheap, one network call, no path simulation. That's both bars — tractable marginal, closed-form objective — cleared.

Before I parameterize, let me check whether this generalizes the things it should generalize, because if it doesn't reduce to plain diffusion and to flow matching I've built a competitor rather than a unification, and I'd want to know that now. Take the data joint to be `p_data(x_0,x_T) = p(x_T|x_0)p_data(x_0)` with `x_T|x_0 ~ N(α_T x_0, σ_T² I)` — i.e. force the "source" to be a noised version of the target, the diffusion setup. If the bridge contains diffusion, then marginally `x_t` should be the ordinary diffusion `N(α_t x_0, σ_t² I)`. Check the algebra first: write `r = SNR_T/SNR_t`. From the bridge sample `x_t = r(α_t/α_T)x_T + α_t x_0(1-r) + σ_t√(1-r) ε₁` and substitute `x_T = α_T x_0 + σ_T ε₂`. The `x_0` coefficient becomes `rα_t + α_t(1-r) = α_t`. The noise becomes `r(α_t/α_T)σ_T ε₂ + σ_t√(1-r) ε₁`, a sum of independent Gaussians. Its first variance is
`r²(α_t²/α_T²)σ_T² = r σ_t²`,
because `r = (α_T²/σ_T²)/(α_t²/σ_t²)`. The second variance is `σ_t²(1-r)`, so the total is `σ_t²`. Thus `x_t = α_t x_0 + σ_t ε`, the diffusion marginal.

This one I can also just sample, which catches any substitution error the symbolic step might hide. Same base, `x_0 = 0.3`, `t = 0.4`: draw `x_T = α_T x_0 + σ_T ε₂` and then `x_t = a_t x_T + b_t x_0 + √v_t ε₁` over 4M samples, and compare moments to the diffusion target:

```
empirical mean(x_t)=0.24532   alpha_t*x0 =0.24562
empirical var (x_t)=0.32934   sigma_t^2  =0.32968
```

and the pivotal noise identity `r²(α_t²/α_T²)σ_T² = 0.094364 = r σ_t²` matches, with the two variances summing to `σ_t² = 0.32968` exactly. So the diffusion marginal really is the `x_T`-Gaussian special case. And the FPE reduces too: marginalize the bridge FPE over `x_T ~ p(x_T)`. The drift's h-term is `g² E_{x_T}[ p(x_t|x_T) ∇log p(x_T|x_t) ]`, and writing the expectation as an integral, `∫ p(x_T) p(x_t|x_T) ∇log p(x_T|x_t) dx_T = p(x_t) ∫ p(x_T|x_t) ∇log p(x_T|x_t) dx_T = p(x_t) ∫ ∇p(x_T|x_t) dx_T = p(x_t) ∇∫ p(x_T|x_t) dx_T = p(x_t) ∇1 = 0`. The h-drift integrates to zero, and what's left is precisely the unconditional diffusion's Fokker-Planck. So plain diffusion is the special case where `x_T` is Gaussian — the h-guidance vanishes on average because there's no information in a noise endpoint to be pulled toward. That's a satisfying sanity check.

And flow matching? Take a VE base (`f = 0`, `σ_t² = c²t`), scale the bridge variance by `c`, and send `c → 0`. The interpolation mean is `m_t = (t/T)x_T + (1 - t/T)x_0`, so `x_t = m_t + c√(t(1-t/T))ε`. The conditional bridge score is `-(x_t-m_t)/(c²t(1-t/T)) = O(1/c)`, so the `-c²·½s` part of the PF-ODE vanishes as `O(c)`. The h-term is
`c² h = c²(x_T-x_t)/(c²(T-t)) = (x_T - m_t)/(T-t) + O(c)`.
But `x_T - m_t = (1-t/T)(x_T-x_0)`, so `c²h = (x_T-x_0)/T + O(c)`. At `T=1`, the limiting drift is `x_T - x_0`. Let me put numbers to the limit to be sure the `c` really does drop out of the h-drift and I'm not fooling myself with an `O(c)` that's secretly `O(1)`. Evaluate `g²h` at the bridge mean for `x_0=0.2`, `x_T=1.3`, `t=0.45`, sweeping `c`:

```
c=1.00   g^2 h = 1.100000   (x_T-x0)/T = 1.100000
c=0.30   g^2 h = 1.100000   (x_T-x0)/T = 1.100000
c=0.10   g^2 h = 1.100000
c=0.01   g^2 h = 1.100000
```

Flat in `c` and equal to `x_T - x_0 = 1.1` — the rectified-flow / OT velocity field. So the deterministic part of my ODE limits onto exactly the flow-matching drift. The one subtlety the numbers underline: at `c → 0` the bridge *score* blows up like `1/c`, so I can't keep regressing the score there — flow matching sidesteps this by regressing the *drift* `x_T - x_0` directly. So flow matching is the noiseless limit of the VE bridge. Two reductions, both confirmed numerically: this is a generalization, not a competitor.

Now I need to *parameterize* `s_θ`, and the diffusion world has a hard-won lesson here I should reuse: don't predict the score or the noise directly. EDM's argument is that if I predict noise and reconstruct `x_0 = x_t - σ F`, the network's errors get amplified by `σ` at high noise; predicting the clean signal with a `σ`-dependent skip is far more stable, and the variance of the `x_0` target does not drift with `t`. So I use a pred-`x_0` form `D_θ(x_t,t) = c_skip x_t + c_out F_θ(c_in x_t; c_noise)` and convert it back to a score. Let
`a_t = (α_t/α_T)(SNR_T/SNR_t)`, `b_t = α_t(1 - SNR_T/SNR_t)`, and `v_t = σ_t²(1 - SNR_T/SNR_t)`, so the bridge sample is `x_t = a_t x_T + b_t x_0 + √v_t ε`. Then replacing `x_0` by `D_θ` in the conditional score gives
`s ≈ -(x_t - (a_t x_T + b_t D_θ))/v_t`.
The canonical schedule code stores the standard-deviation coefficient `c_t = √v_t` — in its `(α,ρ)` parameterization, `c_t = α_t ρ_bar_t ρ_t / ρ_T` — so the implementation divides by `c_t²`. That square matters.

Now I have to *choose* the four scalings, and I want them forced by principle, not taste. The pred-`x_0` loss is `E[ w̃(t) ‖ c_skip x_t + c_out F_θ - x_0 ‖² ]` with `x_t = a_t x_T + b_t x_0 + √v_t ε`. The network input `c_in x_t` should have unit variance. Because the endpoints are correlated data, not data plus independent noise, I have to track `σ_0²`, `σ_T²`, and `σ_{0T}`:
`Var[x_t] = a_t²σ_T² + b_t²σ_0² + 2 a_t b_t σ_{0T} + v_t`.
So
`c_in(t) = 1/√(a_t²σ_T² + b_t²σ_0² + 2 a_t b_t σ_{0T} + v_t)`.
These two endpoint statistics `σ_T, σ_{0T}` are the only new ones versus EDM; they are what make this a paired-data problem rather than noise-to-data.

Now `c_out` and `c_skip`. Pull `c_out` out of the loss after substituting `x_t`:
`‖c_skip x_t + c_out F_θ - x_0‖² = c_out² ‖F_θ - (1/c_out)( (1 - c_skip b_t) x_0 - c_skip(a_t x_T + √v_t ε) )‖²`.
The effective target of `F_θ` should also have unit variance, so
`c_out² = (1 - c_skip b_t)²σ_0² + c_skip²(a_t²σ_T² + v_t) - 2(1 - c_skip b_t)c_skip a_t σ_{0T}`.
Two unknowns, one equation — EDM's tiebreaker is to *minimize* `c_out` over `c_skip`, because `c_out` multiplies the network's output and hence amplifies its error. Differentiate `c_out²` w.r.t. `c_skip` and set to zero:
`-2(1 - c_skip b_t)b_t σ_0² + 2 c_skip(a_t²σ_T² + v_t) - 2(1 - 2 c_skip b_t)a_t σ_{0T} = 0`.
Solving gives
`c_skip = (b_t σ_0² + a_t σ_{0T}) / (a_t²σ_T² + b_t²σ_0² + 2 a_t b_t σ_{0T} + v_t) = (b_t σ_0² + a_t σ_{0T})·c_in²`.
Back-substitute to simplify `c_out²`. Expanding and using the `c_skip` relation,
`c_out² = σ_0² - (b_t σ_0² + a_t σ_{0T})c_skip = σ_0² - (b_t σ_0² + a_t σ_{0T})²·c_in²`.
Put over `1/c_in²`: the numerator is `σ_0²(a_t²σ_T² + b_t²σ_0² + 2 a_t b_t σ_{0T} + v_t) - (b_t σ_0² + a_t σ_{0T})²`. The `b_t²σ_0⁴` cancels, the `2 a_t b_t σ_{0T}σ_0²` cancels, and I get `a_t²(σ_0²σ_T² - σ_{0T}²) + σ_0²v_t`. So
`c_out(t) = √(a_t²(σ_0²σ_T² - σ_{0T}²) + σ_0²v_t)·c_in(t)`,
and the loss weight to make the effective per-sample weight approximately one is `w̃(t) = 1/c_out(t)²`. For `c_noise` I reuse EDM's log-noise conditioning, `c_noise = ¼ log t`, since the time distribution is not materially different. The acid test of this whole derivation is whether it collapses to EDM's published scalings in the diffusion special case — if it doesn't, I've got an error in the covariance bookkeeping. There `x_T = x_0 + Tε`, so `σ_T² = σ_0² + T²`, `σ_{0T} = σ_0²`, and in the VE corner `a_t = t²/T²`, `b_t = 1 - t²/T²`, `v_t = t²(1 - t²/T²)`. Substituting into `c_in`, the cross terms cancel and the radicand reduces to `σ_0² + t²`, so `c_in = 1/√(σ_0² + t²)`. `c_skip = ((1 - t²/T²)σ_0² + (t²/T²)σ_0²)/(σ_0²+t²) = σ_0²/(σ_0²+t²)`. And `c_out`: the radicand `(t⁴/T⁴)(σ_0²(σ_0²+T²) - σ_0⁴) + σ_0²t²(1-t²/T²) = (t⁴/T²)σ_0² + σ_0²t² - σ_0²t⁴/T² = σ_0²t²`, so `c_out = σ_0 t/√(σ_0²+t²)`. To make sure I didn't quietly assume `T=1` or drop a cross term, I evaluate the *general* `c_in/c_skip/c_out` formulas (the ones with `a_t, b_t, v_t, σ_{0T}` in them) numerically against these EDM closed forms, at `T=1.3`, `σ_0=0.6`, over several `t`:

```
t=0.10  c_in diff=0.0e+00  c_skip diff=3.3e-16  c_out diff=1.4e-17
t=0.50  c_in diff=0.0e+00  c_skip diff=1.1e-16  c_out diff=0.0e+00
t=0.90  c_in diff=2.2e-16  c_skip diff=5.6e-17  c_out diff=1.1e-16
t=1.20  c_in diff=0.0e+00  c_skip diff=0.0e+00  c_out diff=1.1e-16
```

All three agree to floating-point rounding at every `t`. So the generalized preconditioning reduces to EDM exactly, with the only genuinely new ingredients being the two endpoint statistics `σ_T, σ_{0T}` — which makes sense, since those are exactly what distinguishes paired data from data-plus-independent-noise.

One more knob falls out of the h-as-guidance observation: since the ODE drift can be widened to `f - g²(½s - ωh)`, the endpoint pull can be scaled like classifier guidance. That is a useful sampler knob when I want to dial how hard the path is pulled toward the endpoint. For the concrete implementation I am grounding here, I keep the h-term unscaled, `ω=1`, so the code remains the direct SDE/ODE drift I derived.

Now sampling is where the obvious thing is wrong. I have a PF-ODE that I can integrate fast with high-order solvers, the way EDM does for generation. So why not just integrate the ODE backward from `x_T`? Because the bridge has a *fixed, given* starting point `x_T = y`, which is real data, not a fresh noise draw. Integrating a deterministic ODE backward from one fixed point produces *one* deterministic "expected" trajectory — the conditional mean path. For a genuinely one-to-many conditional mapping, many plausible targets can share one source, so the conditional mean is a blurry average and a pure ODE will hand me exactly that average. That's the wall: determinism plus a pinned start equals blur. I *need* stochasticity to recover diversity and sharpness.

So I should put noise back. But pure Euler-Maruyama on the reverse SDE is slow and inaccurate per step, and I'm under a tight budget on denoiser calls. I want the SDE's diversity *and* the ODE's per-step accuracy. The predictor-corrector idea from Song fits perfectly: alternate a numerical step that moves through time with a noise-injecting step that re-randomizes without advancing the marginal much. And EDM already showed the clean engineering of this for generation: each step, briefly *churn* — add a controlled bit of noise to bump up the noise level — then take an accurate deterministic step back down. The churn corrects accumulated error and, here, supplies the stochasticity the bridge needs; the deterministic step does the heavy lifting cheaply with a high-order integrator.

Concretely, discretize time with EDM's power-law schedule `t_i = (t_max^{1/ρ} + i/(N-1)(t_min^{1/ρ} - t_max^{1/ρ}))^ρ`, `ρ = 7` — that spacing equalizes truncation error well and is image-friendly (small `ρ` like 3 equalizes truncation error in theory but 5-10 sample better; 7 is the sweet spot). The actual array is decreasing and ends with a trailing zero. At a current time `t_cur = ts[i]`, choose `t_hat = t_cur + r(ts[i+1] - t_cur)` with step ratio `r`. I spend `[t_cur, t_hat]` on a stochastic Euler-Maruyama move, following the reverse SDE with drift factor `(s - h)` and injected `g·√|Δt|·noise`, then spend `[t_hat, ts[i+1]]` on a deterministic Heun move, following the ODE with factor `(½s - h)`. The churn keeps the per-step marginal close to the training marginal while restoring diversity; the Heun step integrates the deterministic part accurately. When `r = 0` I do no churn; for translation I want `r ≈ 1/3` so each step gets a real stochastic kick.

Let me write the per-step drift once, because both the SDE and ODE steps are the same object with one switch. Define, at time `t`, the denoiser output `D = D_θ(x_t,t)` (one network call), the bridge score from pred-`x_0` `s = -(x_t - (a_t x_T + b_t D))/c_t²` in the code's standard-deviation notation, the h-transform `h = -(x_t - (α_t/α_T)x_T)/(α_t² ρ_bar_t²)` with `ρ_bar_t² = ρ_T² - ρ_t²`, and the base drift/diffusion `f, g²`. Then the reverse drift is
`d = f x_t - g²·(κ·s - h)`, with `κ = 1` for the stochastic (SDE) step and `κ = ½` for the deterministic (ODE) step.
That single sign convention — the score `s` carrying the `κ` switch while `h` stays at full strength — is the SDE/ODE distinction I was careful about earlier, now living in one line of code. The Euler update is `x ← x + d·Δt + [stochastic] noise·√|Δt|·√(g²)`; the Heun (2nd-order) update recomputes `d₂` at the predicted endpoint, averages `d' = (d + d₂)/2`, and re-steps. On the *last* interval, where `t_{i+1} = 0`, I take a single Euler step rather than Heun (no valid second evaluation at `t=0`), which also saves a call. Counting calls: each Heun iteration costs the churn-Euler (1 call) plus the Heun predictor+corrector (2 calls) = 3, except the terminal Euler-only iteration. That's how I'd spend a fixed NFE budget — at a high-NFE reference setting, e.g. `ρ=7`, 17 iterations of which 16 are Heun (3 calls) and the last is churn+Euler (2 calls), totals 50 calls — and at a tight budget I'd shrink `N` and the churn accordingly.

Let me write it as code, filling the intermediate coefficients, the preconditioned `x_0` prediction, the per-step reverse update, and the call-scheduling loop:

```python
import torch
from tqdm.auto import tqdm
import torch.distributed as dist

from .nn import append_dims

def get_d(denoiser, noise_schedule, x, x_T, t, stochastic):
    ones = x.new_ones([x.shape[0]])
    f_t, g2_t = [append_dims(item, x.ndim) for item in noise_schedule.get_f_g2(t * ones)]
    alpha_t, alpha_bar_t, _, rho_bar_t = [
        append_dims(item, x.ndim) for item in noise_schedule.get_alpha_rho(t * ones)
    ]
    a_t, b_t, c_t = [append_dims(item, x.ndim) for item in noise_schedule.get_abc(t * ones)]
    denoised = denoiser(x, t * ones)
    grad_logq = -(x - (a_t * x_T + b_t * denoised)) / c_t**2
    grad_logpxTlxt = -(x - alpha_bar_t * x_T) / (alpha_t**2 * rho_bar_t**2)
    d = f_t * x - g2_t * ((0.5 if not stochastic else 1) * grad_logq - grad_logpxTlxt)
    return d, g2_t, denoised


def ddbm_simulate(denoiser, noise_schedule, x, x_T, t_cur, t_next, stochastic, second_order=False):
    dt = t_next - t_cur
    if isinstance(noise_schedule, I2SBNoiseSchedule):
        dt = dt * (noise_schedule.n_timestep - 1)
    d, g2_t, pred_x0 = get_d(denoiser, noise_schedule, x, x_T, t_cur, stochastic)
    x_new = x + d * dt + (0 if not stochastic else 1) * torch.randn_like(x) * (dt.abs() ** 0.5) * g2_t.sqrt()
    if second_order:
        d_2, _, pred_x0 = get_d(denoiser, noise_schedule, x_new, x_T, t_next, stochastic)
        d_prime = (d + d_2) / 2
        x_new = x + d_prime * dt + (0 if not stochastic else 1) * torch.randn_like(x) * (dt.abs() ** 0.5) * g2_t.sqrt()
    return x_new, pred_x0


@torch.no_grad()
def sample_heun(denoiser, diffusion, x, ts, churn_step_ratio=0.0, **kwargs):
    x_T = x
    path, pred_x0, nfe = [], [], 0
    indices = tqdm(range(len(ts) - 1), disable=(dist.get_rank() != 0))
    assert churn_step_ratio < 1

    for _, i in enumerate(indices):
        if churn_step_ratio > 0:
            t_hat = (ts[i + 1] - ts[i]) * churn_step_ratio + ts[i]
            x, _p = ddbm_simulate(denoiser, diffusion.noise_schedule, x, x_T, ts[i], t_hat, stochastic=True)
            nfe += 1
            path.append(x.detach().cpu()); pred_x0.append(_p.detach().cpu())
        else:
            t_hat = ts[i]

        if ts[i + 1] == 0:
            x, _p = ddbm_simulate(denoiser, diffusion.noise_schedule, x, x_T, t_hat, ts[i + 1], stochastic=False)
            nfe += 1
        else:
            x, _p = ddbm_simulate(denoiser, diffusion.noise_schedule, x, x_T, t_hat, ts[i + 1],
                                  stochastic=False, second_order=True)
            nfe += 2
        path.append(x.detach().cpu()); pred_x0.append(_p.detach().cpu())
    return x, path, nfe, pred_x0, ts, None
```

So the chain closes. I started stuck because diffusion's whole formalism is welded to a Gaussian prior and can't natively start from a structured source image, leaving only unprincipled patches for translation. Doob's h-transform can pin a diffusion to a fixed endpoint, and pinning both ends gives a bridge — but a bridge alone isn't generative. I defined the training marginal as the doubly-pinned Gaussian via Bayes, completed the square to get a closed-form bridge whose mean linearly interpolates the endpoints and is Dirac at both ends, then derived the bridge's own Fokker-Planck by feeding the Kolmogorov forward and backward equations through the product rule, finding the drift is the base drift plus `g²h`; reversing it gave the reverse SDE `f - g²(s - h)` and, via the continuity identity that halves only the `q`-score, the PF-ODE `f - g²(½ s - h)`. I learn `s` by denoising bridge score matching against the closed-form conditional score, which the `L₂` minimizer proves recovers the true marginal score. Plain diffusion falls out when the source is a noised target (the h-drift integrates to zero), and rectified flow / OT falls out in the noiseless VE limit (match the straight-line drift), so the framework genuinely contains both. The pred-`x_0` EDM parameterization, with scalings forced by unit-variance of input and target and `c_skip` minimizing `c_out`, generalizes EDM's preconditioning by exactly the two endpoint statistics `σ_T, σ_{0T}` and reduces to it on the diffusion corner. Finally the sampler: a pure ODE from a pinned data start gives blurry conditional means, so I inject noise predictor-corrector style — a scheduled stochastic Euler "churn" step (SDE, `κ=1`) followed by an accurate deterministic Heun step (ODE, `κ=½`), on the EDM `ρ=7` schedule, with a step ratio `r` trading exploration against endpoint sharpness — which restores diversity while keeping the per-step marginal on the bridge, and the whole update is one drift line with `κ` switching the score's coefficient and `h` always at full strength.
