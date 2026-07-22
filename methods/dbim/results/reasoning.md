I have a trained diffusion bridge: a process pinned to start at clean data `x_0` and arrive almost surely at an informative endpoint `x_T` — a sketch, a degraded image, a masked photo — and I want to generate `x_0` from `x_T`. The model was built by Doob's h-transform of an ordinary diffusion, which gives me an analytic forward kernel `q(x_t | x_0, x_T) = N(a_t x_T + b_t x_0, c_t² I)` with the coefficients pinned by the underlying noise schedule, `a_t = (α_t/α_T)(SNR_T/SNR_t)`, `b_t = α_t(1 − SNR_T/SNR_t)`, `c_t² = σ_t²(1 − SNR_T/SNR_t)`, and `SNR_t = α_t²/σ_t²`. So at every time the bridge state is just a weighted mix of the two endpoints plus Gaussian noise of size `c_t`. The network I have learned to predict the bridge score `∇ log q(x_t | x_T)`, or equivalently — and this is the form I'll keep reaching for — a *data predictor* `x_θ(x_t, t, x_T)` that, given a noisy bridge state, estimates the clean `x_0`. Generating means running the reverse process from `t = T` down to `0`.

And the way that's done today is to write the reverse process as a stochastic differential equation, or its probability-flow ODE, and hand it to a generic numerical solver — a Heun method with some churn, EDM-style. It works, but it's slow: for high-resolution translation I'm looking at many network evaluations to get clean samples, and every evaluation is a full U-Net forward pass, so the wall-clock is basically proportional to NFE. Meanwhile *ordinary* diffusion models have dedicated samplers that take much larger steps. That gap is the whole problem, and I want to close it without retraining anything — I want to keep this exact network and just be cleverer about how I walk it from `x_T` to `x_0`.

So why is the generic SDE solver slow? It treats the reverse dynamics as a black box `dx = (stuff)(x, t) dt` and takes small Euler/Heun steps because that's all a black-box solver can safely do. It knows nothing about the *structure* of a bridge. The structure is sitting right there in the forward kernel: at any `t`, the bridge marginal given the endpoints is a known Gaussian with a known mean and a known noise scale `c_t`. A good sampler should exploit that, not rediscover it step by step.

Let me think about where the leverage is. The single most useful thing I know about how this network was trained: the training loss is a score-matching / denoising loss, and that loss depends on the model *only through the per-time marginals* `q(x_t | x_0, x_T)`, not through the full joint distribution over the whole trajectory `q(x_{t_0:T} | x_T)`. The network never saw a trajectory; it saw `(x_0, x_T, t)` triples, sampled `x_t` from the marginal kernel, and learned to invert that one marginal. That means: any inference process I write down that *agrees with these same marginals* is a process the network is already optimal for. The training never committed me to the particular joint that the h-transform SDE happens to define. I have freedom in the joint as long as I respect the marginals.

This is the same lever that made fast sampling work for ordinary diffusion. There, the realization was that the DDPM loss only sees `q(x_t | x_0)`, so one is free to replace the Markovian forward chain with a whole *family* of non-Markovian inference processes `q_σ(x_{t-1} | x_t, x_0)`, indexed by a stochasticity vector `σ`, each engineered to keep the same marginal `q(x_t | x_0) = N(√α_t x_0, (1−α_t) I)`. The reverse step came out as `x_{t-1} = √α_{t-1} x̂_0 + √(1 − α_{t-1} − σ_t²) ε̂ + σ_t ε` — predicted clean data, plus a "direction pointing back toward `x_t`", plus fresh noise — and at `σ = 0` it collapses to a deterministic implicit map that can take big jumps and be inverted. The pretrained network is reused unchanged because the marginals are preserved.

The obvious play is to do the same thing here. But I have to be careful, because the diffusion construction is tied to the data-to-noise kernel `N(√α_t x_0, (1−α_t) I)` — mean is `√α_t x_0`, variance is `1 − α_t`, there is only one endpoint, namely Gaussian noise. My bridge has *two* endpoints: the mean is `a_t x_T + b_t x_0`, with that extra `a_t x_T` term, and the noise scale is `c_t`, not `1 − α_t`. If I naively copy the DDIM formula it has nowhere to put `x_T`, and it has nothing that matches the `c_t → 0` behavior of a bridge. So I can't import the formula; I have to redo the construction for the bridge kernel and *insist* it preserve the bridge marginals.

Let me set up the discretized timesteps I'll actually sample on, `0 = t_0 < t_1 < … < t_{N-1} < t_N = T`. I want a family of joint inference distributions over `x_{t_0:N-1}` given `x_T`, controlled by a per-step variance `ρ ∈ R^{N-1}`, factorized backward in time the way I'll sample — each `x_{t_n}` conditioned on the next-later state `x_{t_{n+1}}` and on `x_0`:
```
q^ρ(x_{t_{0:N-1}} | x_T) = q_0(x_{t_0}) ∏_{n=1}^{N-1} q^ρ(x_{t_n} | x_0, x_{t_{n+1}}, x_T).
```
Now I have to choose each conditional. The local constraint is the endpoint-conditioned bridge marginal: for each fixed `x_0, x_T`, I need `q^ρ(x_{t_n} | x_0, x_T) = q(x_{t_n} | x_0, x_T) = N(a_{t_n} x_T + b_{t_n} x_0, c_{t_n}² I)`; after integrating over `x_0`, this also gives the same `q(x_{t_n} | x_T)` that the bridge score sees. The diffusion case suggests the trick: make the conditional a Gaussian whose mean has a part that *deterministically* points back at the later state through its own deviation from its mean, and whose injected variance is exactly the bookkeeping that keeps the total variance right. So let me try
```
q^ρ(x_{t_n} | x_0, x_{t_{n+1}}, x_T) = N( a_{t_n} x_T + b_{t_n} x_0 + √(c_{t_n}² − ρ_n²) · (x_{t_{n+1}} − a_{t_{n+1}} x_T − b_{t_{n+1}} x_0)/c_{t_{n+1}} ,  ρ_n² I ).
```
Read the mean: `a_{t_n} x_T + b_{t_n} x_0` is exactly the bridge mean at `t_n`, and then I add `√(c_{t_n}² − ρ_n²)` times the *normalized noise component of the later state` — because `(x_{t_{n+1}} − a_{t_{n+1}} x_T − b_{t_{n+1}} x_0)/c_{t_{n+1}}` is precisely the standard-Gaussian `ε` that generated `x_{t_{n+1}}`. So I'm reusing a fraction of the later step's noise as a deterministic "direction," and adding fresh noise of scale `ρ_n`. The total noise at `t_n` will be split between recycled and fresh. The intuition matches DDIM, but every coefficient is the bridge's. I wrote it down quickly; whether it actually preserves the marginal is a separate question I now have to settle, because the whole edifice rests on it.

Let me try to prove `q^ρ(x_{t_n} | x_0, x_T) = q(x_{t_n} | x_0, x_T) = N(a_{t_n} x_T + b_{t_n} x_0, c_{t_n}² I)` by backward induction on `n`. The base case is `n = N − 1`, where `t_{n+1} = T`. I'll restrict `ρ` so that `ρ_{N-1} = c_{t_{N-1}}` — meaning at the very first sampling step the recycled-noise coefficient `√(c_{t_{N-1}}² − ρ_{N-1}²)` is zero. Then the conditional is just `N(a_{t_{N-1}} x_T + b_{t_{N-1}} x_0, c_{t_{N-1}}² I)`, which *is* the bridge kernel — base case done. Now suppose it holds at `n = k`: `q^ρ(x_{t_k} | x_0, x_T) = N(a_{t_k} x_T + b_{t_k} x_0, c_{t_k}² I)`. I want it at `n = k − 1`:
```
q^ρ(x_{t_{k-1}} | x_0, x_T) = ∫ q^ρ(x_{t_{k-1}} | x_0, x_{t_k}, x_T) q^ρ(x_{t_k} | x_0, x_T) dx_{t_k}.
```
This is a Gaussian convolved with a Gaussian — a linear-Gaussian marginalization, the standard "if `y | x ~ N(Mx + m, L)` and `x ~ N(μ, Σ)` then `y ~ N(Mμ + m, L + MΣMᵀ)`" identity. The "input" `x_{t_k}` enters the mean of `x_{t_{k-1}}` linearly through `√(c_{t_{k-1}}² − ρ_{k-1}²)/c_{t_k} · x_{t_k}`. Plug in the mean of `x_{t_k}`, which is `a_{t_k} x_T + b_{t_k} x_0`. Watch the recycled-direction term: its argument is `(mean of x_{t_k}) − a_{t_k} x_T − b_{t_k} x_0 = 0`. The deterministic direction *averages to zero*, because in expectation `x_{t_k}` sits exactly at its bridge mean, so there's no leftover noise to point along. The mean collapses to `a_{t_{k-1}} x_T + b_{t_{k-1}} x_0`. Good — that's the bridge mean at `t_{k-1}`. Now the variance: it's the injected `ρ_{k-1}² I` plus `(√(c_{t_{k-1}}² − ρ_{k-1}²)/c_{t_k})² · c_{t_k}² I` — the squared linear coefficient times the variance of `x_{t_k}`. The `c_{t_k}²` cancels: `(c_{t_{k-1}}² − ρ_{k-1}²)/c_{t_k}² · c_{t_k}² = c_{t_{k-1}}² − ρ_{k-1}²`, and adding `ρ_{k-1}²` gives `c_{t_{k-1}}²`. So `q^ρ(x_{t_{k-1}} | x_0, x_T) = N(a_{t_{k-1}} x_T + b_{t_{k-1}} x_0, c_{t_{k-1}}² I)`, the bridge kernel exactly. Induction closes.

The variance cancellation is exactly the kind of thing I get wrong on paper, so I check it on actual numbers. Pick a concrete 1D schedule — `α_t = e^{−t/2}`, `σ_t = 0.9√(1−e^{−t}) + 10⁻³` on `(0, 1]`, `T = 1` — and fixed endpoints `x_0 = 1.7`, `x_T = −0.4`. I'll run the recursion above with Monte Carlo over six million samples on the interior schedule `1.0 → 0.8 → 0.6 → 0.4 → 0.2`, with `η = 0.5`, and compare the empirical `(mean, std)` at each interior time to the bridge kernel `(a_t x_T + b_t x_0, c_t)`. The first run blows up: the std at `t = 0.6` comes out around `1.13` against a target of `0.44`, and it gets worse as I march down. That's a real bug, and it's instructive — I had written the recycled term as `√(c²−ρ²)/c_{t_{n+1}}` times the *normalized* deviation `(x_{t_{n+1}} − mean)/c_{t_{n+1}}`, dividing by `c_{t_{n+1}}` a second time. The deviation `(x_{t_{n+1}} − a_{t_{n+1}}x_T − b_{t_{n+1}}x_0)` already carries std `c_{t_{n+1}}`; the `1/c_{t_{n+1}}` lives *inside* the coefficient `√(c²−ρ²)/c_{t_{n+1}}`, so it must not be applied to the deviation again. That double-normalization is precisely the trap the closed-form mean hides — the `ε̂` is a normalized object only when I keep the coefficient unnormalized. Fixing it (coefficient `√(c_{t_n}²−ρ_n²)/c_{t_{n+1}}` multiplying the raw deviation), the run now gives, time by time:
```
t=0.80  emp(mean=0.0115 std=0.3580)  bridge(mean=0.0113 std=0.3581)
t=0.60  emp(mean=0.4230 std=0.4372)  bridge(mean=0.4226 std=0.4372)
t=0.40  emp(mean=0.8386 std=0.4373)  bridge(mean=0.8383 std=0.4373)
t=0.20  emp(mean=1.2627 std=0.3585)  bridge(mean=1.2625 std=0.3585)
```
Every interior marginal matches to about `10⁻⁵`, including the non-monotone wobble in `c_t` (it rises then falls), and it holds at `η = 0.5`, an interior value of the dial — not just at the boundary I forced in the base case. So the construction does preserve *every* bridge marginal, and the induction's variance cancellation is real, not a slip. That's the licence to reuse the network — and it also pinned down that the implementation must carry the `1/c_{t_{n+1}}` inside the coefficient, which I'll need to get right in code.

Now — is this thing actually non-Markovian, and does that matter? In the original h-transform bridge, the forward process is Markovian: `q(x_{t_{n+1}} | x_0, x_{t_n}, x_T)` doesn't depend on `x_0` once you know `x_{t_n}` and `x_T`. Here `x_{t_n}` can depend simultaneously on `x_{t_{n+1}}` *and* `x_0` through the recycled-noise term, so the induced forward process can depend on `x_0`. Let me check exactly when. The forward conditional is given by Bayes' rule, and to test the dependence on `x_0` I differentiate `log q^ρ(x_{t_{n+1}} | x_0, x_{t_n}, x_T)` with respect to `x_{t_{n+1}}` and look at the coefficient of `x_0`. Combining the two Gaussian log-densities and collecting, the `x_0` coefficient is `(b_{t_{n+1}} c_{t_n}² − b_{t_n} c_{t_{n+1}} √(c_{t_n}² − ρ_n²)) / (c_{t_{n+1}}² ρ_n²)`. So the process is Markovian iff that numerator vanishes, `b_{t_{n+1}} c_{t_n}² = b_{t_n} c_{t_{n+1}} √(c_{t_n}² − ρ_n²)`. I want the `ρ_n` that solves this. Substituting the closed forms of `a, b, c` and solving symbolically gives `ρ_n = σ_{t_n} √(a_{t_n}²σ_{t_{n+1}}² − a_{t_{n+1}}²σ_{t_n}²) /(a_{t_n}σ_{t_{n+1}})`, and squaring that against `σ_{t_n}²(1 − SNR_{t_{n+1}}/SNR_{t_n})` the two are identical — so `ρ_n = σ_{t_n} √(1 − SNR_{t_{n+1}}/SNR_{t_n})`. So there's a special, *largest* value of the per-step noise at which the chain becomes Markovian. Working through what that value does to the update, it also *cancels the `x_T` term* and reduces the inference to `p_θ(x_{t_n} | x_{t_{n+1}})` with no endpoint dependence — which is the DDPM-style ancestral sampler. At the other extreme, `ρ_n = 0`, there's no fresh noise at all, the update is a deterministic map, and the process is maximally non-Markovian. Everything in between, `0 < ρ_n < σ_{t_n}√(1 − SNR_{t_{n+1}}/SNR_{t_n})`, is non-Markovian with an intermediate noise level.

So I have a one-parameter dial. Let me write the *generative* step by replacing the unknown `x_0` with the network's data prediction `x̂_0 = x_θ(x_{t_{n+1}}, t_{n+1}, x_T)`, evaluated at the later/current state:
```
x_{t_n} = a_{t_n} x_T + b_{t_n} x̂_0 + √(c_{t_n}² − ρ_n²) · ε̂ + ρ_n ε,   ε̂ = (x_{t_{n+1}} − a_{t_{n+1}} x_T − b_{t_{n+1}} x̂_0)/c_{t_{n+1}},   ε ~ N(0, I).
```
This *is* the bridge forward kernel `x_t = a_t x_T + b_t x_0 + c_t ε`, but with `x_0` swapped for the prediction and a fraction of the would-be Gaussian noise replaced by the *predicted* noise `ε̂` carried over from the later state. The fully deterministic case `ρ_n = 0` for all `n` is an implicit probabilistic model — sampling is a fixed deterministic procedure from a latent — so by analogy to the diffusion case I'll call it a *diffusion bridge implicit model*. The `ρ_n = 0` end is the one I expect to be fast and invertible; the Markov end gives me a stochastic sampler when I want diversity.

The marginal check above licenses reusing the network when I plug `x̂_0` in for `x_0`, but only *morally* — it shows the inference marginals match, not that the variational objective of this new generative family has the same minimizer as the score-matching loss the network was trained with. For strictly positive `ρ` I can chase that down explicitly. The variational objective is `J^ρ(θ) = E[log q^ρ(x_{t_{1:N-1}} | x_0, x_T) − log p_θ(x_{t_{0:N-1}} | x_T)]`, and factorizing both joints it becomes a sum of KL divergences between the inference conditional and the generative conditional, plus a reconstruction term. Each KL is between two Gaussians with the *same* covariance `ρ_n² I` and means differing only in `x_0` versus `x̂_0`, so it's `‖mean difference‖² / (2 ρ_n²)`. The mean difference is `(b_{t_n} − √(c_{t_n}² − ρ_n²) b_{t_{n+1}}/c_{t_{n+1}})(x̂_0 − x_0)`; call the scalar in front `d_n`. So the `n`-th KL is `d_n² ‖x̂_0 − x_0‖² / (2 ρ_n²)`. The reconstruction term at `t_1` is `‖x̂_0 − x_0‖²/(2 ρ_0²)` up to a constant, which I fold in by defining `d_0 = 1`. So `J^ρ − C = Σ_n (d_{n-1}²/(2 ρ_{n-1}²)) E‖x_θ(x_{t_n}, t_n, x_T) − x_0‖²`. It's a weighted sum of data-prediction errors over the discretized times. Now convert data-prediction error to *score* error, because that's what the network actually optimizes: from `s_θ = −(x_t − a_t x_T − b_t x_θ)/c_t²` and the true conditional score `∇ log q(x_t|x_0,x_T) = −(x_t − a_t x_T − b_t x_0)/c_t²`, the difference is `(b_t/c_t²)(x_θ − x_0)`, so `‖x_θ − x_0‖² = (c_t⁴/b_t²)‖s_θ − ∇ log q‖²`. Substituting, `J^ρ − C = Σ_n γ(t_n) E‖s_θ − ∇ log q‖²` with `γ(t_n) = d_{n-1}² c_{t_n}⁴ / (2 ρ_{n-1}² b_{t_n}²)`. This is the denoising bridge score-matching loss, just with a different per-time weighting `γ`. And the weighting doesn't change the optimum: if the network has enough capacity and isn't sharing parameters across `t`, each timestep's term is minimized independently, so the global minimizer is the same regardless of `γ` (the losses under two weightings even bound each other by the ratio of min/max weights). So the network trained under the original loss is also the optimal network for the positive-variance family, and the deterministic sampler is the noiseless endpoint of that same marginal-preserving construction. The reuse is justified at the level of the objective, not just the marginals.

Now back to the dial and the design choices. I'll parameterize `ρ` by interpolating linearly between its two boundaries with a scalar `η ∈ [0, 1]`:
```
ρ_n = η · σ_{t_n} √(1 − SNR_{t_{n+1}}/SNR_{t_n}),
```
where this `ρ_n` is the injected standard deviation of the transition, while the code helper's variable named `rho_t` is a different schedule quantity, `σ_t/α_t = 1/sqrt(SNR_t)`. So `η = 0` is the deterministic implicit model and `η = 1` is the Markovian, DDPM-like stochastic sampler, with everything in between a partially-stochastic non-Markovian bridge. Why allow a dial at all instead of just shipping the deterministic version? Because the two ends genuinely trade off. The deterministic map takes clean, sharp steps and is invertible, which is what I want when the source and target are tightly correlated, or when I need to encode/reconstruct/interpolate. But injected noise has a real benefit too: a stochastic step behaves like a Langevin correction, nudging the trajectory back toward the correct marginal and washing out accumulated discretization error, which can help quality on tasks where many outputs are plausible. There's no universally best stochasticity, so I expose it.

Now I hit a wall, and it's a sharp one. Take the deterministic case `η = 0` and look at the *very first* step, where `t_{n+1} = T`. The update has `ε̂ = (x_T − a_T x_T − b_T x̂_0)/c_T`, and `c_T` is the bridge noise scale at the endpoint, which is *zero*: at `t = T` the bridge is pinned exactly at `x_T`, no spread. So I'm dividing by `c_T = 0`. The deterministic first step is singular. And it's not a mere numerical nuisance — it's telling me something true. Under a fixed `x_T`, the state `x_t` for `t < T` is *genuinely stochastic*: the marginal `p(x_t | x_T)` is not a point mass, because a single masked image corresponds to many plausible completions, a single sketch to many handbags. A fully deterministic procedure from a single `x_T` would have to pretend there's one answer, which is wrong. So the singularity is the math refusing to let me erase the bridge's intrinsic stochasticity at the start.

The fix is forced by the boundary condition I already imposed in the marginal-preservation proof. I required `ρ_{N-1} = c_{t_{N-1}}` at the base case — which is exactly the Markovian boundary value at the first step. With that choice, `√(c_{t_{N-1}}² − ρ_{N-1}²) = 0`, the recycled-noise coefficient vanishes, and the `c_{t_{n+1}} = c_T` in the denominator never gets multiplied in — the singular term is annihilated. What's left is a single injection of fresh standard Gaussian noise at the first step, of scale `c_{t_{N-1}}`. I'll call this the *booting noise*: it's the one shot of randomness that accounts for the spread of `x_0` given a fixed `x_T`, and I can read it as the latent variable of the whole generative process. Fix the booting noise and the rest of an `η = 0` run is deterministic and invertible — so the booting noise is the handle for diversity, encoding, and interpolation. So the first step is special and always stochastic (it places the initial state on the bridge via `x = a_{t_{N-1}} x_T + b_{t_{N-1}} x̂_0 + c_{t_{N-1}} ε`); the implementation realizes this by predicting at `T` and sampling the first interior state `ts[0]`, then applying the `η` dial only to later interior transitions. (This is also why my Monte-Carlo check above started the chain by forcing `ρ = c` at the first transition: that boundary is not a free choice, it's the only thing that makes the deterministic limit well-posed.)

There's a mirror-image concern at the *end* of the trajectory. On the final step, fresh injected noise would just be added straight onto the output with no further denoising to clean it up — it would blur the result. So on the last step I drop the fresh-noise term and keep only the deterministic part, which keeps the endpoint sharp. (This is the same reasoning that makes one take a lower-order, no-churn final step in diffusion samplers.)

Now I want to understand the deterministic `η = 0` sampler as a continuous object, because that's where the real acceleration will come from. Set `ρ = 0`, write `t_{n+1} = t`, `t_n = t − Δt`, and look at the update:
```
x_{t-Δt} = a_{t-Δt} x_T + b_{t-Δt} x̂_0 + (c_{t-Δt}/c_t)(x_t − a_t x_T − b_t x̂_0).
```
This is full of `x_T` and `x̂_0` terms that don't obviously want to become a clean `dx/dt`. Let me try dividing the whole thing by `c_{t-Δt}` and regrouping, because the troublesome factor is that `c_t` ratio:
```
x_{t-Δt}/c_{t-Δt} = x_t/c_t + (a_{t-Δt}/c_{t-Δt} − a_t/c_t) x_T + (b_{t-Δt}/c_{t-Δt} − b_t/c_t) x̂_0.
```
That's clean — it's a finite difference of `x_t/c_t` equal to finite differences of `a_t/c_t` and `b_t/c_t` weighting the two endpoints. Since `a_t, b_t, c_t` are smooth functions of `t`, this is the Euler discretization of
```
d(x_t/c_t) = x_T d(a_t/c_t) + x̂_θ(x_t, t, x_T) d(b_t/c_t).
```
So the natural state variable is not `x_t` but `x_t/c_t`, and the natural "time" coordinates are `a_t/c_t` and `b_t/c_t`, not `t`. This is the bridge analogue of the diffusion trick where DDIM became an Euler step on `x̄ = x/√α` with respect to `dσ`. The noise scale `c_t` is exactly what was blowing up the linear part of the dynamics, and dividing it out gives a low-curvature ODE. But I should check it is the *same* trajectory as the bridge probability-flow ODE the original framework writes down — if it weren't, I'd be sampling the wrong distribution. The cleanest test: the noiseless trajectory of the bridge (the `ε = 0` path) has `x_t = a_t x_T + b_t x_0` exactly, so its velocity is just `d/dt(a_t x_T + b_t x_0) = a_t' x_T + b_t' x_0`. My ODE, evaluated on that same mean trajectory with `x̂_θ = x_0`, must produce the identical velocity. Working it out: `d(x/c)/dt = x_T (a/c)' + x_0 (b/c)'` rearranges to `dx/dt = c[x_T (a/c)' + x_0 (b/c)'] + (c'/c) x`. Substituting `x = a x_T + b x_0` and expanding the derivatives — I did this symbolically with the closed forms of `a, b, c` built from a generic smooth `α(t), σ(t)` and `f = (log α)'`, `g² = (σ²)' − 2(log α)'σ²` — the expression simplifies to exactly `a_t' x_T + b_t' x_0`, matching `d/dt(a x_T + b x_0)` to zero residual. So my ODE carries the bridge along the same path; it's just written in coordinates where the linear part is trivial. The conversion *from* my ODE *to* the PF-ODE is a one-line chain rule; going the other way (deriving my neat form starting from the PF-ODE) would need the heavy machinery of exponential integrators — which tells me I found the right coordinates by sticking with the discrete update and letting the ODE fall out, rather than trying to massage the PF-ODE.

That cleanness is the doorway to high-order solvers, and now I want them, because first-order Euler is still leaving accuracy on the table at low NFE. The way fast diffusion solvers get high order is by recognizing the ODE as *semi-linear* — a linear-in-`x` part plus a nonlinear network part — using variation-of-constants to integrate the linear part *exactly*, and then approximating only the integral of the network output, which is smooth. Let me do that for my bridge ODE. Expanding `d(x/c)` gives `dx_t = [A(t) x_t + B_T(t) x_T + B_θ(t) x̂_θ] dt`, where `A(t) = c_t'/c_t`, `B_T(t) = a_t' − a_t c_t'/c_t = a_t d log(a_t/c_t)/dt`, and `B_θ(t) = b_t' − b_t c_t'/c_t = b_t d log(b_t/c_t)/dt`. Variation-of-constants gives `x_t = e^{∫_s^t A} x_s + ∫_s^t e^{∫_τ^t A} [B_T x_T + B_θ x̂_θ] dτ`. The integrating factor is `e^{∫_s^t A} = e^{∫ (log c)'} = c_t/c_s`. The `x_T` integral closes in elementary form because `x_T` is constant: it contributes `(a_t − (c_t/c_s) a_s) x_T`. The only genuinely hard integral is the one with the network output, `c_t ∫ (b_τ/c_τ) x̂_θ d log(b_τ/c_τ)`. So define the change of variable `λ_t = log(b_t/c_t)` — this is the bridge's log-SNR-like coordinate, and `b_t/c_t` is the signal-to-noise amplitude, just as `α_t/σ_t` is the diffusion signal-to-noise amplitude. Then the exact solution from time `t` to `s < t` is
```
x_s = (c_s/c_t) x_t + (a_s − (c_s/c_t) a_t) x_T + c_s ∫_{λ_t}^{λ_s} e^λ x̂_θ(x_{t_λ}, t_λ, x_T) dλ.
```
The linear and endpoint parts are now *exact*; all the approximation error lives in that one exponentially-weighted integral of the smooth network output. I should keep the operative definition as `λ_t = log(b_t/c_t)` — that is what the code computes directly. Since `b_t/c_t = √(SNR_t − SNR_T)`, the closed form is `λ_t = ½ log(SNR_t − SNR_T)`, but the solver only needs `log(b/c)`.

To get order `k`, Taylor-expand `x̂_θ` as a function of `λ` around the current time:
```
x̂_θ(x_{t_λ}, t_λ, x_T) ≈ x̂_t + (λ − λ_t) x̂_t^{(1)} + ½ (λ − λ_t)² x̂_t^{(2)} + …,
```
where `x̂_t^{(k)} = d^k x̂_θ / dλ^k` at the current time, and substitute. The scalar integrals `∫_{λ_t}^{λ_s} (λ − λ_t)^n e^λ dλ` are analytic — repeated integration by parts. Let me actually compute them with `h = λ_s − λ_t` (note `λ` increases as time decreases here, so stepping `t → s` with `s < t` means moving forward in `λ`). Pulling out the `e^{λ_s}` prefactor: `∫ e^λ dλ = e^{λ_s}(1 − e^{-h})`; `∫ (λ − λ_t) e^λ dλ = e^{λ_s}(h − 1 + e^{-h})`; `∫ ½(λ − λ_t)² e^λ dλ = e^{λ_s}(h²/2 − h + 1 − e^{-h})`. These are easy to get sign-wrong, so let me check them numerically with `λ_t = 0.3, h = 0.7`: trapezoidal integration of the three integrands gives `1.36842302, 0.53437426, 0.13160479`, and the closed forms `e^{λ_s}(1−e^{-h})`, `e^{λ_s}(h−1+e^{-h})`, `e^{λ_s}(h²/2−h+1−e^{-h})` give exactly those same values to ~`10⁻¹⁵`. So the bracket is right:
```
∫ ≈ e^{λ_s} [ (1 − e^{-h}) x̂_t + (h − 1 + e^{-h}) x̂_t^{(1)} + (h²/2 − h + 1 − e^{-h}) x̂_t^{(2)} ].
```
Keeping just the first bracket term should be first order, and it had better coincide with my Euler update or something is inconsistent. Let me verify that reduction symbolically rather than wave at it: the first-order step is `x_s = (c_s/c_t) x_t + (a_s − (c_s/c_t) a_t) x_T + c_s e^{λ_s}(1 − e^{-h}) x̂_t`, and with `e^{λ_s} = b_s/c_s` and `e^{-h} = (b_t/c_t)/(b_s/c_s)` it should equal the `η = 0` update `x_s = a_s x_T + b_s x̂_t + (c_s/c_t)(x_t − a_t x_T − b_t x̂_t)`. Subtracting the two symbolic expressions and simplifying returns exactly zero — they are the same map. Good: the high-order machinery degenerates to the Euler/DBIM step at first order, as it must. Keeping two bracket terms is second order, three is third.

I don't have the `λ`-derivatives of the network output directly — that would cost extra NFE — so I estimate them by finite differences over the *past* network outputs I've already computed, which is free. With one previous time `u` (so `s < t < u`, i.e. `u` is an earlier, noisier step), the first derivative is just `x̂_t^{(1)} ≈ (x̂_t − x̂_u)/h_1`, `h_1 = λ_t − λ_u` — a backward difference in `λ`. For third order I keep two previous times `u_1, u_2` and fit a quadratic: with `h_1 = λ_t − λ_{u_1}`, `h_2 = λ_{u_1} − λ_{u_2}`, the standard divided-difference estimates are
```
x̂_t^{(1)} ≈ [ (x̂_t − x̂_{u_1})/h_1 · (2 h_1 + h_2) − (x̂_{u_1} − x̂_{u_2})/h_2 · h_1 ] / (h_1 + h_2),
x̂_t^{(2)} ≈ 2 [ (x̂_t − x̂_{u_1})/h_1 − (x̂_{u_1} − x̂_{u_2})/h_2 ] / (h_1 + h_2).
```
So high order reuses the network-output history — no extra denoiser calls, which is the whole point under a tight NFE budget. The first step has no history, so it's first order; and the recommended last step drops to first order too (lower-order-final) to avoid amplifying noise near the sharp endpoint.

Let me also place this against the methods I came from, to be sure I've found the right generalization and not a fourth thing. In the regime where `t` is small and `SNR_T/SNR_t → 0`, the bridge coefficients degenerate: `a_t → 0`, `b_t → α_t`, `c_t → σ_t`, so the bridge kernel becomes `N(α_t x_0, σ_t² I)` — the ordinary diffusion kernel. I want to know whether the `η = 0` update degenerates to the DDIM update in that limit, and I can check it symbolically. Taking `a → 0, b → α, c → σ`, my update becomes `x_s = α_s x̂_0 + (σ_s/σ_t)(x_t − α_t x̂_0)`. The DDIM update is `x_s = α_s x̂_0 + σ_s ε̂` with `ε̂ = (x_t − α_t x̂_0)/σ_t`, i.e. `α_s x̂_0 + σ_s(x_t − α_t x̂_0)/σ_t`. Subtracting the two and simplifying gives exactly zero — they are identical, with `x̂_0` merely additionally conditioned on `x_T`. So DDIM falls out as the small-`t` limit where the endpoint stops mattering, and DDIM is the special case where the bridge collapses to a one-sided diffusion. The Markovian `η = 1` boundary, I already showed by solving for the Markov condition, is the DDPM-style ancestral sampler. And a degenerate Brownian-bridge schedule with vanishing intermediate noise turns the `η = 1` step into a plain Euler step of a flow `x_s = x_t − (t − s) v_θ`, the flow-matching update. Everything I know sits inside this family at a particular `(η, schedule)`: three independent limits, three known methods, not a fourth ad hoc scheme.

Time to put this into the code that fills the one empty slot in the sampler harness — the per-step transition. I'll write the first-order `η`-controlled sampler first, mapping straight from the update equation, carrying the `1/c_{t_{n+1}}` inside the recycled coefficient the way the marginal check insisted. Inside the harness the loop names the current (larger) time `s` and the next (smaller) target time `t`, so in the update equation above, `s` is `t_{n+1}` and `t` is `t_n`; I'll let `ω_{st}` denote the per-step injected std `ρ_n` computed at the target time `t`, while the schedule's `rho_t` variable is `σ_t/α_t`:

```python
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from .nn import append_dims
from .random_util import BatchedSeedGenerator


@torch.no_grad()
def sample_dbim(denoiser, diffusion, x, ts, eta=1.0, mask=None, seed=None, **kwargs):
    # x is the source endpoint x_T; ts is the decreasing interior schedule; eta in [0, 1].
    x_T = x
    path, pred_x0 = [], []
    ones = x.new_ones([x.shape[0]])
    indices = tqdm(range(len(ts) - 1), disable=(dist.get_rank() != 0))

    nfe = 0
    # First step is forced stochastic: predict x_0 at the endpoint, then seed the initial
    # bridge state with the booting noise (rho_{N-1} = c_{t_{N-1}} defuses the c_T -> 0 singularity).
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise                                   # the booting noise = the latent variable
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)         # keep observed pixels for inpainting
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)   # a_t x_T + b_t x0_hat + c_t * noise
    path.append(x.detach().cpu()); pred_x0.append(x0_hat.detach().cpu()); nfe += 1

    for _, i in enumerate(indices):
        s = ts[i]            # current (larger) time  = t_{n+1}
        t = ts[i + 1]        # next (smaller) time     = t_n

        x0_hat = denoiser(x, s * ones)                    # data prediction x_hat_0 from the current state
        if mask is not None:
            x0_hat = x0_hat * mask + x_T * (1 - mask)

        a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
        a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
        # the schedule's raw noise level sigma/alpha at both times (named rho in the schedule):
        _, _, rho_s, _ = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_alpha_rho(s * ones)]
        alpha_t, _, rho_t, _ = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_alpha_rho(t * ones)]

        # per-step injected std rho_n = eta * sigma_{t_n} * sqrt(1 - SNR_{t_{n+1}}/SNR_{t_n})
        #   = eta * (alpha_t * (sigma_t/alpha_t)) * sqrt(1 - (sigma_t/alpha_t)^2 / (sigma_s/alpha_s)^2)
        omega_st = eta * (alpha_t * rho_t) * (1 - rho_t**2 / rho_s**2).sqrt()

        # recycled-direction coefficient sqrt(c_{t_n}^2 - rho_n^2)/c_{t_{n+1}}, and the two endpoint coeffs
        tmp_var = (c_t**2 - omega_st**2).sqrt() / c_s
        coeff_xs = tmp_var                                # multiplies the current state x_{t_{n+1}}
        coeff_x0_hat = b_t - tmp_var * b_s                # multiplies x_hat_0  (after substituting eps_hat)
        coeff_xT = a_t - tmp_var * a_s                    # multiplies x_T

        noise = generator.randn_like(x0_hat)
        # x_{t_n} = coeff_x0_hat * x_hat_0 + coeff_xT * x_T + coeff_xs * x_{t_{n+1}} + rho_n * eps
        # ... but drop the fresh noise on the final step to keep the endpoint sharp.
        x = (coeff_x0_hat * x0_hat + coeff_xT * x_T + coeff_xs * x
             + (1 if i != len(ts) - 2 else 0) * omega_st * noise)

        path.append(x.detach().cpu()); pred_x0.append(x0_hat.detach().cpu()); nfe += 1

    return x, path, nfe, pred_x0, ts, first_noise
```

And the deterministic high-order variant fills the same slot with the exponential-integrator solution instead, reusing the history of past predictions `x̂_u` to estimate the `λ`-derivatives, dropping to first order on the first and (optionally) last steps. The formula above used `t` for the current larger time and `s` for the next smaller time; the implementation uses the opposite loop names, with `s = ts[i]` current and `t = ts[i+1]` next, so its `h = lambda_t - lambda_s` is the same positive step in `λ`.

```python
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from .nn import append_dims
from .random_util import BatchedSeedGenerator


@torch.no_grad()
def sample_dbim_high_order(denoiser, diffusion, x, ts, mask=None, order=2,
                           lower_order_final=True, seed=None, **kwargs):
    if order not in [2, 3]:
        raise NotImplementedError("Not supported")
    x_T = x
    path, pred_x0 = [], []
    ones = x.new_ones([x.shape[0]])
    indices = tqdm(range(len(ts) - 1), disable=(dist.get_rank() != 0))

    nfe = 0
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)   # booting noise, deterministic afterwards
    path.append(x.detach().cpu()); pred_x0.append(x0_hat.detach().cpu()); nfe += 1

    # history buffers of previous times u and previous predictions x_hat_u (for lambda-derivatives)
    u = diffusion.t_max
    if u == 1.0:
        u -= 5e-5
    u = [u for _ in range(order - 1)]
    xu_hat = [x0_hat.detach().clone() for _ in range(order - 1)]

    for _, i in enumerate(indices):
        s = ts[i]            # current (larger) time
        t = ts[i + 1]        # next (smaller) time

        # First-order step on the very first iteration and (optionally) the very last one.
        if (lower_order_final and i + 1 == len(ts) - 1) or (i == 0):
            a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
            tmp_var = c_t / c_s                                  # deterministic: rho = 0
            coeff_xs, coeff_x0_hat, coeff_xT = tmp_var, b_t - tmp_var * b_s, a_t - tmp_var * a_s
            x0_hat = denoiser(x, s * ones); nfe += 1
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            x = coeff_xs * x + coeff_x0_hat * x0_hat + coeff_xT * x_T

        elif order == 2 or i == 1:                              # second-order exponential-integrator step
            a_u, b_u, c_u = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(u[-1] * ones)]
            a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u, lambda_s, lambda_t = torch.log(b_u / c_u), torch.log(b_s / c_s), torch.log(b_t / c_t)
            x0_hat = denoiser(x, s * ones); nfe += 1
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            h = lambda_t - lambda_s                              # step in lambda; h2 spans the history step
            h2 = lambda_s - lambda_u
            # integral = e^{lambda_t}[(1 - e^{-h}) x_hat + (e^{-h} + h - 1) * x_hat^{(1)}],  x_hat^{(1)} ~ (x_hat - x_hat_u)/h2
            integral = torch.exp(lambda_t) * ((1 - torch.exp(-h)) * x0_hat
                                              + (torch.exp(-h) + h - 1) * (x0_hat - xu_hat[-1]) / h2)
            x = x * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        elif order == 3:                                        # third-order step
            a_u1, b_u1, c_u1 = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(u[-1] * ones)]
            a_u2, b_u2, c_u2 = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(u[-2] * ones)]
            a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u2, lambda_u1, lambda_s, lambda_t = (torch.log(b_u2 / c_u2), torch.log(b_u1 / c_u1),
                                                        torch.log(b_s / c_s), torch.log(b_t / c_t))
            x0_hat = denoiser(x, s * ones); nfe += 1
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            h, h1, h2 = lambda_t - lambda_s, lambda_s - lambda_u1, lambda_u1 - lambda_u2
            # quadratic divided-difference estimates of the first and second lambda-derivatives
            dx0_hat = ((x0_hat - xu_hat[-1]) * (2 * h1 + h2) / h1 - (xu_hat[-1] - xu_hat[-2]) * h1 / h2) / (h1 + h2)
            d2x0_hat = 2 * ((x0_hat - xu_hat[-1]) / h1 - (xu_hat[-1] - xu_hat[-2]) / h2) / (h1 + h2)
            integral = torch.exp(lambda_t) * ((1 - torch.exp(-h)) * x0_hat
                                              + (torch.exp(-h) + h - 1) * dx0_hat
                                              + (h**2 / 2 - h + 1 - torch.exp(-h)) * d2x0_hat)
            x = x * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        u.append(s); u.pop(0)                                   # roll the history forward
        xu_hat.append(x0_hat); xu_hat.pop(0)
        path.append(x.detach().cpu()); pred_x0.append(x0_hat.detach().cpu())

    return x, path, nfe, pred_x0, ts, first_noise
```

Both variants drop into the same harness slot, reusing the pretrained bridge network exactly as trained: the η dial for cheap, tunable stochasticity control, the exponential-integrator solver for the accuracy that plain Euler leaves on the table once the NFE budget gets tight.
