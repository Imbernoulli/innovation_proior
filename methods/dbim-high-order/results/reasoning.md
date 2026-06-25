Let me start from what actually hurts. I have a trained diffusion bridge that does image-to-image translation beautifully — give it an edge map or a masked image as the endpoint `x_T = y` and it produces a photorealistic completion — but to draw one sample I have to simulate the bridge's reverse process, and at high resolution that swallows well over a hundred denoiser evaluations. Each evaluation is one forward pass of a big U-Net; that's the entire cost. Meanwhile, ordinary diffusion models that map noise to data are sampled in about ten calls these days. So my translation model is an order of magnitude slower than it has any right to be, and the gap is not in the network — it's in how I'm marching the dynamics. I want to keep the trained predictor exactly as it is and only change the rule that steps the state forward, until a tiny call budget is plausible. That's the whole game.

Why can't I just borrow the fast diffusion samplers wholesale? Because every one of them is written for a Gaussian prior. DDIM's update has a single data term `sqrt(alpha_{t-1}) x_0` and the marginal it preserves is `N(sqrt(alpha_t) x_0, (1-alpha_t) I)` — there is no `x_T` anywhere, because in ordinary diffusion the endpoint *is* noise. DPM-Solver and its data-prediction cousin solve the diffusion ODE `dx = [f(t)x + (g^2/2sigma) eps_theta] dt`, again with no endpoint condition. My process is pinned at *both* ends. The bridge's forward kernel is `q(x_t|x_0,x_T) = N(a_t x_T + b_t x_0, c_t^2 I)` — three terms, two endpoints, a genuinely different linear structure. So the fast-sampler machinery is the right *kind* of tool, but I can't drop it in. I have to rebuild it for the bridge.

Let me get the bridge coefficients in front of me, because everything will hinge on them. With base schedule `alpha_t, sigma_t` and `SNR_t = alpha_t^2/sigma_t^2`, the bridge from `x_0` to the fixed endpoint `x_T` has

  a_t = (alpha_t/alpha_T)(SNR_T/SNR_t),  b_t = alpha_t (1 - SNR_T/SNR_t),  c_t^2 = sigma_t^2 (1 - SNR_T/SNR_t),

so `x_t = a_t x_T + b_t x_0 + c_t eps`, `eps ~ N(0,I)`. Sanity check the ends: as `t -> T`, `SNR_T/SNR_t -> 1`, so `b_t, c_t -> 0` and `a_t -> 1`, i.e. `x_T -> x_T` exactly, the bridge is pinned. As `t -> 0`, `SNR_T/SNR_t -> 0`, so `a_t -> 0`, `b_t -> alpha_t`, `c_t -> sigma_t`, recovering the ordinary diffusion marginal around `x_0`. Good — the bridge interpolates from "all endpoint" to "noised data." The denoiser I have is the data predictor `x_theta(x_t, t, x_T)` that recovers `x_0`.

Now, how did the fast samplers get fast in the first place, conceptually, so I can reproduce the move? Two ideas. One: reduce stochasticity — instead of simulating an SDE you find a deterministic process with the *same marginals* and march that, which discretizes far better. Two: use higher-order information about the predictor instead of treating each step as a blind Euler jump. The deterministic-process idea has a clean origin I want to reuse: the diffusion training objective depends only on the *marginals* `q(x_t|x_0)`, not on the joint, so you're free to swap in a different inference process that keeps those marginals, and a network trained for the old process still serves the new one. DDIM exploited exactly this with a family of non-Markovian forward processes indexed by a variance vector. The question is whether I can build the analogous family for a bridge, where the marginal I must preserve is `q(x_t|x_T)`.

So let me try to write down a family of joints over the *sampling* timesteps `0 = t_0 < t_1 < ... < t_{N-1} < t_N = T` that all share the bridge marginals. The bridge score, and hence my trained predictor, only ever sees `q(x_t|x_T)`; if I can construct alternative joints with the same per-step marginals, the same predictor is valid for all of them. Index the family by a variance vector `rho in R^{N-1}` and posit, for `1 <= n <= N-1`,

  q^(rho)(x_{t_n} | x_0, x_{t_{n+1}}, x_T) = N( a_{t_n} x_T + b_{t_n} x_0 + sqrt(c_{t_n}^2 - rho_n^2) * (x_{t_{n+1}} - a_{t_{n+1}} x_T - b_{t_{n+1}} x_0)/c_{t_{n+1}},  rho_n^2 I ).

Where did this mean come from? I want `x_{t_n}` to be the bridge value `a_{t_n} x_T + b_{t_n} x_0` plus a noise piece of total variance `c_{t_n}^2`, but I want to *split* that noise: a fraction `rho_n^2` is fresh, and the rest is borrowed from the already-realized noise at the next timestep `t_{n+1}`. The quantity `(x_{t_{n+1}} - a_{t_{n+1}} x_T - b_{t_{n+1}} x_0)/c_{t_{n+1}}` is exactly the standardized noise that produced `x_{t_{n+1}}` — divide out its variance and it's unit Gaussian. Scaling it by `sqrt(c_{t_n}^2 - rho_n^2)` and adding fresh noise of variance `rho_n^2` gives total variance `(c_{t_n}^2 - rho_n^2) + rho_n^2 = c_{t_n}^2`, which is the marginal variance I need. That's the DDIM trick transplanted: reuse the realized direction so that `rho` can dial the freshness down to zero.

Before I trust this I want to check that the variance actually books to `c_{t_n}^2` on a concrete schedule, not just in my head. Take a simple monotone schedule `alpha(t) = cos(pi t/2)`, `sigma(t) = sin(pi t/2)`, `T = 0.999`, and two nodes `t_n = 0.4`, `t_{n+1} = 0.6`. Computing the bridge coefficients there gives `c_{t_n}^2 = 0.34549`. Pick an admissible `rho_n = 0.5 c_{t_n}`; the borrowed-noise direction is the *realized* noise at `t_{n+1}`, which has variance `c_{t_{n+1}}^2`, scaled by `(sqrt(c_{t_n}^2 - rho_n^2)/c_{t_{n+1}})`, so its contribution to the variance is `(c_{t_n}^2 - rho_n^2)/c_{t_{n+1}}^2 * c_{t_{n+1}}^2 = c_{t_n}^2 - rho_n^2`, and adding the fresh `rho_n^2` gives `0.34549` again — exactly `c_{t_n}^2`, to machine precision. So the splitting does reconstruct the marginal variance, for *any* `rho_n` in `[0, c_{t_n}]`. That's encouraging, but a single node only checks the variance; I still need the mean to land right and the recursion to hold all the way down, which a numeric spot-check can't show. So let me actually prove it.

Claim: `q^(rho)(x_{t_n}|x_T) = q(x_{t_n}|x_T)` for all `n`. Since both factor as `int q(x_{t_n}|x_0,x_T) q(x_0|x_T) dx_0` with the same `q(x_0|x_T)`, it's enough to show `q^(rho)(x_{t_n}|x_0,x_T) = q(x_{t_n}|x_0,x_T) = N(a_{t_n} x_T + b_{t_n} x_0, c_{t_n}^2 I)`. Induct downward from `n = N-1`. The base case needs the boundary `rho_{N-1} = c_{t_{N-1}}`: at `n = N-1` we have `t_{n+1} = T`, the `sqrt(c^2 - rho^2) = sqrt(c_{t_{N-1}}^2 - c_{t_{N-1}}^2) = 0` kills the borrowed-noise term, and the conditional collapses to exactly `N(a_{t_{N-1}} x_T + b_{t_{N-1}} x_0, c_{t_{N-1}}^2 I)`. Good — base case holds, and it tells me `rho_{N-1}` is forced. Inductive step: suppose `q^(rho)(x_{t_k}|x_0,x_T) = N(a_{t_k} x_T + b_{t_k} x_0, c_{t_k}^2 I)`. Then

  q^(rho)(x_{t_{k-1}}|x_0,x_T) = int N(x_{t_{k-1}}; mu_{k-1|k}, rho_{k-1}^2 I) N(x_{t_k}; a_{t_k} x_T + b_{t_k} x_0, c_{t_k}^2 I) dx_{t_k},

a Gaussian marginalized over a Gaussian. Using the standard linear-Gaussian marginalization (a conditional `N(M x_{t_k} + d, S)` against `N(x_{t_k}; m, P)` integrates to `N(M m + d, S + M P M^T)`), the mean substitutes `x_{t_k} -> a_{t_k} x_T + b_{t_k} x_0` into the borrowed-noise direction, and that direction's offset is precisely `(a_{t_k} x_T + b_{t_k} x_0 - a_{t_k} x_T - b_{t_k} x_0)/c_{t_k} = 0`, so the mean is just `a_{t_{k-1}} x_T + b_{t_{k-1}} x_0`. The variance is `rho_{k-1}^2 I + (sqrt(c_{t_{k-1}}^2 - rho_{k-1}^2)/c_{t_k})^2 c_{t_k}^2 I = rho_{k-1}^2 + (c_{t_{k-1}}^2 - rho_{k-1}^2) = c_{t_{k-1}}^2`. Both match `q(x_{t_{k-1}}|x_0,x_T)`. Induction closes. So *every* `rho` in this family preserves the marginals, and my pretrained predictor serves all of them. That's the foothold.

One worry before I move on: have I quietly changed the training objective by changing the joint? If the network had to be retrained per `rho`, this would be worthless. For `rho > 0`, the data-prediction generative process built from this family — replace `x_0` by `x_theta` — has a variational objective that, term by term, is a KL between the family's conditional and the model's conditional. Before translating back to the bridge-score loss, the step ending at `t_{n+1}` contributes `(d_n^2/(2 rho_n^2)) ||x_theta(x_{t_{n+1}},t_{n+1},x_T) - x_0||^2`, with `d_n = b_{t_n} - sqrt(c_{t_n}^2 - rho_n^2) b_{t_{n+1}}/c_{t_{n+1}}` and `d_0 = 1` for the endpoint term. Reindexing by the actual denoising timestep `t_n` and using `s_theta = -(x_t - a_t x_T - b_t x_theta)/c_t^2` turns this into `sum_n gamma(t_n) ||s_theta - nabla log q(x_{t_n}|x_0,x_T)||^2`, where `gamma(t_n) = d_{n-1}^2 c_{t_n}^4/(2 rho_{n-1}^2 b_{t_n}^2)`. That's the same denoising bridge score-matching objective, only with per-timestep weights `gamma(t_n)` that depend on `rho`. The question is whether the reweighting changes the optimum. The `gamma(t_n)` are positive (they're squared quantities over a positive `rho^2 b^2`), and each timestep's term is a non-negative loss minimized at the *same* point — `s_theta = nabla log q` — regardless of its positive weight, provided the network's outputs at different `t` aren't tied together. With a fully `t`-conditioned predictor (the usual case) that independence holds and the global minimizer is `rho`-independent. I'll flag the one caveat honestly: if parameters are heavily shared across `t`, the weights trade off the timesteps against each other and the practical optimum could drift — but that's a finite-capacity effect, not a change in the target the loss points at. So to the extent the predictor fits each timestep well, I can reuse the trained one for any admissible `rho` and then take the deterministic limit for sampling. Now I get to *choose* `rho`.

What does the sampler look like for a given `rho`? Plug the predictor in for `x_0` in the family's conditional. From `x_{t_{n+1}}` to `x_{t_n}`:

  x_{t_n} = a_{t_n} x_T + b_{t_n} x0_hat + sqrt(c_{t_n}^2 - rho_n^2) * (x_{t_{n+1}} - a_{t_{n+1}} x_T - b_{t_{n+1}} x0_hat)/c_{t_{n+1}} + rho_n eps,

with `x0_hat = x_theta(x_{t_{n+1}}, t_{n+1}, x_T)`. Read it the way DDIM reads its update: `a_{t_n} x_T + b_{t_n} x0_hat` is the predicted bridge point, the middle term is "the direction we came along," scaled to the new noise level, and `rho_n eps` is whatever fresh stochasticity I want to inject. Two extremes deserve a look. The first candidate is the value of `rho_n` that makes the process Markovian — that would recover something DDPM-like. A Markov forward kernel `q(x_{t_{n+1}}|x_{t_n})` corresponds, in this family, to the borrowed-noise scale collapsing to a pure rescaling of the bridge with no leftover `x_T` cross-term; the standard choice is `rho_n = sigma_{t_n} sqrt(1 - SNR_{t_{n+1}}/SNR_{t_n})`. Let me see what that does to the borrowed scale `sqrt(c_{t_n}^2 - rho_n^2)/c_{t_{n+1}}`. Plugging in and simplifying, `c_{t_n}^2 - rho_n^2 = sigma_{t_n}^4(alpha_{t_{n+1}}^2 sigma_T^2 - alpha_T^2 sigma_{t_{n+1}}^2)/(alpha_{t_n}^2 sigma_T^2 sigma_{t_{n+1}}^2)`, and dividing by `c_{t_{n+1}}^2` the messy factors cancel to leave `sqrt(c_{t_n}^2 - rho_n^2)/c_{t_{n+1}} = alpha_{t_{n+1}} sigma_{t_n}^2/(alpha_{t_n} sigma_{t_{n+1}}^2)` — a clean ratio of base-schedule quantities with no `SNR_T` and no endpoint dependence left in it. That's the tell that this `rho` produces an endpoint-forgetting Markov step. The other extreme, `rho_n = 0`, kills the fresh noise entirely, so the step is a deterministic function of the previous state — an implicit map from a latent to a sample. That deterministic end is the one I expect to discretize well in few steps, since there's no injected noise whose variance a coarse step would mis-handle; the in-between `rho` interpolates the stochasticity. Structurally this `rho`-indexed family with its deterministic limit is the same move DDIM made for the Gaussian case, now carrying two endpoints.

But the deterministic step has a landmine at the very first move that DDIM never had. Look at `n = N-1`, where `t_{n+1} = T`: the denominator `c_{t_{n+1}} = c_T = 0`, because the bridge is pinned at `T` with zero variance — `c_T^2 = sigma_T^2(1 - SNR_T/SNR_T) = 0`. With `rho = 0` I'd be dividing by zero. And this isn't a numerical nuisance — it's telling me something true. Given a fixed `x_T` (a masked image, say), the value `x_t` for `t < T` is genuinely *not* deterministic: there are many plausible completions of one mask, so `p(x_t|x_T)` is not a point mass. A fully deterministic step from the single point `x_T` cannot manufacture that spread. So I must inject stochasticity at step zero specifically. The fix is forced by the same boundary I already needed for the marginal proof: take `rho_{N-1} = c_{t_{N-1}}` at the first step, the Markovian boundary, which makes `sqrt(c^2 - rho^2) = sqrt(c_{t_{N-1}}^2 - c_{t_{N-1}}^2) = 0` and so multiplies the `1/c_T` by zero — the `0/0` resolves to no borrowed term at all. What's left injects one standard Gaussian — a *booting noise* — and from then on I can run `rho = 0`. Note this is the *only* place fresh noise enters once I commit to `rho = 0` afterward: every later step is a deterministic function of its predecessor, so the entire trajectory is a deterministic map of this single drawn `eps`. That makes the booting noise the latent that carries the generation diversity — fix it and I get one completion, vary it and I sweep them. Concretely the first step just draws `eps` and sets `x = a_{t_{N-1}} x_T + b_{t_{N-1}} x0_hat + c_{t_{N-1}} eps`, i.e. it samples the bridge at the first non-terminal node around the predicted `x_0`.

Now I have a fast first-order deterministic sampler. The "use higher-order information" lever is the other half of the acceleration, and to use it I need to know what continuous object my deterministic step is discretizing. Set `rho = 0`, write `t_{n+1} = t`, `t_n = t - dt`, and stare at the deterministic update. Rearranging, it's cleanest to divide through by `c`:

  x_{t-dt}/c_{t-dt} = x_t/c_t + (a_{t-dt}/c_{t-dt} - a_t/c_t) x_T + (b_{t-dt}/c_{t-dt} - b_t/c_t) x_theta(x_t,t,x_T).

The increments on the right are finite differences of `a_t/c_t` and `b_t/c_t`, which are smooth functions of `t`. So this is an Euler discretization of

  d(x_t/c_t) = x_T d(a_t/c_t) + x_theta(x_t,t,x_T) d(b_t/c_t).

That's a remarkably clean ODE. It evolves `x_t/c_t`, not `x_t`, and it's driven by `d(a_t/c_t)` and `d(b_t/c_t)` rather than `dt` directly — though by the chain rule those are just `(a_t/c_t)' dt` and `(b_t/c_t)' dt`, so it is an ODE in `t`. I should check this is the *right* ODE — that it agrees with the bridge's probability-flow ODE — because if it were some other deterministic process I'd be solving the wrong thing. The bridge PF-ODE, with the score replaced by the data predictor, is

  dx_t = [(f + g^2/sigma_t^2 - g^2/(2 c_t^2)) x_t + g^2 a_t/(2 c_t^2) x_T - g^2 b_t/(2 c_t^2) x_theta] dt.

Now expand my clean ODE back into `dx_t`. From `x_t/c_t`, the product rule gives `dx_t/c_t - (c_t'/c_t^2) x_t dt = (a_t/c_t)' x_T dt + (b_t/c_t)' x_theta dt`, i.e.

  dx_t = [ (c_t'/c_t) x_t + (a_t' - a_t c_t'/c_t) x_T + (b_t' - b_t c_t'/c_t) x_theta ] dt.

So I need three log-derivatives. Using `f = (log alpha_t)'`, `g^2 = (sigma_t^2)' - 2 (log alpha_t)' sigma_t^2`, and the handy `(1/SNR_t)' = (sigma_t^2/alpha_t^2)' = g^2/alpha_t^2`: for `c_t^2 = sigma_t^2(1 - SNR_T/SNR_t)`,

  c_t'/c_t = (1/2)(log c_t^2)' = (1/2)[(log sigma_t^2)' + (log(1 - SNR_T/SNR_t))'].

I have `(log sigma_t^2)' = g^2/sigma_t^2 + 2f` and `(log(1 - SNR_T/SNR_t))' = -SNR_T (1/SNR_t)'/(1 - SNR_T/SNR_t) = -(g^2/c_t^2)(SNR_T/SNR_t)`. Putting them together and using `SNR_T/SNR_t = 1 - c_t^2/sigma_t^2`,

  c_t'/c_t = f + g^2/sigma_t^2 - g^2/(2 c_t^2)  ... (*)

— which is exactly the coefficient on `x_t` in the PF-ODE. For `a_t = (alpha_t/alpha_T)(SNR_T/SNR_t)`, `a_t'/a_t = f + g^2/sigma_t^2`, so `a_t' - a_t c_t'/c_t = a_t (a_t'/a_t - c_t'/c_t) = a_t [g^2/(2 c_t^2)] = g^2 a_t/(2 c_t^2)` — matches the `x_T` coefficient. For `b_t = alpha_t(1 - SNR_T/SNR_t)`, `b_t'/b_t = f + g^2/sigma_t^2 - g^2/c_t^2`, so `b_t' - b_t c_t'/c_t = b_t [-g^2/(2 c_t^2)] = -g^2 b_t/(2 c_t^2)` — matches the `x_theta` coefficient. All three line up; my clean ODE *is* the bridge PF-ODE, just written in coordinates where it looks trivial. The deterministic sampler isn't an approximation to the PF-ODE — it's an exact reparameterization, and a much friendlier one.

Why is it friendlier, and how do I exploit that for higher order? Because it's a *semi-linear* ODE and I've already isolated the linear part. Look at the `dx_t` form: the term `(c_t'/c_t) x_t` is linear in `x_t`, integrable in closed form; everything hard is in `x_theta`. The lesson from fast diffusion solvers is to never let a generic integrator chew on the linear part — cancel it analytically with variation-of-constants and only approximate the nonlinear `x_theta` integral. The black-box hybrid Heun sampler the bridge ships with is exactly the thing that *doesn't* do this: it treats the whole right-hand side as one opaque vector field, so it pays discretization error on the easy linear part too, and needs tiny steps. I won't.

Variation-of-constants on `dx_t = [a(t) x_t + (driving terms)] dt` with `a(t) = c_t'/c_t` gives `e^{int_t^s a} = e^{int_t^s (log c)'} = c_s/c_t`. Equivalently, integrating the clean ODE for `x/c` directly from time `t` to time `s < t` gives

  x_s = (c_s/c_t) x_t + (a_s - (c_s/c_t) a_t) x_T + c_s int_t^s x_theta(x_tau, tau, x_T) d(b_tau/c_tau),

and the `x_T` integral has collapsed analytically to the `(a_s - (c_s/c_t) a_t) x_T` term. To make the remaining integral as simple as possible I want the right change of variable — the bridge analogue of the log-SNR `lambda` that linearized the diffusion case. Since `d(b_t/c_t) = (b_t/c_t) d log(b_t/c_t)`, the natural variable is

  lambda_t = log(b_t/c_t).

What is this, intuitively? `b_t/c_t = alpha_t(1 - SNR_T/SNR_t)/[sigma_t sqrt(1 - SNR_T/SNR_t)] = (alpha_t/sigma_t) sqrt(1 - SNR_T/SNR_t)`, so `lambda_t = (1/2) log(SNR_t (1 - SNR_T/SNR_t)) = (1/2) log(SNR_t - SNR_T)` — a *bridge log-SNR*, the half-log of the excess signal-to-noise over the endpoint. Quick numeric check on the same `cos/sin` schedule so I haven't dropped a factor: at `t = 0.2, 0.4, 0.6, 0.8` the direct `log(b_t/c_t)` evaluates to `1.1242, 0.3195, -0.3195, -1.1242`, and `(1/2) log(SNR_t - SNR_T)` gives the same to twelve digits. So the closed form is right, and it plays exactly the role `log(alpha/sigma)` played for ordinary diffusion. In this variable the exact solution from `t` to `s < t` is

  x_s = (c_s/c_t) x_t + (a_s - (c_s/c_t) a_t) x_T + c_s int_{lambda_t}^{lambda_s} e^lambda x_theta(x_{t_lambda}, t_lambda, x_T) dlambda,

where `t_lambda` inverts `lambda_t`. Let me double-check the prefactor is right by re-deriving it as an exponential integrator directly, since I want to be sure the `c_s` outside and the `e^lambda` inside are correct and not flipped. Matching `dx = [a(t)x + b_1(t) x_T + b_2(t) x_theta] dt` to my expanded form: `a(t) = (log c_t)'`, `b_2(t) = b_t (log(b_t/c_t))' = b_t lambda_t'`. Variation-of-constants: `x_s = e^{int_t^s a} x_t + int_t^s e^{int_tau^s a} b_2(tau) x_theta dtau + (x_T part)`. The exponential factor is `c_s/c_tau` inside and `c_s/c_t` outside. The integrand's `x_theta` coefficient is `(c_s/c_tau) b_tau lambda_tau' = c_s (b_tau/c_tau) lambda_tau' = c_s e^{lambda_tau} lambda_tau'`, and `lambda_tau' dtau = dlambda`, so it's `c_s e^lambda x_theta dlambda`. Confirmed: `c_s` outside, `e^lambda` inside. The linear bookkeeping is exact — *all* discretization error now lives in approximating `int e^lambda x_theta dlambda`, and nothing else.

Let me check the floor first: if I approximate that integral by holding `x_theta` constant at its value at `t`, the integral is `(int_{lambda_t}^{lambda_s} e^lambda dlambda) x_theta = (e^{lambda_s} - e^{lambda_t}) x_theta`. Since `e^{lambda} = b/c`, that's `(b_s/c_s - b_t/c_t) x0`, and the `c_s` outside makes the `x0` coefficient `c_s(b_s/c_s - b_t/c_t) = b_s - (c_s/c_t) b_t`. The `x_t` and `x_T` coefficients are `c_s/c_t` and `a_s - (c_s/c_t) a_t` from the linear part. Lining those three coefficients up against the deterministic `rho = 0` step I wrote earlier — `b_s - (c_s/c_t) b_t` on `x0`, `c_s/c_t` on `x_t`, `a_s - (c_s/c_t) a_t` on `x_T` — they are identical term for term. So the zeroth-order exponential-integrator step *is* the first-order DBIM step, not merely "like" it; the exponential integrator contains the Euler step as its lowest order, which is exactly the consistency I'd want before trusting the higher-order terms. That's the floor. To go higher I do what exponential integrators always do: Taylor-expand `x_theta` as a function of `lambda` about the current node, and integrate each Taylor term against `e^lambda` exactly. Write `h = lambda_s - lambda_t` for the step in `lambda` (with `s < t`, `lambda` increases as time decreases, so `h > 0`). I need three integrals, and I'll actually do them rather than wave at them.

The zeroth: `int_{lambda_t}^{lambda_s} e^lambda dlambda = e^{lambda_s} - e^{lambda_t} = e^{lambda_s}(1 - e^{-h})`. 

The first, `int_{lambda_t}^{lambda_s} (lambda - lambda_t) e^lambda dlambda`. Integrate by parts with `u = lambda - lambda_t`, `dv = e^lambda dlambda`: `= [(lambda - lambda_t) e^lambda]_{lambda_t}^{lambda_s} - int e^lambda dlambda = h e^{lambda_s} - (e^{lambda_s} - e^{lambda_t}) = e^{lambda_s}(h - 1 + e^{-h})`. 

The second, `int_{lambda_t}^{lambda_s} ((lambda - lambda_t)^2/2) e^lambda dlambda`. By parts with `u = (lambda - lambda_t)^2/2`: `= [(lambda-lambda_t)^2/2 e^lambda]_{lambda_t}^{lambda_s} - int (lambda - lambda_t) e^lambda dlambda = (h^2/2) e^{lambda_s} - e^{lambda_s}(h - 1 + e^{-h}) = e^{lambda_s}(h^2/2 - h + 1 - e^{-h})`. 

So if I expand `x_theta(lambda) ≈ x_hat_t + (lambda - lambda_t) x_hat_t^{(1)} + ((lambda-lambda_t)^2/2) x_hat_t^{(2)}`, where `x_hat_t^{(k)}` is the `k`-th derivative of the predictor's output with respect to `lambda` at the current node, the integral becomes

  int e^lambda x_theta dlambda ≈ e^{lambda_s}[(1 - e^{-h}) x_hat_t + (h - 1 + e^{-h}) x_hat_t^{(1)} + (h^2/2 - h + 1 - e^{-h}) x_hat_t^{(2)}].

These three coefficients are the φ-functions of exponential integrators; I didn't pick them, the integrals handed them to me. The first-order solver keeps just the `(1 - e^{-h})` term; second order adds the derivative term; third order adds the curvature term. Now I only need the derivatives `x_hat_t^{(1)}`, `x_hat_t^{(2)}`, and here is the decision that actually controls the cost.

How do I estimate a derivative of the predictor in `lambda`? Two roads. The single-step road inserts an extra intermediate timestep between `t` and `s`, evaluates the network there, and finite-differences — but that's an *extra* denoiser call per step, so a `k`-th order single-step method costs `k` calls per step and a budget of `N` calls buys only `N/k` steps. The multistep road, Adams-Bashforth style, finite-differences against the predictor outputs I *already computed* at previous timesteps — those are free, sitting in a buffer. Multistep costs exactly one new call per step, so `N` calls buy `N` steps, and each step's `h` is `~1/k` of the single-step `h`, which shrinks the dropped `O(h^{k+1})` error too. Under a tight call budget, multistep is the obvious win. I keep a small buffer of past predictor outputs and use them.

So suppose I'm at the current node `t` and have one previous node `u` (with `s < t < u` in time, so in `lambda` the order is `lambda_s > lambda_t > lambda_u`). A backward finite difference estimates the first derivative:

  x_hat_t^{(1)} ≈ (x_hat_t - x_hat_u)/h_1,  h_1 = lambda_t - lambda_u > 0.

That's just the slope between the two most recent outputs in `lambda`-coordinates. For the second-order solver I plug this into the integral with the `(1 - e^{-h})` and `(h - 1 + e^{-h})` coefficients. For third order I need a second derivative too, so I keep two previous nodes `u_1, u_2` with `h_1 = lambda_t - lambda_{u_1}`, `h_2 = lambda_{u_1} - lambda_{u_2}`, and fit the unique quadratic through the three points `(lambda_t, x_hat_t), (lambda_{u_1}, x_hat_{u_1}), (lambda_{u_2}, x_hat_{u_2})`, reading off its first and second derivatives at `lambda_t`. Let me actually derive those, because the unequal spacings make it easy to get wrong. Put `lambda_t = 0`, `lambda_{u_1} = -h_1`, `lambda_{u_2} = -(h_1 + h_2)`, fit `p(x) = A x^2 + B x + C`. From `p(0) = x_hat_t` we get `C = x_hat_t`. From the two back-points,

  A h_1^2 - B h_1 + x_hat_t = x_hat_{u_1},   A(h_1+h_2)^2 - B(h_1+h_2) + x_hat_t = x_hat_{u_2}.

Let `D_1 = (x_hat_t - x_hat_{u_1})/h_1` (slope of the near interval) and `D_2 = (x_hat_{u_1} - x_hat_{u_2})/h_2` (slope of the far interval). Solving the linear system, the first derivative `p'(0) = B` and the second `p''(0) = 2A` come out as

  x_hat_t^{(1)} ≈ B = [ D_1 (2 h_1 + h_2) - D_2 h_1 ] / (h_1 + h_2),
  x_hat_t^{(2)} ≈ 2A = 2 ( D_1 - D_2 ) / (h_1 + h_2).

Two sanity checks on these before I wire them in. First, the equal-spacing limit `h_1 = h_2 = h`: the `B` formula collapses to `[D_1(2h+h) - D_2 h]/(2h) = (3D_1 - D_2)/2`, the standard second-order backward slope, and `2A` collapses to `2(D_1 - D_2)/(2h) = (D_1 - D_2)/h`, the standard curvature — both the textbook stencils, so the unequal-spacing version is a correct generalization, not a guess. Second, the quadratic *interpolates*: substituting `B`, `2A`, and `C = x_hat_t` back, `p(-h_1)` returns `x_hat_{u_1}` and `p(-(h_1+h_2))` returns `x_hat_{u_2}` (this is forced — three coefficients fixed by three points). With both checks passing I trust the formulas; and the unequal-spacing weights are what keep the estimate consistent on a non-uniform timestep schedule, which I'll have because the first step is forced to be a special small jump.

Now assemble the step. With the integral approximated, the update from `t` to `s` is

  x_s = (c_s/c_t) x_t + (a_s - (c_s/c_t) a_t) x_T + c_s * I_hat,

where `I_hat` is the bracketed φ-combination above. I should pin the indexing the way the actual loop runs, because the timesteps march *downward* `t_max -> 0`, and it's worth being careful that "current" and "next" don't get flipped. In the loop the current node is the larger-time `s_loop = ts[i]` and the target is the smaller-time `t_loop = ts[i+1]`; the network is evaluated at the current node `s_loop`, and I step to `t_loop`. So in loop-coordinates "current" is `s_loop` and "target" is `t_loop`, which is the reverse of the `s < t` labels I used in the derivation — fine, as long as I'm consistent: the step in `lambda` is `h = lambda_{target} - lambda_{current} = lambda_{t_loop} - lambda_{s_loop} > 0` (since the target is at smaller time, larger `lambda`), the prefactor outside the integral is `e^{lambda_{target}}`, the linear factor is `c_{target}/c_{current}`, and the finite-difference spacings are measured from the current node back into the buffer. Same math, just relabeled to the marching direction.

A couple of edge cases the multistep scheme forces on me. Before the loop even starts I need a stochastic boot sample from `T` to the first non-terminal node, because `c_T = 0` makes the deterministic formula singular. Then the first ordinary loop transition still has no trustworthy history in `lambda` — only the boot predictor at the pinned endpoint, with `lambda(T)` effectively singular — so that loop step must also be first order. At the very last step it's prudent to drop back to first order ("lower order final"): the derivative estimates get noisy as `h` shrinks near the end and the predictor's outputs cluster, and a clean low-order finish avoids amplifying that noise into the final image. So the schedule is: boot sample, order-1 first loop transition, order-`k` multistep in the middle, order-1 at the finish, with the buffer of past outputs filling in between.

Let me also be honest about what `rho` (equivalently a stochasticity scale `eta`) buys, since I made the high-order solver deterministic. The `rho = 0` limit is the one that gives me the ODE, so it is the skeleton on which the exponential integrator can be built. Extra stochasticity remains available in the first-order family and can act like a Langevin correction for accumulated discretization error, but the high-order derivation targets the deterministic middle of the trajectory. The booting noise already supplies the irreducible diversity; after that, the solver I am about to write follows the `rho = 0` ODE.

So let me write the sampler that fills the empty transition slot. The first call seeds the trajectory with the booting noise; then a loop that, at each step, evaluates the predictor once at the current node, and depending on how much history is in the buffer and the chosen order, takes a first-, second-, or third-order exponential-integrator step in the bridge log-SNR variable; the buffer of past outputs and their times is rolled forward each step. Everything is grounded in the bridge primitives I already have — `get_abc` for `a_t, b_t, c_t`, `bridge_sample` for the booting step — and `lambda_t = log(b_t/c_t)`.

```python
import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from ddbm.nn import append_dims
from ddbm.random_util import BatchedSeedGenerator


@torch.no_grad()
def sample_dbim_high_order(
    denoiser,
    diffusion,
    x,
    ts,                       # decreasing schedule t_max -> 0
    mask=None,
    order=2,                  # 2 or 3
    lower_order_final=True,
    seed=None,
    **kwargs,
):
    if order not in [2, 3]:
        raise NotImplementedError("Only order 2 or 3 supported")
    x_T = x
    path, pred_x0, nfe = [], [], 0
    ones = x.new_ones([x.shape[0]])
    indices = tqdm(range(len(ts) - 1), disable=(dist.get_rank() != 0))

    # --- Booting step: deterministic step 0 is singular (c_T = 0), so use the
    #     Markovian boundary rho_{N-1}=c_{t_{N-1}} -> sample the bridge at the first
    #     non-terminal node
    #     around the predicted x_0. The drawn noise is the diversity latent. ---
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)      # keep observed pixels fixed
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)   # a*x_T + b*x0 + c*eps
    path.append(x.detach().cpu()); pred_x0.append(x0_hat.detach().cpu()); nfe += 1

    # --- Multistep buffers: last (order-1) predictor outputs and their lambda's ---
    u = diffusion.t_max
    if u == 1.0:
        u -= 5e-5                                       # avoid lambda(T) = -inf at the pin
    u = [u for _ in range(order - 1)]
    xu_hat = [x0_hat.detach().clone() for _ in range(order - 1)]

    for _, i in enumerate(indices):
        s = ts[i]           # current node (larger time)
        t = ts[i + 1]       # target node  (smaller time)

        # ---------- First-order step: first loop transition after boot, or final step ----------
        if (lower_order_final and i + 1 == len(ts) - 1) or (i == 0):
            a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
            tmp = c_t / c_s                              # exp-integrator linear factor c_target/c_current
            x0_hat = denoiser(x, s * ones); nfe += 1
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            x = tmp * x + (b_t - tmp * b_s) * x0_hat + (a_t - tmp * a_s) * x_T   # rho=0 bridge Euler step

        # ---------- Second-order multistep step ----------
        elif order == 2 or i == 1:
            a_u, b_u, c_u = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(u[-1] * ones)]
            a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u, lambda_s, lambda_t = torch.log(b_u / c_u), torch.log(b_s / c_s), torch.log(b_t / c_t)
            x0_hat = denoiser(x, s * ones); nfe += 1
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            h = lambda_t - lambda_s                      # step in lambda toward target, > 0
            h2 = lambda_s - lambda_u                     # spacing to previous node (= h_1 of derivation)
            # I_hat = e^{lambda_target}[(1-e^{-h}) x0 + (e^{-h}+h-1) * dx0],  dx0 = (x0 - x0_prev)/h_1
            integral = torch.exp(lambda_t) * (
                (1 - torch.exp(-h)) * x0_hat + (torch.exp(-h) + h - 1) * (x0_hat - xu_hat[-1]) / h2
            )
            x = x * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        # ---------- Third-order multistep step ----------
        elif order == 3:
            a_u1, b_u1, c_u1 = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(u[-1] * ones)]
            a_u2, b_u2, c_u2 = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(u[-2] * ones)]
            a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u2, lambda_u1, lambda_s, lambda_t = (
                torch.log(b_u2 / c_u2), torch.log(b_u1 / c_u1), torch.log(b_s / c_s), torch.log(b_t / c_t),
            )
            x0_hat = denoiser(x, s * ones); nfe += 1
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            h = lambda_t - lambda_s
            h1 = lambda_s - lambda_u1                     # near spacing
            h2 = lambda_u1 - lambda_u2                    # far spacing
            D1 = (x0_hat - xu_hat[-1]) / h1               # near slope
            D2 = (xu_hat[-1] - xu_hat[-2]) / h2           # far slope
            dx0_hat = (D1 * (2 * h1 + h2) - D2 * h1) / (h1 + h2)     # quadratic-fit first derivative
            d2x0_hat = 2 * (D1 - D2) / (h1 + h2)                     # quadratic-fit second derivative
            integral = torch.exp(lambda_t) * (
                (1 - torch.exp(-h)) * x0_hat
                + (torch.exp(-h) + h - 1) * dx0_hat
                + (h ** 2 / 2 - h + 1 - torch.exp(-h)) * d2x0_hat
            )
            x = x * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        # roll the multistep buffers forward: push current, drop oldest
        u.append(s); u.pop(0)
        xu_hat.append(x0_hat); xu_hat.pop(0)
        path.append(x.detach().cpu()); pred_x0.append(x0_hat.detach().cpu())

    return x, path, nfe, pred_x0, ts, first_noise
```

Let me trace the causal chain back so I trust it end to end. I was stuck simulating a bridge's reverse SDE/ODE in over a hundred denoiser calls while ordinary diffusion samples in ten, and the fast diffusion tools didn't transfer because they assume a Gaussian prior with no endpoint. The escape was the marginal-only fact: the trained predictor depends only on `q(x_t|x_T)`, so I built a `rho`-indexed family of non-Markovian bridges with the same marginals (proved by downward induction, base case forcing `rho_{N-1} = c_{t_{N-1}}`) and the same reweighted objective (so no retraining), and its `rho = 0` limit is a deterministic implicit sampler — the bridge DDIM. The first deterministic step is singular because the bridge is pinned at `T` with `c_T = 0`, which I fixed with the Markovian boundary at step zero, injecting a booting noise that doubles as the diversity latent. Setting `rho = 0` and taking `dt -> 0` revealed a strikingly clean ODE for `x_t/c_t`, which I verified is exactly the bridge PF-ODE by matching all three log-derivative coefficients. Recognizing it as semi-linear, I cancelled the linear part with variation-of-constants, changed variable to the bridge log-SNR `lambda_t = log(b_t/c_t) = (1/2) log(SNR_t - SNR_T)`, and reduced everything to a single exponentially-weighted integral of the predictor. Taylor-expanding the predictor in `lambda` and integrating each term against `e^lambda` by parts gave the exact φ-coefficients `(1-e^{-h})`, `(h-1+e^{-h})`, `(h^2/2-h+1-e^{-h})`; estimating the derivatives by Adams-Bashforth finite differences of *past* predictor outputs gave the second- and third-order multistep solvers, with a stochastic boot sample, an order-1 first loop transition, and an order-1 finish. The result fills the transition slot exactly, reusing the existing bridge predictor and primitives.
