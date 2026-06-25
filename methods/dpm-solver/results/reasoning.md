The thing eating me alive is the wall-clock. I have a diffusion model that makes gorgeous images, and to make one image it runs the network a thousand times in a row. A GAN runs it once. So the entire practical story of these models hinges on one number — how many sequential network evaluations a sample costs — and right now that number is three orders of magnitude too big. I want to drive it down to about ten, without retraining anything, because the models are expensive and I'd like whatever I build to drop onto any pre-trained network as-is.

Let me get precise about why it's slow, because "it takes many steps" is not a reason, it's a symptom. Sampling is reversing a noising process. The forward process takes data x_0 and, at time t, gives a noisy version distributed as N(α_t x_0, σ_t² I); the pair (α_t, σ_t) is the noise schedule, and the signal-to-noise ratio α_t²/σ_t² falls monotonically from t=0 (clean) to t=T (basically pure noise). The network ε_θ(x_t, t) was trained to predict the noise that was added — equivalently, up to the factor −σ_t, the score ∇_x log q_t(x_t). To sample, I start at noise and undo the process down to t=0.

There are two ways to undo it. One is the reverse-time SDE, and its first-order discretization is exactly the usual ancestral sampler — that's the thousand-step thing. Discretizing an SDE is intrinsically painful: each step injects fresh Brownian noise, and the step size you can get away with is capped by that randomness; push the step size up in high dimension and the thing just doesn't converge. So the SDE is the wrong horse if I want big steps. The other way is deterministic: the probability-flow ODE, which has the same marginal distribution at every time t as the SDE but no noise term at all,

  dx_t/dt = f(t) x_t + (g²(t) / 2σ_t) ε_θ(x_t, t),

solved from T down to 0, where f(t) = d log α_t/dt and g²(t) = dσ_t²/dt − 2(d log α_t/dt)σ_t². The only randomness is now the initial noise sample; the trajectory is an honest deterministic curve. So sampling is solving an ODE, and "make it fast" becomes "solve this ODE accurately in few steps." Once I have that reframing the question is just which solver.

So just throw a good ODE solver at it. People have: take the high-order adaptive Runge–Kutta pair, RK45, hand it the right-hand side, and it gets to good quality in around sixty evaluations instead of a thousand. Big win. But sixty is not ten, and when I shrink the budget toward ten the generic solver falls apart — the samples are garbage. Why? A black-box solver is built to be agnostic; it knows nothing about the function it's integrating, treats the entire right-hand side as one opaque vector field, and spends its error budget uniformly across it. That agnosticism is exactly what I should attack. The right-hand side is not generic. Stare at it.

  dx/dt = f(t) x  +  (g²(t)/2σ_t) ε_θ(x, t).

The first term, f(t) x, is *linear* in x. The second term is the network — genuinely nonlinear, genuinely a black box. So this is a semi-linear ODE: a simple exactly-solvable linear part welded to a hard nonlinear part. And a black-box RK solver, by lumping them, commits discretization error on *both* — including on the linear part, which I could have solved in closed form. That's the waste. Worse than waste: the exact solution of a linear ODE x' = f(t)x is an exponential, e^{∫f}, and when you discretize an exponential with a finite-difference scheme the error can blow up multiplicatively as the step grows. There's a whole folklore in numerical analysis that explicit RK schemes go unstable on semi-linear problems at large step size for exactly this reason. So the diagnosis is sharp: half of my ODE is linear and solvable exactly, and the generic solver is throwing away accuracy approximating something it didn't need to approximate. So the thing to try is: solve the linear part exactly, and spend the entire approximation budget only on the network term. Whether that actually buys order is something I'll have to check, not assume — but it's the obvious first move.

How do I solve the linear part exactly when there's a nonlinear forcing term sitting next to it? This is the textbook variation-of-constants situation. Write the ODE as x' = f(t)x + b(t) where b(t) := (g²(t)/2σ_t) ε_θ(x_t, t) — yes b depends on x through the network, but for the integrating-factor manipulation I just treat the whole second term as an inhomogeneous forcing and carry it along. Multiply by the integrating factor e^{−∫_0^t f(r)dr}:

  d/dt [ e^{−∫_0^t f} x_t ] = e^{−∫_0^t f} ( x_t' − f x_t ) = e^{−∫_0^t f} b(t).

Integrate from s to t:

  e^{−∫_0^t f} x_t − e^{−∫_0^s f} x_s = ∫_s^t e^{−∫_0^τ f} b(τ) dτ,

so, multiplying back by e^{∫_0^t f},

  x_t = e^{∫_s^t f(r)dr} x_s + ∫_s^t e^{∫_τ^t f(r)dr} b(τ) dτ.

Now use what f actually is: ∫_s^t f(r)dr = ∫_s^t d log α_r = log α_t − log α_s, so e^{∫_s^t f} = α_t/α_s. The linear part is now exact:

  x_t = (α_t/α_s) x_s + ∫_s^t (α_t/α_τ) (g²(τ)/2σ_τ) ε_θ(x_τ, τ) dτ.

The first term carries the entire linear dynamics with zero discretization error. Everything I'll ever approximate is now sealed inside that single integral. That already feels right — but the integral is ugly. It tangles the schedule coefficients f, g, σ together with the network. I need to clean it before I can approximate it well.

Let me massage the integrand. The g² term is the irritant. Rewrite it:

  g²(t) = dσ_t²/dt − 2 (d log α_t/dt) σ_t² = 2σ_t² ( d log σ_t/dt − d log α_t/dt ),

using dσ_t²/dt = 2σ_t² · (d log σ_t/dt). Now the combination d log σ_t/dt − d log α_t/dt is the derivative of log σ_t − log α_t = −log(α_t/σ_t). The quantity log(α_t/σ_t) is one half of the log-SNR; SNR is monotone in t, so this is a clean monotone function of t. Name it λ_t := log(α_t/σ_t). Then d log σ_t/dt − d log α_t/dt = −dλ_t/dt, and

  g²(t) = −2σ_t² dλ_t/dt.

Before I lean on this rewrite I should make sure I didn't drop a factor — it's the load-bearing identity for everything downstream. Take a concrete variance-preserving schedule, the standard continuous one with log α_t = −¼ β_d t² − ½ β_min t (β_d = 19.9, β_min = 0.1) and σ_t = √(1 − α_t²). Compute g²(t) two ways: directly from g²(t) = dσ_t²/dt − 2 f(t) σ_t² with f = d log α_t/dt, and from the rewrite −2σ_t² dλ_t/dt, evaluating the derivatives numerically. At t = 0.3 both give 6.0700; at t = 0.7 both give 14.030; at t = 1.0 both give 20.000. They agree to all the digits I printed, so the identity holds and no factor went missing. Substitute it into the integrand's coefficient:

  (α_t/α_τ)(g²(τ)/2σ_τ) = (α_t/α_τ) · ( −2σ_τ² (dλ_τ/dτ) / 2σ_τ ) = −α_t (σ_τ/α_τ) (dλ_τ/dτ).

So

  x_t = (α_t/α_s) x_s − α_t ∫_s^t (dλ_τ/dτ) (σ_τ/α_τ) ε_θ(x_τ, τ) dτ.

Two things just happened. The coefficient σ_τ/α_τ is exactly e^{−λ_τ}, by definition of λ. And there's a dλ_τ/dτ sitting inside the integral, begging me to change the integration variable from τ to λ. Do it. Since λ is strictly monotone in t it has an inverse t = t_λ(λ); let me write x̂_λ for x_{t_λ(λ)} and ε̂_θ(x̂_λ, λ) for ε_θ evaluated along that reparameterized trajectory. The substitution dλ = (dλ_τ/dτ)dτ collapses the messy product into

  x_t = (α_t/α_s) x_s − α_t ∫_{λ_s}^{λ_t} e^{−λ} ε̂_θ(x̂_λ, λ) dλ.

Stop and look at this. Every trace of the noise schedule — f, g, σ_t, all the schedule-specific bookkeeping — has evaporated from inside the integral and been replaced by the single analytic factor e^{−λ}. What's left to approximate is

  ∫_{λ_s}^{λ_t} e^{−λ} ε̂_θ(x̂_λ, λ) dλ,

an integral of (a known exponential) times (the network), in the variable λ. The prefactors α_t/α_s and α_t are exact, computed from the endpoints alone. In fact the whole solution between two times depends only on λ_s, λ_t and the function ε̂_θ — it is invariant to whatever schedule you used to get between them. The only thing I will ever approximate is the integral of the network against e^{−λ}. That is a dramatic reduction in what's left to do, and an exponentially-weighted integral like this is precisely the object the exponential-integrator literature knows how to handle. Defining λ paid off twice here: it's the variable that turns f and g analytic and leaves only the network.

Now approximate the integral. I'm marching in steps; at step i I'm at λ_{t_{i-1}} with a value x̃_{t_{i-1}}, and I want x̃_{t_i} at λ_{t_i}. Write h_i := λ_{t_i} − λ_{t_{i-1}} for the step in λ. The only unknown is ε̂_θ along the interval, and I have its value at the left endpoint. So Taylor-expand ε̂_θ in λ around λ_{t_{i-1}}:

  ε̂_θ(x̂_λ, λ) = Σ_{n=0}^{k-1} ( (λ − λ_{t_{i-1}})^n / n! ) ε̂_θ^{(n)}(x̂_{λ_{t_{i-1}}}, λ_{t_{i-1}}) + O((λ − λ_{t_{i-1}})^k),

where ε̂_θ^{(n)} is the n-th total derivative w.r.t. λ. Plug into the exact step solution

  x_{t_{i-1}→t_i} = (α_{t_i}/α_{t_{i-1}}) x̃_{t_{i-1}} − α_{t_i} ∫_{λ_{t_{i-1}}}^{λ_{t_i}} e^{−λ} ε̂_θ dλ,

and the integral splits into a sum of terms each of the form ε̂_θ^{(n)} times the scalar integral ∫ e^{−λ} (λ − λ_{t_{i-1}})^n / n! dλ:

  x_{t_{i-1}→t_i} = (α_{t_i}/α_{t_{i-1}}) x̃_{t_{i-1}} − α_{t_i} Σ_{n=0}^{k-1} ε̂_θ^{(n)} ∫_{λ_{t_{i-1}}}^{λ_{t_i}} e^{−λ} ((λ − λ_{t_{i-1}})^n/n!) dλ + O(h_i^{k+1}).

Those scalar integrals are elementary — integrate by parts n times and they close in elementary functions. Let me set up the standard book-keeping for them. Substitute λ = λ_{t_{i-1}} + δh, and remember that λ_t is the right endpoint of the step. The exponential-integrator people define

  φ_k(z) := ∫_0^1 e^{(1−δ)z} δ^{k-1}/(k-1)! dδ,   φ_0(z) = e^z,

with the convenient facts φ_k(0) = 1/k! and the recurrence φ_{k+1}(z) = (φ_k(z) − φ_k(0))/z. Carrying the substitution through, the integral against e^{−λ} of the n-th Taylor monomial produces e^{−λ_t}h^{n+1}φ_{n+1}(h). Concretely the whole expansion lands as

  x_t = (α_t/α_s) x_s − σ_t Σ_{k=0}^{n} h^{k+1} φ_{k+1}(h) ε̂_θ^{(k)}(x̂_{λ_s}, λ_s) + O(h^{n+2}),

where the α_t in front turned into σ_t because α_t · e^{−λ_t} = α_t · (σ_t/α_t) = σ_t — the endpoint factor from integrating e^{−λ} gets absorbed and converts α_t into σ_t. The first three φ's are

  φ_1(h) = (e^h − 1)/h,
  φ_2(h) = (e^h − h − 1)/h²,
  φ_3(h) = (e^h − h²/2 − h − 1)/h³,

which I can read off from φ_1(h) = ∫_0^1 e^{(1−δ)h}dδ = (e^h−1)/h and then the recurrence. Two quick sanity checks I can do in my head and on paper before I trust these. First, φ_k(0) = 1/k!: at small h, φ_1 → 1, φ_2 → 1/2, φ_3 → 1/6. Plugging h = 0.01 in numerically gives φ_1 = 1.00502, φ_2 = 0.50167, φ_3 = 0.16708 — i.e. 1, 0.5, 0.1667 to the precision the step allows, so the constants are right. Second, the recurrence: at z = 0.4, (φ_1(z) − 1)/z = 0.573904 and φ_2(z) computed directly is 0.573904; (φ_2(z) − 1/2)/z = 0.184761 and φ_3(z) directly is 0.184761. They match, so my closed forms and the recurrence are mutually consistent. So the approximation comes in orders: keep the first k terms of the Taylor sum and I get a k-th order method. I only ever need to estimate the first few total derivatives of the network in λ — and there's a clean way to do that with extra function evaluations, exactly as in exponential Runge–Kutta.

Take k=1 first, the bare bones: keep only the n=0 term, drop O(h²). The single integral is ∫_{λ_s}^{λ_t} e^{−λ}dλ = e^{−λ_s} − e^{−λ_t}. Then

  α_t (e^{−λ_s} − e^{−λ_t})

let me be careful with signs. e^{−λ_s} = e^{−λ_t} e^{−(λ_s−λ_t)} = e^{−λ_t} e^{h} since h = λ_t − λ_s. So e^{−λ_s} − e^{−λ_t} = e^{−λ_t}(e^h − 1), and α_t e^{−λ_t} = α_t (σ_t/α_t) = σ_t. Therefore α_t ∫ e^{−λ}dλ = σ_t(e^h − 1), and the step is

  x̃_{t_i} = (α_{t_i}/α_{t_{i-1}}) x̃_{t_{i-1}} − σ_{t_i}(e^{h_i} − 1) ε_θ(x̃_{t_{i-1}}, t_{i-1}),   h_i = λ_{t_i} − λ_{t_{i-1}}.

That's an order-1 step: one network call, the linear part exact, the network frozen at the left endpoint. Clean. But I claimed exact-linear-part buys accuracy and I haven't actually checked that — let me put a number on it before going further. On the same VP schedule, take a scalar toy network ε(x,t) = tanh(0.5x + 0.3t) + 0.1t (smooth, mildly nonlinear, so it has honest λ-derivatives) and integrate the true ODE from s=1.0 to a nearby t with a 20000-step RK4 to get ground truth. The order-1 step's error against ground truth, as I halve the step in λ: h=0.488 → err 1.26e−2; h=0.247 → err 2.82e−3; h=0.124 → err 6.67e−4. Each halving of h cuts the error by ≈4.5× then 4.2×, i.e. error ∝ h². A local error of O(h²) is exactly what a first-order method should give per step. So the construction does what I hoped — the bare scheme is genuinely first order, and the exact linear part is carrying its weight.

Now something nags at me. There is already a fast deterministic sampler people use — DDIM — and its single step from t_{i-1} to t_i is

  x̃_{t_i} = (α_{t_i}/α_{t_{i-1}}) x̃_{t_{i-1}} − α_{t_i}( σ_{t_{i-1}}/α_{t_{i-1}} − σ_{t_i}/α_{t_i} ) ε_θ(x̃_{t_{i-1}}, t_{i-1}).

It was derived from a completely different story — a non-Markovian forward process engineered to share the DDPM marginals — and has nothing to do with semi-linear ODEs on its face. But the leading α_{t_i}/α_{t_{i-1}} factor is suspiciously identical to mine. Let me push it. The coefficients σ_{t_{i-1}}/α_{t_{i-1}} and σ_{t_i}/α_{t_i} are just e^{−λ_{t_{i-1}}} and e^{−λ_{t_i}} by the definition of λ. So the DDIM correction term is

  −α_{t_i}( e^{−λ_{t_{i-1}}} − e^{−λ_{t_i}} ) ε_θ = −α_{t_i} e^{−λ_{t_i}}( e^{λ_{t_i}−λ_{t_{i-1}}} − 1 ) ε_θ = −α_{t_i} e^{−λ_{t_i}}(e^{h_i} − 1) ε_θ = −σ_{t_i}(e^{h_i} − 1) ε_θ,

since α_{t_i} e^{−λ_{t_i}} = σ_{t_i}. That is the same as my order-1 step. To make sure I haven't talked myself into an algebraic mirage, I evaluate both update formulas numerically on the toy schedule from x=1.3, s=1.0: to t=0.9 the order-1 step gives 2.0299935878 and DDIM gives 2.0299935878; to t=0.5 both give 20.4982886093. Bit-identical, not just close. So DDIM *is* the first-order member of this family. The deterministic sampler everyone already trusts has been silently exploiting the semi-linearity all along — it solves the linear part exactly, which is why it always beat the plain Euler discretization of the ODE that people couldn't quite explain. And the payoff for me is the reverse direction: this formulation isn't just a re-derivation of DDIM, it tells me how to go to higher order, which DDIM's non-Markovian story never could. That's the opening — push k up.

k=2. Now I need the first total derivative ε̂_θ^{(1)} as well, and I don't have it for free; I have to manufacture it with an extra network evaluation at an intermediate point, the way explicit exponential Runge–Kutta does. Put one intermediate node inside the step at λ_s + r_1 h (a fraction r_1 of the way across in λ). First take an order-1 sub-step to that node to get a predictor u_i there:

  u_i = (α_{s_i}/α_{t_{i-1}}) x̃_{t_{i-1}} − σ_{s_i}(e^{r_1 h_i} − 1) ε_θ(x̃_{t_{i-1}}, t_{i-1}),   s_i = t_λ(λ_{t_{i-1}} + r_1 h_i).

Evaluate the network there, ε_θ(u_i, s_i), and form the finite difference against the left endpoint, which is a first-order estimate of the derivative I'm missing: ε_θ(u_i, s_i) − ε_θ(x̃_{t_{i-1}}, t_{i-1}) ≈ (λ_{s_i} − λ_s) ε̂_θ^{(1)} = r_1 h ε̂_θ^{(1)}. Now build the full step keeping n=0 and n=1 in the φ-expansion,

  x_t = (α_t/α_s)x_s − σ_t h φ_1(h) ε_θ(x_s,s) − σ_t h² φ_2(h) ε̂_θ^{(1)} + O(h³),

and replace ε̂_θ^{(1)} by the finite difference divided by r_1 h. The cleanest assembly that matches the order-2 condition is

  x̃_{t_i} = (α_{t_i}/α_{t_{i-1}}) x̃_{t_{i-1}} − σ_{t_i}(e^{h_i} − 1) ε_θ(x̃_{t_{i-1}}, t_{i-1}) − (σ_{t_i}/(2 r_1))(e^{h_i} − 1)( ε_θ(u_i, s_i) − ε_θ(x̃_{t_{i-1}}, t_{i-1}) ),

since h φ_1(h) = e^h − 1, so the leading correction reuses (e^h − 1). With r_1 = 1/2 the coefficient 1/(2r_1) becomes 1 — a natural default, the midpoint in λ. The constant 1/(2r_1) in front of the finite difference has to be exactly right or I silently lose the order, so I'm going to verify, not assert. The exact solution to order h² is x_t = (α_t/α_s)x_s − σ_t h φ_1(h)ε_θ(x_s,s) − σ_t h²φ_2(h)ε̂_θ^{(1)} + O(h³). My approximation replaces the second-derivative-term factor ε̂_θ^{(1)} by the finite difference, contributing −(σ_t/(2r_1))(e^h−1)(λ_{s_1}−λ_s)ε̂_θ^{(1)} + O(h³), and λ_{s_1} − λ_s = r_1 h. So the difference between exact and approximate, to the relevant order, is

  σ_t [ h² φ_2(h) − (e^h − 1)(r_1 h)/(2 r_1) ] ε̂_θ^{(1)} = σ_t [ h² φ_2(h) − (e^h − 1)h/2 ] ε̂_θ^{(1)}.

Now h²φ_2(h) = e^h − h − 1, so the bracket is

  e^h − h − 1 − (e^h − 1)h/2.

Expand for small h. e^h = 1 + h + h²/2 + h³/6 + …, so e^h − h − 1 = h²/2 + h³/6 + O(h⁴), and (e^h − 1)h/2 = (h + h²/2 + h³/6 + …)·h/2 = h²/2 + h³/4 + O(h⁴). Subtracting: (h²/2 − h²/2) + (h³/6 − h³/4) + O(h⁴) = (2/12 − 3/12)h³ + O(h⁴) = −h³/12 + O(h⁴). So the bracket is −h³/12, not anything larger — I want to flag this to myself because it's easy to misread: the *numerator* 2e^h − h − 2 − he^h that you get from putting the bracket over 2 expands as −h³/6, but the bracket itself is half that, −h³/12. Either way the bracket is O(h³), so the local error from the finite-difference substitution is O(h³). That makes the order-2 step genuinely second order, and the 1/(2r_1) was load-bearing — a different constant would leave an O(h²) bracket and kill the order.

I'll cross-check that numerically rather than trust the small-h algebra alone, because a sign or factor slip in the assembly wouldn't show up in the bracket calculation. Same VP schedule, same toy network, ground truth from a 40000-step RK4, the order-2 step from s=1.0: h=0.488 → err 1.18e−3; h=0.247 → err 1.24e−4; h=0.124 → err 1.42e−5; h=0.0623 → err 1.70e−6. Each halving of h drops the error by ≈9.5×, 8.8×, 8.4× — converging to the factor of 8 that O(h³) local error demands. So the assembled formula, signs and all, really is second order. (And note this is much better than the order-1 errors at the same h, which is the whole point of paying for one extra evaluation.)

k=3. Same idea, now I need ε̂_θ^{(1)} and ε̂_θ^{(2)}, so two intermediate nodes, at r_1 and r_2 of the way across in λ. Predict u at the first node by a first-order sub-step, take a difference D_1 = ε_θ(u_1,s_1) − ε_θ(x_s,s); predict u_2 at the second node using a second-order sub-step that already incorporates D_1 (so u_2 is accurate to O(h³)), take D_2 = ε_θ(u_2,s_2) − ε_θ(x_s,s); then assemble the full step from ε_θ(x_s,s) and D_2:

  s_1 = t_λ(λ_s + r_1 h),  s_2 = t_λ(λ_s + r_2 h),
  u_1 = (α_{s_1}/α_s) x_s − σ_{s_1}(e^{r_1 h} − 1) ε_θ(x_s,s),
  D_1 = ε_θ(u_1,s_1) − ε_θ(x_s,s),
  u_2 = (α_{s_2}/α_s) x_s − σ_{s_2}(e^{r_2 h} − 1) ε_θ(x_s,s) − (σ_{s_2} r_2/r_1)( (e^{r_2 h} − 1)/(r_2 h) − 1 ) D_1,
  D_2 = ε_θ(u_2,s_2) − ε_θ(x_s,s),
  x̃_t = (α_t/α_s) x_s − σ_t(e^h − 1) ε_θ(x_s,s) − (σ_t/r_2)( (e^h − 1)/h − 1 ) D_2.

Why those particular coefficients ((e^{r_2 h}−1)/(r_2 h) − 1 and (e^h−1)/h − 1)? They are the φ_2-type quantities: h φ_1(h) = e^h − 1 gives the first correction, and h² φ_2(h) = e^h − h − 1 means φ_2(h) = (e^h − h − 1)/h² = ((e^h−1)/h − 1)/h, so the factor ((e^h−1)/h − 1) is h φ_2(h) up to the h, exactly what multiplies the first derivative. Now I have two free knots r_1, r_2 and I should pin them by the order condition rather than guess. The exact expansion to O(h⁴) is

  x_t = (α_t/α_s)x_s − σ_t h φ_1(h)ε_θ − σ_t h²φ_2(h)ε̂^{(1)} − σ_t h³φ_3(h)ε̂^{(2)} + O(h⁴).

Substitute D_2 = ε̂^{(1)}(r_2 h) + ½ε̂^{(2)}(r_2 h)² + O(h³) (its Taylor expansion at the left endpoint) into my assembled step. The ε̂^{(1)} coefficient comes out −σ_t·(1/r_2)((e^h−1)/h − 1)·r_2 h = −σ_t((e^h−1) − h) = −σ_t h²φ_2(h), which matches the exact one for any r_2 — good, the first-derivative term is automatically right. The ε̂^{(2)} coefficient is −σ_t·(1/r_2)((e^h−1)/h − 1)·½ r_2² h² = −σ_t·½ r_2 h²·((e^h−1)/h − 1). I need this to equal −σ_t h³φ_3(h) to leading order. Now h³φ_3(h) = e^h − h²/2 − h − 1 = h³/6 + O(h⁴), and ½ r_2 h²((e^h−1)/h − 1) = ½ r_2 h²(h/2 + O(h²)) = (r_2/4)h³ + O(h⁴). Setting r_2/4 = 1/6 gives r_2 = 2/3. Let me confirm that arithmetic numerically: with r_2 = 2/3, compare h³φ_3(h) against ½ r_2 h²((e^h−1)/h − 1): at h=0.3, 4.859e−3 vs 4.986e−3; at h=0.1, 1.709e−4 vs 1.724e−4; at h=0.03, 4.534e−6 vs 4.545e−6 — ratios 1.026, 1.008, 1.003, converging to 1 as h→0, which is exactly "equal to leading order." So r_2 = 2/3 is the value that makes the second-derivative term hit the φ_3 coefficient and lift the method to third order; r_1 stays free and I take r_1 = 1/3 so the two knots sit evenly at 1/3 and 2/3 across the step in λ. Local error O(h⁴), so the order-3 step is third order.

That settles the structure: an order-k step costs k network evaluations and is order k, for k = 1, 2, 3. Call the family DPM-Solver, and DPM-Solver-k for the order-k member. Why stop at 3? Because going to order 4 in the exponential-Runge–Kutta world needs substantially more intermediate stages and the bookkeeping balloons; the marginal return in this few-step regime isn't worth it. Three orders is the sweet spot.

A couple of design choices left. First, how to lay out the time steps. The solution between two times depends only on λ_s, λ_t and the network — it's invariant to the schedule in between — so the honest variable to grid is λ, not t. The simplest choice is uniform in λ: λ_{t_i} = λ_T + (i/M)(λ_0 − λ_T). This is deliberately *not* uniform-in-t, which is what the old samplers used; spacing uniformly in the half-log-SNR is the natural grid given that all the schedule complexity has been pushed into the analytic e^{−λ}. To get from a chosen λ_i back to the time t_i the network wants as input, I invert λ — and for the standard variance-preserving schedules (linear, cosine) the inverse t_λ has a closed form, so that conversion is free. For a tighter step controller I can also borrow the adaptive idea of comparing a lower- and higher-order estimate (e.g. DPM-Solver-2 against DPM-Solver-3) and sizing the next λ-step from their discrepancy, but uniform-λ already samples well in a handful of steps.

Second, when the network was trained at a discrete set of time indices rather than continuously, I just wrap it: map a continuous t to the (possibly non-integer) discrete index the model expects and call it. The smooth time embeddings tolerate the non-integer input, so a discrete-time model also becomes a continuous ε_θ I can solve with. And to use a fixed evaluation budget K well, I take as many order-3 steps as fit and finish with a single order-2 or order-1 step to land exactly on K calls — order-3 steps are the most accurate per network call (the error-scaling numbers above bear that out), so I want as many of them as the budget allows.

One numerical nit before code: I keep computing e^h − 1 with h small, where naive exp(h) − 1 loses precision catastrophically. Use expm1(h) instead — it's built for exactly this.

Now the code should mirror the derivation without silently switching algorithms. I keep the noise schedule object responsible for log α_t, σ_t, λ_t, and t_λ, including the discrete-time interpolation case and the linear continuous-time closed form. I wrap every trained model type into a noise-prediction function, because this original branch approximates an integral of ε_θ; if I instead convert to a data-prediction function I have moved into the later ++ branch with different signs and different propagation factors.

Inside the solver, I name the branch explicitly as `algorithm_type="dpmsolver"`. Then the first-order update uses `phi_1 = expm1(h)` and returns `(α_t/α_s)x - σ_t phi_1 ε_s`. The second-order singlestep update evaluates the first-order predictor at `λ_s + r_1 h` and, for the recommended `solver_type="dpmsolver"`, returns the base first-order step minus `(0.5/r_1)σ_t phi_1(ε_{s1}-ε_s)`. With `r_1=1/2` this collapses to the simple midpoint-looking formula that uses the intermediate noise prediction in the whole nonlinear correction.

The third-order singlestep update uses `r_1=1/3`, `r_2=2/3`, `phi_22 = expm1(r_2 h)/(r_2 h)-1`, and `phi_2 = expm1(h)/h-1`. Its second intermediate point subtracts `(r_2/r_1)σ_{s2}phi_22(ε_{s1}-ε_s)`, and its final correction subtracts `(1/r_2)σ_t phi_2(ε_{s2}-ε_s)`. Those signs are exactly the noise-prediction branch; the data-prediction branch would have `expm1(-h)`, `σ_t/σ_s`, and opposite-looking correction signs.

For a fixed NFE budget I combine orders the same way as the implementation: if `steps % 3 == 0`, use third-order steps, then one second-order step and one first-order step; if the remainder is 1, finish with one first-order step; if the remainder is 2, finish with one second-order step. The time grid I use is uniform in log-SNR, implemented as uniform λ followed by `inverse_lambda`, although the maintained implementation also exposes uniform-time and quadratic-time schedules for practical tuning.

The causal chain, end to end: sampling a DPM is solving its probability-flow ODE; that ODE is semi-linear, so variation of constants solves the linear half exactly and seals every approximation inside one integral; rewriting g² as −2σ²(dλ/dt) (an identity I checked numerically) and changing the integration variable to the half-log-SNR λ collapses all the schedule coefficients into a single analytic e^{−λ}, leaving an exponentially-weighted integral of the network alone; Taylor-expanding the network in λ and integrating against e^{−λ} term by term yields the φ-functions and an order-k method, DPM-Solver-k; order 1 is exactly DDIM (verified bit-identical), revealing why DDIM beat plain Euler; orders 2 and 3 add one and two intermediate evaluations whose finite differences recover the λ-derivatives, with r_1 = 1/2 and r_2 = 2/3 pinned by the order conditions and confirmed by the h²/h³ error scaling; gridding uniformly in λ and spending the budget on order-3 steps gives high-quality samples in around ten network evaluations, training-free.
