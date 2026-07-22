Let me start from the thing that actually bugs me. I have two functions on the hypercube that look almost identical — `h(z) = z1 z2 z3` and `h(z) = z1 + z1 z2 + z1 z2 z3` — both depending on only three of the `d` coordinates, both containing a degree-3 term, both hopeless for any fixed-feature method because that degree-3 term forces `Omega(d^3)` samples. And yet I have this nagging sense, from watching networks train, that SGD treats them completely differently. The staircase one gets learned; the bare monomial just sits there. If that's real, then there's some structural property of the target that decides whether plain SGD on a plain two-layer network can find it, and I want to name that property exactly — necessary and sufficient — not just exhibit one example that works.

So what do I actually have to work with. A two-layer network and one-pass batch-SGD. The first decision is the parametrization, and it matters more than it looks. If I write the output with the `1/sqrt(N)` normalization, then as the width grows the weights barely move during training: the network linearizes around its initialization and becomes a fixed kernel machine, its feature map frozen before it ever sees which coordinates are relevant. That's exactly the regime I want to *escape*, because a frozen feature map can't adapt to the unknown latent subset `I` — it'll pay `d^3` for either of my two functions and won't tell them apart. So I take the other scaling,

```
fhat(x; Theta) = (1/N) sum_{j=1}^N a_j sigma(<w_j, x>),   Theta = (a_j, w_j)_{j in [N]}.
```

With the `1/N` out front, each neuron contributes `O(1/N)` to the output but the weights are free to travel an `O(1)` distance, so the dynamics stays genuinely nonlinear. This is the regime where features can move toward `I`. Train it by square loss, one-pass batch-SGD with batch `b`, possibly different step sizes for the two layers and a little `ell_2` regularization:

```
theta_j^{k+1} = theta_j^k + (1/b) sum_{i in [b]} { y_{ki} - fhat(x_{ki}; Theta^k) } H_k grad_theta [ a_j sigma(<w_j, x_{ki}>) ] - H_k Lambda theta_j^k,
```

with `H_k = diag(eta^a_k, eta^w_k I_d)` and `Lambda = diag(lambda^a, lambda^w I_d)`. Fresh samples every step, so `n = b * (#steps)`.

Now, the network has `N` neurons, but they're exchangeable — the output only sees them through their empirical distribution. So instead of chasing `N(d+1)` coordinates let me track the measure `rhohat^{(N)} = (1/N) sum_j delta_{theta_j}`. The population risk is a functional of that measure alone: expanding `R(rho) = E_x[(f*(x) - fhat(x;rho))^2]` I get a constant `R_# = E[y^2]`, a linear term `-2 int E[y sigma(<w,x>)] a rho`, and a quadratic interaction `int E[sigma(<w,x>) sigma(<w',x>)] a a' rho rho`. The point is it depends on `rho`, not on a labeling of neurons. And the known result here is that in the wide-and-small-step limit one-pass SGD on this object is a Wasserstein gradient flow: the empirical measure converges to `rho_t` solving a continuity equation,

```
partial_t rho_t = nabla_theta . ( rho_t H(t) nabla_theta psi(theta; rho_t) ),
psi(theta; rho_t) = a E_x[ { fhat(x;rho_t) - f*(x) } sigma(<w,x>) ] + (1/2) theta^T Lambda theta,
```

with `H(t) = diag(xi^a(t), xi^w(t) I_d)`. The neurons are a gas of particles descending the risk while conserving mass locally. Good — this is a clean object, but it still lives in `R^{d+1}`, and `d` is enormous. I haven't used sparsity yet.

Here's where I want to lean on the structure of `f*`. The target is `f*(x) = h*(z)` with `z = x_I` the `P` signal coordinates and the rest pure noise. Let me split every input `x = (z, r)` and every weight `w = (u, v)`, with `u in R^P` aligned to the signal and `v in R^{d-P}` aligned to the noise `r`. At initialization the coordinates of `w^0` are iid and symmetric, so flipping any sign of the noise block leaves the initial distribution unchanged, and that means `fhat(x; rho_0)` doesn't actually depend on `r` — averaging over the symmetric `v^0` washes the noise part out. Does that survive training? Let me check whether the PDE preserves it. Take the pushforward `rho_t^{#r}` of `rho_t` under the map that flips `v` by a fixed sign pattern `r`. At `t=0` it equals `rho_0` by the symmetry I just noted. Differentiating, the potential transforms as `psi(phi_r(theta); rho_t) = psi(theta; rho_t^{#r})` because the gradient flow only couples weights through `<w,x>` and I can absorb the flip into the `x`-average — so `rho_t^{#r}` solves the very same PDE with the same initial condition. By uniqueness of the solution, `rho_t^{#r} = rho_t` for all `t`. So the network stays independent of the irrelevant directions throughout training:

```
fhat(x; rho_t) = int a^t E_r[ sigma(<u^t, z> + <v^t, r>) ] rho_t(dtheta^t) =: fhat(z; rho_t).
```

That's a relief, but `<v^t, r>` still drags in the `d - P` noise coordinates. Now `r ~ Unif({+1,-1}^{d-P})` and `<v^t, r>` is a sum of `d - P` independent bounded terms, so for `d` large it's approximately Gaussian: `<v^t, r> ~ ||v^t||_2 G`, `G ~ N(0,1)`, with the error controlled by `max_i |v_i^t| / ||v^t||_2`. At `t=0` that ratio is tiny because the `v_i^0` are iid sub-Gaussian (Berry-Esseen gives the bound), and it stays small up to times of order one. So the only thing the noise block contributes is a single scalar — its norm `s^t := ||v^t||_2` — acting as a Gaussian smoothing width. I can collapse the whole `(d-P)`-dimensional noise weight into one parameter. Introduce effective parameters `thetabar^t = (abar^t, ubar^t, sbar^t) in R^{P+2}` and an effective network

```
fhat(z; rhobar_t) = int abar^t E_G[ sigma(<ubar^t, z> + sbar^t G) ] rhobar_t(dthetabar^t).
```

This is a two-layer network in dimension `P` with an adaptive Gaussian smoothing — and `P` is fixed, free of `d`. Writing out the evolution of `(a^t, u^t, ||v^t||_2)` under the mean-field PDE and taking `d -> infinity`, replacing `<v^t, r>` everywhere by `sbar^t G`, I get a closed dynamics on `P(R^{P+2})`:

```
d/dt abar^t = xi^a(t) E_{z,G}[ (f*(z) - fhat(z;rhobar_t)) sigma(<ubar^t,z> + sbar^t G) ] - xi^a(t) lambda^a abar^t,
d/dt ubar^t = xi^w(t) abar^t E_{z,G}[ (f*(z) - fhat(z;rhobar_t)) sigma'(<ubar^t,z> + sbar^t G) z ] - xi^w(t) lambda^w ubar^t,
d/dt sbar^t = xi^w(t) abar^t E_{z,G}[ (f*(z) - fhat(z;rhobar_t)) sigma'(<ubar^t,z> + sbar^t G) G ] - xi^w(t) lambda^w sbar^t.
```

These are the parameter equations of a *dimension-free* Wasserstein gradient flow on `R(rhobar) = E_z[(h*(z) - fhat(z;rhobar))^2]`. And the initialization is the part I want to stare at. The finite network draws the first-layer coordinates at scale `1/sqrt(d)`: equivalently `sqrt(d) w_{j,l}^0` has a fixed law `mu_w`, so the `P` signal coordinates satisfy `u_i^0 = O(1/sqrt(d)) -> 0`, while the noise norm tends to `sbar^0 = m_2^w := E_{mu_w}[W^2]^{1/2}` and `abar^0 ~ mu_a`. So the dynamics *starts with the first-layer signal weights exactly at zero*. The signal directions begin with no information at all; whether the function gets learned is precisely whether the flow can push `ubar` off the origin and reach zero risk.

Before I rely on this reduction I should make sure it's not a fiction. I want a quantitative bound between actual finite-`N`, finite-`d`, finite-`eta` batch-SGD and this dimension-free flow. I get it in two hops, with the standard mean-field PDE as a bridge. First, SGD to the `d`-dimensional PDE: extending the non-asymptotic propagation-of-chaos analysis for two-layer mean-field training to batch updates and anisotropic step sizes gives, up to time `T`,
`||fhat(.;Theta^k) - fhat(.;rho_{k eta})|| <= K e^{K T^3} { sqrt(log N / N) + (sqrt((d + log N)/b) ∨ 1) sqrt(eta) }`.
Second, the `d`-dimensional PDE to the dimension-free one: a propagation-of-chaos plus Berry-Esseen argument controlling the Gaussian approximation of `<v,r>` gives
`sup_{t <= T} ||fhat(.;rho_t) - fhat(.;rhobar_t)|| <= K e^{K T^7} sqrt((P + log d)/d)`.
Together,
`||fhat(.;Theta^k) - fhat(.;rhobar_{k eta})|| <= e^{K T^7} { sqrt((P + log d)/d) + sqrt(log N/N) + sqrt((d + log N)/b) sqrt(eta) }`.
So as long as I take `d, N, 1/eta` large while keeping `T = eta n / b` bounded, the dimension-free flow is a faithful proxy for SGD. That gives me the equivalence I actually wanted: `h*` is learnable by `O(d)`-sample SGD on the two-layer network exactly when the dimension-free flow, started from `ubar^0 = 0`, can drive the risk to zero. The hard learnability question becomes a question about a gradient flow on `P(R^{P+2})`. That's the whole game now.

Let me attack necessity first, because I think it's where the example separation lives. Why would the bare monomial `z1 z2 z3` be stuck? Look at the first-layer evolution from the `ubar^0 = 0` start:
`d/dt ubar_i^t = abar^t E_z[ (h*(z) - fhat(z;rhobar_t)) sigma'(<ubar^t,z> + sbar^t G) z_i ] - reg`.
A coordinate `i` only leaves zero if it gets a nonzero push from this expectation. So the question is whether the correlation between `z_i` and the residual-weighted gradient is nonzero at the start.

Take a cleaner non-MSP example to build intuition: `h*(z) = z1 + z1 z2 z3`. Forget the regularization and pretend `fhat = 0`, `sbar = 0` for a second, just to see the geometry. Then `d/dt ubar_i = abar E_z[(z1 + z1 z2 z3) sigma'(<ubar,z>) z_i]`. Coordinate 1 sees `z1` correlate with the `z1` term — fine, `u_1` moves. But coordinates 2 and 3 enter only through `z1 z2 z3`, and crucially the equations for `u_2` and `u_3` are *symmetric under swapping 2 and 3*, and they both start at zero, so `u_2^t = u_3^t =: u_{23}^t` for all time. Set `z_{23} = z_2 + z_3` and integrate it out:
`d/dt u_{23}^t = abar^t E_z[ z1 z_{23} sigma'(z1 u_1^t + z_{23} u_{23}^t) ] = abar^t u_{23}^t E_{z1}[ z1 sigma''(z1 u_1^t + r) ]`
for some `r in [-2 u_{23}, 2 u_{23}]` by the mean value theorem. This is a *homogeneous linear* ODE in `u_{23}`: `d/dt u_{23} = (something) * u_{23}`, with `u_{23}^0 = 0`. So `u_{23}^t = 0` forever. The degree-3 term needs `u_2, u_3` nonzero to be represented, but they never budge off zero — there's no first-order signal to lift them, because the only thing that correlates with `z2` (or `z3`) is the joint `z1 z2 z3`, which contributes nothing at second order from a zero start. The function `z1 + z1 z2 z3` permanently misses its `z1 z2 z3` piece, and the risk is bounded below by `hhat({1,2,3})^2 > 0`.

Now I see the general mechanism and I can make it rigorous without the `fhat=0` cheat. Given a non-MSP `h*`, order its Fourier supports greedily and let `Sbar_*` be the largest subset whose supports *can* be added one-new-coordinate-at-a-time; non-MSP means there's a leftover support whose new coordinates — call that leftover index set `Omega` — never appear "alone" with respect to `Sbar_*`. I claim `ubar_i^t = 0` throughout the flow for every `i in Omega`. To control the push on such a coordinate I bound three correlations. First, the contribution of the network's own output: integrating out `z_i`,
`| E_{z_i}[ fhat(z;rhobar_t) sigma'(<z,ubar^t>) z_i ] | <= K(1+t) |ubar_i^t|`,
where the `(1+t)` comes from `|abar^t| <= K(1+t)` (the readout can grow at most linearly), and the rest from differences of `sigma'` across the two values of `z_i`, each bounded by a derivative times `|ubar_i^t|`. Second, for a support `S in Sbar_*` with `i not in S`,
`| E_z[ chi_S(z) sigma'(<z,ubar^t>) z_i ] | <= ||sigma''||_infty |ubar_i^t|`,
again because `z_i` appears in neither `chi_S` nor (to first order) the argument, so the whole thing is proportional to `ubar_i^t`. Third — and this is the crux — for a leftover support `S` (not in `Sbar_*`), there is *another* fresh coordinate `j in Omega ∩ S`, `j != i`, so
`| E_z[ chi_S(z) sigma'(<z,ubar^t>) z_i ] | <= ||sigma''||_infty |ubar_j^t|`,
i.e. the push on `i` is gated by *some other* zero-coordinate. Putting these together with `m_Omega^t := max_{i in Omega} |ubar_i^t|`,
`| d/dt ubar_i^t | = |abar^t| | E_z[ (h*(z) - fhat(z;rhobar_t)) sigma'(<ubar^t,z>) z_i ] | <= K(1+t)^2 m_Omega^t`.
Every coordinate in `Omega` has a time-derivative bounded by a constant times the *maximum* over `Omega`, and they all start at zero, so by Gronwall `m_Omega^t = 0` for all `t`. The signal coordinates in `Omega` are frozen at the origin, every monomial touching them is unlearnable, and `R(rhobar_t) >= sum_{S unlearnable} hhat(S)^2 > 0`, independent of the activation, step schedule, regularization, or noise width. So: no merged-staircase ordering of the supports ⇒ not learnable. The condition is that the supports `{S_1, ..., S_m}` can be ordered so each adds at most one genuinely new coordinate, `|S_i \ ∪_{j<i} S_j| <= 1` — the merged-staircase property. Necessity done, and `z1 z2 z3` (which leaps straight to a degree-3 support from nothing) is the cleanest violator.

Is the converse true — does MSP suffice? Let me try to construct a flow that learns a vanilla staircase `h*(z) = alpha_1 z1 + alpha_2 z1 z2 + ... + alpha_P z1...zP` and watch what the necessity proof's mechanism does in reverse. I'll train layerwise: Phase 1, train only the first layer `ubar` for `t in [0,T1]` (set `xi^a = 0, xi^w = 1`); Phase 2, freeze `ubar` and train only the readout `abar` (set `xi^a = 1, xi^w = 0`). For the analysis I take `mu_a = Unif([-1,1])` and `mu_w = delta_0`, so `sbar^0 = 0` and hence `sbar^t = 0` throughout — that kills the Gaussian-smoothing term and leaves me with a clean ODE in `(a, u)`.

Phase 1 is nonlinear and the residual couples everything, so let me see whether the staircase structure simplifies it. The intuition from necessity was: a coordinate moves only once the supports below it are "lit up." Concretely, what lifts `u_k` off zero is the correlation of `z_k` with the staircase, and the lowest-order such correlation comes from the term `alpha_{1..k} z1...zk` paired against `sigma'` — but to expose `z_k` you need `z1, ..., z_{k-1}` already carried by nonzero `u_1, ..., u_{k-1}`. So I expect a cascade. Let me guess the leading-order behavior and verify. Define a simplified trajectory `huk^t(a)` by the cascade ODE that keeps only the dominant driving term:
`d/dt huk^t = a alpha_{1..k} m_k prod_{j<k} huj^t,   huk^0 = 0,`
where `m_r = sigma^{(r)}(0)`. This comes from Taylor-expanding: `E_z[chi_S(z) sigma^{(i)}(<u,z>)] = m_{|S|+i} prod_{k in S} u_k (1 + O(t)) + O(t^L)` (only the monomial whose support matches `S` survives the orthogonality at leading order, and it pulls out the derivative `m_{|S|+i}`). Solve the cascade from the bottom. `u_1`: `d/dt hu1 = a alpha_1 m_1` (empty product is 1), so `hu1 ~ a alpha_1 m_1 t` — degree-1 weight grows linearly, like `t`. `u_2`: `d/dt hu2 = a alpha_{12} m_2 hu1 ~ a alpha_{12} m_2 (a alpha_1 m_1 t)`, so `hu2 ~ t^2` — quadratic. `u_3`: `d/dt hu3 = a alpha_{123} m_3 hu1 hu2 ~ t * t^2 = t^3`, so `hu3 ~ t^4`. The exponents go `1, 2, 4, ...` — each new stair grows like `t^{2^{k-1}}`, doubling the power each time. Carrying it through, the general form is

```
huk^t(a) = 2^{1 - 2^{k-1}} (a t)^{2^{k-1}} prod_{i in [k]} ( alpha_{1..i} m_i )^{2^{max(k-1-i, 0)}}.
```

So `|u_k^t| = Theta(t^{2^{k-1}})`: the weights light up *sequentially*, lower-degree first, each higher stair an entire order slower than the one below it. This is precisely "climbing the staircase," and now I can see *why* it climbs — coordinate `k` has no first-order driver until the product `u_1 ... u_{k-1}` is nonzero, and once it is, `u_k` is dragged up at the rate set by that product. I should check the true trajectory stays close to this guess. Let `Delta_k^t = sup_{s<=t,a} |u_k^s - huk^s|`. The full derivative `d/dt u_k = a E_z[h* z_k sigma'(<u,z>)] - a E_z[fhat z_k sigma'(<u,z>)]`. The `fhat` term I bound by `K Delta_k^t + O(t^L)` (its Fourier coefficients on supports containing `k` are themselves controlled by `prod_{i in S} u_i`, hence by `Delta_k`). The `h*` term splits into the matching driver (giving `huk`'s derivative up to `Delta`'s) plus lower- and higher-degree leakage, all of which are `O(Delta) + O(t * leading)`. The upshot is `d/dt Delta_k <= K(t^{2^{k-1}} + Delta_k)`, and Gronwall plus the inductive facts (`Delta_1 <= K t^2`, `|huk| = Theta(t^{2^{k-1}})`, `t^L = O(t^{2^{k-1}})` since I chose `L > 2^{P-1}`) give `Delta_k^t = O(t^{2^{k-1}+1})`. So the simplified cascade captures `u_k` to leading order with a strictly higher-order error. The first layer, at the end of Phase 1, has each signal coordinate raised to a known nonzero magnitude with a clean dependence on the random readout `a`.

Now Phase 2, with `ubar` frozen at `ubar^{T1}(a)`. The readout `a` is the only thing moving, and the output is linear in it, so this is convex — a kernel regression with the fixed kernel
`K^{T1}(z, z') = E_{a ~ mu_a}[ sigma(<ubar^{T1}(a), z>) sigma(<ubar^{T1}(a), z'>) ]`.
The residual `g_t(z) = h*(z) - fhat(z;rhobar_t)` obeys `d/dt R(rhobar_t) = - E_{z,z'}[ g_t(z) K^{T1}(z,z') g_t(z') ]`. Going to the Fourier basis, with `g_t = (g_t(S))_S` and the kernel matrix `K^{T1} = (K^{T1}(S,S'))`, since `R = ||g_t||_2^2`,
`d/dt ||g_t||_2^2 = - g_t^T K^{T1} g_t <= - lambda_min(K^{T1}) ||g_t||_2^2`,
so `||g_t||_2^2 <= e^{-lambda_min(K^{T1})(t - T1)} ||g_{T1}||_2^2`. If the kernel matrix is strictly positive definite I get exponential convergence to zero risk, and I'm done by taking `T2 = T1 + log(K/eps)/lambda_min`. So everything reduces to: is `lambda_min(K^{T1}) > 0`?

Use the Phase-1 structure. From the Fourier-coefficient leading order, `E_z[chi_S(z) sigma(<ubar^{T1}(a),z>)] = m_{|S|} nu_S(T1) a^{beta(S)} (1 + O(T1))` where `nu_S(T1) = prod_{k in S} nu_k(T1)`, `nu_k` is the `a`-independent part of `huk`, and the key exponent is `beta(S) = sum_{k in S} 2^{k-1}` — the binary value of the support. Setting `D_S = m_{|S|} nu_S(T1)`, `D = diag(D_S)`,
`K^{T1} = D (M + Delta) D,   M = ( E_{a ~ mu_a}[ a^{beta(S) + beta(S')} ] )_{S,S'},   ||Delta||_op <= C T1 P.`
Look at `M`. The exponents `beta(S)` range over *all* of `{0, 1, ..., 2^P - 1}` as `S` ranges over subsets (binary expansion is a bijection), so `M` is exactly the Gram matrix of the monomials `1, X, X^2, ..., X^{2^P - 1}` in `L^2([-1,1], Unif)`. Those monomials are linearly independent, so `M` is strictly positive definite, `lambda_min(M) > 0` independent of `T1`. Take `T1 <= lambda_min(M)/(2P)` so the perturbation `Delta` can't kill it, and `lambda_min(K^{T1}) >= (min_S D_S^2) lambda_min(M)/2 > 0` (the `D_S` are nonzero precisely because every `m_r != 0` and every `alpha != 0`). So the kernel is positive definite, Phase 2 converges, and the vanilla staircase is learnable. And I notice the load-bearing role of the *random* readout: it's the randomness of `a ~ mu_a` that makes the per-neuron features `a^{beta(S)}` into linearly independent functions of `a` and hence makes `M` full rank. If I'd fixed all the `a` equal, the neurons would be identical and the kernel would collapse. Neuron diversity is doing the work.

This also tells me which activations are allowed. The cascade ODE `d/dt huk = a alpha_{1..k} m_k prod huj` *dies* the instant any `m_k = sigma^{(k)}(0) = 0` — a missing derivative at the origin breaks the chain at that degree and the staircase stalls. So I need `sigma^{(r)}(0) != 0` for `r = 0, 1, ..., P`. A symmetric activation (odd like `tanh`, where all even derivatives at 0 vanish; or even, where all odd ones do) is exactly wrong: it zeroes out half the chain. I want all the low-order derivatives at the origin to be nonzero. The shifted sigmoid `sigma(x) = (1 + e^{-x + 0.5})^{-1} = sigmoid(x - 0.5)` does this — the shift moves the evaluation point off the symmetric center of the logistic, so `m_0, m_1, m_2, ...` are all nonzero, and the cascade can climb all `P` stairs.

For a general MSP function — chains that merge, like `z1 + z1 z2 + z2 z3 + z3 z4` — tracking the single leading order isn't enough; I have to keep more of the polynomial structure. The discrete-time route is cleaner here: take a sufficiently high-degree polynomial activation, write the simplified first-layer weights after the first phase as recurrence-defined polynomials in `(a, the Fourier coefficients alpha_S, the activation coefficients)`, build the corresponding simplified kernel, and ask whether its determinant is identically zero. Using that large powers of distinct polynomials are linearly independent, that determinant is not the zero polynomial for every MSP set structure; anti-concentration under random readout weights and generic Fourier coefficients then makes it nonzero almost surely. For smooth non-polynomial activations, I can get the analogous generic statement only with a one-time random perturbation of the activation before the second phase. So *generic* MSP functions are learnable, but I should not overstate that as "all MSP functions under any smooth activation." The exception is real: a symmetric MSP like `z1 + z2 + z1 z3 + z2 z4` is invariant under the swap `(1,2,3,4) -> (2,1,4,3)`, and that symmetry forces `u_1 = u_2`, `u_3 = u_4` along the flow, collapsing the weights into a subspace that contains no zero-risk solution. The flow finds a symmetric fixed point and stalls. Random Fourier coefficients break such exact symmetries with probability one, which is why the clean statement is "almost surely over the coefficients." So MSP is necessary, and nearly sufficient — sufficient off a measure-zero set, with the activation caveat just noted.

One more thing I want, to know this is actually a separation and not just an internal characterization: that linear methods genuinely cannot do what this network does. For a single degree-`k` monomial over an unknown subset, any linear method (kernel, random features, dimension `q`) needs `min(n,q) = Omega(d^k)` — a subspace-projection bound does it: if the targets are nearly orthogonal to any low-dimensional subspace, the average error stays near 1 until `min(n,q)` is large. The catch is staircases aren't almost-orthogonal and are SQ-easy, so I need the sharper projection bound `min(n,q) >= (eta - kappa) / max_i (1/M) sum_j |<f_i, proj_Omega f_j>|` applied to the family of all coordinate-permutations of a staircase. It yields, for the degree-`P` vanilla staircase, `min(n,q) >= (eta/2) binom(d, floor(eta P /2))`, which is `d^{omega(1)}` once `P` grows even slowly with `d` — superpolynomial — while the network learns it in `d^{O(1)}`. The separation is real.

So the picture closes: the `1/N` mean-field parametrization keeps the dynamics nonlinear so features can move; sparsity collapses the flow to a dimension-free gradient flow that *starts at the origin in the signal directions*; a coordinate only escapes the origin once the supports beneath it are lit, so the network learns Fourier components in order of increasing degree, climbing at rate `t^{2^{k-1}}`; this climb succeeds cleanly for vanilla staircases and generically for MSP supports, and is provably stuck for non-MSP supports; and the whole thing beats every linear method on the same targets. The concrete artifact is the finite training recipe that realizes the same scaling in the sparse-hypercube experiments — a two-layer network in the mean-field scaling with an activation whose derivatives at the origin are all nonzero, continuous random readout weights for neuron diversity, first-layer weights scaled as `sqrt(d) w_{j,l}^0 ~ N(0,1)` so the signal coordinates start near the origin, and plain SGD:

```python
import torch
import torch.nn as nn


def build_model(config) -> nn.Module:
    """Two-layer mean-field network on {+1,-1}^d, output normalized by 1/M."""

    class TwoLayerMeanField(nn.Module):
        def __init__(self, d: int, M: int) -> None:
            super().__init__()
            # First layer d -> M, no bias: input <w,x> stays centered, and the
            # coordinates of w start at O(1/sqrt(d)), so the signal block starts
            # near the saddle of the dimension-free flow.
            self.fc1 = nn.Linear(d, M, bias=False)
            # Readout M -> 1, no bias.
            self.fc2 = nn.Linear(M, 1, bias=False)
            nn.init.normal_(self.fc1.weight, mean=0.0, std=d ** -0.5)
            with torch.no_grad():
                # a_j ~ Unif([-1,1]): random readout weights give neuron diversity,
                # which is what makes the Phase-2 kernel full rank.
                self.fc2.weight.uniform_(-1.0, 1.0)
            self.M = M

        @staticmethod
        def _shifted_sigmoid(u: torch.Tensor) -> torch.Tensor:
            # sigma(x) = (1 + exp(-x + 0.5))^{-1} = sigmoid(x - 0.5).
            # The shift moves evaluation off the logistic's symmetric center so
            # sigma^{(r)}(0) != 0 for all low r -- the cascade that lifts u_k off
            # zero needs every one of those derivatives.
            return torch.sigmoid(u - 0.5)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self._shifted_sigmoid(self.fc1(x))
            return self.fc2(h).view(-1) / self.M    # 1/M mean-field normalization

    return TwoLayerMeanField(d=config.d, M=config.width)


def get_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """Plain SGD, eta = 1/2, no momentum: the bare gradient flow, nothing adaptive."""
    return torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.0)


def train_step(model, optimizer, x, y) -> float:
    """One square-loss step on a fresh batch (one-pass batch-SGD)."""
    optimizer.zero_grad(set_to_none=True)
    pred = model(x).view(-1)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    return float(loss.item())
```
