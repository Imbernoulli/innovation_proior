The lazy baseline told me exactly what is missing, and it told me in the Fourier-recovery column. NTK
landed at `test_mse_h1 = 2.78`, `h2 = 2.99`, `h3 = 1.00` — all three sitting essentially at their
trivial-predictor variances (`3, 3, 1`), with recovery `0.96, 1.00, 1.00`, i.e. it recovered *almost
nothing* on any target. Even `h1`, where I expected the degree-1 piece to be reachable, only fell from `3`
to `2.78`: the frozen `M=100` random-feature basis barely resolved the lone `z1` term and got none of the
`z1z2` or `z1z2z3` pieces. I can read exactly how little it got. If the frozen readout puts coefficient `c`
on `z1` and nothing on the other two monomials, then `test_mse_h1 = (1-c)^2 + 1 + 1` and
`recovery_h1 = (|1-c| + 1 + 1)/3`; solving `(1-c)^2 = 0.78` gives `c ≈ 0.12`, which predicts
`recovery_h1 ≈ (0.88 + 2)/3 ≈ 0.96`, and the measured recovery is `0.956`. The two columns agree on the
same story: the lazy machine captured about a *tenth* of even the degree-1 coefficient and precisely zero
of everything above it. And this is not seed noise — across the three latent subsets `h1` landed at
`2.832, 2.748, 2.772` (spread `< 0.09`), `h2` at `2.957, 3.033, 2.988`, `h3` at `0.9999, 0.9996, 0.99998`;
the floor is rock-stable because it is set by the feature map's degree content, not by which coordinates
happened to be latent. The deceptive `score_h3 = 0.368` is exactly the artifact I flagged — `h3` has
unit variance, so the trivial predictor scores `exp(-1)`; recovery `0.9999` confirms nothing was learned.
So the failure is sharp and structural: the features were frozen *before* the network knew which
coordinates were latent, and a non-adaptive kernel machine cannot put its limited capacity where the
signal is. The fix is not a better optimizer on those features. It is to let the features *move* — and the
parametrization that lets them move is the whole game now.

Before I reach for the `1/M` scaling, I should check the alternatives, because "let the features move" has
more than one embodiment. One option is to keep the lazy kernel machine but simply *widen* it — throw
`Omega(d^2) = 10^4` random features at the degree-2 pieces. But that is off-contract (`config.width` is
fixed at `M=100`) and, more fundamentally, it does not adapt: it would still freeze the map before seeing
`I`, so the degree-3 pieces stay unreachable no matter how wide. A second option is to unfreeze the first
layer but *keep* the `1/sqrt(M)` output. That does not escape the lazy regime either: with `1/sqrt(M)` and
unit-scale readout the network linearizes around its init, the first-layer weights travel a distance that
vanishes as the width grows, and I am back to a fixed kernel machine with extra noise. The distinctive
property I need is `O(1)` weight travel, and the `1/M` mean-field normalization is the specific scaling
that delivers it while keeping the output `O(1)` — each of the `M` neurons contributes `O(1/M)`, but the
per-neuron force is not suppressed by any `1/sqrt(M)`, so the weights genuinely move. That is the one
member of the design space that is both on-contract and actually adaptive, so I take it: `1/M` output,
first layer free, plain SGD with a large step.

There is a third temptation I should dispose of before moving on, because it will come back to haunt the
harder targets: if the whole problem is that the signal weights start near the origin, why not just
*initialize them large* so the flow begins away from that flat spot? Two things kill it. I do not know
`I`, so I cannot preferentially inflate the four signal weights — the only move available is a larger
*isotropic* init, `w ~ N(0, c^2 I_d)` with `c` big. But that inflates the noise block too, and the noise
block has `d-P = 96` coordinates against the signal's `4`: the pre-activation `<w,x>` then has standard
deviation `c·sqrt(d) ≈ 10c`, so a large `c` drives every neuron deep into the saturated tail of `sigma`
where `sigma'(<w,x>) ≈ 0` and the gradients that would move *any* coordinate die. And even setting
saturation aside, isotropy is fatal to the mechanism: by the same noise-sign symmetry that collapses the
flow to `(a,u,s)`, an isotropic init gives the signal directions *zero net first-order push* no matter the
scale — a bigger `u^0` pointing in a random direction is not a bigger `u^0` pointing at the latent
monomial. So inflating the init cannot substitute for the cascade; the only thing that lifts a signal
coordinate is a lower-degree support beneath it already being lit. That is the property I now have to check
target by target, and it is exactly what will separate `h1` from `h2` and `h3`.

Start from the thing the lazy result makes unavoidable. With the `1/sqrt(M)` normalization NTK used, as
the width grows the first-layer weights barely travel: the network linearizes around its initialization
and is a fixed kernel machine, its feature map frozen before it ever sees `I`. That is precisely the
regime I have to *escape*. So I take the other scaling, `fhat(x) = (1/M) sum_j a_j sigma(<w_j,x>)`. With
the `1/M` out front, each neuron contributes `O(1/M)` to the output but the weights are free to travel an
`O(1)` distance, so the dynamics stays genuinely nonlinear. This is the regime where features can rotate
toward the latent subset. Train it by square loss, one-pass batch-SGD with the fixed batch `b=150`, plain
SGD, large step — this is the canonical mean-field two-layer recipe, and the question is which of the
three targets it can actually find.

The network has `M` neurons, but they are exchangeable — the output only sees them through their empirical
distribution. So instead of chasing `M(d+1)` coordinates, track the measure `rho = (1/M) sum_j
delta_{theta_j}`, `theta_j = (a_j, w_j)`. The population risk is a functional of that measure alone:
expanding `R(rho) = E_x[(f*(x) - fhat(x;rho))^2]` gives a constant `E[y^2]`, a linear term in `rho`, and
a quadratic neuron-interaction term — all depending on `rho`, not on a labeling of neurons. In the
wide-and-small-step limit one-pass SGD on this object is a Wasserstein gradient flow: the empirical
measure converges to `rho_t` solving a continuity equation `partial_t rho_t = nabla.(rho_t H nabla
psi(theta;rho_t))`, with `psi(theta;rho) = a E_x[(fhat - f*) sigma(<w,x>)] + reg`. The neurons are a gas
of particles descending the risk. Clean — but it lives in `R^{d+1}` and `d = 100` is large. I have not
used sparsity yet, and sparsity is where the separation between the three targets must live.

Lean on the structure of `f*`. The target is `f*(x) = h*(z)` with `z = x_I` the `P=4` signal coordinates
and the rest pure noise. Split every input `x = (z, r)` and every weight `w = (u, v)`, `u` aligned to the
signal, `v` to the noise. At initialization the coordinates of `w^0` are iid and symmetric, so flipping
any sign of the noise block leaves the initial distribution unchanged, which means `fhat(x;rho_0)` does
not depend on `r`. Does that survive training? Take the pushforward of `rho_t` under a noise-sign flip; at
`t=0` it equals `rho_0`, and because the flow only couples weights through `<w,x>` the flip can be
absorbed into the `x`-average, so the flipped measure solves the same PDE with the same initial condition.
By uniqueness, the network stays independent of the irrelevant directions throughout training. The noise
block then enters only through `<v,r>`, a sum of `d-P` independent bounded terms, approximately Gaussian
`||v||_2 G` for large `d`. So the entire `(d-P)`-dimensional noise weight collapses to a single scalar
`s = ||v||_2` acting as a Gaussian smoothing width. The flow reduces to a *dimension-free* gradient flow
on effective parameters `(a, u, s) in R^{P+2}`, with `fhat(z) = int a E_G[sigma(<u,z> + sG)] rho`. And —
this is the part to stare at — as `d -> infinity` the signal block of the first-layer weight starts at
`u^0 = O(1/sqrt(d)) -> 0`. The dynamics *starts with the signal weights at the origin*. Whether a target
is learned is precisely whether the flow can push `u` off the origin and drive the risk to zero. With
`d=100` this is the relevant near-saddle start.

So attack each target through its first-layer evolution from `u^0 ≈ 0`:
`d/dt u_i = a E_z[(h*(z) - fhat(z)) sigma'(<u,z>+sG) z_i] - reg`. A coordinate `i` only leaves the origin
if the correlation between `z_i` and the residual-weighted gradient is nonzero. This is the lens that
decides `h1` vs `h2` vs `h3`.

Take `h3(z) = z1z2z3` first, because it is the cleanest. Forget regularization and the network output for
a second: `d/dt u_i = a E_z[z1z2z3 · sigma'(<u,z>) z_i]`. By symmetry of `h3` under permuting `{1,2,3}`
and the common origin start, `u_1 = u_2 = u_3 =: u_{123}` for all time. The driving expectation
`E_z[z1z2z3 · sigma'(u_{123}(z1+z2+z3)) z_1]` is a *homogeneous* function of `u_{123}` that vanishes at
`u_{123}=0` to high order — there is no first-order push, because the only thing correlating with `z_1` is
the full triple `z1z2z3`, which contributes nothing at the origin. The coordinates never budge:
`u_{123}^t = 0` forever, and the risk is bounded below by `hat(h3)({1,2,3})^2 = 1`. `h3` is a leap
straight from nothing to a degree-3 support — it is non-MSP, and the flow is frozen at the origin. Plain
mean-field SGD *cannot* learn it, exactly like the lazy baseline, but for a different reason: not "the
features are frozen" but "the gradient at the origin is zero in the signal directions."

Now `h2(z) = z1z2 + z2z3 + z3z4`, the non-MSP chain. The first added support is `{1,2}` with `|S_1| = 2`,
so the first thing the flow would have to do is lift *two* coordinates off the origin at once. Repeat the
homogeneity check: `d/dt u_1 = a E_z[(z1z2 + ...) sigma'(<u,z>) z_1]`, and the only term correlating with
`z_1` near the origin is `z1z2`, whose expectation `E_z[z1z2 sigma'(u_1 z1 + u_2 z2 + ...) z_1] = E[z_2
sigma'(...)]` is gated by `u_2` — and symmetrically `u_2`'s push is gated by `u_1`. Linearizing the pair near the origin,
`d/dt u_1 = a m_2 alpha_{12} u_2 + O(u^2)` and `d/dt u_2 = a m_2 alpha_{12} u_1 + O(u^2)`, i.e.
`d/dt (u_1, u_2) = kappa [[0,1],[1,0]] (u_1, u_2)` with `kappa = a m_2 alpha_{12}` — a *homogeneous* linear
system whose source term is exactly zero at the origin. Its matrix has eigenvalues `±kappa`, so there is
even an unstable direction `u_1 = u_2`; but an unstable eigenvalue only *amplifies an existing
perturbation*, and here the initial perturbation is `(0,0)`. The origin is a fixed point, and `e^{κt}·0 =
0`: the pair never leaves. This is the crisp contrast with `h1`, whose first equation `d/dt u_1 = a m_1
alpha_1` has a *constant, inhomogeneous* source (the degree-1 term `z1`) that forces `u_1` off zero
regardless of where it starts — an inhomogeneous linear ODE, not a homogeneous one. The general
statement: order `h*`'s supports greedily, and for a non-MSP target there is a leftover index set `Omega`
whose new coordinates never appear "alone." Bounding the three relevant correlations (the network's own
output contributes `K(1+t)|u_i|`; a lit support `S` with `i not in S` contributes `||sigma''||_inf |u_i|`;
a leftover support contributes `||sigma''||_inf |u_j|` for *another* zero coordinate `j in Omega`) gives
`|d/dt u_i| <= K(1+t)^2 max_{j in Omega} |u_j|`. Every coordinate in `Omega` has a derivative bounded by
the maximum over `Omega`, all start at zero, so by Gronwall they stay at zero. `h2` leaps by 2 at its
first step; `Omega` is nonempty; mean-field SGD is stuck, `R >= sum_{unlearnable} hat(h2)(S)^2 > 0`. The
non-MSP chain is *not* learnable by this recipe either.

Finally `h1(z) = z1 + z1z2 + z1z2z3`, the vanilla staircase, where I expect the recipe to come alive.
Here the supports order `{1} -> {1,2} -> {1,2,3}`, each adding exactly one new coordinate — MSP / leap-1.
Watch the cascade from the origin. Coordinate `1` enters through the degree-1 term `z1`:
`d/dt u_1 = a · alpha_1 · m_1` (with `m_r = sigma^{(r)}(0)`), so `u_1 ~ a alpha_1 m_1 t` — it grows
*linearly* off the origin, because it has a first-order driver that does not depend on any other
coordinate. Once `u_1 != 0`, coordinate `2` gets a driver through `z1z2`:
`d/dt u_2 = a alpha_{12} m_2 u_1 ~ t`, so `u_2 ~ t^2`. Then `u_1, u_2 != 0` light up coordinate `3`
through `z1z2z3`: `d/dt u_3 = a alpha_{123} m_3 u_1 u_2 ~ t·t^2`, so `u_3 ~ t^4`. The weights light up
*sequentially*, lower degree first, each new stair an entire order slower — `|u_k| = Theta(t^{2^{k-1}})`.
Let me actually integrate the chain to make sure the exponents are what I claimed rather than a
hand-wave. `u_1 ~ c_1 t` (degree-1 driver, constant, gives a linear ramp). Feed that into
`d/dt u_2 ∝ u_1 ∝ t`, integrate: `u_2 ~ t^2`. Feed both into `d/dt u_3 ∝ u_1 u_2 ∝ t·t^2 = t^3`,
integrate: `u_3 ~ t^4`. So the exponents are `1, 2, 4`, i.e. `2^{k-1}` for `k = 1,2,3`, exactly the
`Theta(t^{2^{k-1}})` law — the stairs are not merely slower, they slow *super-linearly*, each one squaring
the timescale of the last. That is the concrete meaning of "orders of magnitude slower": to move `u_3` an
`O(1)` distance I need `t = O(1)` in flow time, but `u_3` spends most of that time crawling at `t^4`, so
the degree-3 stair of `h1` is the piece most at risk of not finishing inside the `T=4000`-step budget even
though it is, in principle, reachable. Let me size this honestly, because `Theta(t^{2^{k-1}})` cuts both
ways: the exponents `1, 2, 4` mean each stair does not merely start later, it *accelerates* later. `u_3`
spends the overwhelming majority of any finite window down in the `t^4` basement before its product driver
`u_1 u_2` has grown enough to yank it up, and there is nothing adaptive here to rescale that tiny gated
gradient — plain SGD takes it at face value. So my honest expectation for `h1` is a *partial* climb: the
degree-1 term (a linear ramp with a constant, un-gated driver) picked up first and most completely, the
degree-2 `z1z2` term (`t^2`) partly, and the degree-3 `z1z2z3` term (`t^4`) barely if at all inside
`T=4000`. Quantitatively that means `test_mse_h1` should fall from NTK's `2.78` but need *not* approach the
`1.0` it would reach if all three stairs finished cleanly (which would strip the degree-1 and degree-2 units
and leave only the degree-3 unit of variance); I expect it to stay above `2`, still carrying most of the
degree-3 monomial and a good part of the degree-2 one — a modest, real improvement, not a collapse to zero.
This is climbing the staircase: a coordinate has no first-order
driver until the product of the lower-degree weights is nonzero, and once it is, it gets dragged up. This
is exactly the mechanism `h3` and `h2` *lack* — they have no degree-1 (or single-new-coordinate) entry
point to start the cascade.

Two design choices keep the cascade alive, and they are why the fill looks the way it does. First, the
activation. The cascade `d/dt u_k = a alpha_{1..k} m_k prod_{j<k} u_j` *dies* the instant any
`m_k = sigma^{(k)}(0) = 0`, so I have to check the derivatives at the origin, not just wave at "symmetry."
Take the tempting default first, `tanh`: it is odd, so `tanh''(0) = 0`, which means `m_2 = 0` and the
degree-2 stair `d/dt u_2 = a alpha_{12} m_2 u_1` never lights — with `tanh` the recipe would stall on `h1`
too, at the `z1z2` step. Worse, the *un-shifted* logistic `sigmoid(x)` has the same disease at exactly the
place I care about: `sigmoid''(x) = sigmoid'(x)(1 - 2 sigmoid(x))`, and at `x = 0`, `sigmoid(0) = 1/2`, so
`1 - 2·(1/2) = 0` and `m_2 = 0` again. Both "obvious" smooth activations kill the second stair. The fix is
the shift: `sigma(x) = sigmoid(x - 0.5)` evaluates the logistic at `-0.5`, where `sigmoid(-0.5) ≈ 0.378`,
so `m_0 ≈ 0.378`, `m_1 = sigma'(0) = 0.378·0.622 ≈ 0.235`, and `m_2 = sigma'(0)(1 - 2·0.378) ≈
0.235·0.245 ≈ 0.058` — all nonzero, and the higher `m_r` stay nonzero for the same off-center reason. So
the `-0.5` shift is not cosmetic; it is precisely what moves evaluation off the logistic's symmetric center
`m_2 = 0`, keeping every low-order derivative alive so the cascade can climb all the way to the degree-3
stair. And the degree-3 stair genuinely needs the *third* derivative: the `z1z2z3` term of `h1` first
appears at order `r=3` in the Taylor expansion of `sigma(<u,z>)`, so its driver carries `m_3 = sigma'''(0)`.
Compute it at the shift to be sure it survives: with `g = sigmoid`, `g''' = g'[(1-2g)^2 - 2g']`, and at the
evaluation point `x = -0.5` where `g = 0.378`, `g' = 0.235`, `(1-2g)^2 = 0.245^2 = 0.060`, so
`m_3 = 0.235·(0.060 - 0.470) ≈ -0.096` — nonzero (the sign is immaterial; only that the driver does not
vanish). So the full chain `m_1, m_2, m_3 = 0.235, 0.058, -0.096` is alive, and every stair of `h1` up to
degree 3 has a live coefficient. Second, the readout. The second (linear) phase, once the features are set, is kernel
regression with `K(z,z') = E_a[sigma(<u(a),z>) sigma(<u(a),z')>)]`; in the Fourier basis this is
`D(M_a + Delta)D` with `M_a = (E_a[a^{beta(S)+beta(S')}])`, `beta(S) = sum_{k in S} 2^{k-1}`. Since
`beta` hits all of `{0,...,2^P - 1}`, `M_a` is the Gram matrix of `1, X, ..., X^{2^P-1}` under random `a`
— full rank *because* `a` is random. It is worth seeing the collapse concretely: if I fix every `a_j` to
the same value `+1`, then `a^{beta(S) + beta(S')} = 1` for all `S, S'`, so `M_a` becomes the all-ones
matrix, rank `1` — the readout can then represent only a single direction and the whole Fourier structure
collapses. Drawing `a ~ Unif({+1,-1})` instead makes `E_a[a^m] = 0` for odd `m` and `1` for even `m`, so
`M_a` picks up off-diagonal structure that separates the monomials and the Gram becomes full rank. That is
the mechanism by which the *random* readout signs, which looked like a throwaway init detail, are what keep
the readout phase well-posed. So: `w ~ N(0,I_d)` (signal weights start near the origin), `a ~ Unif({+1,-1})`
(diversity), shifted sigmoid (live cascade), `1/M` (feature-learning regime), plain SGD at `lr=0.5` (the
bare gradient flow, nothing adaptive). The distilled fill is in the answer.

Now the falsifiable expectations against the lazy numbers. The delta from NTK is the parametrization:
unfreeze the first layer, swap `1/sqrt(M)` for `1/M`, large plain-SGD step. The prediction splits cleanly
by leap. On `h1`, mean-field SGD should do *strictly better than NTK's `2.78`*: it can climb the leap-1
staircase that no fixed feature map reaches, picking up the degree-1 piece quickly and the degree-2 piece
as `u_1` lights `u_2` — so `test_mse_h1` should fall and `score_h1` should rise above NTK's `0.062`, with
the degree-3 piece (the slowest stair, `t^4`) the question mark within the `T=4000` budget. On `h2`, I
predict it *fails*, sitting near the trivial `test_mse_h2 ≈ 3` just as NTK did — the leap-2 first support
freezes the cascade at the origin, so `score_h2` should be no better than NTK's `0.050`. The number `3` is
not a guess: `h2` is a sum of three distinct, orthogonal, unit-magnitude parity monomials, so
`E[h2^2] = 1+1+1 = 3`, and if the flow lifts *no* signal coordinate the risk sits exactly at that constant-
predictor variance. On `h3`, I predict it *fails identically* to NTK, `test_mse_h3 ≈ 1` (the single
degree-3 monomial has `E[h3^2] = 1`), `score_h3 ≈ exp(-1) ≈ 0.37`, recovery near `1` — same trivial-
predictor number, different reason (zero gradient at the origin, not frozen features). If
mean-field SGD beats NTK on `h1` but ties it on `h2` and `h3`, the diagnosis for the next rung is already
written: feature learning helps, but plain SGD climbs only *one* new coordinate at a time, so any leap
greater than one stalls it. The cure is not more SGD; it is to *manufacture* the staircase the targets do
not have — a saddle-escaping schedule that supplies the missing lower stairs so the coordinates the bare
flow leaves at the origin finally get a first-order driver. That is the next step.
