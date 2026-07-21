The lazy baseline told me exactly what is missing, and it told me in the Fourier-recovery column. NTK
landed at `test_mse_h1 = 2.78`, `h2 = 2.99`, `h3 = 1.00` — all three essentially at their trivial-predictor
variances (`3, 3, 1`), with recovery `0.96, 1.00, 1.00`, i.e. it recovered almost nothing. Even `h1`, where
I expected the degree-1 piece to be reachable, only fell from `3` to `2.78`: the frozen `M=100` random-
feature basis barely resolved the lone `z1` term and got none of `z1z2` or `z1z2z3`. I can read exactly how
little it got — if the frozen readout puts coefficient `c` on `z1` and nothing else, then
`test_mse_h1 = (1-c)^2 + 1 + 1` and `recovery_h1 = (|1-c| + 1 + 1)/3`; solving `(1-c)^2 = 0.78` gives
`c ≈ 0.12`, predicting `recovery_h1 ≈ 0.96` against the measured `0.956`. The two columns agree: the lazy
machine captured about a tenth of even the degree-1 coefficient and precisely zero above it. And it is not
seed noise — across the three latent subsets `h1` landed at `2.832, 2.748, 2.772` (spread `< 0.09`), the
floor rock-stable because it is set by the feature map's degree content, not by which coordinates are
latent. The deceptive `score_h3 = 0.368` is exactly the artifact I flagged: `h3` has unit variance, so the
trivial predictor scores `exp(-1)`, and recovery `0.9999` confirms nothing was learned. The failure is
sharp and structural — the features were frozen before the network knew which coordinates were latent. The
fix is to let the features *move*, and the parametrization that lets them move is the whole game now.

"Let the features move" has more than one embodiment, and only one is both on-contract and actually
adaptive. Widening the lazy kernel machine to `Omega(d^2)` features is off-contract and still freezes the
map before seeing `I`, so degree-3 stays unreachable no matter how wide. Unfreezing the first layer but
keeping the `1/sqrt(M)` output does not escape the lazy regime either: with `1/sqrt(M)` and unit-scale
readout the network linearizes around init, the first-layer weights travel a distance that vanishes as
width grows, and I am back to a fixed kernel machine with extra noise. The property I need is `O(1)` weight
travel, and the `1/M` mean-field normalization is the specific scaling that delivers it while keeping the
output `O(1)`: each of the `M` neurons contributes `O(1/M)`, but the per-neuron force is not suppressed by
any `1/sqrt(M)`, so the weights genuinely move. So: `1/M` output, first layer free, plain SGD, large step —
`fhat(x) = (1/M) sum_j a_j sigma(<w_j,x>)`, trained by square loss on one-pass batches, the canonical
mean-field recipe. The question is which of the three targets it can find.

There is a third temptation to dispose of, because it comes back on the harder targets: if the whole
problem is that the signal weights start near the origin, why not *initialize them large* so the flow begins
away from that flat spot? Two things kill it. I do not know `I`, so I cannot inflate the four signal weights
selectively — the only move is a larger *isotropic* init `w ~ N(0, c^2 I_d)`. That inflates the noise block
too, and it has `d-P = 96` coordinates against the signal's `4`: the pre-activation `<w,x>` then has std
`c·sqrt(d) ≈ 10c`, so a large `c` drives every neuron into the saturated tail where `sigma'(<w,x>) ≈ 0` and
the gradients that would move any coordinate die. And even setting saturation aside, isotropy is fatal to
the mechanism: by the noise-sign symmetry that collapses the flow to `(a,u,s)` below, an isotropic init
gives the signal directions *zero net first-order push* at any scale — a bigger `u^0` pointing in a random
direction is not a bigger `u^0` pointing at the latent monomial. Inflating the init cannot substitute for
the cascade; the only thing that lifts a signal coordinate is a lower-degree support beneath it already
being lit. That is the property I now have to check target by target.

The `M` neurons are exchangeable — the output only sees them through their empirical distribution — so
instead of chasing `M(d+1)` coordinates I track the measure `rho = (1/M) sum_j delta_{theta_j}`,
`theta_j = (a_j, w_j)`. The population risk `R(rho) = E_x[(f*(x) - fhat(x;rho))^2]` is a functional of that
measure alone: a constant `E[y^2]`, a linear term, and a quadratic neuron-interaction term. In the
wide-and-small-step limit one-pass SGD on this object is a Wasserstein gradient flow — the empirical measure
solves a continuity equation `partial_t rho_t = nabla·(rho_t nabla psi(theta;rho_t))`, the neurons a gas of
particles descending the risk. Clean, but it lives in `R^{d+1}` with `d=100` large, and I have not used
sparsity yet — which is where the separation between the three targets must live.

Lean on the structure of `f*`. The target is `f*(x) = h*(z)`, `z = x_I` the `P=4` signal coordinates, the
rest pure noise. Split `x = (z, r)` and `w = (u, v)`, `u` aligned to signal, `v` to noise. At init the
coordinates of `w^0` are iid symmetric, so flipping any noise-block sign leaves the distribution unchanged,
which means `fhat(x;rho_0)` does not depend on `r`. It survives training: the noise-sign-flipped pushforward
of `rho_t` equals `rho_0` at `t=0` and, because the flow couples weights only through `<w,x>`, solves the
same PDE with the same initial condition — by uniqueness the network stays independent of the irrelevant
directions throughout. The noise block then enters only through `<v,r>`, a sum of `d-P` bounded terms,
approximately `||v||_2 G` (Gaussian) for large `d`, so the whole `(d-P)`-dimensional noise weight collapses
to a single scalar `s = ||v||_2` acting as a Gaussian smoothing width. The flow reduces to a *dimension-free*
gradient flow on `(a, u, s) in R^{P+2}`, with `fhat(z) = int a E_G[sigma(<u,z> + sG)] rho`. And as
`d -> infinity` the signal block starts at `u^0 = O(1/sqrt(d)) -> 0`: the dynamics *starts with the signal
weights at the origin*, and whether a target is learned is precisely whether the flow can push `u` off it.

So attack each target through `d/dt u_i = a E_z[(h*(z) - fhat(z)) sigma'(<u,z>+sG) z_i] - reg` from
`u^0 ≈ 0`. A coordinate `i` leaves the origin only if the correlation between `z_i` and the residual-weighted
gradient is nonzero.

Take `h3(z) = z1z2z3` first. By symmetry under permuting `{1,2,3}` and the common origin start,
`u_1 = u_2 = u_3 =: u_{123}`, and the driving expectation `E_z[z1z2z3 · sigma'(u_{123}(z1+z2+z3)) z_1]` is a
*homogeneous* function of `u_{123}` that vanishes to high order at `0` — the only thing correlating with
`z_1` is the full triple `z1z2z3`, which contributes nothing at the origin. The coordinates never budge:
`u_{123}^t = 0` forever, and the risk is bounded below by `hat(h3)({1,2,3})^2 = 1`. `h3` is a leap straight
from nothing to degree-3 — non-MSP — and the flow is frozen. Plain mean-field SGD cannot learn it, like the
lazy baseline but for a different reason: not "the features are frozen" but "the gradient at the origin is
zero in the signal directions."

Now `h2(z) = z1z2 + z2z3 + z3z4`, the non-MSP chain. Its first support is `{1,2}` with `|S_1| = 2`, so the
flow would have to lift *two* coordinates at once. Near the origin the only term correlating with `z_1` is
`z1z2`, whose push `E_z[z1z2 sigma'(...) z_1] = E[z_2 sigma'(...)]` is gated by `u_2`, and symmetrically
`u_2`'s push is gated by `u_1`. Linearizing the pair,
`d/dt (u_1, u_2) = kappa [[0,1],[1,0]] (u_1, u_2)` with `kappa = a m_2 alpha_{12}` — a *homogeneous* linear
system, source exactly zero at the origin. Its eigenvalues are `±kappa`, so there is even an unstable
direction `u_1 = u_2`; but an unstable eigenvalue only amplifies an existing perturbation, and here the
initial perturbation is `(0,0)`, `e^{κt}·0 = 0`, so the pair never leaves. Contrast `h1`, whose first
equation `d/dt u_1 = a m_1 alpha_1` has a *constant, inhomogeneous* source — the degree-1 term — that forces
`u_1` off zero regardless of where it starts. The general statement: order `h*`'s supports greedily, and a
non-MSP target leaves an index set `Omega` whose new coordinates never appear alone; bounding the relevant
correlations gives `|d/dt u_i| <= K(1+t)^2 max_{j in Omega} |u_j|`, so every coordinate in `Omega` has a
derivative bounded by the maximum over `Omega`, all start at zero, and by Gronwall they stay at zero. `h2`
leaps by 2 at its first step; `Omega` is nonempty; mean-field SGD is stuck at `R >= sum_{unlearnable}
hat(h2)(S)^2 > 0`.

Finally `h1(z) = z1 + z1z2 + z1z2z3`, where I expect the recipe to come alive. The supports order
`{1} -> {1,2} -> {1,2,3}`, each adding one new coordinate — MSP / leap-1. Coordinate `1` enters through the
degree-1 term: `d/dt u_1 = a alpha_1 m_1` (with `m_r = sigma^{(r)}(0)`), a constant driver, so `u_1 ~ t`.
Once `u_1 != 0`, coordinate `2` gets a driver through `z1z2`: `d/dt u_2 = a alpha_{12} m_2 u_1 ~ t`, so
`u_2 ~ t^2`. Then `u_1, u_2 != 0` light coordinate `3` through `z1z2z3`:
`d/dt u_3 = a alpha_{123} m_3 u_1 u_2 ~ t·t^2`, so `u_3 ~ t^4`. The exponents are `1, 2, 4`, i.e.
`Theta(t^{2^{k-1}})` — each stair not merely slower but *super-linearly* slower, squaring the timescale of
the last. That is the concrete meaning of "orders of magnitude slower": `u_3` spends the overwhelming
majority of any finite window down in the `t^4` basement before its product driver `u_1 u_2` grows enough to
yank it up, and plain SGD takes that tiny gated gradient at face value with nothing adaptive to rescale it.
So my honest expectation for `h1` is a *partial* climb: the degree-1 term picked up first and most fully,
the degree-2 term partly, the degree-3 term barely if at all inside `T=4000`. Quantitatively `test_mse_h1`
should fall from NTK's `2.78` but need not approach the `1.0` a clean climb of all three stairs would reach;
I expect it to stay above `2`, still carrying most of the degree-3 monomial and a good part of the degree-2
one — a modest, real improvement, not a collapse. This is the mechanism `h3` and `h2` lack: no
single-new-coordinate entry point to start the cascade.

Two design choices keep the cascade alive. First, the activation: the chain `d/dt u_k = a alpha_{1..k} m_k
prod_{j<k} u_j` dies the instant any `m_k = sigma^{(k)}(0) = 0`, so I have to check the derivatives at the
origin. `tanh` is odd, `tanh''(0) = 0`, so `m_2 = 0` and the degree-2 stair never lights — it would stall
`h1` at the `z1z2` step. The un-shifted logistic has the same disease: `sigmoid''(x) = sigmoid'(x)(1 - 2
sigmoid(x))`, and at `0`, `sigmoid(0) = 1/2`, so `1 - 2·(1/2) = 0`, `m_2 = 0` again. Both "obvious" smooth
activations kill the second stair. The shift fixes it: `sigma(x) = sigmoid(x - 0.5)` evaluates at `-0.5`
where `sigmoid(-0.5) ≈ 0.378`, giving `m_1 = 0.378·0.622 ≈ 0.235`, `m_2 = 0.235·(1 - 2·0.378) ≈ 0.058`, and
`m_3 = g'[(1-2g)^2 - 2g']` at the same point `≈ 0.235·(0.060 - 0.470) ≈ -0.096` (sign immaterial) — the full
chain `m_1, m_2, m_3` alive, so every stair of `h1` up to degree 3 has a live driver. The `-0.5` shift is
not cosmetic; it moves evaluation off the logistic's symmetric center where `m_2` vanishes.

Second, the readout. The second (linear) phase, once features are set, is kernel regression with
`K(z,z') = E_a[sigma(<u(a),z>) sigma(<u(a),z')>)]`; in the Fourier basis this is `D(M_a + Delta)D` with
`M_a = (E_a[a^{beta(S)+beta(S')}])`, `beta(S) = sum_{k in S} 2^{k-1}`. Since `beta` hits all of
`{0,...,2^P-1}`, `M_a` is the Gram matrix of `1, X, ..., X^{2^P-1}` under random `a` — full rank *because*
`a` is random. If instead every `a_j = +1`, then `a^{beta(S)+beta(S')} = 1` everywhere, `M_a` is the
all-ones matrix, rank `1`, and the readout can represent only a single direction. Drawing `a ~ Unif({+1,-1})`
gives `E_a[a^m] = 0` for odd `m`, `1` for even, so `M_a` picks up off-diagonal structure that separates the
monomials and the Gram becomes full rank. So the *random* readout signs, an apparent throwaway init detail,
are what keep the readout phase well-posed. Collecting: `w ~ N(0,I_d)`, `a ~ Unif({+1,-1})`, shifted sigmoid,
`1/M` output, plain SGD at `lr=0.5` — the bare gradient flow, nothing adaptive.

So the prediction splits by leap: `h1` beats NTK — `test_mse_h1` below `2.78` and `score_h1` above `0.062`,
with the degree-3 stair (`t^4`) the question mark inside budget — while `h2` and `h3` stay pinned at the
trivial variances `≈3` and `≈1` (`score` `0.050` and `exp(-1) ≈ 0.37`), recovery near `1`, failing now for a
new reason: not frozen features but zero gradient at the origin. If that pattern holds, the diagnosis for the
next step is written — plain SGD climbs only *one* new coordinate at a time, so any leap greater than one
stalls, and the cure is to *manufacture* the missing lower stairs so the stuck coordinates get a first-order
driver.
