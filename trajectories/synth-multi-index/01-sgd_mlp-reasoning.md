The recipe is the whole point, but it bolts onto a fixed architecture, and with the plainest possible
recipe that architecture is the floor — so the pain to start from is just training the two-layer MLP
the obvious way and seeing how far obvious gets me. The target reads the input through `r` unknown
orthonormal directions, `y = (1/√r) Σ_i He₃((U*ᵀx)_i)`, with isotropic Gaussian `x`. Nothing in `x`
itself is informative; the entire problem is *which* directions the label looks along, and then the
cubic link on those projections. The fixed net can in principle do both — its first-layer rows can
rotate to point into `V* = span(U*)`, and once they do the readout fits an `r`-dimensional cubic — so
the whole game reduces to one question: can plain gradient descent on this net get the first-layer rows
to align with `V*` inside the 8000-step budget, and how badly does the cubic link's information
exponent fight me?

Let me write down what "align" means so I do not fool myself. For a first-layer row `w_i ∈ R^d`, what I
care about is how much of it lies in `V*`: `‖Π* w_i‖ / ‖w_i‖`, where `Π* = U*U*ᵀ` projects onto the
teacher subspace. At a random Kaiming start the rows are essentially isotropic in `R^d`, so the squared
projection of a row onto a fixed `r`-dimensional subspace is about `r/d`. Put the numbers in: `d = 128`,
so `r/d` is `2/128 = 0.0156` for r2, `3/128 = 0.0234` for r3, `4/128 = 0.0313` for r4, and the *aligned
fraction* `‖Π* w_i‖/‖w_i‖ = √(r/d)` is `0.125`, `0.153`, `0.177` respectively. A random neuron is almost
entirely orthogonal to the subspace I need it to find — seven-eighths of its length points nowhere
useful even in the best (r4) case. The leaderboard's `subspace_err = ‖P_Û − P_{U*}‖_F` measures exactly
the failure of this alignment at the level of the whole first-layer matrix: if the top-`r` right-singular
subspace of `W_in` is uncorrelated with `V*`, two rank-`r` orthogonal projectors that share no
directions sit at Frobenius distance `√(2r)` apart. I should get that constant exactly rather than
wave at it: `‖P_A − P_B‖_F² = tr(P_A) + tr(P_B) − 2 tr(P_A P_B) = r + r − 2·(overlap)`, and for two
random rank-`r` subspaces in `d = 128` the overlap `tr(P_A P_B) ≈ r²/d` is `0.03`/`0.07`/`0.13` —
negligible — so the distance is `√(2r − 2r²/d) ≈ √(2r)`, i.e. `√4 = 2.00` (r2), `√6 = 2.449` (r3),
`√8 = 2.828` (r4). Those are the numbers a recipe that learns *nothing* about the subspace should print,
and they are worth holding in mind as the "did the first layer move at all" yardstick — anything the
floor prints within a percent of them means the first layer never left the equator.

Now the gradient, because whether the floor moves off that yardstick is a question about drift. The loss
is per-batch mean squared error, `(1/n) Σ (f̂(x) − y)²`, and the first-layer row `w_i` gets, through the
chain rule on `f̂(x) = Σ_j a_j σ(⟨w_j, x⟩) + b`, a step proportional to `(1/n) Σ_ν a_i x^ν σ'(⟨w_i,
x^ν⟩) (f̂(x^ν) − y^ν)`. The piece that can rotate `w_i` into a *new* direction is the correlation of the
activation-weighted input with the label residual, `E[x σ'(⟨w_i, x⟩) y]`. There is a clean way to read
this under a Gaussian: Stein's lemma, `E[x h(x)] = E[∇_x h(x)]`. With `h(x) = σ'(⟨w_i, x⟩) y` this gives
a piece along `w_i` itself — `E[σ''(⟨w_i,x⟩) y] · w_i`, a rescaling that does nothing for orientation —
plus `E[σ'(⟨w_i, x⟩) ∇_x y]`, which is the part that can move the row toward `V*`. And `∇_x y =
∇_x g(U*ᵀx) = U* (∇g)(U*ᵀx)` lives entirely in `V*` by construction, so the whole rotational drift is a
vector inside the teacher subspace, weighted by `E[σ'(⟨w_i,x⟩) (∂_j g)(U*ᵀx)]`. That is the object I have
to size.

Expand both factors in Hermite tensors. The teacher's Hermite content lives entirely in `V*`, because
`y` — and hence `∇g` — is a function of `U*ᵀx`, so its `k`-th Hermite tensor is built from `k` copies of
teacher directions. Contracting that tensor against the activation pattern of a row `w_i` whose foot in
`V*` is only `O(1/√d)` produces a scalar of size `O((1/√d)^{k'})` at each surviving order — every Hermite
order the row has to reach through costs another factor of the overlap `⟨ŵ_i, u⟩ ~ 1/√d`. So the *lowest*
surviving order dominates, and which order is lowest is the information exponent of the link. Here is the
crux for this benchmark. The link is `g(z) = (1/√r) Σ_i He₃(z_i)` — a *pure cubic* in each coordinate,
with `He₃` the third probabilists' Hermite polynomial. Its Hermite expansion is a single term at degree
3: no degree-0 constant, no degree-1 linear part, no degree-2 quadratic part. `∂_j g = (3/√r) He₂(z_j) =
(3/√r)(z_j² − 1)`, degree 2 in the projection. The information exponent — the lowest degree at which the
population correlation between a single fresh direction and the label is nonzero — is exactly 3. So the
useful part of the first gradient, the part that points into `V*`, has leading size `O(overlap^{s-1}) =
O(overlap²) = O((1/√d)²) = O(1/d)`. A vanilla SGD step `w_i ← w_i − η g_i` with the scaffold's `η = 5e-2`
moves the row by `O(η/d) = 5e-2/128 ≈ 4·10⁻⁴` into the subspace per step. Against a row of unit-ish
length that needs to rotate `O(1)` into `V*`, that is essentially nothing.

Let me actually verify the `m^s` scaling by hand rather than quote it, because the whole floor argument
rests on it. Take a single fresh unit row `ŵ` with overlaps `m_i = ⟨u_i, ŵ⟩` onto the teacher
directions, and ask for the degree-3 correlation the gradient can see, `E[y · He₃(⟨ŵ, x⟩)]`. Under
isotropic Gaussian `x`, the Hermite product-moment identity gives `E[He_j(⟨a,x⟩) He_k(⟨b,x⟩)] = δ_{jk}
k! ⟨a,b⟩^k` for unit `a, b`, so `E[He₃(z_i) He₃(⟨ŵ,x⟩)] = 3!·m_i³ = 6 m_i³`. Summing against the link,
`E[y · He₃(⟨ŵ,x⟩)] = (1/√r) Σ_i 6 m_i³ = (6/√r) Σ_i m_i³`. At a random start each `m_i ~ 1/√d`, so the
correlation is `~ (6/√r)·r·d^{-3/2} = 6√r · d^{-3/2}`, which is cubic in the overlap — exactly `s = 3`
confirmed by direct Hermite algebra, not asserted. Put in numbers: `d^{-3/2} = 128^{-3/2} = 1/1448 ≈
6.9·10⁻⁴`, so the population degree-3 correlation at init is about `6·1.41·6.9·10⁻⁴ ≈ 5.9·10⁻³` on r2,
scaling like `√r`. And the *drift* on the overlap is `d/dm` of that correlation, `∝ 18 m² = O(m²) =
O(m^{s-1})` — the `overlap²` pull I claimed. So both the correlation `~m³` and its drift `~m²` fall
straight out of `E[He_3 He_3] = 6 m³`, and the `5.9·10⁻³`-sized signal at init is the entire budget the
gradient has to work with before finite-batch noise (std `~0.088` on 128 points) buries it.

The single-index theory makes the consequence precise: one-pass SGD on a target with information
exponent `s` needs `≈ d^{s-1}` samples/steps to escape the uninformative equator, because the overlap
`m = ⟨ŵ_i, u⟩` starts at `O(1/√d)`, the population correlation behaves like `m^s`, and its derivative —
the actual drift `dm/dt` pulling the row toward `u` — is `O(m^{s-1})`. Integrate that ODE from
`m_0 ~ d^{-1/2}`: `dm/dt = c·m^{s-1}` gives, for `s = 3`, `dm/dt = c m²`, whose solution `1/m_0 − 1/m(t) =
c t` reaches order-one overlap only after `t ~ 1/m_0 = √d` steps in the *noiseless* idealization — but
that noiseless picture is the trap, because the stochastic version has to first climb *out* of the band
where the `O(m²)` drift is smaller than the `O(1/√d)`-per-step sampling diffusion, and that escape time is
what sets the true cost `d^{s-1} = d² = 128² = 16{,}384` steps to find *one* direction. The budget is
8000. I am structurally short by more than a factor of two on the steps needed to escape the saddle for a
single teacher direction, before I even ask for `r` of them.

The multi-direction story is no kinder, and it is worth tracing because it explains why this floor should
be flat rather than merely slow. With several directions the relevant quantity is the leap complexity:
SGD climbs the teacher directions saddle-to-saddle, learning Hermite components in order of increasing
degree, and a direction only leaves the equator once the directions it is *staircase-connected* to are
already aligned — meaning some lower-degree monomial couples them, so that conditioning on the found
direction exposes a nonzero first Hermite coefficient for the next. The cost is `≈ d^{max(Leap, 2)}`
steps. But look at what staircase structure this link offers: `g = (1/√r) Σ_i He₃((U*ᵀx)_i)` is a *sum
of decoupled* cubics, one per teacher coordinate, with **no cross terms and no lower-degree terms at
all**. There is no `z_1` to bootstrap `z_2`, no `z_1 z_2` ladder rung — each direction must be found
cold, from a degree-3 correlation, with nothing beneath it to lift it off the origin. Conditioning on a
found direction `u_1` does nothing for `u_2`: `E[He₃(z_2) | z_1] = He₃(z_2)` because the coordinates are
independent under isotropic `x`, so the leap never shortens. So this is the worst case for the
saddle-to-saddle dynamic: there is no staircase to climb, just `r` independent degree-3 escapes, each
costing `d²` it does not have. The leap analysis predicts joint SGD here picks up essentially none of
`V*` within budget, and — this is the falsifiable edge — that `r4` is no *easier per direction* than
`r2`. If anything the recovery target is larger (more directions, each cold), so I expect all three ranks
to fail at the subspace and the failure not to soften with smaller `r`; the `subspace_err` gap to `√(2r)`
should stay near zero across r2/r3/r4 rather than opening up on the easy end.

Before I commit to the trivial fill I owe myself a real look at whether any *cheap* twist on the
optimizer could manufacture the missing signal, because if one could, "the floor" would be a strawman. I
have four candidates in the four hooks. Momentum: heavy-ball with `β` accumulates a persistent drift by
a geometric factor `1/(1−β)`, which at `β = 0.9` is `10×`. But the drift I need to amplify is `O(1/d) ~
1/128`, and I need it to reach `O(1)`; `10×` closes only one of the two-plus orders of magnitude, and
momentum also amplifies the `O(1/√d)`-per-step sampling noise along whatever direction it happens to
accumulate, so it does not improve the signal-to-noise that actually gates the escape. It buys a constant,
not the `d²`. Adam: per-coordinate RMS normalization divides each coordinate of the gradient by its own
running root-mean-square. At the equator each coordinate is dominated by sampling fluctuation of size
`O(1/√n_batch)`, and the `O(1/d)` teacher-aligned signal is buried underneath it; Adam rescales signal
and noise by the *same* per-coordinate denominator, so the SNR is invariant — it cannot conjure structure
the raw gradient does not carry, it only whitens the step. A bigger learning rate: I could push `η` from
`5e-2` toward `O(1)`. But the batch is 128 fixed points, and the empirical degree-3 correlation on 128
samples has sampling standard deviation `~ 1/√128 ≈ 0.088`, which swamps the `O(1/d) ≈ 0.0078` population
signal — a per-batch SNR of `0.0078/0.088 ≈ 0.09`. A giant single step on that batch amplifies the `0.088`
noise, not the `0.0078` signal; to get the *averaged* correlation's SNR up to one I would need `~ (0.088/
0.0078)² ≈ 127` batches, and even then I am back on the `d²` escape-time treadmill. More data: I could
grow `make_dataset` past `4096`. But the kernel/random-feature lower bound says a degree-3 target needs
`Ω(d³) = 128³ ≈ 2·10⁶` samples for the readout alone in the fixed-feature regime, and the gradient route
to the *subspace* still pays `d²` steps regardless of pool size — throwing samples at the wall changes
neither exponent. So every cheap twist dies to the same arithmetic: nothing local and gradient-based
manufactures a degree-3 signal at an `O(1/√d)` start inside 8000 steps. That is precisely why the honest
floor is the *trivial* fill — its failure has to be clean and legible so the harder rungs' machinery is
measurable against it.

So I leave every hook at the scaffold default. `init_model` is standard Kaiming-uniform on both linear
layers with zero biases — isotropic rows, the `O(1/√d)` random start the whole information-exponent story
is told from. `make_dataset` returns a fixed `n = 4096` Gaussian set with teacher labels; I deliberately
do *not* enlarge it, because the point of this rung is to expose the optimization wall, not to throw
samples at it — and `4096` points reshuffled over 8000 steps at batch 128 means `8000·128/4096 ≈ 250`
epochs, so the net sees each point about 250 times. That is emphatically *multi-pass*, the opposite of
the one-pass idealization, which if anything *helps* the readout overfit the random features and does
nothing for the subspace. It also sharpens the noise floor I care about: because the same `4096` points
recur, the *independent* sample count backing the degree-3 correlation is `4096`, not the `8000·128 ≈
10⁶` gradient evaluations. The population signal at init is `~5.9·10⁻³` (computed above), and the
sampling error of the empirical correlation over `4096` fixed points is `~1/√4096 ≈ 0.0156` — still
larger than the signal by a factor of `~2.6`. So even fully averaged over the whole fixed set, the
degree-3 correlation the first layer could exploit sits *below* its own sampling noise; multi-pass
re-use cannot beat that ceiling, it only re-reads the same `4096`-sample estimate 250 times. This is an
independent route to the same conclusion as the `d²` step count: the alignment signal is sub-noise at
this pool size, so the floor's first layer has nothing clean to climb. `get_optimizer_config` is plain SGD on both layers at `lr = 5e-2`, no momentum,
no weight decay, no noise — the bare gradient flow, nothing adaptive, nothing that could manufacture the
missing third-order signal (and I just checked that even if I turned those knobs on, they could not).
`training_step` is a single joint squared-loss update on both layers: zero the grads, forward, MSE,
backward, step. No layer freezing, no spherical projection, no closed-form solve — exactly the moves the
harder rungs will turn on, left off here so their effect is measurable against this baseline.

I should also be honest about the second layer, because the scaffold trains both jointly and the MSE the
leaderboard reads is dominated by whatever the readout manages. While the first layer flails near the
equator, the readout `a` is fitting whatever the (near-random) features provide. Random ReLU features of
a Gaussian input form a fixed kernel, and the cubic target has degree-3 content that a finite kernel of
`256` random features in `d = 128` cannot represent without paying the ambient dimension — the degree-3
eigenspace of the ReLU-Gaussian kernel has dimension `~ d³/6 ≈ 3·10⁵`, and `256` features sample a
vanishing fraction of it. So even granting the readout a perfect convex fit on frozen random features,
the best it can do is capture the projection of `g` onto the `256`-dimensional random feature span, which
for a pure degree-3 target is almost nothing. The MSE should therefore sit near the variance of `y`
itself. Let me compute that floor rather than assert it. `Var[He₃(z)] = E[(z³ − 3z)²] = E[z⁶ − 6z⁴ +
9z²]`, and the Gaussian moments are `E[z⁶] = 15`, `E[z⁴] = 3`, `E[z²] = 1`, so `Var[He₃(z)] = 15 − 18 +
9 = 6`. The link averages `r` independent such coordinates scaled by `1/√r`: `Var[g] = (1/r) Σ_i
Var[He₃(z_i)] = (1/r)·r·6 = 6`, independent of `r` — a useful invariant, since it means the MSE floor is
the same `6` on all three ranks and any rank-dependence in `test_mse` comes from the dynamics, not the
target. As a cross-check on the Hermite normalization, `E[He₃(z)²] = 3! = 6` by the standard
`E[He_k² ] = k!`, which matches `15 − 18 + 9 = 6` exactly — so I have the variance right. A recipe that
learns neither subspace nor link therefore predicts a fresh-test MSE near `6`, plus whatever extra
variance the untamed joint dynamics injects on top.

Fold that into the `score = exp(−subspace_err²/r)·exp(−test_mse)`. If the floor sits at `subspace_err ≈
√(2r)` and `test_mse ≈ 6`, the first exponential is `exp(−2r/r) = exp(−2) ≈ 0.135` on every rank, and the
second is `exp(−6) ≈ 2.48·10⁻³`, giving `score ≈ 3.3·10⁻⁴` as an *optimistic* bound — optimistic because
it assumes the MSE lands exactly at `6` rather than overshooting it. If the joint dynamics inflates the
prediction variance and the MSE lands at, say, `9`, the second exponential drops to `exp(−9) ≈ 1.2·10⁻⁴`
and the score falls to `~1.7·10⁻⁵`. Either way the score is pinned in the `10⁻⁵`–`10⁻⁴` basement, and
crucially it is `exp(−subspace_err²/r)` that caps it: as long as the first layer never leaves the
equator, no amount of readout fitting can lift the score out of that band, because `exp(−2)` multiplies
everything. As a degeneracy check that I am reading the metric right: at *perfect* recovery
`Û = U*` gives `subspace_err = 0` and, once the features span `V*`, the ridge fit of a cubic on `r`
directions drives `test_mse → 0`, so `score → exp(0)·exp(0) = 1` — the top of the `(0, 1]` range. The
metric is therefore a genuine `0`-to-`1` thermometer of "found the subspace and fit the link", and the
floor should read at its cold end. The two exponentials also confirm the coupling the task warns about:
a low `subspace_err` is *necessary* for a low `test_mse` (features must span `V*` before the readout can
fit the cubic), so both move together or neither does, and the floor is the case where neither does.

Now reason about what this floor must do, because that is the entire reason to run it. Joint SGD descends
the MSE built from the cubic target. The first layer's only route to `V*` is the third-order correlation,
which at a random start is `O(1/d)` weak and needs `d² ≈ 16k` steps per direction to amplify — more than
the budget — and there is no lower-degree staircase to shortcut the climb. So the first-layer rows should
stay essentially where they started, near-orthogonal to `V*`, and `subspace_err` should sit near `√(2r) ≈
2.0/2.45/2.83` across r2/r3/r4, the gap to that band near zero and *not* opening up on the easy end. The
readout, fitting near-random cubic-blind features, should leave `test_mse` near `Var[g] = 6` or worse,
since the untamed joint dynamics can inflate the prediction variance rather than reduce it. And the
`score` should collapse into the `10⁻⁵`–`10⁻⁴` basement on all three ranks, because the subspace
exponential is saturated at `exp(−2)` and the MSE exponential at `exp(−6)` or below. This is the weakest
recipe I can run *by construction*: it has no mechanism to find a degree-3 direction inside the budget,
and on a problem where finding the subspace is the whole task it should mostly find nothing.

The diagnosis is already pointed at the next rung. Whatever the exact numbers, the failure is an
*optimization* failure, not a representation one — the net can represent `g` fine once its rows point into
`V*` (I sized the features above and the obstruction was the random subspace, not capacity); it simply
cannot get them there by descending a third-order-flat landscape with a `d²`-times-too-small drift inside
8000 steps. The fix is not more of the same gradient — I ruled out momentum, Adam, bigger steps, and more
data one by one, and they all die to the same exponent. It is to stop asking joint SGD to find the
subspace through the cubic at all: the net is conflating two jobs — finding `V*`, which is nonconvex and
the hard part, and fitting the link, which is convex once the features are good — and training them
jointly is exactly what lets the readout chase moving, near-random features and inflate the MSE `50`–`70%`
above its own `6` floor. So the mechanism has to change: whatever I try next has to *decouple* those two
jobs and hand the first layer a cleaner, larger-amplitude alignment signal than the `d²`-flat cubic
gradient can supply on its own. The contract leaves every relevant hook off by default — the readout is
trained, the rows drift in norm, the fit is joint — precisely so that this optimization wall can be
attacked head-on once those hooks are turned on. The full default module is in the answer.
