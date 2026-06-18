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
projection of a row onto a fixed `r`-dimensional subspace is about `r/d` — with `d = 128` and `r ∈
{2,3,4}` that ratio starts at roughly `0.016`–`0.031`, i.e. `‖Π* w_i‖ ~ √(r/d) = O(1/√d)`. A random
neuron is almost entirely orthogonal to the subspace I need it to find. The leaderboard's
`subspace_err = ‖P_Û − P_{U*}‖_F` measures exactly the failure of this alignment at the level of the
whole first-layer matrix: if the top-`r` right-singular subspace of `W_in` is uncorrelated with `V*`,
two rank-`r` orthogonal projectors that share no directions sit at Frobenius distance `√(2r)` apart —
about `2.0` for `r=2`, `2.4` for `r=3`, `2.8` for `r=4`. Those are the numbers a recipe that learns
*nothing* about the subspace should print, and they are worth holding in mind as the "did the first
layer move at all" yardstick.

Now the gradient. The loss is per-batch mean squared error, `(1/n) Σ (f̂(x) − y)²`, and the
first-layer row `w_i` gets, through the chain rule on `f̂(x) = Σ_j a_j σ(⟨w_j, x⟩) + b`,

a step proportional to `(1/n) Σ_ν a_i x^ν σ'(⟨w_i, x^ν⟩) (f̂(x^ν) − y^ν)`. The piece that can rotate
`w_i` into a *new* direction is the correlation of the activation-weighted input with the label
residual, `E[x σ'(⟨w_i, x⟩) y]`. There is a clean way to read this under a Gaussian: Stein's lemma,
`E[x h(x)] = E[∇_x h(x)]`. With `h(x) = σ'(⟨w_i, x⟩) y` this gives a piece along `w_i` itself (a
rescaling, not alignment) plus `E[σ'(⟨w_i, x⟩) ∇_x y]`, which is the part that can move the row toward
`V*`. Expand both in Hermite tensors. The teacher's Hermite content lives entirely in `V*`, because
`y` is a function of `U*ᵀx`, so its `k`-th Hermite tensor is built from `k` copies of teacher
directions. Contracting that tensor against `k` copies of a row `w_i` whose foot in `V*` is only
`O(1/√d)` produces a scalar of size `O((1/√d)^k)`: every Hermite order costs another factor of
`1/√d`. So the *lowest* surviving order dominates, and which order is lowest is the information
exponent of the link.

Here is the crux for this benchmark. The link is `g(z) = (1/√r) Σ_i He₃(z_i)` — a *pure cubic* in each
coordinate, with `He₃` the third probabilists' Hermite polynomial. It has no linear part, no quadratic
part: its first nonzero Hermite coefficient is at degree 3. The information exponent is exactly 3. So
the useful part of the first gradient — the part that points into `V*` — has size `O((1/√d)^{3-1}) =
O(1/d)`. A vanilla SGD step `w_i ← w_i − η g_i` with the scaffold's `η = 5e-2` moves the row by `O(η/d)`
into the subspace, which against `d = 128` is essentially nothing. The single-index theory makes the
consequence precise: one-pass SGD on a target with information exponent `s` needs `≈ d^{s-1}`
samples/steps to escape the uninformative equator, because the overlap `⟨w_i, u⟩` starts at `O(1/√d)`,
the population correlation behaves like `overlap^s`, and its derivative — the actual drift pulling the
row toward `u` — is `O(overlap^{s-1})`. For `s = 3` that is `d^{s-1} = d² = 128² ≈ 16{,}384` steps to
find *one* direction, and the budget is 8000. I am structurally short of the steps needed to escape the
saddle even for a single teacher direction, let alone `r` of them.

The multi-direction story is no kinder, and it is worth tracing because it explains why this floor
should be flat rather than merely slow. With several directions the relevant quantity is the leap
complexity: SGD climbs the teacher directions saddle-to-saddle, learning Hermite components in order of
increasing degree, and a direction only leaves the equator once the directions it is
*staircase-connected* to are already aligned — meaning some lower-degree monomial couples them, so that
conditioning on the found direction exposes a nonzero first Hermite coefficient for the next. The cost
is `≈ d^{max(Leap, 2)}` steps. But look at what staircase structure this link offers: `g = (1/√r) Σ_i
He₃((U*ᵀx)_i)` is a *sum of decoupled* cubics, one per teacher coordinate, with **no cross terms and no
lower-degree terms at all**. There is no `z_1` to bootstrap `z_2`, no `z_1 z_2` ladder rung — each
direction must be found cold, from a degree-3 correlation, with nothing beneath it to lift it off the
origin. So this is the worst case for the saddle-to-saddle dynamic: there is no staircase to climb,
just `r` independent degree-3 escapes, each costing `d²` it does not have. The leap analysis predicts
that joint SGD here picks up essentially none of `V*` within budget, and that `r4` (four cold degree-3
directions) is no easier per-direction than `r2` — if anything the recovery target is larger, so I
expect all three ranks to fail at the subspace and the failure not to soften with smaller `r`.

I should also be honest about the second layer, because the scaffold trains both jointly. While the
first layer flails near the equator, the readout `a` is fitting whatever the (near-random) features
provide. Random ReLU features of a Gaussian input form a fixed kernel, and the cubic target has
degree-3 content that a finite kernel of `256` random features in `d = 128` cannot represent without
paying the ambient dimension. So even granting the readout a perfect convex fit on frozen random
features, the best it can do is capture the low-order (here, zero) projection of `g` onto the random
feature span — which for a pure cubic is almost nothing. The MSE should therefore sit near the variance
of `y` itself. Let me estimate that floor: `Var[He₃(z)] = E[(z³−3z)²] = E[z⁶ − 6z⁴ + 9z²] = 15 − 18 +
9 = 6` for a standard Gaussian coordinate, and `g = (1/√r) Σ_i He₃(z_i)` averages `r` independent such
terms scaled by `1/√r`, giving `Var[g] = (1/r)·r·6 = 6`. So a recipe that learns neither the subspace
nor the link predicts a fresh-test MSE near `6`, plus whatever extra variance the untamed joint
dynamics injects. A near-`6`-or-worse `test_mse` paired with a near-`√(2r)` `subspace_err` is the
signature of the floor, and the `score = exp(−subspace_err²/r)·exp(−test_mse)` collapses to nearly zero
under both — `exp(−2r/r)·exp(−6) = exp(−2)·exp(−6) ≈ 3·10⁻⁴` as an optimistic bound, smaller once the
MSE overshoots 6.

Given all of that, the step-1 recipe is the *trivial* fill, and trivial is the right choice precisely
because its failure has to be clean and legible. I leave every hook at the scaffold default. `init_model`
is standard Kaiming-uniform on both linear layers with zero biases — isotropic rows, the `O(1/√d)`
random start the whole information-exponent story is told from. `make_dataset` returns a fixed `n =
4096` Gaussian set with teacher labels; I deliberately do *not* enlarge it, because the point of this
rung is to expose the optimization wall, not to throw samples at it — and `4096` points reshuffled over
8000 steps means the net sees each point about twice, so this is closer to multi-pass than to the
one-pass idealization, which if anything *helps* the readout overfit the random features and does
nothing for the subspace. `get_optimizer_config` is plain SGD on both layers at `lr = 5e-2`, no
momentum, no weight decay, no noise — the bare gradient flow, nothing adaptive, nothing that could
manufacture the missing third-order signal. `training_step` is a single joint squared-loss update on
both layers: zero the grads, forward, MSE, backward, step. No layer freezing, no spherical projection,
no closed-form solve — exactly the moves the harder rungs will turn on, left off here so their effect
is measurable against this baseline.

Now reason about what this floor must do, because that is the entire reason to run it. Joint SGD
descends the MSE built from the cubic target. The first layer's only route to `V*` is the third-order
correlation, which at a random start is `O(1/d)` weak and needs `d² ≈ 16k` steps per direction to
amplify — more than the budget — and there is no lower-degree staircase to shortcut the climb. So the
first-layer rows should stay essentially where they started, near-orthogonal to `V*`, and
`subspace_err` should sit near `√(2r) ≈ 2.0/2.4/2.8` across `r2/r3/r4`. The readout, fitting near-random
cubic-blind features, should leave `test_mse` near `Var[g] = 6` or worse, since the untamed joint
dynamics can inflate the prediction variance rather than reduce it. And the `score` should collapse to
near zero on all three ranks, because both exponentials are saturated against it. This is the weakest
recipe I can run *by construction*: it has no mechanism to find a degree-3 direction inside the budget,
and on a problem where finding the subspace is the whole task it should mostly find nothing.

The diagnosis is already pointed at the next rung. Whatever the exact numbers, the failure is an
*optimization* failure, not a representation one — the net can represent `g` fine once its rows point
into `V*`; it simply cannot get them there by descending a third-order-flat landscape with a
million-times-too-small drift inside 8000 steps. The fix is not more of the same gradient. It is to
stop asking joint SGD to find the subspace through the cubic at all: separate the two jobs the net is
conflating, freeze the readout and give the first layer a cleaner, larger-amplitude alignment signal,
then fit the link in closed form once the features are good. That is the two-stage move, and it is what
the next rung turns on — every hook the default leaves off (frozen readout, spherical rows, a stage-2
ridge solve) exists in the contract precisely so the floor's optimization wall can be attacked head-on.
The full default module is in the answer.
