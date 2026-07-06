The scaffold ships with the worst defensible initializer — every weight an i.i.d. draw from `N(0, 0.01²)`,
a variance picked for nothing in particular — so the floor I have to beat is not "no initializer" but "an
initializer that ignores the layer's shape." Before I reach for anything clever I want to put a number on how
badly that floor fails, because the size of the failure sets how aggressive the fix has to be. Take the
cleanest case, the VGG stack, and strip BN out of my head for a moment so I can watch the raw weights act:
it is a plain chain of thirteen `Conv-BN-ReLU` triples, and ignoring BN it is a product of linear maps
`W_L … W_2 W_1` acting on the input. A single 3×3 conv with 64 input channels has `fan_in = 64·9 = 576`
incoming connections per output, so its forward second-moment map multiplies the input variance by
`fan_in · Var(W) = 576 · 10⁻⁴ ≈ 0.058` — the signal variance is cut by a factor of roughly seventeen at
every layer. Compound that over thirteen convs and the attenuation is `0.058¹³ ≈ 10⁻¹⁶`: a signal that
enters at unit scale reaches the classifier sixteen orders of magnitude down. That is not "small," it is
annihilated, and it is annihilated *geometrically*, at a rate I can read straight off the fan and the
variance. And the same collapse runs the other way. The backward pass through a linear layer multiplies the
incoming gradient by `Wᵀ`, whose second-moment map scales gradient variance by `fan_out · Var(W)`; for the
square 64-channel conv `fan_out = 576` too, so the backward factor is also `≈ 0.058`, and the gradient that
reaches the first conv is down by the same `10⁻¹⁶`. So the scaffold does not merely start the forward signal
dead — it starts the gradient dead, which is worse, because a dead gradient means SGD cannot even find the
direction to climb out. BN in the real network re-standardizes the forward activations so the net still
passes *something*, but that does not rescue the weights: they are still two orders of magnitude below any
shape-aware scale, and the pre-BN conv response and the gradient carried backward by `Wᵀ` both inherit that
scale before BN's per-channel rescaling can touch them. An initializer's one job is to set the weight scale
right; the scaffold sets it wrong by a factor of a hundred regardless of what BN does downstream. Symmetric,
depth-blind, and exactly the vanishing-signal start that 200 epochs of SGD struggle to dig out of.

The fan-scaling lineage — LeCun, Xavier, He — fixes the *average* of this story. It chooses the variance of
the entries of `W` so that the *expected* squared output norm equals that of the input, layer by layer:
Xavier targets `Var(W) = 2/(fan_in+fan_out)` for a symmetric unit-derivative nonlinearity, He corrects the
factor of two that ReLU eats. This is real progress — it makes the *mean* signal norm depth-stable — but it
is a statement about a single number, the second moment, and a matrix is not a number. A random Gaussian
scaled to the right variance still spreads its singular values. For a square `n×n` Gaussian rescaled by
`1/√n` the singular values follow the Marchenko–Pastur law on `[0, 2]`, and I do not have to take that on
faith — sampling one at `n = 1000` gives a smallest singular value near `0.0008` and a largest near `1.98`,
a condition number in the thousands. So even a perfectly He-scaled square layer amplifies some input
directions by close to two and crushes others to almost nothing; it is norm-preserving *on average* and
badly anisotropic in the worst case. And the worst case is what compounds. A product of `L` independent
Gaussian matrices does not just repeat one layer's spread — the log-singular-values of the product add up, so
the gap between the most-amplified and most-annihilated direction grows *linearly* in `L`. Concretely: if a
single He-Gaussian conv already puts a factor of `~2500` between its stiffest and softest direction, then
after only a handful of layers the product's condition number is astronomical, and there exists an input
direction that arrives at the head amplified by orders of magnitude while an orthogonal direction arrives
crushed to nothing. Fan-scaling stabilizes the mean and leaves the spectrum to chance, and the spectrum is
where depth does its damage.

Saxe, McClelland & Ganguli (2014) make the stronger demand, and it is the demand I want at the bottom of this
ladder: don't just match the mean, preserve the *whole* norm — every singular value exactly one. A matrix
with all singular values equal to one is orthogonal (or, for non-square, semi-orthogonal): `WᵀW = I` on the
smaller dimension. Such a map is an isometry — it rotates or reflects its input without changing any length —
and the composition of orthogonal maps is again orthogonal. So in a deep *linear* net an orthogonal init
gives exact dynamical isometry: the forward signal norm is conserved at every depth, and because the backward
Jacobian of a linear net is the transposed product, itself a product of orthogonal matrices, so is the
gradient norm. No vanishing, no exploding, at any depth, exactly rather than on average. Contrast the two
spectra directly: the He-Gaussian product's condition number blows up with depth as the log-spectrum spreads,
while the orthogonal product's condition number is pinned at exactly one at *every* depth, because a product
of isometries is an isometry. That is the qualitatively stronger guarantee fan-scaling cannot make, and it is
why orthogonality is the right first rung: it is the most that initialization-by-norm-preservation can
promise, and I want to anchor the climb at the strongest single form of the signal-preservation idea before I
start trading it away.

Now I have to translate "orthogonal" to the objects the scaffold actually hands me, because a conv weight is
not a matrix — it is a 4-D tensor of shape `(out_channels, in_channels, kh, kw)`. The standard move, the one
`nn.init.orthogonal_` implements, is to view the tensor as a 2-D matrix by flattening all but the first
dimension: rows are `out_channels`, columns are `in_channels·kh·kw`. Fill that matrix with a random
(semi-)orthogonal factor — sample a Gaussian of that shape, take its QR factorization, keep the orthogonal
factor `Q`, and fix the sign convention so the result is deterministic up to the random draw — and reshape
back. The cost is a single QR per layer at initialization; even VGG's widest conv flattens to a `512 × 4608`
matrix whose QR is a millisecond one-time expense against a 200-epoch schedule, so I pay it without a second
thought. In the usual case the flattened matrix is far wider than tall (`in·kh·kw ≫ out`), so the result is
*row*-orthogonal, `WWᵀ = I` on the output channels: every output channel gets a unit-norm, mutually-orthogonal
filter, preserving norm in the forward direction I care about most. I should be honest that this is not a
literally norm-preserving *convolution*. There is a construction that is — the delta / circular orthogonal
convolution, which pads a Kronecker-delta spatial kernel with an orthogonal channel matrix so the operator is
a true isometry on the feature map — and it is worth pausing on it because it is the theoretically cleaner
object. I reject it here for three concrete reasons: it is not what `nn.init.orthogonal_` exposes, so I would
be hand-building a delta kernel per layer; it starts every conv as a near-identity spatial response, which
fights the very 3×3 receptive-field structure the network is meant to learn; and a delta-in-space kernel
interacts awkwardly with BN's per-channel whitening. The matrix-level scheme gives the practical content of
the orthogonal idea — every filter unit-norm and mutually orthogonal — and it is the one the edit surface
supports without touching the graph, so I take it knowingly.

There is a shape subtlety in the flattening that I should trace rather than wave past, because it is the same
subtlety that will later decide MobileNetV2's fate, and I want to see it first on a benign case. Not every
conv flattens *wide*. VGG's very first conv takes the three input channels of the image, so its weight is
`(64, 3, 3, 3)`, flattening to a `(64, 27)` matrix — *taller* than wide, `out = 64 > 27 = in·kh·kw`. On a
tall matrix `orthogonal_` cannot make the 64 rows mutually orthogonal (they live in a 27-dimensional column
space and there are 64 of them), so it does the only thing it can: it makes the 27 *columns* orthonormal,
`WᵀW = I₂₇`. Is that a failure? I check what it means for the operator. The linear map here genuinely *is*
the dense matmul `W · patch`, where `patch ∈ ℝ²⁷` is the im2col vector of one receptive field, and a matrix
with orthonormal columns satisfies `‖W·patch‖² = patchᵀ WᵀW patch = ‖patch‖²` — an *isometric embedding* of
ℝ²⁷ into ℝ⁶⁴. So the tall first conv preserves the forward norm perfectly; the tallness costs nothing here.
The lesson I extract is precise and I will need it: `orthogonal_` on a tall matrix is fine *exactly when the
flattened matrix is the operator*, because then column-orthonormality is genuine input-norm preservation. It
is only a problem when the flattened matrix is a *fiction* — when the real operator is not that dense matmul.
Hold that; it is the whole MobileNetV2 story.

There is one scale subtlety I cannot skip, and it is exactly where orthogonality and ReLU collide. A pure
orthogonal map preserves norm — perfect for a linear net — but every conv here is followed by ReLU, which
discards the negative half of its pre-activations. For a zero-mean Gaussian pre-activation `z ~ N(0, σ²)`,
`E[relu(z)²] = σ²/2` exactly — half the variance survives — and I can confirm the factor rather than assert
it: drawing two million samples of `z ~ N(0, 9)` gives `E[relu(z)²] ≈ 4.500`, right on `σ²/2 = 4.5`. So an
exactly-orthogonal conv (singular values one) followed by ReLU halves the activation variance every layer;
over VGG's thirteen convs that is `(1/2)¹³ ≈ 1.2·10⁻⁴`, the same ReLU leak that forced He to put a factor of
two into the variance. The remedy is to scale the orthogonal matrix by a gain that undoes the halving, which
is `√2` — precisely `nn.init.calculate_gain('relu')`. Multiplying by `√2` sets every singular value to `√2`,
so each layer amplifies the pre-ReLU signal by `√2` and ReLU pulls it back to unit: the post-activation norm
holds across depth. And the backward pass inherits the same fix symmetrically — the ReLU derivative is a 0/1
mask that zeroes half the entries of the back-propagated gradient in expectation, halving its variance too,
and the `√2` gain on `Wᵀ` compensates it the same way it compensates the forward halving. It is worth seeing
that this is not a *different* target from He but the same one reached differently. A row-orthonormal matrix
scaled by `√2` has each row of squared norm two, spread over `fan_in = in·kh·kw` entries, so the per-entry RMS
is `√(2/fan_in)` — for the 576-fan layer that is `√(2/576) ≈ 0.0589`, which is exactly He's `fan_in` standard
deviation. So orthogonal-with-`√2`-gain *is* He in the second moment, with the singular-value spectrum
additionally pinned to a single value instead of scattered across Marchenko–Pastur. I am buying the same
variance plus spectral control, and the spectral control is the only thing I am paying extra for. I apply the
same rule to the `Linear` layers, since the classifier head and VGG's hidden FC sit behind ReLUs of the same
kind; biases go to zero, because at initialization there is no reason to offset any unit and a nonzero bias
would only inject an arbitrary shift into the carefully balanced pre-activations.

BatchNorm is the one piece I deliberately leave at its conventional setting `(γ=1, β=0)`, and I want to be
exact about why, because whether it makes the whole scheme redundant is the single most important question
this rung has to answer. At initialization BN whitens its input to unit variance per channel and then applies
the identity affine, so it neither amplifies nor suppresses — it re-standardizes the running statistics while
the `γ=1` affine stays out of the way of the orthogonal scaling I imposed on the convs. The tension is real
and I should name it precisely rather than hope it away: BN re-standardizes the forward variance at *every*
layer regardless of how I initialized the conv, so the *per-channel forward variance* — the diagonal of the
activation covariance — is something BN pins for free, and to that extent the forward-norm half of my
orthogonal guarantee is delivered downstream whether or not I paid for it. But — and this is the crack I am
betting on — BN standardizes each channel *independently*; it is a diagonal rescaling of the activations. It
does nothing to the *off-diagonal conditioning* of the weight map `W` itself. A He-Gaussian conv whose
singular values span `[0.0008, 1.98]` still, after BN, feeds the next layer through a map that stretches some
directions and crushes others; BN equalizes the per-channel *variances* but cannot re-round the *shape* of
`W`'s action on the incoming activation vector. Orthogonal init removes that anisotropy at the source, before
BN ever sees it. So the honest hypothesis is: on a deep plain stack, where the full spectrum is well-defined
and every layer's conditioning compounds, orthogonality should retain a real if modest edge that BN's diagonal
rescaling cannot supply; where the architecture defeats the spectrum for other reasons, the edge should
vanish and BN's free per-channel standardization should be all that survives. I am not going to resolve this
by getting clever with BN — I leave it at the neutral identity affine and add no bespoke scaling of my own on
top of it — because the whole point of the first rung is to test the orthogonal hypothesis in isolation, with
the simplest BN that lets the conv init speak. So the
rule is one uniform pass with no architecture branching, no depth arithmetic, no special treatment of
shortcuts: every `Conv2d` and `Linear` gets `orthogonal_` with gain `√2` and zero bias, every `BatchNorm2d`
gets `(1, 0)`.

Walking the three targets tells me where this should and should not pay off, and the cross-architecture spread
is what the leaderboard will actually grade. VGG-16-BN is the friendliest — a deep plain `Conv-BN-ReLU` chain
with no shortcuts, exactly the topology orthogonality was derived for — so if the orthogonal spectrum has an
edge anywhere it is here, where the full spectrum is genuinely well-defined and the clean per-layer
conditioning has nothing working against it. This is precisely the case where my BN-crack argument predicts a
survivable advantage: BN pins each channel's variance, but the compounding conditioning of thirteen stacked
convs is exactly what orthogonality removes and BN cannot. ResNet-56 is more delicate. Its main path is a
running sum `x_{l+1} = x_l + F_l(x_l)`, where each `F_l` is a residual branch of two 3×3 convs each ending in
BN. Orthogonality controls each conv *inside* `F_l`, but it says nothing about how the branch outputs
*accumulate* down the additive path. Let me make the accumulation quantitative using the BN I just fixed at
`γ=1`: the branch ends `conv2 → bn2`, and with `γ=1` that BN standardizes the branch output to unit variance
per channel, so each `F_l(x_l)` enters the sum at roughly unit scale. If the branches are roughly independent
at init, the main-path variance grows additively: after `L` blocks it is about `Var(x_0) + Σ_l Var(F_l) ≈
(1+L)·Var(x_0)`. ResNet-56 on CIFAR has 27 `BasicBlock`s, so taken at face value that is a factor of `28` in
variance, a `√28 ≈ 5.3×` inflation of the signal standard deviation by the time it reaches the head. The true
number is somewhat gentler — the net has three stages and a projection shortcut at each stage boundary
partially resets the running sum — but the *mechanism* does not care about the exact count: the signal
inflates with depth, and it inflates because branches *add*, not because any single conv is mis-scaled.
Orthogonality buys clean per-branch maps but not residual-accumulation control, so I expect it to be merely
ordinary on ResNet — in the fan-scaling neighborhood, not above it, because the thing that limits a 56-layer
res-net is the accumulation my per-conv isometry never touches.

MobileNetV2 is the hardest case and the one worth naming most carefully, because it is where the elegant
theory has the least grip on the actual weights, and it is exactly the fiction-versus-operator distinction I
flagged on VGG's first conv, now turned malignant. Its workhorse is the depthwise 3×3 conv, a conv with
`groups = channels` so each output channel sees exactly one input channel. Its weight tensor is `(channels,
1, 3, 3)`, flattening to shape `(channels, 9)` — far *taller* than wide. On a tall matrix `orthogonal_` can
only make the nine *columns* orthonormal: `WᵀW = I₉` is achievable, but `WWᵀ` is a `channels × channels`
matrix of rank at most nine, so for any realistic channel count in the hundreds it can *never* equal the
identity — row orthogonality is not degraded, it is impossible. But the rank deficiency is the *shallow*
problem. The deep problem is that, unlike VGG's tall first conv, here the flattened `(channels, 9)` matrix is
*not the operator*. A depthwise conv never forms the dense product `W · patch`; it is block-diagonal, output
channel `c` is just `x_c` convolved with its own 3×3 filter `f_c`, with no mixing across channels at all. So
when `orthogonal_` orthonormalizes the nine columns, it is imposing `Σ_c f_c(j) f_c(k) = δ_{jk}` — a
*cross-channel* condition tying together filters that the operator keeps completely independent. It is
orthonormalizing an axis the convolution is physically indifferent to, and getting *nothing* in return: the
per-filter norm behavior that actually governs a depthwise layer is the frequency response of each `f_c`
alone, which the column-orthonormalization neither sets nor respects. That is the layer type where I would
expect orthogonal init to lose its advantage outright, possibly to fall *behind* a plain fan-scaled Gaussian
that at least gives each filter the right per-tap variance without pretending to an isometry it cannot
deliver — and MobileNetV2 is built almost entirely out of that layer type.

So the step-1 edit is the literal orthogonal fill: iterate `model.modules()`; for every `Conv2d` apply
`orthogonal_` with gain `√2`; for every `Linear` the same with zero bias; for every `BatchNorm2d` set
`(weight=1, bias=0)`. No architecture branching, no depth arithmetic, no special treatment of shortcuts — one
uniform rule, the honest way to test the orthogonal hypothesis before complicating it. What I expect, stated
so the next rung has something concrete to push against: on VGG-16-BN, competitive and quite possibly best,
because the clean spectrum is its best shot on the topology it was built for and the compounding conditioning
is the one thing BN's diagonal rescaling cannot hand me for free; on ResNet-56, unremarkable, because
orthogonality fixes per-layer conditioning but not the residual accumulation that dominates a 56-layer net, so
it should land in the fan-scaling neighborhood; on MobileNetV2/FashionMNIST, weakest, because the depthwise
convs defeat the matrix-orthogonality construction on the exact layer that dominates the network, and defeat
it not by degrading it but by making it orthonormalize a fictitious axis. If that cross-architecture pattern
holds — fine on the plain stack, ordinary on the residual net, weakest on the depthwise mobile net — then the
diagnosis for the next rung writes itself. Two things would be implicated: BatchNorm already re-standardizes
per-channel variance at every layer, so the per-layer isometry is partly redundant wherever BN sits, and the
architectures that matter are dominated by either residual accumulation or depthwise filters that orthogonality
cannot touch. That would point the next move away from the expensive full-spectrum control and back toward
getting the *second moment* exactly right for ReLU — the He variance — applied uniformly and cheaply, to see
whether that simpler, BN-friendly target already matches the orthogonal spectrum it cost so much to impose. The
distilled module and its exact code live in the answer.
