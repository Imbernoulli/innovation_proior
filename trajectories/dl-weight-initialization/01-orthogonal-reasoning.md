The scaffold ships with the worst defensible initializer — every weight an i.i.d. draw from `N(0, 0.01²)`,
a variance picked for nothing in particular — so the floor I have to beat is not "no initializer" but "an
initializer that ignores the layer's shape." The size of that failure sets how aggressive the fix has to be,
so I put a number on it. Take the cleanest case, the VGG stack, and strip BN out for a moment: it is a chain
of thirteen `Conv-BN-ReLU` triples, a product of linear maps `W_L … W_2 W_1`. A single 3×3 conv with 64
input channels has `fan_in = 64·9 = 576`, so its forward second-moment map multiplies input variance by
`fan_in · Var(W) = 576 · 10⁻⁴ ≈ 0.058` — a factor of seventeen cut per layer. Over thirteen convs that
compounds to `0.058¹³ ≈ 10⁻¹⁶`: a unit-scale input reaches the classifier sixteen orders of magnitude down,
and it dies geometrically, at a rate I read straight off the fan and the variance. The backward pass is the
same story — back through a linear layer the gradient variance scales by `fan_out · Var(W)`, and for the
square 64-channel conv `fan_out = 576` too, so the gradient reaching the first conv is down by the same
`10⁻¹⁶`. A dead gradient is worse than a dead forward signal, because SGD cannot even find the direction to
climb out. BN re-standardizes the forward activations so the net passes *something*, but it does not rescue
the weights: they are still two orders of magnitude below any shape-aware scale, and the pre-BN conv response
and the backward `Wᵀ` gradient both inherit that scale before BN's per-channel rescaling can touch them. The
scaffold sets the one thing an initializer must set — the weight scale — wrong by a factor of a hundred.

The fan-scaling lineage — LeCun, Xavier, He — fixes the *average* of this story: choose `Var(W)` so the
expected squared output norm matches the input layer by layer (Xavier's `2/(fan_in+fan_out)`, He's correction
for the factor of two ReLU eats). That makes the *mean* signal norm depth-stable — but it is a statement
about a single number, and a matrix is not a number. A square `n×n` Gaussian rescaled by `1/√n` has singular
values spread across the Marchenko–Pastur law on `[0, 2]` — at `n = 1000`, a smallest singular value near
`0.0008` and a largest near `1.98`, a condition number in the thousands. So even a perfectly He-scaled square
layer amplifies some directions by nearly two and crushes others to almost nothing; norm-preserving on
average, badly anisotropic in the worst case. And the worst case compounds: a product of `L` independent
Gaussians adds its log-singular-values, so the gap between most-amplified and most-annihilated direction grows
*linearly* in `L`. If one He-Gaussian conv already puts a factor of ~2500 between its stiffest and softest
direction, a few layers of product have an astronomical condition number — one input direction arrives at the
head amplified by orders of magnitude while an orthogonal one arrives crushed. Fan-scaling stabilizes the
mean and leaves the spectrum to chance, and the spectrum is where depth does its damage.

Saxe, McClelland & Ganguli (2014) make the stronger demand: don't just match the mean, preserve the *whole*
norm — every singular value exactly one. A matrix with all singular values one is orthogonal (semi-orthogonal
for non-square): `WᵀW = I` on the smaller dimension, an isometry that rotates without changing any length,
and a composition of isometries is again an isometry. So in a deep *linear* net an orthogonal init gives exact
dynamical isometry — forward norm conserved at every depth, and because the backward Jacobian is the
transposed product of the same orthogonal matrices, gradient norm conserved too. No vanishing, no exploding,
at any depth, exactly rather than on average. Where the He-Gaussian product's condition number blows up with
depth, the orthogonal product's stays pinned at one. That is the strongest guarantee
initialization-by-norm-preservation can make, so it is the right place to start: test the
most aggressive form of the signal-preservation idea before trading any of it away.

Now translate "orthogonal" to the objects the scaffold hands me, because a conv weight is a 4-D tensor
`(out_channels, in_channels, kh, kw)`, not a matrix. The standard move — what `nn.init.orthogonal_` does —
views it as 2-D by flattening all but the first dimension (rows `out_channels`, columns `in_channels·kh·kw`),
fills that matrix with a random (semi-)orthogonal factor via QR of a Gaussian, and reshapes back. The cost is
one QR per layer; even VGG's widest conv flattens to `512 × 4608`, a millisecond against a 200-epoch schedule.
In the usual case the flattened matrix is far wider than tall (`in·kh·kw ≫ out`), so the result is
*row*-orthogonal, `WWᵀ = I` on the output channels: every output channel a unit-norm, mutually-orthogonal
filter, preserving norm forward. This is not a literally norm-preserving *convolution* — there is a
construction that is, the delta/circular orthogonal conv that pads a Kronecker-delta spatial kernel with an
orthogonal channel matrix — but I reject it: it is not what `orthogonal_` exposes, it starts every conv as a
near-identity spatial response that fights the 3×3 receptive-field structure the network must learn, and a
delta-in-space kernel interacts awkwardly with BN's per-channel whitening. The matrix scheme gives the
practical content — every filter unit-norm and mutually orthogonal — without touching the graph.

One shape subtlety recurs with teeth on MobileNetV2, so I work it out here. Not every conv
flattens wide. VGG's first conv takes the three image channels, weight `(64, 3, 3, 3)`, flattening to
`(64, 27)` — *taller* than wide, `out = 64 > 27`. On a tall matrix `orthogonal_` cannot make 64 rows mutually
orthogonal (they live in a 27-dimensional space), so it makes the 27 *columns* orthonormal, `WᵀW = I₂₇`. Is
that a failure? Here the operator genuinely *is* the dense matmul `W · patch`, where `patch ∈ ℝ²⁷` is the
im2col vector of one receptive field, and orthonormal columns give
`‖W·patch‖² = patchᵀ WᵀW patch = ‖patch‖²` — an isometric embedding of ℝ²⁷ into ℝ⁶⁴. The tall first conv
preserves forward norm perfectly. The lesson is precise: `orthogonal_` on a tall matrix is fine *exactly when
the flattened matrix is the operator*, because then column-orthonormality is genuine input-norm preservation.
It fails only when the flattened matrix is a *fiction* — when the real operator is not that dense matmul.

The scale subtlety I cannot skip is where orthogonality and ReLU collide. A pure orthogonal map preserves norm
— perfect for a linear net — but every conv here is followed by ReLU, which discards the negative half of its
pre-activations. For a zero-mean Gaussian pre-activation `z ~ N(0, σ²)`, `E[relu(z)²] = σ²/2` exactly: half
the variance survives. So an exactly-orthogonal conv followed by ReLU halves the activation variance every
layer; over thirteen VGG convs that is `(1/2)¹³ ≈ 10⁻⁴`, the same ReLU leak that put the factor of two in He.
The remedy is a gain that undoes the halving, `√2` — precisely `nn.init.calculate_gain('relu')` — which sets
every singular value to `√2`: each layer amplifies the pre-ReLU signal by `√2`, ReLU pulls it back to unit, and
the post-activation norm holds across depth. The backward pass inherits the fix symmetrically: the ReLU
derivative masks out half the back-propagated gradient in expectation, and the `√2` gain on `Wᵀ` compensates
it the same way. This is not a different target from He but the same one reached differently — a row-orthonormal
matrix scaled by `√2` has each row of squared norm two over `fan_in` entries, so per-entry RMS `√(2/fan_in)`,
which is `√(2/576) ≈ 0.0589` on the 576-fan layer, exactly He's standard deviation. I buy He's second moment
*plus* the singular-value spectrum pinned to one value instead of scattered across Marchenko–Pastur, and the
spectral control is the only thing I pay extra for. The same rule goes on `Linear` layers, which sit behind
ReLUs of the same kind; biases go to zero, since at init no unit deserves an offset and a nonzero bias only
shifts the balanced pre-activations.

BatchNorm I leave at the conventional `(γ=1, β=0)`, and whether that makes the whole scheme redundant is the
question I most need to answer here. At init BN whitens its input to unit variance per channel and
applies the identity affine, so it neither amplifies nor suppresses. The tension is real: BN re-standardizes
the forward variance at *every* layer regardless of the conv init, so the *per-channel forward variance* — the
diagonal of the activation covariance — is pinned for free, and to that extent the forward-norm half of my
guarantee is delivered downstream whether I paid for it or not. But BN standardizes each channel
*independently*; it is a diagonal rescaling. It does nothing to the *off-diagonal conditioning* of the weight
map `W`. A He-Gaussian conv whose singular values span `[0.0008, 1.98]` still, after BN, feeds the next layer
through a map that stretches some directions and crushes others; BN equalizes per-channel variances but cannot
re-round the *shape* of `W`'s action. Orthogonal init removes that anisotropy at the source. So the honest
hypothesis: on a deep plain stack, where the full spectrum is well-defined and every layer's conditioning
compounds, orthogonality should hold a real if modest edge BN cannot supply; where the architecture defeats the
spectrum for other reasons, the edge should vanish and BN's free standardization is all that survives. I
resolve nothing by getting clever with BN — I leave the neutral identity affine so the conv init speaks in
isolation. The rule is one uniform pass, no branching, no depth arithmetic, no special-casing shortcuts: every
`Conv2d` and `Linear` gets `orthogonal_` with gain `√2` and zero bias, every `BatchNorm2d` gets `(1, 0)`.

Walking the three targets tells me where this should and should not pay off. VGG-16-BN is friendliest — a deep
plain chain, exactly the topology orthogonality was derived for, and the one place the compounding per-layer
conditioning that BN cannot fix is genuinely present — so if the spectrum has an edge anywhere it is here.
ResNet-56 is more delicate: its main path is a running sum `x_{l+1} = x_l + F_l(x_l)`, each `F_l` two 3×3 convs
ending `conv2 → bn2`. Orthogonality controls each conv *inside* `F_l` but says nothing about how branch outputs
*accumulate*. With `γ=1`, `bn2` standardizes each branch to unit variance, so each `F_l` enters the sum at unit
scale; if the branches are roughly independent, the main-path variance grows additively, `≈ (1+L)·Var(x_0)`.
With 27 `BasicBlock`s that is a factor of `28` in variance, `√28 ≈ 5.3×` in standard deviation at the head —
milder in truth because the three stage-boundary projection shortcuts partially reset the sum, but the
mechanism inflates with depth regardless. Orthogonality buys clean per-branch maps, not accumulation control,
so I expect it merely ordinary on ResNet, in the fan-scaling neighborhood.

MobileNetV2 is the hardest case, and it is the fiction-versus-operator distinction from VGG's first conv
turned malignant. Its workhorse is the depthwise 3×3 conv, `groups = channels`, each output channel seeing
exactly one input channel. The weight is `(channels, 1, 3, 3)`, flattening to `(channels, 9)` — far taller
than wide, so `orthogonal_` can only orthonormalize the nine *columns*; `WWᵀ` is `channels × channels` of rank
at most nine, never the identity for realistic channel counts. But rank deficiency is the shallow problem. The
deep problem is that the flattened `(channels, 9)` matrix is *not the operator*: a depthwise conv never forms
`W · patch`, it is block-diagonal, output channel `c` is just `x_c` convolved with its own filter `f_c`, no
cross-channel mixing at all. So orthonormalizing the columns imposes `Σ_c f_c(j) f_c(k) = δ_{jk}` — a
cross-channel condition on filters the operator keeps independent, orthonormalizing an axis the convolution is
physically indifferent to, and getting nothing in return: what actually governs a depthwise layer is each
`f_c`'s own frequency response, which column-orthonormalization neither sets nor respects. That is the layer
where I would expect orthogonal init to lose outright, possibly falling *behind* a plain fan-scaled Gaussian
that at least gives each filter the right per-tap variance — and MobileNetV2 is built almost entirely from it.

So my first edit is the literal orthogonal fill: iterate `model.modules()`, `orthogonal_` with gain `√2` on
every `Conv2d` and `Linear` (zero bias), `(1, 0)` on every `BatchNorm2d`, uniform, the honest way to test the
orthogonal hypothesis before complicating it. If the predicted pattern holds — best on VGG, ordinary on ResNet,
weakest on MobileNetV2 — it implicates two things to attack next: BN already re-standardizes per-channel
variance, so the per-layer isometry is partly redundant, and the architectures that matter are dominated by
residual accumulation or depthwise filters orthogonality cannot touch — pointing back toward getting the
*second moment* right for ReLU cheaply and uniformly. The full module is in the answer.
