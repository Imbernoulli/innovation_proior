The scaffold ships with the worst defensible initializer — every weight an i.i.d. draw from `N(0, 0.01²)`,
a variance picked for nothing in particular — so the floor I have to beat is not "no initializer" but "an
initializer that ignores the layer's shape." I want to start the ladder from the first principled rung, the
one that asks the strongest single question you can ask of an untrained network: not "is the variance right
on average?" but "does a signal survive being pushed through this layer at all?" That is the orthogonal
question, and it is the natural place to anchor the climb because it is the most aggressive form of the
signal-preservation idea — it controls the entire spectrum of each weight map, not just its second moment.

Start from what actually goes wrong in a deep net at step zero. Take the forward pass through the VGG stack,
which is the cleanest case because it is a plain chain with no shortcuts: input `x`, then thirteen
`Conv-BN-ReLU` triples. Ignore BN for a moment and ask what a stack of linear maps `W_L … W_2 W_1` does to
the norm of an input vector. The answer is governed by the singular values of each `W_i`. If a layer's
largest singular value exceeds one, it stretches; if its smallest is below one, it shrinks; compose dozens of
such maps and the product's singular values spread out geometrically, so a generic input is either blown up
or crushed to nearly nothing by the time it reaches the classifier. The scaffold default makes this acute:
`N(0, 0.01²)` gives a conv map whose singular values are all tiny, so the signal is multiplied by a small
number at every layer and arrives at the head essentially dead. Symmetric, depth-blind, and exactly the kind
of vanishing-signal start that 200 epochs of SGD struggles to dig out of.

The fan-scaling lineage — LeCun, Xavier, He — fixes the *average* of this story. It chooses the variance of
the entries of `W` so that the *expected* squared norm of the output equals that of the input, layer by
layer. Xavier targets `Var(W) = 2/(fan_in+fan_out)` for a symmetric unit-derivative nonlinearity; He
corrects the factor of two that ReLU eats by zeroing half its inputs, giving `Var(W) = 2/fan`. This is real
progress — it makes the *mean* signal norm depth-stable — but it is a statement about one number, the second
moment. A random Gaussian matrix scaled to the right variance still has a spread of singular values (for a
square `n×n` Gaussian the singular values follow the Marchenko–Pastur law, ranging from near 0 up to ~2 in
the scaled units). So even with He scaling the *individual* directions are not preserved: some directions in
the input space are amplified, others nearly annihilated, and over depth this anisotropy compounds. The mean
is stable; the worst case is not.

Saxe, McClelland & Ganguli (2014) make the stronger demand, and it is the demand I want at the bottom of this
ladder: don't just match the mean, preserve the *whole* norm — every singular value exactly one. A matrix
with all singular values equal to one is orthogonal (or, for non-square, semi-orthogonal): `WᵀW = I` on the
smaller dimension. Such a map is an isometry — it rotates/reflects its input without changing any length.
Chain orthogonal maps and the composition is still orthogonal, so in a deep *linear* net an orthogonal init
gives perfect dynamical isometry: the forward signal norm is exactly conserved at every depth, and — because
the backward Jacobian of a linear net is the transpose product, also a product of orthogonal matrices — so is
the gradient norm. No vanishing, no exploding, at any depth, exactly rather than on average. That is a
qualitatively stronger guarantee than fan-scaling's mean-matching, which is why it is the right first rung:
it is the most that initialization-by-norm-preservation can promise.

Now I have to translate "orthogonal" to the objects the scaffold actually hands me, because a conv weight is
not a matrix — it is a 4-D tensor of shape `(out_channels, in_channels, kh, kw)`. The standard move, and the
one PyTorch's `nn.init.orthogonal_` implements, is to view the tensor as a 2-D matrix by flattening all but
the first dimension: rows = `out_channels`, columns = `in_channels·kh·kw`. Then fill that matrix with a random
(semi-)orthogonal matrix — sample a Gaussian, take its QR (or SVD) factorization, keep the orthogonal factor,
fix the sign convention — and reshape back. When the flattened matrix is wider than it is tall (the usual
case: `in·kh·kw` ≫ `out`), the result is *row*-orthogonal: `W Wᵀ = I` on the output channels. That preserves
the norm in the over-determined direction, which is the forward-pass direction I care about most. It is not a
literally norm-preserving *convolution* — true orthogonal convolutions need the delta/circular construction —
but at the matrix level it gives every output channel a unit-norm, mutually-orthogonal filter, which is the
practical content of the idea and what the task's `nn.init.orthogonal_` exposes. I accept that gap knowingly;
the matrix-level scheme is the one the edit surface supports without altering the graph.

There is a scale subtlety I cannot skip, and it is exactly where orthogonality and ReLU collide. A pure
orthogonal map preserves norm — perfect for a linear net. But every layer here is followed by ReLU, which
discards the negative half of its pre-activations and so, in expectation, halves the variance of what flows
forward. If I initialize each conv as exactly orthogonal (singular values one) and then ReLU after it, the
*activation* norm decays by roughly √2 per layer — the same ReLU leak that forced He to put a factor of two
into the variance. So I scale the orthogonal matrix by a gain. The gain that compensates ReLU's variance halving
is `√2`, which is precisely `nn.init.calculate_gain('relu')`. Multiplying an orthogonal matrix by `√2` sets
every singular value to `√2`, so each layer amplifies the pre-ReLU signal by √2 and ReLU pulls it back to
unit — the post-activation norm holds across depth. This is the orthogonal analogue of He scaling: same
target second moment, but achieved with a controlled spectrum (every singular value identical at √2) instead
of a Gaussian's spread. I apply the same orthogonal-with-`√2`-gain rule to the `Linear` layers too, since the
classifier head and VGG's hidden FC sit behind ReLUs of the same kind; biases go to zero, since at
initialization there is no reason to offset any unit and a nonzero bias would just inject an arbitrary shift
into the carefully balanced pre-activations.

BatchNorm is the one piece I leave at its conventional setting, and I want to be explicit about why, because
it is the same decision the fan-scaling baselines make and it matters for how this rung behaves. Each
`BatchNorm2d` has an affine `(weight γ, bias β)`. The neutral, identity-like setting is `γ=1, β=0`: at
initialization BN whitens its input to unit variance per channel and then applies the identity affine, so it
neither amplifies nor suppresses. That keeps BN out of the way of the orthogonal scaling I just did on the
convs — the conv sets the spectrum, BN re-standardizes the running statistics, and the `γ=1` affine doesn't
fight it. I deliberately do *not* try anything cleverer with BN at this rung (no zero-γ, no per-block
scaling); the whole point of the first rung is to test the orthogonal hypothesis in isolation, with the
simplest BN that lets the conv init speak.

Walk the three target architectures with this rule and predict where it should and shouldn't pay off, because
the cross-architecture spread is what the leaderboard will actually grade. **VGG-16-BN** is the friendliest:
a deep plain chain of `Conv-BN-ReLU` with no shortcuts, exactly the topology orthogonality was derived for.
Here the orthogonal spectrum should give the cleanest conditioning of any per-layer scheme, and if
orthogonal init has an edge anywhere it is here. **ResNet-56** is more delicate. Its main path is a sum of
residual branches: `x_{l+1} = x_l + F_l(x_l)`. Orthogonality controls each conv *inside* `F_l`, but it says
nothing about how the *branch outputs accumulate* down the additive main path. If each branch starts with
unit-scale output (which orthogonal-√2 convs give), then the main-path variance grows roughly linearly with
the number of blocks — 27 of them — so the signal entering the head is inflated by depth, and BN has to
absorb that growth. Orthogonality buys clean per-branch maps but not residual-accumulation control; I expect
it to be merely fine on ResNet, not special. **MobileNetV2** is the hardest case for this scheme and worth
naming. Its workhorse is the *depthwise* 3×3 conv: a conv with `groups = channels`, so each output channel
sees exactly one input channel. Flatten its weight to a matrix and you get shape `(channels, 1·3·3)` =
`(channels, 9)` — far *taller* than wide. `nn.init.orthogonal_` on a tall matrix can only make the *columns*
orthonormal (nine orthonormal vectors in `channels`-space); it cannot make the many rows mutually orthogonal,
so the per-channel-filter norm-preservation story that justified the whole approach simply doesn't hold for
depthwise layers. The orthogonal guarantee degrades to "nine well-conditioned directions" out of a tensor
that has one filter per channel — exactly the layer type where I'd expect orthogonal init to lose its
advantage, possibly to fall *behind* a plain fan-scaled Gaussian that at least gets the per-filter variance
right without pretending to an isometry it can't deliver.

So the step-1 edit is the literal orthogonal fill: iterate `model.modules()`; for every `Conv2d` apply
`orthogonal_` with gain `√2`; for every `Linear` the same with zero bias; for every `BatchNorm2d` set
`(weight=1, bias=0)`. No architecture branching, no depth arithmetic, no special treatment of shortcuts —
one uniform rule, which is the honest way to test the orthogonal hypothesis before complicating it. The
distilled module and its exact code live in the answer.

What I expect when this runs, stated so the next rung has something to push against. On VGG-16-BN, the
topology orthogonality was built for, this should be at least competitive — the clean spectrum is its best
shot. On ResNet-56 I expect it to be unremarkable: orthogonality fixes per-layer conditioning but not the
residual accumulation that is the dominant effect in a 56-layer res-net, so it should land in the same
neighborhood as plain fan-scaling rather than above it. On MobileNetV2/FashionMNIST I am genuinely worried —
the depthwise convs defeat the matrix-orthogonality construction, so this is the architecture where the
elegant theory has the least grip on the actual weights, and I would not be surprised to see it trail there.
If that cross-architecture pattern holds — fine on the plain stack, ordinary on the residual net, weakest on
the depthwise mobile net — then the diagnosis for the next rung writes itself: norm-preservation of the
*whole spectrum* is the wrong thing to over-invest in when (a) BatchNorm already re-standardizes variance at
every layer so the per-layer isometry is partly redundant, and (b) the architectures that matter are
dominated by either residual accumulation or depthwise filters that orthogonality cannot touch. The move
after this is to stop chasing the full spectrum and go back to getting the *second moment* exactly right for
ReLU — the He variance — applied uniformly and cheaply, and see whether that simpler, BN-friendly target
already matches or beats the orthogonal spectrum it cost so much to impose.
