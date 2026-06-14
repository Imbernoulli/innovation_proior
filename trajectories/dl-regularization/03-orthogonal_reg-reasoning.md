The confidence penalty did what I expected and where I expected it. It came back at 72.66 on
ResNet-56/CIFAR-100, 74.31 on VGG-16-BN/CIFAR-100, and 94.90 on MobileNetV2/FashionMNIST — above the
DropBlock floor on every one of the three pairs (72.45, 73.37, 94.69). The diagnosis I made entering this
rung is confirmed in detail: the largest gain landed on VGG-16-BN, exactly as predicted, +0.94 from 73.37
to 74.31 — the architecture whose over-fitting lives in a dense head my conv-only weight penalty could
never reach but whose softmax the entropy term acts on directly. ResNet-56 moved up too, +0.21 to 72.66,
a smaller gain consistent with residual nets already being well-regularized. MobileNetV2/FashionMNIST
inched up +0.21 to 94.90, the smallest move, consistent with a near-saturated task. So acting on the
*output distribution* was the right move, and it confirms that over-confidence — the peaked, low-entropy
softmax — was a live failure that the weight-shape penalty left untouched.

But look at where the confidence penalty is *weak*, because that is what tells me the next move.
ResNet-56 only reached 72.66 — a thin +0.21, barely above the DropBlock floor. And the absolute numbers
are still well short of what these architectures can do; a +0.2-to-+0.9 nudge from an output penalty
leaves the bulk of the generalization gap on the table. The reason is structural: the entropy penalty
acts purely on the *final* output distribution and says nothing about the *internal* conditioning of the
network — how a signal at the bottom is transformed on its way to the top, and how a gradient is
transformed on its way back down. ResNet-56 is the deepest of the three, 56 layers of `3×3` convs, and a
deep net's behavior is dominated by what happens *inside* — by the spectra of the weight matrices it
multiplies a signal by, layer after layer. The output penalty cannot see that. So the confidence
penalty's smallest gain being on the deepest network is not noise; it is the signature of a regularizer
that fixes the output but ignores the propagation. The next rung has to act on the *internal* weight
matrices, but not the way DropBlock did — not their spatial shape, but their *spectrum*.

Let me reason from what actually goes wrong inside a deep conv net, because that is where the lever is. A
signal enters at the bottom and gets multiplied by one layer's weights, then a nonlinearity, then the
next layer's weights, up to the logits; a gradient does the same in reverse, the error at the top
multiplied by the transposed weight matrices on its way down to the early layers. So both the forward
activation and the backward gradient are, at heart, a vector hit by a long product of matrices. A product
of matrices does to a vector roughly what a product of numbers does to a scalar: if the typical
multiplicative factor is below one, the vector shrinks geometrically with depth — vanishing gradients in
the early layers; if above one, it grows geometrically — explosion. The thing controlling the catastrophe
is spectral: the singular values of each weight matrix decide whether long products contract, stay
balanced, or blow up. This is the diagnosis Pascanu, Mikolov and Bengio made precise on recurrent nets,
the cleanest case because an unrolled RNN is a feedforward net with the same matrix tied across layers:
the long-range gradient factors into a product of per-step Jacobians, each bounded by the spectral norm
of the recurrent matrix times the bounded nonlinearity derivative, so if those spectral norms drift off
one the product contracts or has room to explode. The remedy the analysis points at is not subtle: I want
the relevant singular values of my weight matrices to stay near one.

What makes a matrix's singular values all exactly one? A square matrix with that property is exactly
norm-preserving — it stretches no direction — which for a square matrix means it is orthogonal, with the
two equivalent conditions `W^T W = I` (columns orthonormal) or `W W^T = I` (rows orthonormal). Saxe,
McClelland and Ganguli worked out why this is the right configuration: orthogonal weights give *dynamical
isometry* that survives depth, because a product of orthogonal matrices is itself orthogonal, so a deep
stack of orthogonal layers has an end-to-end map with all singular values still exactly one, no matter how
deep. The naive alternative — scaled Gaussian init that preserves norm *on average* (Glorot–Bengio) —
fails exactly the test that matters: norm preservation in expectation is not a flat spectrum. The squared
singular values of a scaled Gaussian matrix follow the Marchenko–Pastur law, a real spread that does not
vanish with size, and under composition the product develops a wildly kurtotic spectrum — most singular
values crushed toward zero, a thin tail blowing up. Such a product still preserves the norm of a typical
vector, but anisotropically: any component of a backpropagated error lying in the crushed subspace is
annihilated, so the early layers get no gradient there. Orthogonal has none of that spread.

So I could initialize orthogonal and be done — except orthogonal initialization is a condition on the
weights at step zero, and nothing holds it there. The instant the data gradient takes its first step it
pulls `W` off the orthogonal manifold in whatever direction reduces the task loss, with zero regard for
the singular spectrum. By mid-training the spectrum has spread back out, exactly during the long stretch
where most weight movement happens. I need a *standing force* that keeps acting throughout training, not a
one-time placement. And I have just been burned by the opposite mistake on rung one — a force that
arrived too late (the DropBlock penalty reached strength only when the cosine schedule had nearly stopped
the weights) — so a force that acts at every step is precisely what I want.

The heavy-handed standing force would be a hard constraint: confine each weight matrix to the orthogonal
(Stiefel) manifold and re-orthogonalize after every step with a QR or SVD. That keeps the spectrum exactly
flat but costs a matrix factorization of every weight matrix on every single iteration on a GPU — a
non-starter in the inner loop, especially across three full 200-epoch runs where step cost matters (the
same cheapness concern that made the confidence penalty attractive). And there is a problem specific to
convolutions I must confront head-on: conv weights are not square. A `Conv2d` weight is
`[out_channels, in_channels, k, k]`; to talk about its spectrum I flatten it to a 2D matrix with output
channels as rows and everything else collapsed into columns, `W ∈ R^{out × (in·k·k)}`. That is almost
never square — usually the fan-in `in·k·k` exceeds `out`, so the matrix is short and wide. Now the two
orthogonality conditions stop being equivalent: `W^T W = I` asks the `in·k·k` columns to be orthonormal in
`R^{out}`, impossible when there are more columns than the dimension they live in; `W W^T = I` asks the
`out` rows — exactly the `out` filters — to be orthonormal in `R^{in·k·k}`, which *is* feasible for a wide
matrix. So for a wide conv weight the achievable thing is to make the output-channel filters mutually
orthonormal: `W W^T = I`. And that reads beautifully on its own terms — no two filters point in the same
direction, the filters are decorrelated, none is a wasted redundant copy, and each has unit norm so none
has collapsed. The rectangular reality forces the feasible Gram and hands me a clean interpretation for
free.

A hard constraint is too expensive and clumsy for rectangular weights, but I do not need `W` exactly
orthogonal every step — I need it to stay *close*, so the spectrum stays roughly flat. "Close to" is the
language of a penalty, not a constraint. So I add a term that grows as `W` drifts from orthogonal, hand it
to the SGD I am already running, and let the data loss and the penalty negotiate — no factorization,
fully differentiable, acting every step. I turn "how far is `W` from orthogonal" into a scalar via the
residual of the Gram from the identity: `R = W W^T − I`. This is the right thing to measure, and not just
by analogy. The `(i, j)` entry of `W W^T` is the dot product of filter `i` and filter `j`, so the
off-diagonal entries of `R` are exactly the pairwise filter correlations I want driven to zero, and the
diagonal entries are `||filter_i||^2 − 1`, exactly the magnitude-collapse I want prevented. One matrix `R`
simultaneously encodes "decorrelate the filters" (off-diagonals) and "keep each filter unit-norm"
(diagonal). Contrast the alternatives: penalizing `||W||` pulls every singular value toward zero — that is
plain L2 weight decay, already on through the optimizer, controlling scale not shape, and it is *exactly*
the kind of magnitude control rung one piled on redundantly. The Gram residual controls shape — the flat
spectrum — which is the complementary thing L2 cannot do.

Measured how? The smooth choice with a clean closed-form gradient is the squared Frobenius norm of the
residual, `||W W^T − I||_F^2 = sum_{i,j} R_{ij}^2`. Squaring is deliberate: the Frobenius norm itself has
a square root and is non-differentiable at zero, while squaring removes the root, makes the penalty a
smooth polynomial in `W`, and grows fast when a residual entry gets large — pushing hardest exactly on the
worst-off filters. Its gradient is `4(W W^T − I)W`: two matrix multiplies, no SVD, no eigendecomposition,
which autograd computes for me when I just write the forward penalty. So the wall that killed the hard
constraint — a factorization every step — is gone; I traded an exact projection for a cheap gradient nudge
toward the manifold, which is all I need to keep the spectrum from drifting, and it costs essentially
nothing per step, the same cheapness that mattered for the confidence penalty.

How hard should I push? This is a soft encouragement, not a law, so it gets a small coefficient `lambda`.
The data loss is what I actually care about; the orthogonality term gently keeps the weights near a good
manifold while the data loss does the real work. If `lambda` is too large I optimize for orthogonal but
task-blind filters and lose accuracy. And it coexists with — does not replace — the optimizer's L2 weight
decay: L2 shrinks all singular values (scale), this flattens the feasible nonzero singular values toward
one (shape); they pull on different things. A small `lambda = 1e-4` is the natural setting — a fraction of
a percent of the loss scale, the same conservative order as the coefficients on the earlier rungs, and the
lightest touch that still keeps the spectrum from drifting. I sum the penalty over every conv filter bank
in the network, each layer penalized independently, since each layer's own propagation matters; and I
select layers by `'conv' in name and 'weight' in name and p.dim() == 4`, which picks exactly the
4D convolutional weights and skips biases, BatchNorm scales, and the linear classifier weights — the
orthogonality claim is about conv filter banks. I do *not* schedule this. Unlike rung one, where the
warm-up was what made the penalty arrive too late, this force should act from the first step: the whole
point is that the data gradient drags `W` off the manifold immediately, so the standing force has to be
there immediately to oppose it. No delayed start, no ramp.

Why should this help generalization and not just stabilize gradients? Two mechanisms, both readable from
`R`. The off-diagonal part drives filter–filter correlations toward zero, so filters span distinct
directions instead of clustering — the layer stops wasting capacity on near-duplicate filters, and more of
the nominal parameters do independent work. The diagonal part pushes `||filter||^2` toward one, keeping
filter magnitudes from collapsing, so capacity stays usable instead of dying toward an uninformative
origin. Both are about using the model's capacity fully and keeping the representation well-conditioned —
a generalization story, not merely a gradient-flow story. And this is exactly the lever the confidence
penalty lacked: it acts *inside* the network, on the propagation, on the deepest models where the entropy
penalty was weakest.

So the fill iterates the conv filter banks, reshapes each weight to `[out_channels, in·k·k]` so rows are
filters, builds `W W^T`, subtracts an identity on the same device, squares the residual and sums it, and
scales by `lambda`. Autograd supplies the `4(W W^T − I)W` gradient. It reads only `model` (for the conv
weights), ignores `inputs`, `outputs`, `targets` except to land the scalar on the right device, and
changes nothing else — same architecture, init, SGD, cosine schedule, evaluation. The full scaffold body
is in the answer.

Now the falsifiable expectations against the confidence penalty's numbers. The mechanism is finally
internal — it acts on the weight spectrum that governs propagation through depth — so I expect the gains
to concentrate where the output penalty was weakest: **ResNet-56**, the deepest network, where the
confidence penalty managed only +0.21 to 72.66 because it never saw the internal conditioning. I expect
the Gram penalty to beat 72.66 on ResNet-56 and to be the *strongest* of the three regularizers there,
plausibly into the low-73s — the deepest model is where flattening the layer-wise spectra should pay off
most. On VGG-16-BN I expect a result near, but not clearly above, the confidence penalty's 74.31: VGG's
real capacity sink is its dense classifier head, which this conv-only penalty does not touch, so the
output penalty's direct grip on that head's softmax may match or slightly edge a conv-spectrum penalty
there — I would be unsurprised to land just under 74.31, in the low-74s. On MobileNetV2/FashionMNIST I
expect a near-tie around the confidence penalty's 94.90, since the task is saturated and the depthwise
convs (single-channel filters) give the Gram penalty little to decorrelate — high-94s, within a fraction
of a point either way. Netting out across the three pairs, I expect this rung to edge ahead overall by
winning the hardest, deepest model decisively while trading the others roughly evenly — making it the
strongest baseline on the ladder. If instead it *lost* to the confidence penalty on ResNet-56, that would
falsify my reading that the deepest network's shortfall was about internal spectral conditioning rather
than output over-confidence, and would tell me the propagation story is doing less work here than the
output story. I do not expect that: the confidence penalty's own weakest point was its deepest model, and
a flat weight spectrum is exactly what a deep stack needs.
