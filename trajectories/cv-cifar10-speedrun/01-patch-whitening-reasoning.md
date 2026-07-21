The baseline trains cleanly and reaches 94% — but it needs 45 epochs and 18.3 A100-seconds to get
there, and before I change a line I want to know where those 18.3 seconds go, because the prior art
tells me most of them are recoverable. David Page's ResNet pipeline reaches the same 94% bar in about 10
A100-seconds, hlb-CIFAR10 in 6.3, on the same fixed hardware and the same fixed data. Nothing about the
problem changed between them and me — same 50,000 images, same A100, same bar — so the entire gap from
18.3 down to ~6 is *method*, not hardware. A factor of three does not come from shaving a few percent
off each step; it comes from removing whole epochs of work that never needed to happen. So the question
is which epochs are wasted, and the honest way to find out is to ask what each layer is being *asked* to
learn in the first few epochs, and whether that work is actually about the labels.

Start at the very front. The first conv takes the 3-channel image and convolves it with a small kernel.
On CIFAR-10 the three colour channels are strongly correlated — red, green and blue at a pixel move
together far more often than not — and adjacent pixels are strongly correlated too, because natural
images are smooth almost everywhere and edges are rare. So the raw input is highly redundant: most of
the variance lives in a few directions (overall brightness, low-frequency colour gradients), and the
directions that actually discriminate one class from another — edges, fine texture, object boundaries —
sit in the low-variance tail. A randomly initialized first conv has to *discover* this structure by
gradient descent, slowly rotating its filters until they line up with the informative directions. That
rotation is pure overhead: it is identical for every CIFAR-10 run, it has nothing to do with which
images are cats and which are trucks, and the network burns its first several epochs on it before it can
begin learning anything class-specific.

The cost of that is quantitative, and the quantity is what makes it worth attacking. If the input's
covariance has eigenvalues λ₁ ≥ λ₂ ≥ …, gradient descent moves along each eigen-direction at a rate
proportional to that direction's eigenvalue: big directions get a strong gradient and move fast, small
directions crawl. The number of steps to make progress on the weakest direction scales linearly with
the condition number κ = λ_max / λ_min, because GD on a quadratic contracts the worst direction's error
by only ≈ 1 − 2/κ per step. For natural-image patches the spread between the brightness direction and
the high-frequency directions is two to three orders of magnitude — κ in the hundreds. So the
fine-texture directions the classifier actually needs are learning a hundred to a thousand times slower
than the brightness direction it barely needs. That is the mechanism behind "45 epochs": not that the
labels are hard, but that the weak-but-informative input directions are learned at 1/κ the rate of the
loud, useless ones.

Can I hand the network a *decorrelated* input for free, before training starts, instead of paying for it
in epochs? The classical answer is whitening: estimate the covariance Σ, take its eigenbasis Σ = U Λ Uᵀ,
and apply the linear map W = Λ^(−1/2) Uᵀ. The transformed input Wx then has covariance W Σ Wᵀ = I —
every direction carries unit variance, the redundancy is gone, and the condition number is exactly 1.
With κ = 1 the weak and strong directions learn at the same rate, and the first layer has nothing left
to rotate into: it starts already decorrelated.

There is a catch that fixes the *shape* of the fix: a convolution acts not on whole images but on small
spatial patches, so the object to whiten is the distribution of *patches* the first conv sees. For a
2×2×3 kernel each patch is a 12-dimensional vector, so I whiten a 12-dimensional distribution with a
12×12 covariance — a tiny object to eigendecompose. And whitening a patch is a linear map from the
12-dim patch to a 12-dim whitened patch, and *sliding a fixed linear map across an image is exactly what
a convolution does*. The whitening transform is itself a 2×2 convolution. So rather than bolt a
preprocessing stage in front of the network, I write the whitening matrix directly into the first conv's
weights and freeze them.

Freezing is a real design choice. The obvious alternative is to *initialize* the first conv from the
whitening transform but let it keep training — but the decorrelation is a label-independent property of
CIFAR-10 images, identical across every run, so there is nothing class-specific for gradient descent to
improve about it. Leaving it trainable would only let minibatch noise drift the carefully-conditioned
first layer back toward a noisier configuration, spending gradient to un-do the conditioning I installed.
Freezing costs nothing in expressivity (the layers after it are free to learn whatever they like on a
whitened input) and buys a real backward-pass saving I will come back to. (Doing the whitening as a
fixed preprocessing multiply outside the network works too, but adds an op and a materialized
intermediate; folding it into a conv that already exists is strictly cheaper.)

Now a subtlety that fixes the *width* of the layer. The thing after this conv is a GELU, which is
asymmetric — it passes positives nearly linearly and squashes negatives toward zero. If the whitening
conv emits only the whitened responses v·x, then wherever a whitened coordinate is negative the GELU
clips it and the *sign* of that coordinate — half its information — is lost. So I also emit the negation
of each whitened filter, giving both +v·x and −v·x for every eigen-direction; whichever sign is positive
survives the GELU. That is why the layer has 2 × 3 × k² outputs: the factor 3·k² is the number of
whitening directions (12 for a 2×2×3 patch), and the factor of two is the positive/negative pair. For
k = 2 that is 24 output channels — exactly the 24-channel first conv the baseline network already has, so the
doubled-width whitening layer drops in without changing any downstream shape.

The one real knob is the ε floor. The scaling is 1/√(λ_i + ε): I divide each eigen-direction by the
square root of its variance. But the smallest eigenvalues correspond to nearly-flat patches —
near-constant 2×2 regions dominated by sensor noise — where λ_i can be tiny, say around 10⁻⁵. Without a
floor that direction is amplified by 1/√(10⁻⁵) ≈ 316, so whitening would multiply essentially noise by
three hundred and hand the network a loud noise channel it then has to *learn to ignore*, re-introducing
wasted epochs of a different kind. With ε = 5×10⁻⁴ the same direction is amplified by at most
1/√(5×10⁻⁴) ≈ 45 instead of 316. That is the whole job of ε — regularize the smallest eigenvalues so a
near-singular flat-patch direction cannot explode, at the cost of leaving those directions slightly
under-whitened (harmless, since they carry almost no signal). It is small enough not to touch the
informative mid-spectrum and large enough to tame the tail; 5×10⁻⁴ is my starting point and the one
thing I would tune.

One choice of *flavour*: writing the scaled eigenvectors themselves as the filter rows is PCA whitening
(W = Λ^(−1/2) Uᵀ), which whitens and rotates into the eigenbasis; ZCA (W = U Λ^(−1/2) Uᵀ) whitens while
rotating back to the pixel basis. Both give identity covariance; they differ only by an orthogonal
rotation U on the output, which does not matter here because the next thing is a trainable conv stack
that can compose any orthogonal map for free — so I take the simpler PCA form the code naturally
produces. And I keep *all* 12 directions rather than dropping the low-variance tail: the classifier may
want the weak edge directions, it just wants them on equal footing, which is what rescaling-without-
truncating delivers.

The initialization is short. Gather every 2×2 patch with `unfold`, flatten to a 12-vector, form the
empirical covariance XᵀX/n, eigendecompose with `torch.linalg.eigh` (symmetric), scale the eigenvectors
by 1/√(λ+ε), and write `cat(scaled, −scaled)` into the frozen conv weight:

```python
def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))
```

The shape bookkeeping in the middle function is easy to get wrong: `eigh` returns eigenvalues ascending
and eigenvectors as *columns*, so `eigenvectors.T` puts each on a row, and `reshape(12, 3, 2, 2)` turns
each length-12 eigenvector back into a 3×2×2 filter — the inverse of the flattening I did to build the
patches. The `flip(0)` reverses ascending into descending so the loudest direction lands in channel 0
(cosmetic for a frozen layer, but tidy), and `cat((scaled, −scaled))` stacks the 12 filters and their
negations into the 24 output channels, matching the (24, 3, 2, 2) frozen conv exactly.

The cost is genuinely negligible, which matters because "free" is load-bearing. I estimate the
covariance from the first 5000 training images: each 32×32 image yields 31² = 961 overlapping 2×2
patches, so ~4.8 million patches, and forming XᵀX is a ~7×10⁸ multiply-add matmul — well under a
millisecond of A100 time — followed by an instantaneous 12×12 eigendecomposition. Estimating 78 free
covariance parameters from ~5 million samples is overwhelmingly over-determined. Against an 18-second
budget the whole initialization does not register.

Whitening is complementary to the per-channel normalization the loader already does, not redundant with
it. Per-channel normalization fixes the *marginal* scale of each of the three colour channels
independently but does nothing about the *correlations* — it leaves R/G/B moving together and
neighbouring pixels correlated, so the 12×12 patch covariance stays far from identity with all its
off-diagonal structure and eigenvalue spread intact. Whitening is precisely the step that zeroes those
off-diagonals and flattens the spectrum. This is also why I estimate the covariance from the
*normalized* images — I want the whitening transform consistent with the exact input distribution the
conv will see at run time.

One piece rides along with the frozen conv: I give it a *learnable* bias even though its weights never
move. Whitening fixes the linear transform but says nothing about *where* along each whitened coordinate
the GELU should threshold, and that operating point is worth learning. So the bias trains for a few
epochs and then I freeze its gradient too — not for accuracy (the thresholds have settled by then) but
for the backward pass: once the entire first layer has `requires_grad = False`, autograd no longer
propagates gradient into it, trimming a little off every backward step for the rest of training. That is
the second, smaller dividend of freezing, on top of not wasting gradient re-learning a fixed transform.

So the bet, stated so the epoch count can falsify it: the 45-epoch baseline was 45 epochs largely
because its input was un-whitened, so the first layer spent a linear-in-κ number of steps rotating
itself into a decorrelating basis before the network could get to work. Whitening removes that entirely
— the layer starts at κ = 1 — so I expect this to be the single largest speedup of anything I can add,
and if the un-whitened input really was the dominant cost the epoch count should *more than halve* from
45, with the seconds following. The failure mode I am watching is noise amplification: if ε is too small
and the low-variance directions blow up into noise the network must suppress, accuracy could sag below
the bar even as epochs drop. The ε floor is the hedge, and the test is clean — epochs-to-94% falling
sharply while the mean holds at 94% confirms the conditioning argument. The full code is in the answer.
