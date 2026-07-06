The baseline trains cleanly and reaches 94% — but it needs 45 epochs and 18.3 A100-seconds to get
there, and before I change a line I want to know where those 18.3 seconds actually go, because the
prior art tells me most of them are recoverable. David Page's ResNet pipeline reaches the same 94% bar
in about 10 A100-seconds, and hlb-CIFAR10 does it in 6.3, on the same fixed hardware and the same fixed
data. Nothing about the problem changed between them and me — same 50,000 images, same A100, same bar —
so the entire gap from 18.3 down to ~6 is *method*, not hardware. That is close to a factor of three
sitting on the table, and a factor of three does not come from shaving a few percent off each step; it
comes from removing whole epochs of work that never needed to happen. So the question is which epochs
are wasted, and the honest way to find out is to ask what each layer is being *asked* to learn in the
first few epochs, and whether that work is actually about the labels.

Start at the very front. The first layer takes the 3-channel image and convolves it with a small
kernel. On CIFAR-10 the three colour channels are strongly correlated — red, green and blue at a pixel
move together far more often than they move independently — and adjacent pixels are strongly correlated
too, because natural images are smooth almost everywhere and edges are rare. So the raw input the
network sees is highly redundant: most of the variance lives in a few directions (overall brightness,
low-frequency colour gradients), and the directions that actually discriminate one class from another —
edges, fine texture, the boundary of an object against its background — sit in the low-variance tail. A
randomly initialized first conv has no knowledge of this. It has to *discover* the structure by gradient
descent, slowly rotating its filters until they line up with the informative directions and stop wasting
their output range on the redundant ones. That rotation is pure overhead. It is identical for every
CIFAR-10 run — it has nothing to do with which images are cats and which are trucks — and the network
burns its first several epochs on it before it can begin learning anything class-specific.

Let me make the "poorly conditioned" claim quantitative, because that is the whole argument. Suppose the
first layer is effectively solving a least-squares-like problem in its input, and the input's covariance
has eigenvalues λ₁ ≥ λ₂ ≥ … The speed at which gradient descent can move along each eigen-direction is
proportional to that direction's eigenvalue: the big directions get a strong gradient and move fast, the
small directions get a weak gradient and crawl. The relevant number is the condition number κ = λ_max /
λ_min. Gradient descent on a quadratic with condition number κ contracts the error along the worst
direction by a factor of roughly (κ − 1)/(κ + 1) ≈ 1 − 2/κ per step, so the number of steps to make real
progress on the weakest direction scales *linearly* with κ. For natural-image patches the eigenvalue
spread between the DC/brightness direction and the high-frequency directions is easily two to three
orders of magnitude — call it κ in the hundreds. That is the mechanism behind "45 epochs": it is not
that the labels are hard, it is that the weak-but-informative directions of the input are being learned
at 1/κ the rate of the loud, useless ones. If κ ≈ 100–1000, the fine-texture directions the classifier
actually needs are learning a hundred to a thousand times slower than the brightness direction it barely
needs at all. That is a lot of epochs spent on conditioning, not on classification.

So the question becomes whether I can hand the network a *decorrelated* input for free, before training
starts, instead of paying for it in epochs. The classical answer to "decorrelate a signal" is whitening:
estimate the covariance Σ of the input, find its eigenbasis Σ = U Λ Uᵀ, and apply the linear map W =
Λ^(−1/2) Uᵀ. Let me check that this really does what I want rather than just asserting it. The covariance
of the transformed input Wx is W Σ Wᵀ = Λ^(−1/2) Uᵀ (U Λ Uᵀ) U Λ^(−1/2) = Λ^(−1/2) Λ Λ^(−1/2) = I, using
UᵀU = I. So after whitening the input has *identity* covariance — every direction carries exactly unit
variance, the redundancy is gone, and the condition number is exactly 1. With κ = 1 the weak directions
and the strong directions learn at the same rate, and the first layer no longer has anything to rotate
into: it starts already decorrelated. That is the entire prize, and the algebra says it is real and
exact, not approximate.

There is a catch that determines the *shape* of the fix: a convolution does not act on whole images, it
acts on small spatial patches. The object I should whiten is therefore not the 3072-dimensional image
but the distribution of *patches* the first conv sees. For a 2×2×3 kernel each patch is a 12-dimensional
vector (2·2·3 = 12), so the thing to whiten is a 12-dimensional distribution with a 12×12 covariance —
a tiny object I can eigendecompose in microseconds. And here is the part that makes this fold cleanly
into the network rather than bolting a preprocessing stage in front of it: whitening a patch is a linear
map from the 12-dim patch to a 12-dim whitened patch, and *sliding a fixed linear map across an image is
exactly what a convolution does*. The whitening transform is itself a 2×2 convolution. So I do not need
a separate preprocessing op that costs its own forward pass; I can write the whitening matrix directly
into the first conv's weights and freeze them there.

That "freeze" is a real design choice and I should defend it against the alternative. The obvious
alternative is to *initialize* the first conv from the whitening transform but let it keep training. The
argument against is exactly the argument that motivated whitening in the first place: the decorrelation
is a label-independent property of CIFAR-10 images, identical across every run, so there is nothing
class-specific for gradient descent to improve about it. Leaving it trainable would only let the
optimizer drift the carefully-conditioned first layer back toward some noisier configuration under the
pressure of minibatch noise, spending gradient to un-do the very conditioning I installed. Freezing it
costs nothing in expressivity (the layers after it are free to learn whatever they like on top of a
whitened input) and it buys a real efficiency win in the backward pass, which I will come back to. A
second alternative — do the whitening as a fixed preprocessing tensor multiply outside the network —
also works mathematically but adds an op and a materialized intermediate; folding it into a conv that
already exists is strictly cheaper. So: fold the whitening into the first conv, and freeze it.

Now a subtlety that changes the *width* of the layer. The thing after this conv is a GELU, which is not
symmetric — it suppresses negative pre-activations, passing positives through nearly linearly and
squashing negatives toward zero. If the whitening conv emits only the whitened responses v·x for each
eigen-direction v, then for any patch where a whitened coordinate happens to be negative, the GELU
clips it, and the *sign* of that whitened coordinate — which is half the information in it — is lost.
The fix is to also emit the negation of each whitened filter, so the layer produces both +v·x and −v·x
for every eigen-direction. Then whichever sign of a given coordinate is positive survives the GELU, and
the information is preserved on one branch or the other. This is why the layer has 2 × 3 × k² outputs
for a k×k kernel: the factor 3·k² is the number of whitening directions (12 for a 2×2×3 patch), and the
factor of two is the positive/negative pair. For k = 2 that is 2 · 3 · 4 = 24 output channels — which is
exactly the 24-channel first conv the scaffold already has, so the doubled-width whitening layer drops
into the existing architecture without changing any downstream shape. Good; the width bookkeeping checks
out against the substrate rather than fighting it.

The one real knob is the ε floor, and I can size it by asking what goes wrong without it. The scaling is
1/√(λ_i + ε): I divide each eigen-direction by the square root of its variance so it comes out at unit
variance. But the smallest eigenvalues correspond to nearly-flat patches — near-constant 2×2 regions,
which are dominated by sensor noise rather than signal — and there λ_i can be tiny. Suppose the smallest
eigenvalue is around 10⁻⁵. Without a floor, that direction is amplified by 1/√(10⁻⁵) ≈ 316, so the
whitening conv would multiply what is essentially noise by three hundred and hand the network a loud
noise channel it then has to *learn to ignore* — re-introducing wasted epochs of a different kind. With
ε = 5×10⁻⁴ added before the square root, the same direction is amplified by at most 1/√(5×10⁻⁴) ≈ 45
instead of 316: the floor caps the amplification at roughly a seventh of the unregularized blow-up. That
is the whole job of ε — it regularizes the smallest eigenvalues so a near-singular flat-patch direction
cannot explode, at the cost of leaving those directions slightly under-whitened (which is harmless,
because they carry almost no signal anyway). ε is small enough not to touch the informative mid-spectrum
directions and large enough to tame the tail; it is the one thing I would tune, and 5×10⁻⁴ is my
starting point.

Before I commit, let me trace a stripped-down two-direction version by hand to make sure the mechanics do
what I think. Take a toy input whose patch covariance has just two eigen-directions with variances λ₁ = 1
(say the brightness direction) and λ₂ = 0.01 (say a high-frequency edge direction), κ = 100. The
whitening filter for direction i is the unit eigenvector u_i scaled by 1/√(λ_i + ε). With ε negligible
against these values, the brightness filter is u₁·1/√1 = u₁, essentially untouched, while the edge
filter is u₂·1/√0.01 = 10·u₂, amplified tenfold. So the transform *lifts the weak edge direction up to
the same output variance as the loud brightness direction* — a patch that was 1% edge and 99% brightness
in variance comes out 50/50, and the downstream layers now see the edge information at equal strength
instead of buried 100× below the brightness. That is precisely the redundancy removal I wanted, made
concrete: the amplification factor is √κ = 10 on the weakest direction, exactly enough to equalize the
two. And it shows why ε matters — if λ₂ had been 10⁻⁵ instead of 0.01, the amplification would be 316×
rather than 10×, which is the flat-patch blow-up the floor exists to cap.

One design point I want to be explicit about, because there are two flavours of whitening and I am
choosing one. Writing the *scaled eigenvectors themselves* as the filter rows means W = Λ^(−1/2) Uᵀ —
PCA whitening, which whitens *and* rotates the patch into the eigenbasis, so each output channel is one
clean eigen-direction. The alternative, ZCA whitening W = U Λ^(−1/2) Uᵀ, whitens while rotating back to
the original pixel basis, so filters look like little sharpening stencils. Mathematically both give
identity covariance (both satisfy WΣWᵀ = I); the difference is only an orthogonal rotation U on the
output. Here that rotation does not matter, because the very next thing is a trainable conv stack that
can compose any orthogonal map it likes for free — so I take the simpler PCA form the code naturally
produces. And note I keep *all* 12 directions rather than dropping the low-variance tail the way PCA
dimensionality-reduction would: the classifier downstream may well want the weak edge directions, it
just wants them on equal footing, which is exactly what rescaling-without-truncating delivers.

With the design fixed, the initialization is short. I gather every 2×2 patch with an `unfold` over the
spatial dimensions, flatten each to a 12-vector, form the empirical covariance XᵀX/n, eigendecompose it
with `torch.linalg.eigh` (symmetric, so eigh not svd), scale the eigenvectors by 1/√(λ+ε), and write
`cat(scaled, −scaled)` into the frozen conv weight:

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

The shape bookkeeping in that middle function is worth checking once, because a transpose in the wrong
place would silently scramble the filters. `eigh` returns the 12 eigenvalues in *ascending* order and
the eigenvectors as *columns* of a 12×12 matrix. Taking `eigenvectors.T` puts each eigenvector on a row,
and `reshape(c*h*w, c, h, w) = reshape(12, 3, 2, 2)` turns each length-12 eigenvector back into a 3×2×2
conv filter — the inverse of the flattening I did to build the patches, so the spatial/channel layout is
restored consistently. The `flip(0)` on both returns reverses the ascending order into *descending*, so
the strongest (highest-variance) whitening direction becomes output channel 0 and the flat-patch tail
lands last; the ordering is cosmetic for a frozen layer but keeps the loudest directions in the low
channel indices, which is tidy. Then `cat((scaled, −scaled))` stacks the 12 filters and their 12
negations into the 24 output channels. So the weight tensor is (24, 3, 2, 2), matching the frozen conv
exactly — the reshape round-trips and the width lands on 24 as derived.

Let me convince myself the cost of this is genuinely negligible, because "free" is a load-bearing part
of the pitch — if the estimate itself ate a second, the whole idea would be self-defeating. I estimate
the covariance from the first 5000 training images. Each 32×32 image yields (32 − 2 + 1)² = 31² = 961
overlapping 2×2 patches, so 5000 images give about 4.8 million patches. Forming XᵀX is a (12 × 4.8M) by
(4.8M × 12) matmul, roughly 4.8M · 12 · 12 ≈ 7×10⁸ multiply-adds — well under a millisecond of A100
matmul time — and the eigendecomposition is of a 12×12 matrix, which is instantaneous. 5000 images is
plenty: a covariance is a second-order statistic and converges fast with sample count, and I am
estimating only 12·13/2 = 78 free parameters from ~5 million samples, an overwhelmingly over-determined
fit. So the whole initialization is one small matmul plus one tiny eigendecomposition, done once before
the training loop really begins; against an 18-second budget it does not register.

It is worth being clear about what whitening adds *on top of* the normalization the loader already does,
because otherwise it looks redundant. The input images are already per-channel mean/std normalized before
they reach the network — `train_loader.normalize` subtracts each channel's mean and divides by its
standard deviation. But per-channel normalization only fixes the *marginal* scale of each of the three
colour channels independently; it does nothing about the *correlations* — it leaves red-green-blue still
moving together and leaves neighbouring pixels still correlated. In covariance terms, per-channel
normalization sets the three diagonal blocks' overall scale but leaves the 12×12 patch covariance far
from identity, with all its off-diagonal structure intact and its eigenvalue spread untouched. Whitening
is precisely the step that zeroes those off-diagonals and flattens the spectrum. So the two are
complementary, not redundant: normalization handles per-channel scale, whitening handles the joint
patch-level correlation, and it is the latter that carries the conditioning win. This is also why I
estimate the covariance from the *normalized* images (`train_loader.normalize(train_loader.images[:5000])`)
rather than the raw ones — I want the whitening transform to be consistent with the exact input
distribution the conv will see at run time.

One more piece rides along with the frozen conv: I give it a *learnable* bias even though its weights
never move. Whitening fixes the linear transform, but it says nothing about *where* along each whitened
coordinate the GELU should threshold — and that operating point is worth learning, because it sets how
much of each whitened direction survives the nonlinearity. So the bias trains for a few epochs and then
I freeze its gradient too. Freezing it late is not about accuracy — by then the thresholds have settled
— it is about the backward pass: once the entire first layer (weights already frozen, now bias too) has
`requires_grad = False`, the autograd graph no longer has to propagate gradient back into it, which
trims a little off every backward step for the rest of training. That is the second, smaller dividend of
freezing, on top of not wasting gradient re-learning a fixed transform.

Now the prediction, stated so the epoch count can falsify it. The bet is that the 45-epoch baseline was
45 epochs largely because its input was un-whitened, so the first layer spent a linear-in-κ number of
steps rotating itself into a decorrelating basis before the network could get to work. Whitening removes
that entirely — the layer starts at κ = 1, the algebra above says exactly so — so the network should
begin learning class structure from step one and the whole training should compress into far fewer
epochs. Because this attacks the *conditioning of the input* directly, at the front of the network where
the redundancy is largest, rather than trimming overhead somewhere downstream, I expect it to be the
single largest speedup of anything I can add: if the un-whitened input really was the dominant cost, the
epoch count should *more than halve* from 45, and the seconds should follow it down (with a small extra
saving once the frozen layer stops taking backward gradient). The concrete failure mode I am watching
for is the noise-amplification one — if ε is too small and the high-frequency, low-variance directions
get blown up into noise the network has to learn to suppress, accuracy could sag even as epochs drop,
and I would see the mean fall below the 94% bar. The ε floor is my hedge against exactly that, and the
test is clean: if the epochs-to-94% drop sharply while the mean accuracy holds at the bar, the bet is
good; if accuracy falls, ε needs raising. The concrete change is the frozen doubled-width whitening
first conv plus its `init_whitening_conv`; the full code is in the answer, and the epochs-to-94% table
is what will tell me whether the conditioning argument was right.
