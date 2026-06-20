The baseline trains cleanly and reaches 94% — but it needs 45 epochs and 18.3 A100-seconds to get
there, and almost all of that is spent in the early epochs doing something I can see is wasteful if I
look at what the first convolution is actually being asked to learn. The first layer takes a 3-channel
image and convolves it with a small kernel. On CIFAR-10 the three colour channels are strongly
correlated and adjacent pixels are strongly correlated, so the raw input the network sees is highly
redundant: most of the variance lives in a few directions (overall brightness, low-frequency colour),
and the directions that actually discriminate classes — edges, fine texture — sit in the low-variance
tail. A randomly initialized first conv has to *discover* this structure by gradient descent, slowly
rotating its filters until they decorrelate the input. That rotation is pure overhead: it is the same
for every CIFAR-10 run, it has nothing to do with the labels, and the network burns its first several
epochs on it before it can start learning anything class-specific. The optimization is poorly
conditioned at the input because the input is not white.

So the question is whether I can hand the network a decorrelated input *for free*, before training
starts, instead of paying for it in epochs. The classical answer to "decorrelate a signal" is
whitening: estimate the covariance of the input, find its eigenbasis, and rescale each eigen-direction
by the inverse square root of its eigenvalue so that every direction has unit variance. After that
transform the input has identity covariance — the redundancy is gone and every direction carries equal
weight. The catch is that a convolution does not act on whole images; it acts on small spatial patches.
The right object to whiten is therefore the distribution of *patches*. For a 2×2×3 kernel, each patch
is a 12-dimensional vector; I can collect every 2×2 patch from a few thousand training images, estimate
the 12×12 patch covariance, eigendecompose it, and build a linear transform that whitens patches.

The beautiful part is that this whitening transform *is itself a convolution*. Whitening a patch is a
linear map from the 12-dim patch to a 12-dim whitened patch; sliding that linear map across the image
is exactly what a 2×2 conv with 12 output channels does. So I do not need to bolt a preprocessing stage
in front of the network — I can fold the whitening directly into the first convolution's weights and
*freeze* them. Concretely: stack the eigenvectors of the patch covariance, each scaled by
1/√(λ_i + ε), into the conv kernel. The ε is a floor added to the eigenvalues before the inverse
square root; without it a near-singular patch-covariance direction (a flat patch) would be divided by
almost zero and blow up, so ε regularizes the smallest eigenvalues and is the one knob I tune.

There is a sign subtlety I have to handle. The next thing after this conv is a GELU nonlinearity, which
is not symmetric — it suppresses negative pre-activations. If I emit only the whitened responses, half
the information (the sign of each whitened coordinate) gets clipped by the GELU and is lost. The fix is
to also emit the *negation* of each whitened filter, so the layer produces both `+v·x` and `−v·x` for
every eigen-direction. Then whichever sign survives the GELU, the information is preserved. That is why
the layer has `2 × 3 × k²` outputs for a k×k kernel — the factor of two is the positive/negative pair.

Let me write the initialization. I collect patches with an `unfold` over the spatial dims, form the
empirical covariance `Xᵀ X / n`, eigendecompose it with `torch.linalg.eigh`, scale the eigenvectors by
`1/√(λ+ε)`, and write `cat(scaled, −scaled)` into the frozen conv weight:

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

The layer's weights are not updated during training — `requires_grad = False` on the kernel. Two design
choices come with it. First, I add a *learnable bias* to this otherwise-frozen conv, because the
whitening fixes the linear transform but the network can still benefit from learning where to threshold
the whitened responses; I train that bias for a few epochs and then freeze its gradient too, which
costs nothing in accuracy but speeds up the backward pass once frozen (the gradient no longer has to
flow back through the first layer). Second, the ε floor: keeping it small but nonzero is what prevents
a singular patch-covariance from producing exploding filter norms.

I estimate the covariance from only the first 5000 training images, which is plenty — covariance is a
second-order statistic and converges fast — and the whole initialization is a single eigendecomposition,
microseconds of A100 time. So the cost is essentially zero and it happens once, before the timer's
training loop really gets going.

Now the prediction, against the baseline's 45 epochs / 18.3 seconds. The bet is that handing the
network a pre-whitened input removes the early-epoch decorrelation work entirely: the first layer
starts already doing the job a random init would spend epochs learning, so the optimization is
well-conditioned from step one and the whole training should compress into far fewer epochs. I expect
this to be the single largest speedup of anything I can add, because it attacks the conditioning of the
input directly rather than shaving overhead at the margins. If the baseline needed 45 epochs largely
because the input was un-whitened, whitening it should *more than halve* the epoch count. The risk is
that whitening amplifies the low-variance, high-frequency directions — exactly the ones dominated by
sensor noise — and the 1/√(λ+ε) scaling could blow those up into noise the network then has to learn to
ignore; the ε floor is my hedge against that, and if accuracy holds while epochs drop, the bet is good.
The concrete change is the frozen-whitening first conv plus its `init_whitening_conv`; the full code is
in the answer.
