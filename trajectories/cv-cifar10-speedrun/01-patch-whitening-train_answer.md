The unwhitened baseline reaches 94% cleanly, but it needs 45 epochs and 18.3 A100-seconds to get there, and most of that time is spent in the early epochs doing something I can see is wasteful once I look at what the first convolution is being asked to learn. On CIFAR-10 the three colour channels are strongly correlated and adjacent pixels are strongly correlated, so the raw input is highly redundant: most of its variance lives in a few directions (overall brightness, low-frequency colour), while the directions that actually discriminate classes — edges, fine texture — sit in the low-variance tail. A randomly initialized first conv has to *discover* this structure by gradient descent, slowly rotating its filters until they decorrelate the input. That rotation is pure overhead: it is identical for every CIFAR-10 run, it has nothing to do with the labels, and the network burns its first several epochs on it before it can begin learning anything class-specific. The optimization is poorly conditioned at the input simply because the input is not white.

I propose to remove that overhead entirely by handing the network a decorrelated input *for free*, before training starts, via a **frozen patch-whitening initialization** of the first convolution. The classical way to decorrelate a signal is whitening: estimate the covariance of the input, find its eigenbasis, and rescale each eigen-direction by the inverse square root of its eigenvalue so that every direction ends up with unit variance and the redundancy is gone. A convolution does not act on whole images, though — it acts on small spatial patches — so the right object to whiten is the distribution of *patches*. For a $2\times2\times3$ kernel each patch is a 12-dimensional vector; I collect every $2\times2$ patch from a few thousand training images, estimate the $12\times12$ patch covariance, eigendecompose it, and build the linear map that whitens patches. The key observation is that this whitening map *is itself a convolution*: whitening a patch is a linear map from the 12-dim patch to a 12-dim whitened patch, and sliding that map across the image is exactly what a $2\times2$ conv does. So rather than bolt a preprocessing stage in front of the network, I fold the whitening directly into the first conv's weights and *freeze* them — I stack the eigenvectors of the patch covariance, each scaled by $1/\sqrt{\lambda_i + \varepsilon}$, into the conv kernel.

Two design choices make this work in practice. First the $\varepsilon$ floor: without it, a near-singular patch-covariance direction (a flat patch, $\lambda_i \approx 0$) would be divided by almost zero and the corresponding filter norm would blow up, amplifying what is essentially sensor noise into a signal the network would then have to learn to ignore. Adding $\varepsilon$ to the eigenvalues before the inverse square root regularizes the smallest eigenvalues and is the one knob I tune; I use $\varepsilon = 5\times10^{-4}$. Second, the sign symmetry: the layer is followed by a GELU, which is asymmetric and suppresses negative pre-activations, so if I emit only the whitened responses then half the information — the sign of each whitened coordinate — gets clipped away. The fix is to also emit the *negation* of each whitened filter, so the layer produces both $+v\cdot x$ and $-v\cdot x$ for every eigen-direction and whichever sign survives the GELU carries the information through. That is why the layer has $2\times3\times k^2$ outputs for a $k\times k$ kernel — for $k=2$, a width of 24 — the factor of two being the positive/negative pair. The kernel weights themselves never train (`requires_grad = False`), but I do add a *learnable bias* to this otherwise-frozen conv: the whitening fixes the linear transform, yet the network can still benefit from learning where to threshold the whitened responses before the GELU. I train that bias for a few epochs and then freeze its gradient too, which costs nothing in accuracy and speeds up the backward pass once the gradient no longer has to flow back through the first layer.

To build the initialization I gather patches with an `unfold` over the spatial dims, form the empirical covariance $X^\top X / n$, eigendecompose it with `torch.linalg.eigh`, scale the eigenvectors by $1/\sqrt{\lambda + \varepsilon}$, and write `cat(scaled, −scaled)` into the frozen conv weight. The covariance is estimated from only the first 5000 training images — it is a second-order statistic and converges fast — and the whole thing is a single eigendecomposition, microseconds of A100 time, done once before the training loop really begins. Because a whitened input has identity patch-covariance, the optimization is well-conditioned from step one: the network no longer spends epochs rotating its first layer into a decorrelating basis and can start learning class structure immediately. This attacks the conditioning of the input directly rather than shaving overhead at the margins, so I expect it to be the single largest speedup of anything I can add — more than halving the epoch count while accuracy holds at the bar.

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

# in make_net(): frozen, doubled-width first conv, no padding, learnable bias
whiten_kernel_size = 2
whiten_width = 2 * 3 * whiten_kernel_size**2   # = 24
net = nn.Sequential(
    Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
    nn.GELU(),
    ...
)
net[0].weight.requires_grad = False

# in main(): initialize from the first 5000 training images, before the training loop
train_images = train_loader.normalize(train_loader.images[:5000])
init_whitening_conv(model[0], train_images)
```
