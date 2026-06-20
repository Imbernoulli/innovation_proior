**Problem (from baseline).** The unwhitened baseline reaches 94% but needs 45 epochs / 18.3 A100-seconds,
because the raw 3-channel CIFAR-10 input is highly redundant (correlated channels, correlated neighboring
pixels) and the first convolution must *learn* to decorrelate it by gradient descent — wasting the early
epochs on a label-independent rotation that is identical for every run.

**Key idea.** Whiten the input *for free* by folding a frozen patch-whitening transform into the first
convolution. Estimate the covariance of 2×2×3 image patches from 5000 training images, eigendecompose it,
and set the first conv's (frozen) weights to the eigenvectors scaled by 1/√(λ+ε) — the whitening map *is* a
convolution. Emit both the filters and their negation, so the asymmetric GELU after it cannot clip away the
sign of any whitened coordinate. Add a learnable bias on this conv, trained for a few epochs then frozen to
speed up the backward pass.

**Why it works.** A whitened input has identity patch-covariance, so the optimization is well-conditioned
from step one: the network no longer spends epochs rotating its first layer into a decorrelating basis and
can start learning class structure immediately. It is the single most impactful feature in the speedrun —
it attacks input conditioning directly rather than trimming overhead. The ε floor on the eigenvalues guards
against a singular (flat-patch) direction being divided by ~0 and exploding into amplified noise.

**Change / code.** Frozen first conv `Conv(3, 2·3·k², k, padding=0, bias=True)` with `requires_grad=False`
on the weight; initialized by `init_whitening_conv` below; the bias trains for `whiten_bias_epochs` then
freezes. The whitening estimate uses `unfold` to gather patches, `XᵀX/n` for the covariance, and
`torch.linalg.eigh` for the eigenbasis.

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
