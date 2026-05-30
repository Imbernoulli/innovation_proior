Let me start from what's actually going wrong, because the symptoms are strange. I'm training a feed-forward generator to stylize images in one pass — the amortized version of Gatys' slow optimization, the way Texture Nets and Johnson's network do it: a conv/pool/upsample/batch-norm net g(x, z) trained on a pile of content images to minimize the perceptual loss L(x_0, x_t, g(x_t, z_t)), Gram-matrix style loss from shallow VGG layers plus content loss from deep ones. It's fast, but the pictures are visibly worse than Gatys, and two things about the failure are weird.

First: more training data makes it *worse*. A generator trained on sixteen images looks better than one trained on thousands; the best results come from few images and stopping early. That's backwards from everything I expect of a neural net — more data should help. Second: the ugliest artifacts sit right on the *border* of the image, where I zero-pad before each convolution, and no amount of cleverer padding fixes them. Both smell like the same thing — the network is being asked to learn a function it's structurally bad at, so it overfits a few examples (hence few-images-is-better, early-stopping-helps) and falls apart at the boundary where the padding lies to it.

So instead of fiddling with data or padding, let me ask what function the generator is actually being forced to learn, and whether part of it is something a conv stack is just the wrong tool for.

Here's a property of stylization I can state independently of any network. Look at the style loss: it transfers the *style* image's statistics onto the content, and one of those statistics is contrast. If I take a content photo and a low-contrast copy of the same photo and stylize both, I get essentially the *same* stylized image — the output contrast is set by the style, not the content. So the correct generator must, somewhere inside it, *throw away the content image's contrast*. Its output has to be invariant to how bright-or-flat the input was.

Now, is "discard the input's contrast" an easy thing for a conv/ReLU/pool stack to compute? Write down the simplest version of contrast normalization: for each image t and channel i, divide every pixel by the spatial sum of that channel,

  y_tijk = x_tijk / (Σ_l Σ_m x_tilm).

That's a per-image global division — every spatial location's output depends on a sum over *all* spatial locations of that image. Convolutions are local and ReLUs are pointwise; to build a global normalization out of them the network has to approximate "sum over the whole image, then divide everywhere by it" through many layers. That's a genuinely awkward, highly nonlinear function to express as conv+ReLU. So this is plausibly *the* hard part of the objective — the generator is burning its capacity (and overfitting) trying to learn a contrast normalization it can't represent cleanly. If that's right, I shouldn't make the net learn it. I should *build it into the architecture*.

And I almost have the right primitive already, because the generator contains a normalization layer: batch norm. Let me write it down for a tensor x ∈ R^{T×C×W×H} (t = image in the batch, i = channel, j,k = spatial). Batch norm normalizes each channel i using statistics pooled over the *whole batch*:

  y_tijk = (x_tijk − μ_i) / √(σ_i² + ε),  μ_i = (1/(HWT)) Σ_t Σ_l Σ_m x_tilm,  σ_i² = (1/(HWT)) Σ_t Σ_l Σ_m (x_tilm − μ_i)².

That subtracts a mean and divides by a standard deviation per channel — structurally it's a normalization, the right shape. But the statistics are pooled over all T images of the batch *and* all spatial positions. So when batch norm processes image t, it doesn't remove image t's *own* contrast — it removes a blend of the whole batch's contrast. Image t's individual contrast survives, mixed in. And at test time it gets worse: batch norm freezes and uses fixed training-set running statistics, so a single test image is normalized by *training-set averages*, not by anything about itself. So batch norm, as is, does not do the one thing I need — strip *this* image's contrast — and the generator is left to learn the residual. That lines up with the failures.

The fix falls right out of the comparison. The contrast I want to remove is per-image. Batch norm averages the moments over the batch index t. So just *don't* average over t — compute the mean and variance for each image separately, over its spatial extent only, per channel:

  y_tijk = (x_tijk − μ_ti) / √(σ_ti² + ε),  μ_ti = (1/(HW)) Σ_l Σ_m x_tilm,  σ_ti² = (1/(HW)) Σ_l Σ_m (x_tilm − μ_ti)².

The only change from batch norm is the subscript: μ_ti, σ_ti instead of μ_i, σ_i — the batch sum is gone. Now each image's each channel is centered and scaled by its *own* spatial statistics. That subtracts the image's own mean and divides by its own spatial standard deviation, per channel — which is exactly a contrast normalization of that single image, the operation Eq. 1 was reaching for, now expressed as a clean per-instance moment instead of an awkward conv approximation. Because it's computed per instance, call it instance normalization. The generator no longer has to learn to discard content contrast — the architecture does it, identically for every image, so there's nothing data-dependent left to overfit, and the few-images-beats-many pathology should dissolve. I keep a per-channel affine γ, β afterward, same as batch norm, so the network can rescale the normalized signal where it needs to.

There's a second decision hiding here, and it's just as important as dropping the batch sum: what happens at *test time*. Batch norm switches behavior between train and test — batch statistics during training, frozen population statistics at inference, and the layer is often folded away entirely. For instance norm I want to do something different: apply the *same* per-instance computation at test time too. Two reasons, and they reinforce each other. The principled one: at inference I stylize a *single* image, and the whole point is to normalize *that image's* contrast — I must compute its own μ_ti, σ_ti, not substitute some training-set average that has nothing to do with it. The structural one: instance norm has no batch dependence and no population statistics to accumulate — its output is a deterministic function of the single input image — so there is simply no "training statistics vs test statistics" distinction to make. Train and test are automatically identical. That consistency is exactly what I want for a generator whose job at test time is to contrast-normalize whatever one image it's handed.

So the recipe is: replace every batch normalization in the generator with instance normalization, and keep it active at test time. One subscript changed, one train/test behavior simplified.

Implementing it is almost free given that a fast batch-norm primitive already exists. Batch norm normalizes per channel across the batch-and-spatial axes; I want to normalize each (image, channel) slice across spatial only. I can borrow the batch-norm machinery by *reshaping*: take the (N, C, H, W) tensor and view it as (1, N·C, H, W) — fold the batch into the channel axis so there are now N·C "channels" and a batch of one. Apply spatial batch normalization over that: with batch size 1, each of the N·C channels is normalized over its own (H, W) — which is precisely per-image, per-channel spatial normalization. Repeat the length-C affine parameters N times to line up with the N·C flattened channels, run it, and view the output back to (N, C, H, W). Same fast kernel, different bookkeeping. (In a modern framework this is just an InstanceNorm2d layer that normalizes each sample's each channel over H×W and does not track running statistics, so it behaves the same in train and eval.)

```python
import torch
import torch.nn as nn


class InstanceNormalization(nn.Module):
    """Per-(image, channel) normalization over spatial extent, identical at
    train and test. Realized by folding batch into channels and reusing the
    batch-norm kernel: (N, C, H, W) -> (1, N*C, H, W)."""

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))    # gamma, per channel
            self.bias = nn.Parameter(torch.zeros(num_features))     # beta, per channel

    def forward(self, x):
        n, c, h, w = x.size()
        x_flat = x.contiguous().view(1, n * c, h, w)        # fold batch into channels
        weight = self.weight.repeat(n) if self.affine else None   # tile affine N times
        bias = self.bias.repeat(n) if self.affine else None
        # batch size 1 + N*C channels => each (image,channel) normalized over (h,w);
        # training=True always, so no running stats -> train == test
        out = nn.functional.batch_norm(
            x_flat, running_mean=None, running_var=None,
            weight=weight, bias=bias, training=True, eps=self.eps)
        return out.view(n, c, h, w)


# In the generator, swap every batch-norm for InstanceNormalization and keep it
# active at test time:
#   conv -> InstanceNormalization(C) -> ReLU -> ...
# Equivalent in a modern framework: nn.InstanceNorm2d(C, eps=1e-5, affine=True),
# which normalizes each sample's each channel over H,W and tracks no running stats.
```

The chain: a feed-forward stylizer was learning worse from more data and breaking at the borders, as if forced to learn a function it's bad at. Stylization output is independent of the content image's contrast, so the generator must discard that contrast — a per-image global normalization that conv/ReLU/pool can't express cleanly, so it was being learned and overfit. Batch norm is almost the right primitive but pools its statistics across the batch and uses frozen population statistics at test, so it never removes a single image's own contrast. Dropping the batch index from the moments — normalizing each image's each channel over its own spatial extent (μ_ti, σ_ti) — bakes the contrast normalization into the architecture, and keeping that same per-instance computation at test time makes train and test identical, which is exactly what single-image inference needs. Implemented by reshaping (N,C,H,W)→(1,N·C,H,W) and reusing the batch-norm kernel, it is a one-line change that swaps batch normalization for instance normalization everywhere in the generator.
