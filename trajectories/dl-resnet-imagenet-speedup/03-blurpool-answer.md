**Problem (from step 2).** ResNet-50 repeatedly downsamples (strided convs, the stem max-pool) with no
low-pass filter in front, so high spatial frequencies *alias* — folding back as noise and making the
network needlessly brittle to one-pixel shifts. For natural images that brittleness is lost accuracy.
Having banked a free throughput win, the next move is cheap accuracy *headroom* (which later pays for a
shorter schedule).

**Key idea — BlurPool.** Anti-alias every downsample: low-pass filter *before* subsampling. Replace each
strided `Conv2d` with "blur-then-conv" and each `MaxPool2d` with "take the window-max, blur, then reduce."
The filter is a fixed, non-learnable binomial (Gaussian-approximation) kernel `(1/16)[[1,2,1],[2,4,2],
[1,2,1]]`, applied per-channel as a depthwise conv with "same" padding. Blur *first* (not after the conv)
so the conv's multiply-adds stay at their original count and only the cheap blur is added. Skip the
input-facing strided conv (fewer than ~16 input channels) so the raw image isn't extra-downsampled.

**Why it works.** Subsampling without a low-pass filter aliases high frequencies into the coarse map
(Nyquist); blurring first removes the content the lower rate can't represent, so the fold-back can't happen.
This improves shift-invariance and cleans the representation, buying a top-1 gain on ImageNet for a small
throughput cost — a pareto improvement on ResNet-50 / ImageNet. Orthogonal to channels-last (memory layout)
and the optimizer, so it composes cleanly.

**Change / code.** Model surgery replaces strided convs and max-pools with blur-first equivalents; the
core is the fixed 2D blur conv and the policy that only rewrites strided convs with enough input channels.

```python
def _default_2d_filter():
    default_filter = torch.tensor([[[
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ]]]) * 1 / 16.0
    return default_filter

def blur_2d(input, channels=-1, stride=1, filter=None):
    if filter is None:
        filter = _default_2d_filter()
    padding = _padding_for_filt_2d_same(filter)
    if channels < 1:  # depthwise: one filter per channel
        _, channels, h, w = input.shape
        if (filter.shape[0] == 1) and (channels > 1):
            filter = filter.repeat((channels, 1, 1, 1))
    return F.conv2d(input, filter, None, _pair(stride), _pair(padding), _pair(1), channels)

def apply_blurpool(model, replace_convs=True, replace_maxpools=True,
                   blur_first=True, min_channels=16, optimizers=None):
    transforms = {}
    if replace_maxpools:
        transforms[torch.nn.MaxPool2d] = BlurMaxPool2d.from_maxpool2d
    if replace_convs:
        transforms[torch.nn.Conv2d] = functools.partial(
            _maybe_replace_strided_conv2d, blur_first=blur_first, min_channels=min_channels)
    module_surgery.replace_module_classes(model, optimizers=optimizers, policies=transforms)

def _maybe_replace_strided_conv2d(module, module_index, blur_first, min_channels=16):
    # Only blurpool strided convs with enough input channels (skip the stem input conv)
    if np.max(module.stride) > 1 and module.in_channels >= min_channels:
        return BlurConv2d.from_conv2d(module, module_index, blur_first=blur_first)
    return None
```
