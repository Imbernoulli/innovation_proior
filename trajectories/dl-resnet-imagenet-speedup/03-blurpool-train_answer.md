Channels Last banked a free throughput win at zero accuracy cost, but throughput alone does not move the frontier — what lets me go fast is *accuracy headroom*, because every extra point of top-1 I can earn is a point I can later spend by shortening the schedule and still landing on 76.6%. So the next move is a quality lever that is nearly free in time. Where is ResNet-50 leaving accuracy on the table for structural reasons I can fix cheaply? The downsampling. A ResNet repeatedly halves spatial resolution — through strided convolutions and the early max-pool — and downsampling is exactly where aliasing happens. There is a basic signal-processing fact here: if you subsample a signal without first removing the high-frequency content the lower sample rate cannot represent, those high frequencies *fold back* and masquerade as low frequencies. A stride-2 conv or a stride-2 max-pool is subsampling the feature map by $2\times$ with no low-pass filter in front, so the high-spatial-frequency content of the activations folds into the downsampled map as garbage. This hurts two ways: it injects aliased noise into the representation, and — more tellingly — it makes the network *not shift-invariant*. Shift the input by one pixel and the aliased strided ops can change the downsampled maps by far more than one pixel's worth. For natural images, where a one-pixel jitter is meaningless, that brittleness is pure lost accuracy.

The fix I propose is **BlurPool**, and it is textbook anti-aliasing: low-pass filter *before* you subsample. The Nyquist story says blur away the frequencies above the new sample rate's limit, then downsample, and the fold-back cannot happen. So I insert a small spatial low-pass filter in front of every downsampling operation. For ResNet-50's two kinds of downsampling this takes two concrete forms. For the stem max-pool: keep the window-max (that is the nonlinearity I want) but *blur before actually reducing the spatial size* — decouple "take the max over a window" from "throw away spatial positions" and slip a low-pass filter between them. For strided convs: a stride-2 conv is a conv followed by subsampling, so I replace it with "blur the input first, then run the original convolution," which anti-aliases the downsample while leaving the conv's multiply-adds untouched.

The filter is fixed, not learned — a small, cheap, separable binomial blur, the discrete approximation to a Gaussian, applied per-channel as a depthwise convolution with the same kernel shared across all channels:

$$\frac{1}{16}\begin{bmatrix}1 & 2 & 1\\ 2 & 4 & 2\\ 1 & 2 & 1\end{bmatrix},$$

the outer product of $[1,2,1]$ with itself, normalized to sum to one. It is the smallest symmetric low-pass that meaningfully attenuates the highest spatial frequencies, adds no learnable parameters, and with "same" padding sits cleanly in front of the existing stride.

Two design choices are load-bearing. The first is the order for strided convs: blur *first*, not after. If I blur first, the original stride-2 conv still operates at its original cost and I have added only the cheap blur. If I blur *after* the conv, the conv runs at the higher pre-downsample resolution and its multiply-adds are paid at full size, multiplying the conv cost by the stride area ($4\times$ for stride 2). Blur-after more closely matches the original anti-aliasing formulation and yields roughly a $0.1\%$ larger accuracy gain on ResNet-50/ImageNet, but in exchange for about a $10\%$ slowdown — not the trade I want this early, so I use `blur_first=True`. The second is a guard rail: do *not* blur the input-facing strided conv. Blurpooling a conv that sees the raw image amounts to downsampling the input itself by more than my preprocessing intended, throwing away detail I meant to keep. The clean heuristic is to skip strided convs with fewer than $\sim 16$ input channels, which in practice means leaving the stem's input conv alone.

I get this into the model by surgery: walk the module tree and replace every `MaxPool2d` with a blur-aware max-pool and every qualifying strided `Conv2d` with a blur-then-conv module. The cost is a little throughput — a depthwise blur added in front of each downsample, roughly doubling the max-pool data movement, but with blur-first the conv math is unchanged so the overhead is small. The benefit is the accuracy I came for: anti-aliasing improves shift-invariance and cleans the representation, buying $+0.5$ to $1\%$ top-1 on ImageNet across networks (Zhang, 2019) and, on ResNet-50/ImageNet, a pareto improvement — a real accuracy bump for a small throughput cost. It is orthogonal to channels-last (memory layout) and the optimizer, so it composes cleanly, and that accuracy bump is schedule-shortening fuel for later.

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
