BlurPool and label smoothing each added accuracy at little or no throughput cost, and the whole reason I wanted that headroom was to *spend* it on speed. So now I go after the biggest throughput lever I can find and accept that it may cost a little accuracy, because I have some to give back. Where does the compute go in a ResNet-50 step? It is dominated by convolutions, and the cost of every convolution scales with the spatial size of the feature maps — roughly with $H\cdot W$. Feed the network a $224\times224$ image and every conv runs over maps proportional to that area; feed it $112\times112$ and the maps are a quarter the area, so the convolutions are about $4\times$ cheaper. Compute per step is, to first order, proportional to the input resolution squared — an enormous knob, and right now I pay full price for it on every single step from the very first. Is that full price worth it on every step? Think about what the network learns early versus late. At the start the weights are near-random and the network is learning coarse, low-frequency structure — gross shapes, dominant colors, rough textures. None of that needs fine pixel detail; a downsampled image carries the coarse structure perfectly. The fine, high-frequency detail — the texture distinguishing two similar dog breeds, the small diagnostic part of an object — matters for the *last* bit of accuracy, and the network cannot exploit it until late in training, once the coarse features are in place. So I am paying for full-resolution detail during exactly the phase when the network cannot yet use it.

The method I propose is **Progressive Image Resizing**: train on smaller images early, when coarse features are being learned, and grow the resolution back to full size by the time the network is ready to use fine detail. The schedule has three phases. A first phase holds at a reduced scale — each side $\times 0.5$, so $112\times112$, roughly a $4\times$ cheaper conv cost. Then a ramp grows the image size linearly back toward full. Then a final fine-tuning phase runs at full $224\times224$, so the network finishes seeing exactly the images it will be evaluated on. The knobs are an initial scale (start at $0.5$ of each side), a delay fraction (how long to hold at the initial size before ramping, keeping the first chunk of training cheap), a finetune fraction (reserve the last chunk for full size), and a size increment so the chosen sizes snap to sensible multiples. For ResNet-50 on ImageNet the settings that land are `initial_scale=0.5`, `delay_fraction=0.5`, `finetune_fraction=0.2`, `size_increment=4`, `mode='resize'`.

The resize operation itself matters. On ImageNet the right move is *downsampling by interpolation*, not cropping. Cropping would change which part of the image the network sees and throw away content — bad when scale is variable and the whole image matters, as on ImageNet, where objects appear at many sizes and you need all of the image each time. Bilinear/nearest interpolation to a smaller size keeps the whole image at lower resolution: the coarse structure is preserved and only the fine detail is dropped, which is exactly the detail the early network cannot use anyway. (For a dataset like CIFAR, where objects are uniform in scale and already low-res, cropping is better — but that is not this task.) The resize is cheap and goes on-device: take the batch of inputs and interpolate them down by the current scale factor right before they hit the model. For pure classification I resize only the inputs, not the targets — the label is a class index, unaffected by image size.

Now the cost I am accepting, honestly. Smaller images carry less information, so training on them can slightly hurt final accuracy for the same number of steps, because the network spends much of training never seeing full detail. But this still wins, and it belongs right after I banked headroom, because it speeds up each early step so much that I can afford to *train for more steps* to recover the accuracy and still finish in less wall-clock time than the full-resolution baseline. The throughput gain during the small-image phase is large enough to buy back the steps, and the headroom from BlurPool plus label smoothing absorbs the residual dip — what is left is a big reduction in time-to-target. Two interaction risks I keep in mind. First, this changes input size, so it composes with any other image-size-changing method (ColOut, CutOut, selective backprop with downsampling) with diminishing returns — none are in the stack yet, but I will remember when I add them. Second, and this one is a real gotcha: by making the early steps so cheap on the GPU, I may shift the bottleneck *off* the GPU and *onto* the data loader or CPU image processing, so the speedup during the small-image phase is capped by data loading. That flags the data pipeline as the thing a later rung will have to balance. The core is the batch-resize below: downscale the input batch by the scheduled scale factor before the forward pass, via a nearest/bilinear interpolation transform, short-circuiting when the factor is $\geq 1$.

```python
def resize_batch(input, target, scale_factor, mode='resize', resize_targets=False):
    assert input.dim() > 2, 'Input dimensionality not large enough for resizing'
    # Short-circuit if nothing should be done
    if scale_factor >= 1:
        return input, target
    ...
    if mode.lower() == 'crop' and resize_targets is False:
        resize_transform = _make_crop(tensor=input, scale_factor=scale_factor)
        X_sized, y_sized = resize_transform(input), target
    elif mode.lower() == 'resize':
        resize_transform = _make_resize(scale_factor=scale_factor)
        X_sized = resize_transform(input)
        if resize_targets:
            y_sized = resize_transform(y_sized)
    ...
    return X_sized, y_sized

def _make_resize(scale_factor: float):
    """Makes an interpolation transform at the specified scale factor."""
    return partial(F.interpolate, scale_factor=scale_factor, mode='nearest',
                   recompute_scale_factor=False)
```
