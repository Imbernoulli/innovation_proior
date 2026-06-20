**Problem (from step 4).** A ResNet-50 step is dominated by convolutions whose cost scales with feature-map
area (≈ H·W), yet the recipe pays full 224×224 price from step zero — exactly when the early network is
learning only coarse, low-frequency structure that a downsampled image carries fine. Full-resolution detail
is wasted early. Time to spend the banked accuracy headroom on the biggest throughput lever.

**Key idea — Progressive Resizing.** Train on smaller images early and grow them back to full size by a set
point in training. Three phases: hold at an initial reduced scale (each side ×0.5 → ~4× cheaper convs),
linearly ramp the size up, then a final fine-tuning phase at full 224×224. Resize by *interpolation* (not
cropping) so the whole image is kept — only fine detail is dropped, which the early network can't use
anyway — and resize on-device, just the inputs (targets are class indices, unaffected).

**Why it works.** Early training learns coarse features that don't need pixel detail; small images make those
cheap steps physically cheap, so the average step cost drops and the schedule finishes much faster in
wall-clock time. It can slightly lower quality *per step*, but the throughput gain lets you train more steps,
recover accuracy, and still finish faster — and the headroom from BlurPool + label smoothing absorbs the
dip. Two flagged interactions: it composes with other input-size-changing methods (ColOut, etc.) with
diminishing returns, and by making the early GPU work cheap it can shift the bottleneck onto the data
loader/CPU (a flag for a later rung). Suggested ResNet-50 config: `initial_scale=0.5`, `finetune_fraction=
0.2`, `delay_fraction=0.5`, `size_increment=4`, `mode='resize'`.

**Change / code.** The core is `resize_batch` — downscale the input batch by the current scale factor before
the forward pass, via a nearest/bilinear interpolation transform.

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
