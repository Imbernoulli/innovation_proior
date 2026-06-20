**Problem (from step 1).** The optimizer is now honest, but the floor recipe is GPU-bound on convolutions
and PyTorch stores activations in NCHW, while NVIDIA tensor cores run convolutions natively in NHWC. So
every `Conv2d` in ResNet-50 pays a transpose to NHWC and back — pure overhead, repeated across dozens of
convs, every forward and backward. Need a speedup that costs no accuracy and no regularization budget.

**Key idea.** Change only the *memory format* of the model's tensors to channels-last (NHWC), not their
logical shape. Cast the whole model to `torch.channels_last` at the start of training; the first conv
converts the first activation to match, and because channels-last-aware ops preserve their input's format,
NHWC then persists through every activation and gradient for the rest of the network. The transposes happen
once at the input instead of around every conv.

**Why it works.** NHWC vs NCHW is purely where bytes sit in memory — the logical tensor, the convolution
computed, and every gradient are bit-for-bit identical. So this is a systems-level change that does not
alter the math or the outcome of training in any way: a free throughput win, no accuracy risk, no new
hyperparameter. The conv kernels get the layout the hardware wants without cuDNN transposing first. (The
one caveat: a layer that can't support channels-last forces a convert-back-and-forth around it; a vanilla
ResNet-50 is a clean conv stack and has no such layer, so the format persists end to end.)

**Change / code.** One model-surgery call — convert the model to channels-last memory format. No dataloader
changes; the optimizer, loss, data, schedule, and accuracy are unchanged from step 1.

```python
def apply_channels_last(model: torch.nn.Module) -> None:
    """Changes the memory format of the model to ``torch.channels_last``.

    This usually yields improved GPU utilization.
    """
    model.to(memory_format=torch.channels_last)  # type: ignore
```
