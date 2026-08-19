**Problem.** Three ResNet-block conventions have never been re-derived for
the block this ladder has actually built (narrow -> depthwise 7x7 -> 1x1
expand 4x -> 1x1 project -> narrow): an activation after every convolution,
a normalization before every activation, and BatchNorm specifically as that
normalization. A Transformer sublayer pair normalizes and activates far
more sparingly. Test each convention against the Transformer reference,
one at a time, as five sequential, cumulative sub-steps on the rung-5
(kernel 7x7) baseline, holding the training recipe fixed.

**Changes (against the rung 5 baseline, applied in order, each measured
before the next is added).**

1. ReLU -> GELU (same position, same count, one-line swap).
2. Drop to a single activation per block: remove the activations after the
   depthwise conv and after the projection 1x1, keep only the one between
   the two 1x1s (mirrors the Transformer feedforward sublayer's one
   nonlinearity).
3. Drop to a single normalization per block: remove all but the norm
   immediately before the two 1x1 convs (mirrors pre-norm placement before
   the "heavy lifting" sublayer).
4. Substitute that remaining norm from BatchNorm to LayerNorm.
5. Add separate downsampling: pull resolution change out of the block
   entirely into a standalone 2x2 stride-2 convolution between stages, and
   bracket every point where resolution changes abruptly with a
   normalization layer — before each new downsampling conv, right after
   the stem, and right after the final global pooling.

```python
# rung 6: micro design — activation count, norm count, norm kind, separate downsampling.
import torch
import torch.nn as nn
import torch.nn.functional as F

class MicroDesignBlock(nn.Module):
    """narrow -> depthwise(7) -> norm -> 1x1 expand 4x -> act -> 1x1 project -> narrow.
    Sub-step 1-2 use GELU and a single activation; sub-step 3-4 collapse to a
    single norm and swap it to LayerNorm; downsampling (sub-step 5) is pulled
    out of this block entirely -- see SeparateDownsample below.
    """
    expand_ratio = 4

    def __init__(self, dim, norm_layer, act_layer=nn.GELU):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm = norm_layer(dim)                      # single norm, after dw, before the 1x1 pair
        hidden = dim * self.expand_ratio
        self.pw_expand = nn.Conv2d(dim, hidden, 1)
        self.act = act_layer()                            # single activation, between the two 1x1s
        self.pw_project = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        identity = x
        out = self.dw(x)                                  # no activation, no norm here (sub-steps 2-3)
        out = self.norm(out)
        out = self.pw_expand(out)
        out = self.act(out)                                # the one nonlinearity in the block
        out = self.pw_project(out)                          # no activation here either (sub-step 2)
        return out + identity                                # no stride/downsample handled in-block

class SeparateDownsample(nn.Module):
    """Standalone 2x2 stride-2 conv between stages, bracketed by a norm on the
    reduced-resolution side -- the fix for training the un-normalized boundary
    that in-block downsampling used to cover implicitly."""
    def __init__(self, dim_in, dim_out, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim_in)
        self.reduce = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)

    def forward(self, x):
        return self.reduce(self.norm(x))

# LayerNorm placed after the stem and after the final global pool too,
# bracketing every abrupt resolution/statistics change in the network.
```

**Test.** Five sequential measured points against the rung-5 (kernel 7x7)
baseline: GELU swap; single activation; single normalization; BN->LN;
separate downsampling with bracketing norms. Expect the first sub-step
close to flat; the middle three (single activation, single norm, BN->LN)
to carry the real weight, with the BN->LN sign genuinely uncertain given
the documented precedent against naive LN substitution in an unmodified
ResNet; the last (separate downsampling) most likely to either help
meaningfully or destabilize training, depending on whether the bracketing
normalization is sufficient. Whatever the cumulative sequence lands on is
the architecture this exploration commits to.
