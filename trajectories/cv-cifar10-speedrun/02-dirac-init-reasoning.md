Whitening the first conv cut me from 45 epochs to 21, from 18.3 to 8.0 seconds — the input is now
well-conditioned. But the win is concentrated at the input layer. Everything *after* the first conv is
still randomly initialized, and I think the same kind of waste I just removed at layer one is hiding in
the deeper layers, only in a different form. Let me look at what the deeper convolutions are doing in
the first few epochs.

The network is a stack: whitened first conv, then three blocks, each block being two or more 3×3 convs
with BatchNorm and GELU. Standard initialization (Kaiming/Xavier) sets each of those 3×3 convs to small
random weights chosen so that activation variance is preserved through the layer. That keeps signals
from exploding or vanishing, which is necessary — but it means that at step zero, *every deep layer is
a random scrambling of its input*. The signal that the whitened first conv worked so hard to present
cleanly gets immediately mangled by a stack of random rotations, and the network has to spend epochs
un-scrambling them: learning, layer by layer, to first approximate passing information *through* before
it can learn to transform it usefully. A deep random init does not preserve the input; it destroys it
and makes the network rebuild a path from scratch.

What I actually want at initialization is the opposite of a random scramble. I want each deep conv to
*pass its input through unchanged* at step zero — to be the identity — so the clean whitened signal
propagates all the way to the head untouched, and then gradient descent only has to learn the
*deviations* from identity that help, instead of first learning identity and then learning the
deviation. This is the same logic as a residual connection (start near identity, learn the correction),
but I can bake it directly into the conv weights without adding a skip path. A 3×3 convolution can
represent the identity exactly: a kernel that is 1 at the center tap of the diagonal input→output
channel and 0 everywhere else copies each input channel straight to the corresponding output channel.
That is a Dirac-delta kernel, and PyTorch has `torch.nn.init.dirac_` for exactly this.

There is a shape wrinkle. The identity only makes literal sense when output channels ≥ input channels —
I can copy each of the M input channels to one of the N ≥ M outputs and leave the remaining N−M outputs
to their default random init (those are the "extra" feature detectors the layer is free to learn). And
it only makes sense for the convs *after* the first: the first conv is the whitening layer, which is
already doing its special job and is frozen, so I leave it alone. For every convolution after the first,
I initialize its first M filters as an identity transform of the input and leave the rest random:

```python
class Conv(nn.Conv2d):
    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])
```

`w[:w.size(1)]` is the slice of the first `in_channels` output filters; `dirac_` writes the
identity-passthrough kernel into them. The rest of the weight tensor keeps its Kaiming values. So each
deep layer starts as "identity on the first M channels, random on the extra ones" — information passes
through on init, and the network only has to learn what to *add*.

This is a cleaner version of an idea that has been tried before with mixed weights — schemes that blend
the identity into a fraction of the random init rather than using a pure identity-on-first-M slice. I
should keep it simple: a pure Dirac on the first M filters, nothing mixed in. The reason to prefer the
simple version is that the whole point is to make the deep layers *transparent* at init, and any
admixture of random weights into the identity path partially re-scrambles the very signal I am trying to
preserve. Pure passthrough on the first M, pure freedom on the rest.

I should be honest about how big a win to expect. The whitening fix attacked the input, where the
redundancy was largest and the conditioning worst, so it bought a huge jump (45→21 epochs). Dirac init
attacks the *deeper* layers, where the signal has already been whitened once and the random scrambling,
while still wasteful, is a smaller fraction of the total work. So I expect a real but more modest
speedup — the deep layers no longer wasting their early epochs relearning identity, the network
converging in meaningfully fewer epochs than 21, but not another halving. The mechanism is the same as
the residual-network intuition (identity is the easy default, learn the correction), applied as an
initialization rather than an architecture change, so I expect it to compose cleanly with whitening
rather than fight it: whitening conditions the input, Dirac preserves it downstream. If it works,
epochs-to-94% should drop again with accuracy held; if the extra-channel random filters end up doing
the heavy lifting anyway, the gain would be small. The one-line change is `dirac_` on the first M
filters of every conv after the first; the code is in the answer.
