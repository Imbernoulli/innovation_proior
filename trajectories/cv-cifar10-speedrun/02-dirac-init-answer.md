**Problem (from step 1).** Whitening fixed the input layer (21 epochs / 8.0 s), but every deep 3×3 conv is
still randomly initialized. At step zero each deep layer is a random scramble of its input, so the clean
whitened signal gets mangled and the network spends early epochs relearning, layer by layer, to pass
information *through* before it can learn to transform it usefully.

**Key idea.** Initialize every convolution *after the first* as a partial identity (Dirac) transform. A 3×3
kernel that is 1 at the center diagonal tap copies each input channel straight to the matching output
channel. For a conv with M inputs and N ≥ M outputs, set the first M filters to this identity and leave the
remaining N−M filters at their default random init. The whitened first conv is left alone (it is frozen and
doing its own job). Information now passes through on init; gradient descent only has to learn the
*deviation* from identity — the residual-network intuition baked into initialization instead of an added
skip path.

**Why it works.** A near-identity deep stack preserves the well-conditioned whitened signal all the way to
the head, so no epochs are wasted reconstructing a passthrough path. It composes with whitening rather than
fighting it: whitening conditions the input, Dirac keeps it intact downstream. A *pure* identity on the
first M filters (no random admixture, unlike earlier mixed-init schemes) is preferred — any blended-in
random weight would partially re-scramble the signal being preserved. In PyTorch this is exactly
`torch.nn.init.dirac_(w[:w.size(1)])`.

**Change / code.** One line in the conv's `reset_parameters`, applied to all convs except the frozen
whitening layer.

```python
class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])   # first `in_channels` filters = identity passthrough
```
