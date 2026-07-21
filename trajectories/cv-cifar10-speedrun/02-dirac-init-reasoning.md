Whitening the first conv cut me from 45 epochs to 21, from 18.3 to 8.0 A100-seconds, and the numbers
tell me *how* the win arrived. The epoch count fell by 2.14×, the wall-clock by 2.29× — not quite the
same, and the small discrepancy is informative: per-epoch cost went from 18.3/45 = 0.407 s to
8.0/21 = 0.381 s, about 6–7% cheaper. That 7% is exactly the saving I expected from freezing the first
layer — once its weights and bias stop taking gradient the backward pass has slightly less to do. So the
decomposition is clean: ~93% of the whitening speedup was *removing epochs* (conditioning) and ~7% was
*cheaper epochs* (frozen backward). The lesson for what comes next: the remaining 8.0 seconds is still
overwhelmingly epoch-bound — 21 epochs at 0.381 s, and at ~49 steps per epoch that is only about a
thousand optimizer steps — so the next big cut again has to come from *epochs I don't need to run*, not
from a faster kernel.

The whitening win was concentrated at the input layer because that is where I looked. Everything *after*
the first conv is still randomly initialized, and I suspect the same waste is hiding in the deeper
layers wearing a different disguise. Standard Kaiming/Xavier init sets each of those six deep convs to
small random weights scaled so activation variance is preserved through the layer. Preserving variance
keeps signals from exploding or vanishing, but it is a statement about *magnitudes*, not *content*: a
variance-preserving random init does not preserve the signal, it *rotates* it. A Kaiming 3×3 conv from C
to C channels implements, at step zero, a near-random orthogonal-ish map that sends the input into a
random basis while keeping its norm. So every deep layer is a random scrambling of its input. The clean,
whitened signal the first conv worked to present gets mangled by the second layer's random rotation,
then the third, then the fourth — six random rotations stacked before anything reaches the head — and
the network then has to *un-scramble* them, learning layer by layer to first pass information *through*
before it can transform it usefully. Random deep init does not merely fail to help; it actively destroys
the input structure and forces the network to rebuild a path from scratch.

How far is random init from a passthrough? The identity operation for a 3×3 conv is a very specific,
very *sparse* tensor: a kernel that is 1 at the center tap of each diagonal input→output channel and 0
elsewhere — exactly C nonzero entries out of 9C² weights. A Kaiming random init has essentially none of
its mass there; it is an O(1)-operator-norm distance from that sparse target in 9C²-dimensional weight
space. So each deep layer starts a full random rotation away from "do nothing," and gradient descent
must walk it back toward the sparse target before the layer can be useful — six independent
walks-toward-identity in the early epochs, none about the labels. That is the deep-layer analogue of the
input-decorrelation overhead whitening just removed.

What I want is the opposite of a random scramble: each deep conv should *pass its input through
unchanged* at step zero, so the whitened signal propagates to the head untouched and gradient descent
only has to learn the *deviations* from identity that help. This is the residual intuition (start near
identity, learn the correction), and there is more than one way to make a stack start near identity.
Explicit skip connections — output = x + F(x) — make "do nothing" the structural default but change the
architecture (an add, a stored activation, a second path per block). A data-dependent init like LSUV or
orthogonal init fixes the *scale* layer by layer but leaves each layer a rotation, not a passthrough, so
it addresses the magnitude problem BatchNorm already handles here, not the un-scrambling one. The third
option is to bake the identity directly into the conv weights at init and change nothing about the
architecture. In a training that is now only ~21 epochs and racing to get shorter, the weights never
travel far from their init, so the *structural* insurance a permanent skip provides against wandering
into a badly-conditioned configuration is insurance I am unlikely to need. The cheaper, more surgical
move is to fix the *starting point* and leave the architecture alone — a one-line change to the init,
free at run time. So: Dirac init on the conv weights, no architectural skip.

A 3×3 conv represents the identity exactly via the Dirac-delta kernel — 1 at the center tap of the
diagonal channel, 0 elsewhere — which copies each input channel straight to the matching output channel,
and `torch.nn.init.dirac_` writes exactly that; with `padding='same'` the spatial size is preserved, so
the output equals the input pixel-for-pixel on the copied channels, a true identity at step zero. The
*block* is not the identity even when its conv is, and I should not pretend otherwise: after the Dirac
conv, BatchNorm (scale frozen at 1) subtracts the running mean and divides by the running std, an affine
rescaling that leaves an already-whitened unit-variance signal close to itself; then GELU is a genuine
nonlinearity, so the block as a whole is not identity. My claim is narrower and it is the one that
matters: with a random conv the signal arriving at each GELU is a *scrambled* version of the whitened
input, so the nonlinearity fires on garbage and the network must first learn to un-scramble; with Dirac
the signal arriving is the *clean* whitened input, so the nonlinearity fires on real features from step
one and the network only has to learn how to bend them. The GELU is supposed to be there — it is the
source of the expressive power — the point is only that it should be applied to clean signal, not noise,
at init, and the same holds across the `MaxPool2d(2)` in each block.

The scrambling I am removing compounds with depth. If the whitened first conv places a
class-discriminating edge feature into some channel at unit strength, a random 3×3 conv into a same-width
layer spreads it: the output along any channel is a random combination of all C input channels, so the
fraction of that channel's variance coming from my edge feature is ~1/C — buried under the other C−1
directions. Two random layers spread it across C² combinations, three across C³. The feature is not
destroyed (a rotation is invertible) but *diffused*, and un-diffusing it is precisely the early-epoch
work I want to delete. Dirac keeps the edge feature sitting in its own channel at unit strength through
every layer — 1/1, not 1/C — so nothing has to be re-concentrated before learning can begin.

This is also why "fewer epochs" should follow rather than just "a nicer init." With random deep layers,
the logits at step zero are a scrambled projection of a scrambled signal — essentially noise, the
cross-entropy near its maximum (ln 10 ≈ 2.30, softened by the 0.2 label smoothing) — and the first
gradients are dominated by the need to un-scramble rather than by any class signal. With Dirac the
whitened input propagates cleanly to the flattened head, so the step-zero logits are already a
structured linear readout of real features: the starting loss is lower and, more importantly, the
starting gradient points at class-relevant directions from step one instead of at
passthrough-reconstruction directions. The optimizer spends its ~1,000-step budget on classification
from the start rather than burning the first few hundred steps re-establishing a signal path.

The identity only makes literal sense when outputs ≥ inputs, so I check the stack: 24→64 (24 identity
filters, 40 free), 64→64 (all 64 identity), 64→128 (64 identity, 64 free), 128→128 (all identity),
128→256 (128 identity, 128 free), 256→256 (all identity). Every layer has N ≥ M, so the partial-identity
init is well-defined everywhere, widening layers getting passthrough plus free capacity and same-width
layers a pure identity. The slice `w[:w.size(1)]` is the first `in_channels` output filters; `dirac_`
writes the identity into them and the rest keep their Kaiming values.

```python
class Conv(nn.Conv2d):
    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])
```

The first conv is excluded explicitly, and not arbitrarily: it is frozen, its *job* is decorrelation
rather than passthrough (I want it to actively transform into a whitened basis, the opposite of copying
through), and it is a 2×2 stride-shrinking 3→24 conv, so "identity via the center tap" is not even
defined for it. So the rule is precisely "Dirac on every conv *after* the first," and the two inits do
complementary jobs on adjacent layers: the first conv transforms, everything after it passes through,
and gradient descent learns the deviations from that starting configuration. I use a *pure* identity on
the first M filters, not a blend α·identity + (1−α)·random: mixing in a fraction (1−α) of random weight
would re-inject exactly the scramble I am removing, at reduced amplitude, so the signal-to-scramble ratio
at each layer's output would only be ~1:1 and compound over six layers. Pure passthrough on the first M,
pure freedom on the rest — no admixture. The `bias.data.zero_()` is the same idea: a nonzero bias would
shift the passed-through signal off its operating point before the GELU.

I expect this to be real but smaller than whitening. Whitening attacked the *input*, where the
redundancy was largest (κ in the hundreds) and the conditioning worst, so it bought the 45 → 21 jump.
Dirac attacks the *deeper* layers, and by the time the signal reaches them it has already been whitened
once — so the random-scrambling waste is real but operating on a signal that is no longer pathologically
ill-conditioned, a smaller fraction of the remaining work. So epochs-to-94% should drop below 21 by a
handful rather than another halving, and the two fixes should *compose* — whitening conditions the
input, Dirac keeps that conditioning intact downstream, sequential stages of one signal path. Per-epoch
cost should be flat, since Dirac is an initialization and adds nothing to any forward or backward step:
unlike whitening's 7% per-epoch dividend from freezing a layer, this change's whole effect should land in
the epoch column. If instead the widening layers' free capacity was quietly covering for the missing
passthrough, the gain would be slim and the epoch count would barely move. The one-line change is
`dirac_` on the first M filters of every conv after the first; the code is in the answer.
