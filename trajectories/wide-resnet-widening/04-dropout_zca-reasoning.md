The grid gives a clear picture once I read across it. At every depth I tested, error keeps falling as
`k` rises across the tested range — 40-layer networks go 6.85 -> 5.33 -> 4.97 -> 4.66 on CIFAR-10 as
`k` climbs 1 -> 2 -> 4 -> 8, no saturation or reversal inside this budget. And at fixed large `k` (8 or
10, the values shared across every depth), the picture is more textured: going from 16 to 22 to 28
layers keeps helping — 16-8 at 4.81 down to 22-8 at 4.38 down to 28-10 at 4.17 — but pushing the same
`k=8`-or-`k=10` family to 40 layers reverses it: 40-8 posts 4.66, worse than 22-8's 4.38 despite being
both deeper and having twice the parameters (35.7M vs 17.2M). So depth still helps *up to a point* even
in the wide regime, then stops helping and starts hurting — the degradation problem thin networks show
at extreme depth has a wide-regime analogue, just at a much shallower absolute depth than the thin
case. The best cell in the whole grid is 28-10: 4.17% CIFAR-10, 20.50% CIFAR-100, at 36.5M parameters.
That is genuinely the comparison I was watching for: a 28-layer network landing at essentially the
same parameter count as the thin 1001-layer, 10.2M-parameter reference is not quite matched — 36.5M
against 10.2M is over three times more parameters, not "comparable." So this cell alone doesn't yet
settle the "depth-to-width ratio is arbitrary" question the way I'd want; 40-4 at 8.9M parameters is
much closer in scale to the 1001-layer reference and is the cell I'd actually want for that specific
argument. But that's a separate comparison from what this rung is about — right now I have a concrete
best-found architecture, 28-10, and a real design question sitting on top of it that the grid
deliberately didn't touch: regularization.

36.5M parameters is a lot — two orders of magnitude more than the thin-baseline end of this same
family (0.6M at 40-1) — and every one of the ten grid cells was trained with only flip-and-crop
augmentation and no regularization beyond whatever batch normalization itself provides. I already have
a directly measured reason not to assume that's enough. On an earlier CIFAR-10 network built
specifically to test whether batch normalization substitutes for dropout, stacking the two together
reached 92.44% accuracy, and removing either one dropped it to 91.4% — a real, already-measured gap,
not a theoretical worry, and it runs against the tidy claim that normalization alone should make
dropout redundant. That result was on a plain VGG-style network, not a residual one, so it doesn't
transfer mechanically to this architecture, but it's enough to keep dropout on the table rather than
assuming batch norm has already closed the question. The natural next move, given a design axis
(widening) that just multiplied parameter count by nearly two orders of magnitude with no added
regularization, is to check whether adding dropout to the best-found configuration helps, hurts, or
does nothing.

Where dropout goes is not a free choice, and I don't have to guess at the failure mode by pure
argument — there's a directly relevant, already-measured result to reason from. On a residual network
of comparable depth to this family's deeper end, dropout at a 0.5 ratio applied on the output of the
identity shortcut failed to converge to a good solution at all — more than 20% test error against a
6.61% baseline on the same architecture. The stated mechanism is specific enough to generalize from:
dropout on the shortcut statistically imposes an expected scale factor (0.5, at that ratio) on the one
path that's supposed to carry the signal through unimpeded, and the more general claim covers scaling,
gating, and even a 1x1 convolution standing in for identity — any multiplicative manipulation on the
shortcut, not dropout specifically, hampers information propagation because the shortcut is the most
direct path signal has through the whole stack, and anything that damps it damps everything
downstream. That rules the shortcut out as a location for dropout categorically, independent of ratio
or of this being a different architecture than the one that result was measured on — the mechanism is
about what a shortcut *is*, not about the specific network it was tested on. So dropout has to live
inside the residual branch, between the block's two convolutions, and the natural place given the
block's own structure (`BN -> ReLU -> conv -> BN -> ReLU -> conv`) is right after the second BN/ReLU
and before the second convolution: it perturbs the residual transformation on its way through, without
ever touching the identity path, and it hands the *next* block's batch normalization a less stale
activation distribution to normalize against than it would see from an un-perturbed branch.

I don't yet know what dropout probability is right for this architecture and this data regime, and I'm
not going to guess a number and hope — I'll cross-validate it rather than import the identity-mapping
result's 0.5 ratio, since that ratio was measured for shortcut placement (where it's already known to
fail) and there's no reason to assume the same number is right for branch placement in a much wider
network. Whatever value cross-validation selects, I'll apply it to exactly the best cell the grid just
found — 28-10 — since that's the concrete configuration whose regularization is actually in question,
rather than re-running the whole grid with dropout added everywhere, which would be a much larger
experiment for a question I can answer more cheaply by checking the one config that matters most
right now.

I genuinely don't know which way this goes. The self-account result argues dropout should help once
there's enough capacity to overfit into, and 36.5M parameters against 50,000 CIFAR-10 training images
is a lot of capacity. But that result was measured on a different architecture with different
normalization interactions, batch normalization already does some regularizing on its own, and dropout
interacting with batch normalization inside a *residual* branch — where the branch output gets added
back to an unperturbed shortcut rather than being the network's only path forward — is a genuinely
different setting than either of the two priors I'm drawing on. So I'm not predicting a clean win; I'm
predicting the placement is right (branch, not shortcut, so it shouldn't catastrophically fail the way
the identity-mappings result did) and leaving the magnitude of the effect, and even its sign, to the
measurement. Test: WRN-28-10 with dropout inserted between the two branch convolutions, cross-validated
probability, same ZCA-preprocessed CIFAR-10/CIFAR-100 protocol as the grid, compared directly against
the no-dropout 28-10 cell already measured.
