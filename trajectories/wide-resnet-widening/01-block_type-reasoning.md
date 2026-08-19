Before touching the width axis at all, I want to settle what a residual block's own convolutions are
buying. The scaffold gives me a two-convolution `BN -> ReLU -> conv -> BN -> ReLU -> conv` basic block
by default, both convolutions `3x3`, same plane count throughout (no bottleneck). That default is
inherited, not derived — it is the block the pre-activation baseline happens to use, and before I
decide how to spend a training budget across depth, block-internal convolutions, and channel width, I
should check whether the block's internal shape is even well-chosen at the granularity I can control
without changing spatial resolution or the group structure.

Three knobs are on the table for increasing a block's representational power: more convolutions per
block, more feature planes per convolution, and larger spatial kernels. I can dismiss the third
immediately and for a reason external to this architecture: VGG and Inception already ran the
larger-vs-stacked-small-filters question to the ground, and stacked `3x3` convolutions came out ahead
of larger single filters at matched receptive field. Reopening that question inside a residual block
would just be re-deriving old evidence with extra steps, so I fix every kernel at `3x3` or `1x1` for
the whole design process and never touch kernel size again after this paragraph.

That leaves convolution count per block and channel width. I don't want to conflate them in one
experiment, because a block that gets more convolutions AND more channels in the same trial tells me
nothing about which change did the work. So I split the question in two, and this rung handles the
first half only: holding the *number* of `3x3`-equivalent convolutions roughly fixed, does the
specific pattern of `3x3` and `1x1` layers inside a block matter? Call a block's kernel-size list `M`
(so the current default is `M = (3,3)`), and write `B(M)` for the resulting block. The basic block
`B(3,3)` spends two full `3x3` convolutions per residual unit. A `1x1` convolution over the same plane
count is roughly `1/9` the FLOPs and parameters of a `3x3` one (no bottleneck reduction — planes stay
equal across the block by construction, so this is purely about spatial extent), so substituting one
`3x3` for a `1x1` frees a real amount of budget per block. The open question is whether that freed
budget is better spent on more blocks (more residual connections, more depth) or whether the `3x3`
being replaced was doing indispensable representational work that a `1x1` cannot cheaply recover.

I don't have a clean a-priori answer, so I want to lay out the candidates rather than guess one and
move on. Beyond the basic `B(3,3)`, the combinatorially reasonable one- and two-`1x1` variants that
keep at least one `3x3` (a block with *zero* `3x3` convolutions can't grow a receptive field per unit
and isn't a fair test of "does the internal pattern matter", it's a different question) are: `B(3,1,3)`
— sandwich a cheap `1x1` between two `3x3`s, effectively a slightly deepened block at modest extra
cost; `B(1,3,1)` — the mirror image, expand-contract with a `1x1` on each side of the single `3x3`,
which is structurally the "straightened" version of the classic bottleneck (same operations, but
without the plane-count reduction that a real bottleneck uses to buy cheapness — here it's still
representational-power-per-block I'm probing, not compression); `B(1,3)` and `B(3,1)` — alternate a
single `1x1` and a single `3x3`, differing only in which comes first; and `B(3,1,1)` — one `3x3`
followed by two `1x1`s, which is close in spirit to a Network-in-Network block bolted onto a residual
unit. Six candidates including the incumbent. I'm not trying to be exhaustive over all possible kernel
strings — I'm trying to cover the natural one-`3x3` and two-with-one-cheap variants that plausibly
trade capacity for cost in either direction, which this list does.

A fair comparison has to control for parameter count, and the six blocks are not equally expensive at
equal depth: a block with only one `3x3` is cheaper than the two-`3x3` basic block, so a network built
from `B(3,1)`-style blocks can afford more of them for the same total budget. I'll widen everything by
the same modest amount (`k=2`) so the comparison isn't happening at the degenerate `k=1` thin regime,
and I'll pick the depth for each block type so that total parameter count lands in a similar
neighborhood — deeper for the cheaper one-`3x3` blocks, shallower for the two-`3x3` blocks — rather
than fixing depth and letting parameter count vary freely, since a variant that "wins" mainly by
having more parameters wouldn't tell me anything about the block shape itself. Concretely: for the
four blocks with only one `3x3` convolution (`B(1,3,1)`, `B(3,1)`, `B(1,3)`, `B(3,1,1)`) I'll use the
same depth so they're directly comparable to each other and to a `k=2` version of the current 40-layer
convention; for the two-`3x3` blocks (`B(3,3)`, `B(3,1,3)`) I'll use shallower networks so their
parameter counts land in the same rough band instead of running far ahead on both convolutions and
depth simultaneously. Training every configuration to convergence multiple times and reporting the
median guards against a single lucky or unlucky seed being mistaken for a structural result — a
one-run comparison across six candidates would risk exactly that kind of noise-driven ranking. I'll
also record wall-clock time per epoch alongside accuracy: even at matched *parameter* count, a block
with more sequential small ops is not the same computational shape as one with fewer, larger ops, and
if I'm about to make an argument later about spending budget on width rather than depth because GPUs
parallelize large tensors better, I want that argument grounded in a real timing measurement from the
very first experiment, not asserted only once at the end.

What I expect going in, stated as a genuine prediction rather than a foregone conclusion: I don't
think the internal kernel pattern is where the big leverage lives. The scaffold's basic block already
encodes the field's converged wisdom (pre-activation, no bottleneck, small filters), so my prior is
that most of these six variants land close together, and if a `1x1`-substituted variant does come out
ahead it will be by a small margin that's more about parameter efficiency than about a real
representational gap. But I don't get to just assert that — it's exactly the kind of claim that needs
a controlled number before I spend the rest of this design process assuming `B(3,3)` and moving on to
depth and width. So: train `B(1,3,1)`, `B(3,1)`, `B(1,3)`, `B(3,1,1)` at one shared depth and `k=2`;
train `B(3,3)` and `B(3,1,3)` at their own matched-parameter depths, also `k=2`; measure CIFAR-10 test
error (median of 5 runs) and per-epoch training time for all six. If the result is close, I fix the
block internals at whatever comes out on top (most likely `B(3,3)`, on prior) and stop spending
experiments on this axis; if one variant clearly dominates, that becomes the new default block for
every rung after this one.
