## Research Question

Large decoder-only Transformers are built by repeating the same pattern: a residual stream carries the
model state through attention and feed-forward sublayers, and normalization is placed either before or
after each residual addition. The residual path is the reason very deep stacks can train, but it also fixes
how strongly each layer's input and output are connected. The question is whether that connection rule
can be made learnable and input-aware while leaving attention, the feed-forward network, embeddings,
and the output head unchanged.

## The Residual Highway

The ordinary Pre-Norm block can be written as `h_k = h_{k-1} + T_k(Norm(h_{k-1}))`. Its appeal is the
identity route: a signal can move forward and a gradient can move backward through many layers without
being repeatedly transformed. That same identity route is rigid. Every block preserves the previous
state with coefficient one and adds the new branch output with coefficient one, regardless of layer,
token, or the current state of the representation.

Post-Norm changes the placement: `h_k = Norm(h_{k-1} + T_k(h_{k-1}))`. Now the combined state is
renormalized after each addition, so a new layer output does not become small merely because the
unnormalized residual stream has grown. But the residual highway is no longer a clean identity route;
normalization sits on the path that gradients must repeatedly cross.

## The Normalization Seesaw

Pre-Norm and Post-Norm therefore solve opposite halves of the same problem. Pre-Norm keeps gradients
stable, but deeper-layer features tend to become similar because the normalized branch contribution is
added into an accumulated unnormalized stream. Post-Norm keeps layer outputs more balanced in the
forward representation, but its repeated normalization on the highway makes deep optimization harder.

The issue is structural rather than only an initialization accident. A single residual stream gives one
connection pattern per block. Increasing or decreasing a scalar branch weight can move the model along
the trade-off, but it cannot separately preserve a clean long path and give recent branch outputs a
different influence pattern.

## Existing Pressure Points

Prior residual and normalization variants expose the requirements for a better connection rule. Residual
networks show that an identity path is essential for deep optimization. Layer normalization and RMSNorm
show that scale control is needed, but normalization placement changes the gradient route. Transformer
training analyses identify the Pre-Norm/Post-Norm trade-off as a stability-versus-collapse problem.
Two-stream or widened-state variants show that extra state can be useful, but the extra state must remain
cheap enough for billion-parameter pretraining and must not require changing the sublayer definitions.

Parallel transformer blocks and two-hop residual variants add another clue: useful depth structure is not
always purely sequential. Sometimes attention and feed-forward outputs behave as if they should bypass
or run beside neighboring computations. A replacement residual rule should be able to express these
arrangements without hard-coding one fixed layout.

## Replacement Constraints

The desired rule has to be a drop-in replacement at each residual site. It must accept the same sublayer
`T`, provide a single input vector to that sublayer, return a hidden state of the same model dimension to
the surrounding Transformer, and reduce to the ordinary Pre-Norm behavior at initialization so training
does not start from an unfamiliar architecture.

It also has to keep overhead negligible. The new connection parameters should be tiny compared with
attention and feed-forward weights; the additional computation should be a small mixing operation rather
than another large projection; and any extra activations should be manageable by recomputation. Finally,
the rule must let connection strengths vary across depth, and preferably across tokens, because the best
way to route information through a deep stack need not be the same for every layer or input.
